#!/usr/bin/env python3
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Conditionally import PyTorch Lightning or newer Lightning package
try:
    import pytorch_lightning as pl
    HAS_LIGHTNING = True
except ImportError:
    try:
        import lightning.pytorch as pl
        HAS_LIGHTNING = True
    except ImportError:
        # Fallback dummy class if Lightning is not installed in the workspace
        class pl:
            class LightningModule:
                pass
        HAS_LIGHTNING = False

# ==============================================================================
# Relaxed Equivariant Graph Neural Network (rEGNN) Layer
# ==============================================================================

class rEGNNLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, m_dim=64):
        """
        Implementation of the relaxed EGNN (rEGNN) layer from the MiDi paper.
        Uses translation-invariant coordinate features relative to the center of mass (0).
        """
        super().__init__()
        # Message MLP: cat(h_i, h_j, delta_r, y_ij) -> message_ij
        # delta_r consists of: ||r_i - r_j||, ||r_i||, ||r_j||, cos(r_i, r_j) (4 dimensions)
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + 4 + edge_dim, m_dim),
            nn.SiLU(),
            nn.Linear(m_dim, m_dim),
            nn.SiLU()
        )
        
        # Node update MLP: cat(h_i, msg_i) -> h_new
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + m_dim, m_dim),
            nn.SiLU(),
            nn.Linear(m_dim, node_dim)
        )
        
        # Coordinate scalar message MLP: msg_ij -> scalar factor
        self.coord_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim),
            nn.SiLU(),
            nn.Linear(m_dim, 1, bias=False)
        )

    def forward(self, h, R, Y, node_mask=None):
        """
        Args:
            h (Tensor): Node features of shape [B, N, node_dim]
            R (Tensor): Coordinates of shape [B, N, 3] (assumed zero-centered)
            Y (Tensor): Edge features of shape [B, N, N, edge_dim]
            node_mask (Tensor, optional): Mask of shape [B, N] where 1 is active, 0 is padded.
            
        Returns:
            h_new (Tensor): Updated node features [B, N, node_dim]
            R_new (Tensor): Updated coordinates [B, N, 3]
        """
        B, N, _ = R.shape
        
        if node_mask is None:
            node_mask = torch.ones((B, N), dtype=h.dtype, device=h.device)
            
        # 1. Compute invariant coordinate features (delta_r)
        # R_diff_ij = r_i - r_j
        R_diff = R.unsqueeze(2) - R.unsqueeze(1)  # [B, N, N, 3]
        dist = torch.norm(R_diff, p=2, dim=-1, keepdim=True)  # [B, N, N, 1]
        
        norm_node = torch.norm(R, p=2, dim=-1, keepdim=True)  # [B, N, 1]
        norm_i = norm_node.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, 1]
        norm_j = norm_node.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, 1]
        
        dot = torch.sum(R.unsqueeze(2) * R.unsqueeze(1), dim=-1, keepdim=True)  # [B, N, N, 1]
        cos = dot / (norm_i * norm_j + 1e-6)  # [B, N, N, 1]
        
        delta_r = torch.cat([dist, norm_i, norm_j, cos], dim=-1)  # [B, N, N, 4]
        
        # 2. Form messages
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, node_dim]
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, node_dim]
        
        inputs = torch.cat([h_i, h_j, delta_r, Y], dim=-1)  # [B, N, N, node_dim*2 + 4 + edge_dim]
        msg = self.message_mlp(inputs)  # [B, N, N, m_dim]
        
        # Mask out messages for padded nodes
        mask = (node_mask.unsqueeze(2) * node_mask.unsqueeze(1)).unsqueeze(-1)  # [B, N, N, 1]
        msg = msg * mask
        
        # 3. Coordinate Update
        coord_weights = self.coord_mlp(msg)  # [B, N, N, 1]
        coord_weights = coord_weights * mask
        
        # r_i <- r_i + sum_j (coord_weights * (r_j - r_i))
        # Note: (r_j - r_i) is -R_diff
        coord_msg = coord_weights * R_diff  # [B, N, N, 3]
        R_update = coord_msg.sum(dim=2)  # [B, N, 3]
        
        R_new = R + R_update
        
        # Projection onto zero-CoM subspace
        masked_R = R_new * node_mask.unsqueeze(-1)
        nodes_count = node_mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6
        mean_R = masked_R.sum(dim=1, keepdim=True) / nodes_count
        R_new = (R_new - mean_R) * node_mask.unsqueeze(-1)
        
        # 4. Node Features Update
        msg_sum = msg.sum(dim=2)  # [B, N, m_dim]
        node_inputs = torch.cat([h, msg_sum], dim=-1)  # [B, N, node_dim + m_dim]
        h_new = h + self.node_mlp(node_inputs)
        h_new = h_new * node_mask.unsqueeze(-1)
        
        return h_new, R_new

# ==============================================================================
# Graph Transformer Network (Equivariant)
# ==============================================================================

class rEGNNTransformer(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=64, num_layers=4,
                 num_node_classes=2):
        """
        Transformer style architecture with stacked rEGNN layers.
        
        Args:
            node_in_dim (int): Dimensionality of input node features (num_node_classes
                for one-hot categorical input).
            edge_in_dim (int): Dimensionality of input edge features.
            hidden_dim (int): Hidden dimension for all layers.
            num_layers (int): Number of rEGNN layers.
            num_node_classes (int): Number of node categories for the classification head.
        """
        super().__init__()
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_in_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            rEGNNLayer(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Final prediction heads
        # Coordinate head: outputs a SCALAR per node to preserve rotational equivariance.
        # A 3D output would scale axes independently, breaking equivariance.
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Edge head: predicts reconstructed adjacency matrices (binary logits)
        self.edge_head = nn.Linear(hidden_dim, 1)
        # Node category head: predicts class logits (Active vs Virtual)
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_node_classes)
        )

    def forward(self, X_t, R_t, Y_t, node_mask=None):
        """
        Args:
            X_t (Tensor): Noisy node categories [B, N, num_node_classes] (soft one-hot)
            R_t (Tensor): Noisy coordinates [B, N, 3]
            Y_t (Tensor): Noisy edge adjacency [B, N, N, edge_in_dim]
            node_mask (Tensor, optional): Mask of shape [B, N].
                During training this is None (all nodes processed, including virtual).
                During inference with known active nodes, can be provided.
            
        Returns:
            R_pred (Tensor): Reconstructed coordinates [B, N, 3]
            Y_pred (Tensor): Reconstructed edge logits [B, N, N]
            X_pred (Tensor): Reconstructed node category logits [B, N, num_node_classes]
        """
        h = self.node_embed(X_t)
        Y = self.edge_embed(Y_t)
        
        R = R_t
        for layer in self.layers:
            h, R = layer(h, R, Y, node_mask)
            
        # Predict coordinate adjustment
        # Equivariant PosMLP: scalar * unit_direction preserves rotational equivariance
        norm_R = torch.norm(R, p=2, dim=-1, keepdim=True)  # [B, N, 1]
        scale = self.coord_head(h)  # [B, N, 1] — scalar per node
        R_pred = R + (R / (norm_R + 1e-6)) * scale
        
        # Center the final predicted coordinates
        if node_mask is not None:
            masked_R = R_pred * node_mask.unsqueeze(-1)
            nodes_count = node_mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6
            mean_R = masked_R.sum(dim=1, keepdim=True) / nodes_count
            R_pred = (R_pred - mean_R) * node_mask.unsqueeze(-1)
            
        # Predict edge logits
        # Compute edge representations by concatenating node representations
        h_i = h.unsqueeze(2).expand(-1, -1, h.size(1), -1)
        h_j = h.unsqueeze(1).expand(-1, h.size(1), -1, -1)
        edge_rep = h_i + h_j  # Symmetric relation
        Y_pred = self.edge_head(edge_rep).squeeze(-1)  # [B, N, N]
        
        # Force symmetry on predicted adjacency matrix
        Y_pred = 0.5 * (Y_pred + Y_pred.transpose(1, 2))
        
        # Predict node category logits
        X_pred = self.node_head(h)  # [B, N, num_node_classes]
        
        return R_pred, Y_pred, X_pred

# ==============================================================================
# PyTorch Lightning Diffusion Module
# ==============================================================================

class CityJSONDiffusionModule(pl.LightningModule):
    def __init__(self, num_node_classes=2, hidden_dim=64, num_layers=4, T=500,
                 lr=1e-3, discrete_noise_type="uniform", n_max=64):
        """
        Denoising Diffusion Probabilistic Model (DDPM) written in PyTorch Lightning.
        Models:
        - Coordinates R as continuous variables using Gaussian noise.
        - Edges Y as discrete binary features using proper discrete diffusion.
        - Node categories X as discrete categorical features using discrete diffusion.
        
        Args:
            num_node_classes (int): Number of node categories (2: Active, Virtual).
            hidden_dim (int): Hidden dimension for the rEGNN transformer.
            num_layers (int): Number of rEGNN layers.
            T (int): Number of diffusion timesteps.
            lr (float): Learning rate.
            discrete_noise_type (str): Type of discrete noise schedule for edges and
                node categories. One of: "uniform", "absorbing", "discretized_gaussian".
            n_max (int): Maximum number of nodes per graph. Used during sampling to
                determine the graph size. Should match the dataset's N_max.
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.T = T
        self.lr = lr
        self.n_max = n_max
        self.num_node_classes = num_node_classes
        
        assert discrete_noise_type in ("uniform", "absorbing", "discretized_gaussian"), \
            f"Unknown discrete_noise_type: {discrete_noise_type}. " \
            f"Must be one of: 'uniform', 'absorbing', 'discretized_gaussian'."
        self.discrete_noise_type = discrete_noise_type
        
        # Denoising network
        # Node input: X (one-hot categorical node features, dim = num_node_classes)
        # Edge input: Y (binary edge maps)
        self.network = rEGNNTransformer(
            node_in_dim=num_node_classes,
            edge_in_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_node_classes=num_node_classes,
        )
        
        # Cosine noise schedule for coordinates (also used for discrete diffusion alpha_bar)
        self.register_buffer("alphas_bar", self._cosine_noise_schedule(T))
        
    def _cosine_noise_schedule(self, T, s=0.008):
        """
        Computes cosine alpha_bar schedule for continuous coordinates.
        """
        steps = T + 1
        t = torch.linspace(0, T, steps, dtype=torch.float32)
        alphas_bar = torch.cos(((t / T + s) / (1 + s)) * math.pi * 0.5) ** 2
        return alphas_bar / alphas_bar[0]

    # ------------------------------------------------------------------
    # Discrete diffusion helpers (binary edges)
    # ------------------------------------------------------------------

    def _discrete_forward_diffuse(self, Y0, alpha_bar):
        """
        Forward diffusion process for discrete binary edges.
        
        Args:
            Y0 (Tensor): Clean binary edge features [B, N, N, 1] with values in {0, 1}.
            alpha_bar (Tensor): Alpha-bar schedule values [B, 1, 1] for current timestep.
            
        Returns:
            Yt (Tensor): Corrupted edge features [B, N, N, 1].
        """
        if self.discrete_noise_type == "uniform":
            # Uniform discrete diffusion (D3PM / MiDi):
            # Each edge stays at its clean value with prob alpha_bar,
            # otherwise flips to a uniform random binary state {0, 1}.
            flip_mask = torch.rand_like(Y0) > alpha_bar.unsqueeze(-1)  # True = corrupt
            uniform_sample = (torch.rand_like(Y0) > 0.5).float()
            Yt = torch.where(flip_mask, uniform_sample, Y0)
            
        elif self.discrete_noise_type == "absorbing":
            # Absorbing state discrete diffusion (D3PM absorbing):
            # Each edge stays at its clean value with prob alpha_bar,
            # otherwise transitions to the absorbing mask state (0.5).
            absorb_mask = torch.rand_like(Y0) > alpha_bar.unsqueeze(-1)  # True = absorb
            mask_value = torch.full_like(Y0, 0.5)
            Yt = torch.where(absorb_mask, mask_value, Y0)
            
        elif self.discrete_noise_type == "discretized_gaussian":
            # Discretized Gaussian (Multinomial Diffusion, Hoogeboom et al.):
            # Map binary {0, 1} -> logit space {-1, +1}, add Gaussian noise,
            # threshold at 0 to recover corrupted binary state.
            y_logit = 2.0 * Y0 - 1.0  # {0, 1} -> {-1, +1}
            sqrt_alpha = torch.sqrt(alpha_bar).unsqueeze(-1)
            sigma = torch.sqrt(1.0 - alpha_bar).unsqueeze(-1)
            eps = torch.randn_like(Y0)
            z = sqrt_alpha * y_logit + sigma * eps
            Yt = (z > 0.0).float()
        else:
            raise ValueError(f"Unknown discrete_noise_type: {self.discrete_noise_type}")
            
        return Yt
    
    def _discrete_posterior_sample(self, Y_pred_logits, Yt, alpha_bar_t, alpha_bar_prev, t_idx):
        """
        Sample Y_{t-1} from the discrete posterior p(Y_{t-1} | Y_t, Y_pred).
        
        Args:
            Y_pred_logits (Tensor): Predicted clean edge logits [B, N, N].
            Yt (Tensor): Current noised edges [B, N, N, 1].
            alpha_bar_t (Tensor): Alpha-bar at timestep t (scalar).
            alpha_bar_prev (Tensor): Alpha-bar at timestep t-1 (scalar).
            t_idx (int): Current timestep index.
            
        Returns:
            Y_prev (Tensor): Sampled edges at t-1 [B, N, N, 1].
        """
        Y_pred_prob = torch.sigmoid(Y_pred_logits).unsqueeze(-1)  # [B, N, N, 1]
        
        if self.discrete_noise_type == "uniform":
            # Bayes rule for uniform transition matrix:
            # p(Y_{t-1}=1 | Y_t, Y_0_pred) ∝ q(Y_t | Y_{t-1}=1) * p(Y_0=1)
            # q(Y_t=y | Y_{t-1}=c) = alpha_t * delta(y, c) + (1 - alpha_t) * 0.5
            alpha_t = alpha_bar_t / alpha_bar_prev
            
            # Likelihood of Y_t given Y_{t-1} = 1 or 0
            # p(Y_t | Y_{t-1}=1) = alpha_t * Y_t + (1 - alpha_t) * 0.5
            # p(Y_t | Y_{t-1}=0) = alpha_t * (1 - Y_t) + (1 - alpha_t) * 0.5
            likelihood_1 = alpha_t * Yt + (1.0 - alpha_t) * 0.5
            likelihood_0 = alpha_t * (1.0 - Yt) + (1.0 - alpha_t) * 0.5
            
            # Posterior via Bayes rule
            unnorm_1 = likelihood_1 * Y_pred_prob
            unnorm_0 = likelihood_0 * (1.0 - Y_pred_prob)
            
            posterior_prob = unnorm_1 / (unnorm_1 + unnorm_0 + 1e-8)
            
            if t_idx > 1:
                Y_prev = (torch.rand_like(posterior_prob) < posterior_prob).float()
            else:
                Y_prev = (posterior_prob > 0.5).float()
                
        elif self.discrete_noise_type == "absorbing":
            # Absorbing state posterior:
            # If Y_t is in mask state (0.5), sample from predicted distribution.
            # If Y_t is not masked, it stays as-is (already clean).
            is_masked = (Yt == 0.5)
            
            if t_idx > 1:
                # For masked positions: with prob (alpha_bar_prev), reveal to predicted;
                # otherwise keep masked.
                reveal_mask = torch.rand_like(Y_pred_prob) < alpha_bar_prev
                sampled = (torch.rand_like(Y_pred_prob) < Y_pred_prob).float()
                # Reveal to predicted value or stay masked
                unmasked_val = torch.where(reveal_mask, sampled, torch.full_like(Yt, 0.5))
                Y_prev = torch.where(is_masked, unmasked_val, Yt)
            else:
                # Final step: unmask everything using predicted probabilities
                sampled = (Y_pred_prob > 0.5).float()
                Y_prev = torch.where(is_masked, sampled, Yt)
                
        elif self.discrete_noise_type == "discretized_gaussian":
            # Discretized Gaussian posterior:
            # Compute posterior in logit space using Gaussian CDF.
            y_pred_logit = 2.0 * Y_pred_prob - 1.0  # predicted logit in {-1, +1}
            
            if t_idx > 1:
                sqrt_alpha_prev = torch.sqrt(alpha_bar_prev)
                sigma_prev = torch.sqrt(1.0 - alpha_bar_prev)
                
                # Probability that z_{t-1} > 0 given predicted clean state
                # z_{t-1} = sqrt(alpha_bar_{t-1}) * y_logit_pred + sigma_{t-1} * eps
                # P(z > 0) = Phi(sqrt(alpha_bar_{t-1}) * y_logit_pred / sigma_{t-1})
                logit_arg = sqrt_alpha_prev * y_pred_logit / (sigma_prev + 1e-8)
                # Use the normal CDF: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
                posterior_prob = 0.5 * (1.0 + torch.erf(logit_arg / math.sqrt(2.0)))
                
                Y_prev = (torch.rand_like(posterior_prob) < posterior_prob).float()
            else:
                # Final step: threshold the predicted probabilities
                Y_prev = (Y_pred_prob > 0.5).float()
        else:
            raise ValueError(f"Unknown discrete_noise_type: {self.discrete_noise_type}")
            
        return Y_prev

    # ------------------------------------------------------------------
    # Categorical diffusion helpers (node categories)
    # ------------------------------------------------------------------

    def _categorical_forward_diffuse(self, X0, alpha_bar):
        """
        Forward diffusion for categorical node features (one-hot).
        
        Args:
            X0 (Tensor): Clean one-hot node categories [B, N, C].
            alpha_bar (Tensor): Alpha-bar values [B, 1, 1] for current timestep.
            
        Returns:
            Xt (Tensor): Corrupted node categories [B, N, C] (soft probabilities or one-hot).
        """
        C = X0.shape[-1]
        
        if self.discrete_noise_type == "uniform":
            # Uniform categorical diffusion:
            # Stay at clean class with prob alpha_bar, else uniform over C classes.
            # Result is a hard one-hot after sampling.
            keep_mask = torch.rand(X0.shape[:-1], device=X0.device).unsqueeze(-1) < alpha_bar  # [B, N, 1]
            # Sample uniform class indices
            uniform_indices = torch.randint(0, C, X0.shape[:-1], device=X0.device)  # [B, N]
            uniform_onehot = F.one_hot(uniform_indices, C).float()  # [B, N, C]
            Xt = torch.where(keep_mask.expand_as(X0), X0, uniform_onehot)
            
        elif self.discrete_noise_type == "absorbing":
            # Absorbing categorical diffusion:
            # Stay at clean class with prob alpha_bar, else → uniform vector [1/C, ..., 1/C]
            # (the "mask" state in probability space).
            absorb_mask = torch.rand(X0.shape[:-1], device=X0.device).unsqueeze(-1) > alpha_bar  # [B, N, 1]
            mask_state = torch.full_like(X0, 1.0 / C)
            Xt = torch.where(absorb_mask.expand_as(X0), mask_state, X0)
            
        elif self.discrete_noise_type == "discretized_gaussian":
            # Discretized Gaussian for categoricals:
            # Add Gaussian noise to the one-hot vector, re-normalize via softmax
            # to get a soft categorical, then sample a hard one-hot via Gumbel.
            sigma = torch.sqrt(1.0 - alpha_bar)  # [B, 1, 1]
            sqrt_alpha = torch.sqrt(alpha_bar)    # [B, 1, 1]
            noise = torch.randn_like(X0) * sigma
            z = sqrt_alpha * X0 + noise
            # Straight-through: sample argmax as hard one-hot
            hard_indices = z.argmax(dim=-1)  # [B, N]
            Xt = F.one_hot(hard_indices, C).float()  # [B, N, C]
        else:
            raise ValueError(f"Unknown discrete_noise_type: {self.discrete_noise_type}")
            
        return Xt

    def _categorical_posterior_sample(self, X_pred_logits, Xt, alpha_bar_t,
                                      alpha_bar_prev, t_idx):
        """
        Sample X_{t-1} from the categorical posterior p(X_{t-1} | X_t, X_pred).
        
        Args:
            X_pred_logits (Tensor): Predicted clean node category logits [B, N, C].
            Xt (Tensor): Current noised node categories [B, N, C] (one-hot or soft).
            alpha_bar_t (Tensor): Alpha-bar at timestep t (scalar).
            alpha_bar_prev (Tensor): Alpha-bar at timestep t-1 (scalar).
            t_idx (int): Current timestep index.
            
        Returns:
            X_prev (Tensor): Sampled node categories at t-1 [B, N, C] (one-hot).
        """
        C = X_pred_logits.shape[-1]
        X_pred_prob = F.softmax(X_pred_logits, dim=-1)  # [B, N, C]
        
        if self.discrete_noise_type == "uniform":
            # Bayes rule for uniform categorical transition:
            # q(X_t | X_{t-1}=c) = alpha_t * delta(X_t, c) + (1 - alpha_t) / C
            alpha_t = alpha_bar_t / alpha_bar_prev
            
            # For each possible X_{t-1} class c:
            # likelihood(c) = alpha_t * X_t[c] + (1 - alpha_t) / C
            # posterior(c) ∝ likelihood(c) * X_pred_prob[c]
            likelihood = alpha_t * Xt + (1.0 - alpha_t) / C  # [B, N, C]
            unnorm = likelihood * X_pred_prob  # [B, N, C]
            posterior = unnorm / (unnorm.sum(dim=-1, keepdim=True) + 1e-8)
            
            if t_idx > 1:
                sampled_indices = torch.multinomial(
                    posterior.view(-1, C), num_samples=1
                ).view(X_pred_logits.shape[:-1])  # [B, N]
                X_prev = F.one_hot(sampled_indices, C).float()
            else:
                X_prev = F.one_hot(posterior.argmax(dim=-1), C).float()
                
        elif self.discrete_noise_type == "absorbing":
            # Absorbing posterior for categoricals:
            # If Xt is in mask state (uniform 1/C), sample from predicted.
            # Otherwise keep as-is.
            is_masked = (Xt.max(dim=-1).values < (1.0 / C + 0.01))  # [B, N]
            
            if t_idx > 1:
                # Reveal with prob proportional to schedule
                reveal = torch.rand(is_masked.shape, device=Xt.device) < alpha_bar_prev
                sampled_indices = torch.multinomial(
                    X_pred_prob.view(-1, C), num_samples=1
                ).view(X_pred_logits.shape[:-1])  # [B, N]
                sampled_onehot = F.one_hot(sampled_indices, C).float()
                mask_state = torch.full_like(Xt, 1.0 / C)
                # Where masked: reveal or stay masked
                revealed = torch.where(
                    reveal.unsqueeze(-1).expand_as(Xt), sampled_onehot, mask_state
                )
                X_prev = torch.where(is_masked.unsqueeze(-1).expand_as(Xt), revealed, Xt)
            else:
                # Final step: unmask everything
                final_indices = X_pred_prob.argmax(dim=-1)
                final_onehot = F.one_hot(final_indices, C).float()
                X_prev = torch.where(
                    is_masked.unsqueeze(-1).expand_as(Xt), final_onehot, Xt
                )
                
        elif self.discrete_noise_type == "discretized_gaussian":
            # Discretized Gaussian posterior for categoricals:
            # Apply noise scaled to t-1 to the predicted clean distribution, sample argmax.
            if t_idx > 1:
                sqrt_alpha_prev = torch.sqrt(alpha_bar_prev)
                sigma_prev = torch.sqrt(1.0 - alpha_bar_prev)
                noise = torch.randn_like(X_pred_prob) * sigma_prev
                z = sqrt_alpha_prev * X_pred_prob + noise
                sampled_indices = z.argmax(dim=-1)
                X_prev = F.one_hot(sampled_indices, C).float()
            else:
                X_prev = F.one_hot(X_pred_prob.argmax(dim=-1), C).float()
        else:
            raise ValueError(f"Unknown discrete_noise_type: {self.discrete_noise_type}")
            
        return X_prev

    # ------------------------------------------------------------------
    # Forward / Training / Validation
    # ------------------------------------------------------------------

    def forward(self, X_t, R_t, Y_t, node_mask=None):
        return self.network(X_t, R_t, Y_t, node_mask)

    def training_step(self, batch, batch_idx):
        """
        Calculates hybrid diffusion loss:
        - Coordinates: MSE (continuous, masked to active nodes)
        - Edges: Binary Cross-Entropy (discrete)
        - Node categories: Cross-Entropy (discrete)
        
        Batch contains:
        - 'x': node coordinates [B, N_max, 3]
        - 'node_categories': one-hot node categories [B, N_max, 2]
        - 'y': dense adjacency matrices [B, N_max, N_max, 1]
        - 'node_mask': active node flags [B, N_max]
        """
        # Batch unpacking
        R0 = batch["x"]                    # Coordinates [B, N_max, 3]
        X0 = batch["node_categories"]      # One-hot categories [B, N_max, 2]
        Y0 = batch["y"]                    # Adjacency [B, N_max, N_max, 1]
        node_mask = batch["node_mask"]     # Active mask [B, N_max]
        B, N, _ = R0.shape
            
        # 1. Sample time step t
        t = torch.randint(1, self.T + 1, (B,), device=self.device)
        
        # 2. Corrupt coordinates R0 -> Rt (continuous Gaussian, active nodes only)
        alpha_bar = self.alphas_bar[t].view(B, 1, 1)
        sigma_bar = torch.sqrt(1.0 - alpha_bar)
        
        eps = torch.randn_like(R0)
        # Center noise on CoM subspace (over active nodes only)
        eps = eps * node_mask.unsqueeze(-1)
        nodes_count = node_mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6
        eps = eps - eps.sum(dim=1, keepdim=True) / nodes_count
            
        Rt = alpha_bar * R0 + sigma_bar * eps
        
        # 3. Corrupt adjacency matrices Y0 -> Yt (discrete diffusion)
        Yt = self._discrete_forward_diffuse(Y0, alpha_bar)
        
        # 4. Corrupt node categories X0 -> Xt (categorical discrete diffusion)
        Xt = self._categorical_forward_diffuse(X0, alpha_bar)
        
        # 5. Predict clean graph from noised state
        # node_mask=None: the network sees ALL nodes (including virtual) to learn
        # to classify which are real vs virtual
        R_pred, Y_pred, X_pred = self(Xt, Rt, Yt, node_mask=None)
        
        # 6. Compute loss components
        # Coordinates loss (MSE, masked to ACTIVE nodes only — virtual coords are meaningless)
        coord_loss = F.mse_loss(
            R_pred * node_mask.unsqueeze(-1),
            R0 * node_mask.unsqueeze(-1),
            reduction='sum'
        )
        coord_loss = coord_loss / (node_mask.sum() * 3.0 + 1e-6)
            
        # Adjacency loss (Binary Cross Entropy, over ALL node pairs)
        # The network should predict zero edges for virtual nodes
        target_edges = Y0.squeeze(-1)  # [B, N, N]
        edge_loss = F.binary_cross_entropy_with_logits(Y_pred, target_edges, reduction='mean')
            
        # Node category loss (Cross-Entropy, over ALL nodes)
        # The network must learn to classify Active vs Virtual
        node_target = X0.argmax(dim=-1)  # [B, N] class indices
        node_loss = F.cross_entropy(
            X_pred.view(-1, self.num_node_classes),
            node_target.view(-1),
            reduction='mean'
        )
        
        total_loss = 3.0 * coord_loss + 2.0 * edge_loss + 1.0 * node_loss
        
        # Logging
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_coord_mse", coord_loss, on_epoch=True, prog_bar=True)
        self.log("train_edge_bce", edge_loss, on_epoch=True)
        self.log("train_node_ce", node_loss, on_epoch=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        R0 = batch["x"]
        X0 = batch["node_categories"]
        Y0 = batch["y"]
        node_mask = batch["node_mask"]
        B, N, _ = R0.shape
            
        # Use t = T/2 for a representative validation evaluation
        t = torch.full((B,), self.T // 2, dtype=torch.long, device=self.device)
        alpha_bar = self.alphas_bar[t].view(B, 1, 1)
        sigma_bar = torch.sqrt(1.0 - alpha_bar)
        
        eps = torch.randn_like(R0)
        eps = eps * node_mask.unsqueeze(-1)
        nodes_count = node_mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6
        eps = eps - eps.sum(dim=1, keepdim=True) / nodes_count
            
        Rt = alpha_bar * R0 + sigma_bar * eps
        Yt = self._discrete_forward_diffuse(Y0, alpha_bar)
        Xt = self._categorical_forward_diffuse(X0, alpha_bar)
        
        R_pred, Y_pred, X_pred = self(Xt, Rt, Yt, node_mask=None)
        
        # Coordinate metric (active nodes only)
        val_mse = F.mse_loss(
            R_pred * node_mask.unsqueeze(-1),
            R0 * node_mask.unsqueeze(-1),
            reduction='sum'
        )
        val_mse = val_mse / (node_mask.sum() * 3.0 + 1e-6)
        
        # Node classification accuracy
        node_target = X0.argmax(dim=-1)  # [B, N]
        node_preds = X_pred.argmax(dim=-1)  # [B, N]
        node_acc = (node_preds == node_target).float().mean()
            
        self.log("val_coord_mse", val_mse, on_epoch=True, prog_bar=True)
        self.log("val_node_acc", node_acc, on_epoch=True, prog_bar=True)
        return val_mse

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, batch_size=1):
        """
        Unconditional generation sampling loop.
        Generates N_max nodes and jointly denoises coordinates R, edges Y,
        and node categories X. Returns active mask to identify real nodes.
        
        Args:
            batch_size (int): Number of graphs to generate.
            
        Returns:
            Rt (Tensor): Final coordinates [B, N_max, 3]
            edge_probs (Tensor): Edge probabilities [B, N_max, N_max]
            active_mask (Tensor): Boolean mask [B, N_max] where True = Active node
        """
        self.eval()
        N = self.n_max
        C = self.num_node_classes
        
        # 1. Initialize from pure noise
        # Coordinates: Standard Gaussian centered on CoM
        Rt = torch.randn((batch_size, N, 3), device=self.device)
        Rt = Rt - Rt.mean(dim=1, keepdim=True)
        
        # Edges: Initialize according to discrete noise type
        if self.discrete_noise_type == "uniform":
            Yt = (torch.rand((batch_size, N, N, 1), device=self.device) > 0.5).float()
        elif self.discrete_noise_type == "absorbing":
            Yt = torch.full((batch_size, N, N, 1), 0.5, device=self.device)
        elif self.discrete_noise_type == "discretized_gaussian":
            Yt = (torch.rand((batch_size, N, N, 1), device=self.device) > 0.5).float()
        
        # Node categories: Initialize according to discrete noise type
        if self.discrete_noise_type == "uniform":
            rand_indices = torch.randint(0, C, (batch_size, N), device=self.device)
            Xt = F.one_hot(rand_indices, C).float()
        elif self.discrete_noise_type == "absorbing":
            Xt = torch.full((batch_size, N, C), 1.0 / C, device=self.device)
        elif self.discrete_noise_type == "discretized_gaussian":
            rand_indices = torch.randint(0, C, (batch_size, N), device=self.device)
            Xt = F.one_hot(rand_indices, C).float()
        
        # 2. Iterative Denoising Loop
        for t_idx in reversed(range(1, self.T + 1)):
            # Run denoising network to predict clean components
            # node_mask=None: network processes all N_max nodes
            R_pred, Y_pred, X_pred = self(Xt, Rt, Yt, node_mask=None)
            
            # --- Coordinate DDPM update ---
            alpha_bar_t = self.alphas_bar[t_idx]
            alpha_bar_prev = self.alphas_bar[t_idx - 1]
            
            alpha_t = alpha_bar_t / alpha_bar_prev
            beta_t = 1.0 - alpha_t
            
            # Posterior mean coordinate calculation
            coef_pred = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
            coef_t = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            
            mu = coef_pred.view(-1, 1, 1) * R_pred + coef_t.view(-1, 1, 1) * Rt
            
            if t_idx > 1:
                sigma = torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * beta_t)
                z = torch.randn_like(Rt)
                z = z - z.mean(dim=1, keepdim=True)  # project noise to CoM
                Rt = mu + sigma.view(-1, 1, 1) * z
            else:
                Rt = mu
                
            # --- Discrete posterior for edges ---
            Yt = self._discrete_posterior_sample(Y_pred, Yt, alpha_bar_t, alpha_bar_prev, t_idx)
            
            # --- Categorical posterior for node categories ---
            Xt = self._categorical_posterior_sample(X_pred, Xt, alpha_bar_t, alpha_bar_prev, t_idx)
            
        # Final predictions
        edge_probs = torch.sigmoid(Y_pred)  # [B, N_max, N_max]
        node_classes = X_pred.argmax(dim=-1)  # [B, N_max] — 0=Active, 1=Virtual
        active_mask = (node_classes == 0)  # [B, N_max] boolean
        
        return Rt, edge_probs, active_mask

    # ------------------------------------------------------------------
    # CityJSON generation (batched export with virtual node filtering)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_cityjson(self, batch_size=1, threshold=0.5):
        """
        Generates buildings via diffusion sampling and exports each as a CityJSON dict.
        
        Filters out Virtual nodes predicted by the network before passing to the
        CityJSON exporter. Handles the [B, N_max, ...] → [N_active, ...] conversion.
        
        Args:
            batch_size (int): Number of buildings to generate.
            threshold (float): Edge probability threshold for adjacency.
            
        Returns:
            results (list[dict]): List of CityJSON dictionaries, one per building.
        """
        # Lazy import to avoid circular dependencies
        try:
            from post_process import graph_to_cityjson
        except ImportError:
            from src.post_process import graph_to_cityjson
        
        Rt, edge_probs, active_mask = self.sample(batch_size=batch_size)
        
        results = []
        for i in range(batch_size):
            # Get indices of active (non-virtual) nodes
            mask_i = active_mask[i]  # [N_max] boolean
            active_indices = mask_i.nonzero(as_tuple=True)[0]
            
            if len(active_indices) < 3:
                # Need at least 3 nodes to form a polygon
                print(f"Warning: Building {i} has only {len(active_indices)} active nodes, skipping.")
                continue
            
            # Slice out only active nodes and their sub-adjacency matrix
            nodes = Rt[i, active_indices].cpu().numpy()  # [N_active, 3]
            edges = edge_probs[i][active_indices][:, active_indices].cpu().numpy()  # [N_active, N_active]
            
            cj = graph_to_cityjson(
                nodes, edges,
                threshold=threshold,
                building_id=f"generated_building_{i}"
            )
            if cj:
                results.append(cj)
                
        return results
