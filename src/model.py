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
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=64, num_layers=4):
        """
        Transformer style architecture with stacked rEGNN layers.
        """
        super().__init__()
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_in_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            rEGNNLayer(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Final prediction heads
        # Node head: predicts coordinate displacements or coordinates directly
        self.coord_head = nn.Linear(hidden_dim, 3)
        # Edge head: predicts reconstructed adjacency matrices (binary logits)
        self.edge_head = nn.Linear(hidden_dim, 1)

    def forward(self, X_t, R_t, Y_t, node_mask=None):
        """
        Args:
            X_t (Tensor): Noisy node categories/features [B, N, node_in_dim]
            R_t (Tensor): Noisy coordinates [B, N, 3]
            Y_t (Tensor): Noisy edge adjacency [B, N, N, edge_in_dim]
            node_mask (Tensor, optional): Mask of shape [B, N]
            
        Returns:
            R_pred (Tensor): Reconstructed coordinates [B, N, 3]
            Y_pred (Tensor): Reconstructed edge logits [B, N, N]
        """
        h = self.node_embed(X_t)
        Y = self.edge_embed(Y_t)
        
        R = R_t
        for layer in self.layers:
            h, R = layer(h, R, Y, node_mask)
            
        # Predict coordinate adjustment
        # PosMLP rotation-equivariant coordinate calculation
        norm_R = torch.norm(R, p=2, dim=-1, keepdim=True)  # [B, N, 1]
        scale = self.coord_head(h).unsqueeze(-1)  # [B, N, 3, 1] -> broadcast scale along dimensions
        
        # PosMLP(R) = R * scale
        R_pred = R + (R / (norm_R + 1e-6)) * self.coord_head(h)
        
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
        
        return R_pred, Y_pred

# ==============================================================================
# PyTorch Lightning Diffusion Module
# ==============================================================================

class CityJSONDiffusionModule(pl.LightningModule):
    def __init__(self, node_dim=1, hidden_dim=64, num_layers=4, T=500, lr=1e-3):
        """
        Denoising Diffusion Probabilistic Model (DDPM) written in PyTorch Lightning.
        Models:
        - Coordinates R as continuous variables using Gaussian noise.
        - Edges Y as binary features using discrete transitions.
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.T = T
        self.lr = lr
        
        # Denoising network
        # Node input: X (categories/features, here simply dummy node dimensions)
        # Edge input: Y (binary edge maps)
        self.network = rEGNNTransformer(
            node_in_dim=node_dim,
            edge_in_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        # Cosine noise schedule for coordinates
        self.register_buffer("alphas_bar", self._cosine_noise_schedule(T))
        
    def _cosine_noise_schedule(self, T, s=0.008):
        """
        Computes cosine alpha_bar schedule for continuous coordinates.
        """
        steps = T + 1
        t = torch.linspace(0, T, steps, dtype=torch.float32)
        alphas_bar = torch.cos(((t / T + s) / (1 + s)) * math.pi * 0.5) ** 2
        return alphas_bar / alphas_bar[0]

    def forward(self, X_t, R_t, Y_t, node_mask=None):
        return self.network(X_t, R_t, Y_t, node_mask)

    def training_step(self, batch, batch_idx):
        """
        Calculates diffusion loss: Coordinates MSE and Adjacency binary cross-entropy.
        Batch contains:
        - 'x': node coordinates [B, N, 3]
        - 'edge_index': edge list or dense adjacency matrices [B, N, N]
        - 'node_mask': active node flags [B, N]
        """
        # Batch unpacking
        R0 = batch["x"]  # Coordinates [B, N, 3]
        node_mask = batch.get("node_mask")  # [B, N]
        B, N, _ = R0.shape
        
        # Reconstruct dense Y0 from batch (if edge_index is present)
        # For this example, we assume batch provides a dense Y0 matrix of shape [B, N, N, 1]
        Y0 = batch.get("y")  # Adjacency matrices [B, N, N, 1]
        if Y0 is None:
            # Fallback if sparse
            Y0 = torch.zeros((B, N, N, 1), dtype=torch.float32, device=self.device)
            # (Note: dataset loader can construct this dense matrix)
            
        # Dummy node features (types/classes)
        X0 = torch.ones((B, N, self.hparams.node_dim), dtype=torch.float32, device=self.device)
        if node_mask is not None:
            X0 = X0 * node_mask.unsqueeze(-1)
            
        # 1. Sample time step t
        t = torch.randint(1, self.T + 1, (B,), device=self.device)
        
        # 2. Corrupt coordinates R0 -> Rt
        # R_t = alpha_bar_t * R_0 + sigma_bar_t * noise
        alpha_bar = self.alphas_bar[t].view(B, 1, 1)
        sigma_bar = torch.sqrt(1.0 - alpha_bar)
        
        eps = torch.randn_like(R0)
        # Center noise on CoM subspace
        if node_mask is not None:
            eps = eps * node_mask.unsqueeze(-1)
            nodes_count = node_mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6
            eps = eps - eps.sum(dim=1, keepdim=True) / nodes_count
            
        Rt = alpha_bar * R0 + sigma_bar * eps
        
        # 3. Corrupt adjacency matrices Y0 -> Yt (discrete diffusion)
        # For simplicity in this initial implementation, we use a continuous approximation 
        # for binary edge diffusion (BCE loss over continuous noised logits)
        Y_noise = torch.rand_like(Y0)
        # Interpolate between Y0 and uniform noise based on alpha schedule
        Yt = alpha_bar.unsqueeze(-1) * Y0 + (1.0 - alpha_bar.unsqueeze(-1)) * Y_noise
        
        # 4. Predict clean graph from noised state
        R_pred, Y_pred = self(X0, Rt, Yt, node_mask)
        
        # 5. Compute loss components
        # Coordinates loss (MSE)
        if node_mask is not None:
            coord_loss = F.mse_loss(R_pred * node_mask.unsqueeze(-1), R0 * node_mask.unsqueeze(-1), reduction='sum')
            coord_loss = coord_loss / (node_mask.sum() * 3.0)
        else:
            coord_loss = F.mse_loss(R_pred, R0)
            
        # Adjacency loss (Binary Cross Entropy)
        # Y_pred shape [B, N, N], Y0 shape [B, N, N, 1]
        target_edges = Y0.squeeze(-1)  # [B, N, N]
        if node_mask is not None:
            edge_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)
            edge_loss = F.binary_cross_entropy_with_logits(Y_pred, target_edges, reduction='none')
            edge_loss = (edge_loss * edge_mask).sum() / (edge_mask.sum() + 1e-6)
        else:
            edge_loss = F.binary_cross_entropy_with_logits(Y_pred, target_edges)
            
        total_loss = 3.0 * coord_loss + 2.0 * edge_loss
        
        # Logging
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_coord_mse", coord_loss, on_epoch=True, prog_bar=True)
        self.log("train_edge_bce", edge_loss, on_epoch=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        R0 = batch["x"]
        node_mask = batch.get("node_mask")
        B, N, _ = R0.shape
        Y0 = batch.get("y")
        if Y0 is None:
            Y0 = torch.zeros((B, N, N, 1), dtype=torch.float32, device=self.device)
            
        X0 = torch.ones((B, N, self.hparams.node_dim), dtype=torch.float32, device=self.device)
        if node_mask is not None:
            X0 = X0 * node_mask.unsqueeze(-1)
            
        # Use t = T/2 for a representative validation evaluation
        t = torch.full((B,), self.T // 2, dtype=torch.long, device=self.device)
        alpha_bar = self.alphas_bar[t].view(B, 1, 1)
        sigma_bar = torch.sqrt(1.0 - alpha_bar)
        
        eps = torch.randn_like(R0)
        if node_mask is not None:
            eps = eps * node_mask.unsqueeze(-1)
            nodes_count = node_mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6
            eps = eps - eps.sum(dim=1, keepdim=True) / nodes_count
            
        Rt = alpha_bar * R0 + sigma_bar * eps
        Y_noise = torch.rand_like(Y0)
        Yt = alpha_bar.unsqueeze(-1) * Y0 + (1.0 - alpha_bar.unsqueeze(-1)) * Y_noise
        
        R_pred, Y_pred = self(X0, Rt, Yt, node_mask)
        
        # Calculate metric
        if node_mask is not None:
            val_mse = F.mse_loss(R_pred * node_mask.unsqueeze(-1), R0 * node_mask.unsqueeze(-1), reduction='sum')
            val_mse = val_mse / (node_mask.sum() * 3.0)
        else:
            val_mse = F.mse_loss(R_pred, R0)
            
        self.log("val_coord_mse", val_mse, on_epoch=True, prog_bar=True)
        return val_mse

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    @torch.no_grad()
    def sample(self, num_nodes, batch_size=1):
        """
        Unconditional generation sampling loop.
        Generates coordinates R and adjacency Y from pure noise.
        """
        self.eval()
        N = num_nodes
        
        # 1. Initialize from noise
        # Coordinates: Standard Gaussian centered on CoM
        Rt = torch.randn((batch_size, N, 3), device=self.device)
        Rt = Rt - Rt.mean(dim=1, keepdim=True)
        
        # Edges: Random continuous noise
        Yt = torch.rand((batch_size, N, N, 1), device=self.device)
        
        X = torch.ones((batch_size, N, self.hparams.node_dim), dtype=torch.float32, device=self.device)
        
        # 2. Iterative Denoising Loop
        for t_idx in reversed(range(1, self.T + 1)):
            t = torch.full((batch_size,), t_idx, dtype=torch.long, device=self.device)
            
            # Run denoising network to predict clean components
            R_pred, Y_pred = self(X, Rt, Yt)
            
            # Dynamic stepping (DDPM style update)
            # R^{t-1} = mu_t(R_t, R_pred) + sigma_t * noise
            alpha_bar_t = self.alphas_bar[t_idx]
            alpha_bar_prev = self.alphas_bar[t_idx - 1]
            
            alpha_t = alpha_bar_t / alpha_bar_prev
            beta_t = 1.0 - alpha_t
            
            # Posterior mean coordinate calculation
            coef_pred = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
            coef_t = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            
            mu = coef_pred.view(-1, 1, 1) * R_pred + coef_t.view(-1, 1, 1) * Rt
            
            if t_idx > 1:
                # Add noise
                sigma = torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * beta_t)
                z = torch.randn_like(Rt)
                z = z - z.mean(dim=1, keepdim=True)  # project noise to CoM
                Rt = mu + sigma.view(-1, 1, 1) * z
            else:
                Rt = mu
                
            # Update Yt (simple interpolation transition update)
            # Edges transition back
            Y_pred_prob = torch.sigmoid(Y_pred).unsqueeze(-1)
            Yt = alpha_bar_prev.view(-1, 1, 1, 1) * Y_pred_prob + (1.0 - alpha_bar_prev.view(-1, 1, 1, 1)) * torch.rand_like(Yt)
            
        # Final thresholding
        edge_probs = torch.sigmoid(Y_pred)  # [B, N, N]
        
        return Rt, edge_probs
