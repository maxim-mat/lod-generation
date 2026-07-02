import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L

from src.models.regnn import rEGNNTransformer

logger = logging.getLogger(__name__)

class CityJSONDiffusionModule(L.LightningModule):
    def __init__(self, num_node_classes=2, hidden_dim=64, num_layers=4, T=500,
                 lr=1e-3, discrete_noise_type="uniform", n_max=64,
                 lr_scheduler="none", lr_decay_steps=50, lr_decay_rate=0.5):
        """
        Denoising Diffusion Probabilistic Model (DDPM) written in PyTorch Lightning.
        Models:
        - Coordinates R as continuous variables using Gaussian noise.
        - Edges Y as discrete binary features using proper discrete diffusion.
        - Node categories X as discrete categorical features using discrete diffusion.
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.T = T
        self.lr = lr
        self.n_max = n_max
        self.num_node_classes = num_node_classes
        self.lr_scheduler = lr_scheduler
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = lr_decay_rate
        
        assert discrete_noise_type in ("uniform", "absorbing", "discretized_gaussian"), \
            f"Unknown discrete_noise_type: {discrete_noise_type}. " \
            f"Must be one of: 'uniform', 'absorbing', 'discretized_gaussian'."
        self.discrete_noise_type = discrete_noise_type
        
        # Denoising network
        self.network = rEGNNTransformer(
            node_in_dim=num_node_classes,
            edge_in_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_node_classes=num_node_classes,
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

    # ------------------------------------------------------------------
    # Discrete diffusion helpers (binary edges)
    # ------------------------------------------------------------------

    def _discrete_forward_diffuse(self, Y0, alpha_bar):
        """
        Forward diffusion process for discrete binary edges.
        """
        if self.discrete_noise_type == "uniform":
            flip_mask = torch.rand_like(Y0) > alpha_bar.unsqueeze(-1)  # True = corrupt
            uniform_sample = (torch.rand_like(Y0) > 0.5).float()
            Yt = torch.where(flip_mask, uniform_sample, Y0)
            
        elif self.discrete_noise_type == "absorbing":
            absorb_mask = torch.rand_like(Y0) > alpha_bar.unsqueeze(-1)  # True = absorb
            mask_value = torch.full_like(Y0, 0.5)
            Yt = torch.where(absorb_mask, mask_value, Y0)
            
        elif self.discrete_noise_type == "discretized_gaussian":
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
        """
        Y_pred_prob = torch.sigmoid(Y_pred_logits).unsqueeze(-1)  # [B, N, N, 1]
        
        if self.discrete_noise_type == "uniform":
            alpha_t = alpha_bar_t / alpha_bar_prev
            likelihood_1 = alpha_t * Yt + (1.0 - alpha_t) * 0.5
            likelihood_0 = alpha_t * (1.0 - Yt) + (1.0 - alpha_t) * 0.5
            
            unnorm_1 = likelihood_1 * Y_pred_prob
            unnorm_0 = likelihood_0 * (1.0 - Y_pred_prob)
            
            posterior_prob = unnorm_1 / (unnorm_1 + unnorm_0 + 1e-8)
            
            if t_idx > 1:
                Y_prev = (torch.rand_like(posterior_prob) < posterior_prob).float()
            else:
                Y_prev = (posterior_prob > 0.5).float()
                
        elif self.discrete_noise_type == "absorbing":
            is_masked = (Yt == 0.5)
            
            if t_idx > 1:
                reveal_mask = torch.rand_like(Y_pred_prob) < alpha_bar_prev
                sampled = (torch.rand_like(Y_pred_prob) < Y_pred_prob).float()
                unmasked_val = torch.where(reveal_mask, sampled, torch.full_like(Yt, 0.5))
                Y_prev = torch.where(is_masked, unmasked_val, Yt)
            else:
                sampled = (Y_pred_prob > 0.5).float()
                Y_prev = torch.where(is_masked, sampled, Yt)
                
        elif self.discrete_noise_type == "discretized_gaussian":
            y_pred_logit = 2.0 * Y_pred_prob - 1.0
            
            if t_idx > 1:
                sqrt_alpha_prev = torch.sqrt(alpha_bar_prev)
                sigma_prev = torch.sqrt(1.0 - alpha_bar_prev)
                
                logit_arg = sqrt_alpha_prev * y_pred_logit / (sigma_prev + 1e-8)
                posterior_prob = 0.5 * (1.0 + torch.erf(logit_arg / math.sqrt(2.0)))
                
                Y_prev = (torch.rand_like(posterior_prob) < posterior_prob).float()
            else:
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
        """
        C = X0.shape[-1]
        
        if self.discrete_noise_type == "uniform":
            keep_mask = torch.rand(X0.shape[:-1], device=X0.device).unsqueeze(-1) < alpha_bar  # [B, N, 1]
            uniform_indices = torch.randint(0, C, X0.shape[:-1], device=X0.device)  # [B, N]
            uniform_onehot = F.one_hot(uniform_indices, C).float()  # [B, N, C]
            Xt = torch.where(keep_mask.expand_as(X0), X0, uniform_onehot)
            
        elif self.discrete_noise_type == "absorbing":
            absorb_mask = torch.rand(X0.shape[:-1], device=X0.device).unsqueeze(-1) > alpha_bar  # [B, N, 1]
            mask_state = torch.full_like(X0, 1.0 / C)
            Xt = torch.where(absorb_mask.expand_as(X0), mask_state, X0)
            
        elif self.discrete_noise_type == "discretized_gaussian":
            sigma = torch.sqrt(1.0 - alpha_bar)  # [B, 1, 1]
            sqrt_alpha = torch.sqrt(alpha_bar)    # [B, 1, 1]
            noise = torch.randn_like(X0) * sigma
            z = sqrt_alpha * X0 + noise
            hard_indices = z.argmax(dim=-1)  # [B, N]
            Xt = F.one_hot(hard_indices, C).float()  # [B, N, C]
        else:
            raise ValueError(f"Unknown discrete_noise_type: {self.discrete_noise_type}")
            
        return Xt

    def _categorical_posterior_sample(self, X_pred_logits, Xt, alpha_bar_t,
                                      alpha_bar_prev, t_idx):
        """
        Sample X_{t-1} from the categorical posterior p(X_{t-1} | X_t, X_pred).
        """
        C = X_pred_logits.shape[-1]
        X_pred_prob = F.softmax(X_pred_logits, dim=-1)  # [B, N, C]
        
        if self.discrete_noise_type == "uniform":
            alpha_t = alpha_bar_t / alpha_bar_prev
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
            is_masked = (Xt.max(dim=-1).values < (1.0 / C + 0.01))  # [B, N]
            
            if t_idx > 1:
                reveal = torch.rand(is_masked.shape, device=Xt.device) < alpha_bar_prev
                sampled_indices = torch.multinomial(
                    X_pred_prob.view(-1, C), num_samples=1
                ).view(X_pred_logits.shape[:-1])  # [B, N]
                sampled_onehot = F.one_hot(sampled_indices, C).float()
                mask_state = torch.full_like(Xt, 1.0 / C)
                revealed = torch.where(
                    reveal.unsqueeze(-1).expand_as(Xt), sampled_onehot, mask_state
                )
                X_prev = torch.where(is_masked.unsqueeze(-1).expand_as(Xt), revealed, Xt)
            else:
                final_indices = X_pred_prob.argmax(dim=-1)
                final_onehot = F.one_hot(final_indices, C).float()
                X_prev = torch.where(
                    is_masked.unsqueeze(-1).expand_as(Xt), final_onehot, Xt
                )
                
        elif self.discrete_noise_type == "discretized_gaussian":
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
        Calculates hybrid diffusion loss.
        """
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
        eps = eps * node_mask.unsqueeze(-1)
        nodes_count = node_mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6
        eps = eps - eps.sum(dim=1, keepdim=True) / nodes_count
            
        Rt = alpha_bar * R0 + sigma_bar * eps
        
        # 3. Corrupt adjacency matrices Y0 -> Yt (discrete diffusion)
        Yt = self._discrete_forward_diffuse(Y0, alpha_bar)
        
        # 4. Corrupt node categories X0 -> Xt (categorical discrete diffusion)
        Xt = self._categorical_forward_diffuse(X0, alpha_bar)
        
        # 5. Predict clean graph from noised state
        R_pred, Y_pred, X_pred = self(Xt, Rt, Yt, node_mask=None)
        
        # 6. Compute loss components
        coord_loss = F.mse_loss(
            R_pred * node_mask.unsqueeze(-1),
            R0 * node_mask.unsqueeze(-1),
            reduction='sum'
        )
        coord_loss = coord_loss / (node_mask.sum() * 3.0 + 1e-6)
            
        target_edges = Y0.squeeze(-1)  # [B, N, N]
        edge_loss = F.binary_cross_entropy_with_logits(Y_pred, target_edges, reduction='mean')
            
        node_target = X0.argmax(dim=-1)  # [B, N]
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
        
        val_mse = F.mse_loss(
            R_pred * node_mask.unsqueeze(-1),
            R0 * node_mask.unsqueeze(-1),
            reduction='sum'
        )
        val_mse = val_mse / (node_mask.sum() * 3.0 + 1e-6)
        
        node_target = X0.argmax(dim=-1)  # [B, N]
        node_preds = X_pred.argmax(dim=-1)  # [B, N]
        node_acc = (node_preds == node_target).float().mean()
            
        self.log("val_coord_mse", val_mse, on_epoch=True, prog_bar=True)
        self.log("val_node_acc", node_acc, on_epoch=True, prog_bar=True)
        return val_mse

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        
        if self.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs if self.trainer else 100,
                eta_min=1e-6,
            )
        elif self.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.lr_decay_steps,
                gamma=self.lr_decay_rate,
            )
        elif self.lr_scheduler == "none":
            return optimizer
        else:
            raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler}")
            
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_coord_mse",
                "interval": "epoch",
                "frequency": 1,
            }
        }

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, batch_size=1):
        """
        Unconditional generation sampling loop.
        """
        self.eval()
        N = self.n_max
        C = self.num_node_classes
        
        Rt = torch.randn((batch_size, N, 3), device=self.device)
        Rt = Rt - Rt.mean(dim=1, keepdim=True)
        
        if self.discrete_noise_type == "uniform":
            Yt = (torch.rand((batch_size, N, N, 1), device=self.device) > 0.5).float()
        elif self.discrete_noise_type == "absorbing":
            Yt = torch.full((batch_size, N, N, 1), 0.5, device=self.device)
        elif self.discrete_noise_type == "discretized_gaussian":
            Yt = (torch.rand((batch_size, N, N, 1), device=self.device) > 0.5).float()
        
        if self.discrete_noise_type == "uniform":
            rand_indices = torch.randint(0, C, (batch_size, N), device=self.device)
            Xt = F.one_hot(rand_indices, C).float()
        elif self.discrete_noise_type == "absorbing":
            Xt = torch.full((batch_size, N, C), 1.0 / C, device=self.device)
        elif self.discrete_noise_type == "discretized_gaussian":
            rand_indices = torch.randint(0, C, (batch_size, N), device=self.device)
            Xt = F.one_hot(rand_indices, C).float()
        
        for t_idx in reversed(range(1, self.T + 1)):
            R_pred, Y_pred, X_pred = self(Xt, Rt, Yt, node_mask=None)
            
            alpha_bar_t = self.alphas_bar[t_idx]
            alpha_bar_prev = self.alphas_bar[t_idx - 1]
            
            alpha_t = alpha_bar_t / alpha_bar_prev
            beta_t = 1.0 - alpha_t
            
            coef_pred = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
            coef_t = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            
            mu = coef_pred.view(-1, 1, 1) * R_pred + coef_t.view(-1, 1, 1) * Rt
            
            if t_idx > 1:
                sigma = torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * beta_t)
                z = torch.randn_like(Rt)
                z = z - z.mean(dim=1, keepdim=True)
                Rt = mu + sigma.view(-1, 1, 1) * z
            else:
                Rt = mu
                
            Yt = self._discrete_posterior_sample(Y_pred, Yt, alpha_bar_t, alpha_bar_prev, t_idx)
            Xt = self._categorical_posterior_sample(X_pred, Xt, alpha_bar_t, alpha_bar_prev, t_idx)
            
        edge_probs = torch.sigmoid(Y_pred)
        node_classes = X_pred.argmax(dim=-1)
        active_mask = (node_classes == 0)
        
        return Rt, edge_probs, active_mask

    # ------------------------------------------------------------------
    # CityJSON generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_cityjson(self, batch_size=1, threshold=0.5):
        """
        Generates buildings via diffusion sampling and exports each as a CityJSON dict.
        """
        from src.post_process.post_process import graph_to_cityjson
        
        Rt, edge_probs, active_mask = self.sample(batch_size=batch_size)
        
        results = []
        for i in range(batch_size):
            mask_i = active_mask[i]
            active_indices = mask_i.nonzero(as_tuple=True)[0]
            
            if len(active_indices) < 3:
                logger.warning(f"Building {i} has only {len(active_indices)} active nodes, skipping.")
                continue
            
            nodes = Rt[i, active_indices].cpu().numpy()
            edges = edge_probs[i][active_indices][:, active_indices].cpu().numpy()
            
            cj = graph_to_cityjson(
                nodes, edges,
                threshold=threshold,
                building_id=f"generated_building_{i}"
            )
            if cj:
                results.append(cj)
                
        return results
