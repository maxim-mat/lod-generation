"""Mixed graph/3D denoising diffusion for CityJSON building graphs.

Follows MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation
(Vignac et al., arXiv:2302.09048), https://github.com/cvignac/MiDi

Adaptations for this dataset:
  * MiDi's atom types become the Levi graph's node classes (vertex / ground /
    roof / wall face / off); its bond types become the three edge classes
    (off / vertex-vertex / vertex-face). Formal charges are dropped.
  * MiDi samples the node count from the training distribution and pads with a
    node mask. Here every one of the n_max nodes takes part in message passing,
    and the Off category is a genuinely diffused class that the network must
    predict -- that is how the generated building's size is chosen. The node
    mask from the dataset marks the vertex (coordinate-carrying) nodes and is
    only used to centre the coordinates and to report coordinate metrics.
"""
import logging

import lightning as L
import torch
import torch.nn.functional as F

from src.models.noise import GraphNoiseModel, zero_diagonal
from src.models.regnn import rEGNNTransformer

logger = logging.getLogger(__name__)


class CityJSONDiffusionModule(L.LightningModule):
    def __init__(self, num_node_classes=5, num_edge_classes=3,
                 hidden_dim=64, num_layers=4, T=500,
                 lr=1e-3, discrete_noise_type="marginal", n_max=64,
                 lr_scheduler="none", lr_decay_steps=50, lr_decay_rate=0.5,
                 edge_dim=32, global_dim=32, n_head=8, dropout=0.1,
                 x_marginals=None, e_marginals=None,
                 nu_pos=2.5, nu_x=1.0, nu_e=1.5,
                 lambda_pos=3.0, lambda_x=0.4, lambda_e=2.0,
                 coord_scale=1.0, equivariance="so2", z_shift=0.0,
                 time_embed="scalar"):
        """
        Args:
            discrete_noise_type (str): 'marginal' (limit distribution = training
                class marginals, MiDi's default) or 'uniform'.
            x_marginals (Tensor, optional): [num_node_classes] class frequencies
                of the training split. Defaults to uniform.
            e_marginals (Tensor, optional): [num_edge_classes] edge class
                frequencies of the training split. Defaults to uniform.
            nu_pos, nu_x, nu_e (float): Cosine schedule exponents per feature.
                nu_pos > nu_e destroys coordinates faster than graph structure.
            lambda_pos, lambda_x, lambda_e (float): Loss weights.
            coord_scale (float): Metres per unit of the model's coordinate space.
                Targets are divided by it so the data matches the unit-variance
                Gaussian the position diffusion noises towards; `generate_cityjson`
                multiplies by it to return metres. Obtain it from
                `CityJSONDataModule.compute_coord_scale()`. Saved as a
                hyperparameter, so inference restores it from the checkpoint.
            equivariance (str): 'so2' (default), 'se2' (xy-only CoM, absolute z)
                or 'o3'. Threads into both the network and the noise model.
            z_shift (float): se2 only — metres subtracted from z before
                `coord_scale` scaling (train-split mean vertex height, from
                `CityJSONDataModule.compute_z_shift()`); added back by
                `generate_cityjson`. Forced to 0.0 for non-se2 modes. Saved as
                a hyperparameter like `coord_scale`.
            time_embed (str): 'scalar' (raw t/T, MiDi's design) or 'sinusoidal'
                (Fourier lift of t/T before the global-feature MLP).
        """
        super().__init__()
        self.save_hyperparameters()

        if coord_scale <= 0:
            raise ValueError(f"coord_scale must be positive, got {coord_scale}.")

        self.T = T
        self.lr = lr
        self.n_max = n_max
        self.coord_scale = float(coord_scale)
        # Batches dropped for a non-finite loss. Not a buffer: it describes this
        # run, not the model, and must not travel with a checkpoint.
        self._nonfinite_skips = 0
        self.num_node_classes = num_node_classes
        self.num_edge_classes = num_edge_classes
        self.equivariance = equivariance
        self.z_shift = float(z_shift) if equivariance == "se2" else 0.0
        self.lr_scheduler = lr_scheduler
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = lr_decay_rate
        self.lambda_pos = lambda_pos
        self.lambda_x = lambda_x
        self.lambda_e = lambda_e

        if x_marginals is None:
            x_marginals = torch.ones(num_node_classes) / num_node_classes
        if e_marginals is None:
            e_marginals = torch.ones(num_edge_classes) / num_edge_classes
        x_marginals = torch.as_tensor(x_marginals, dtype=torch.float32)
        e_marginals = torch.as_tensor(e_marginals, dtype=torch.float32)

        # ponytail: one knob drives both X and E transitions. GraphNoiseModel keeps
        # them separate so they can be ablated independently -- add two config
        # fields when that ablation is actually run.
        self.noise = GraphNoiseModel(
            T=T, x_marginals=x_marginals, e_marginals=e_marginals,
            transition_x=discrete_noise_type, transition_e=discrete_noise_type,
            nu_pos=nu_pos, nu_x=nu_x, nu_e=nu_e,
            xy_only_com=equivariance == "se2",
        )

        self.network = rEGNNTransformer(
            num_node_classes=num_node_classes,
            num_edge_classes=num_edge_classes,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            global_dim=global_dim,
            n_head=n_head,
            num_layers=num_layers,
            dropout=dropout,
            equivariance=equivariance,
            time_embed=time_embed,
        )

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _centre_positions(R0, node_categories, xy_only=False, z_shift=0.0):
        """Centre real (vertex + face) nodes, leaving Off nodes at 0.

        The network is constrained to emit zero-CoM coordinates (PositionsMLP and
        every layer re-centre over all N slots), so the target must live on the
        same subspace or the coordinate loss has an irreducible floor. Centering
        over the real nodes and pinning Off slots to zero makes the all-N mean
        exactly zero. The real mask is 1 - P(last class): the last class is
        Off/Virtual in both the 5-class and legacy 2-class conventions.

        Under se2 (`xy_only=True`) only the xy mean is removed — z keeps its
        absolute value minus `z_shift` (the train-split mean vertex height), the
        moment-matching analogue of the discrete marginal priors.
        """
        real = (1.0 - node_categories[..., -1]).unsqueeze(-1)     # [B, N, 1]
        num_real = real.sum(dim=1, keepdim=True).clamp(min=1)
        mean = (R0 * real).sum(dim=1, keepdim=True) / num_real
        if xy_only:
            mean = torch.cat(
                (mean[..., :2], torch.full_like(mean[..., 2:], z_shift)), dim=-1
            )
        return (R0 - mean) * real

    def _prepare(self, batch):
        """Unpack a batch into clean (pos, X, E) plus the network's node mask.

        Positions come back in scaled units (metres / `coord_scale`), which is the
        space the whole diffusion -- and therefore every coordinate metric -- lives in.
        """
        R0 = self._centre_positions(
            batch["x"], batch["node_categories"],
            xy_only=self.equivariance == "se2", z_shift=self.z_shift,
        ) / self.coord_scale
        X0 = batch["node_categories"]
        E0 = zero_diagonal(
            F.one_hot(batch["y"].squeeze(-1).long(), self.num_edge_classes).float()
        )
        # Every node participates: Off is a class, not padding.
        net_mask = torch.ones_like(batch["node_mask"])
        return R0, X0, E0, net_mask

    # ------------------------------------------------------------------
    # Forward / losses
    # ------------------------------------------------------------------

    def forward(self, X_t, R_t, Y_t, t_int, node_mask=None):
        t_norm = t_int.float() / self.T
        return self.network(X_t, R_t, Y_t, t_norm, node_mask)

    def _shared_step(self, batch, t_int=None):
        R0, X0, E0, net_mask = self._prepare(batch)
        z = self.noise.apply_noise(R0, X0, E0, net_mask, t_int=t_int)

        R_pred, E_pred, X_pred = self(
            z["X_t"], z["pos_t"], z["E_t"], z["t_int"], node_mask=net_mask
        )

        # Positions: MiDi predicts x0 directly and regresses it with an MSE.
        coord_loss = F.mse_loss(R_pred, R0)

        node_loss = F.cross_entropy(
            X_pred.reshape(-1, self.num_node_classes), X0.argmax(dim=-1).reshape(-1)
        )

        # Self-loops are not modelled, so they must not enter the edge loss.
        n = E0.shape[1]
        off_diag = ~torch.eye(n, dtype=torch.bool, device=E0.device)
        edge_loss = F.cross_entropy(
            E_pred[:, off_diag].reshape(-1, self.num_edge_classes),
            E0[:, off_diag].argmax(dim=-1).reshape(-1),
        )

        total = self.lambda_pos * coord_loss + self.lambda_x * node_loss + self.lambda_e * edge_loss
        return total, coord_loss, node_loss, edge_loss, X_pred, R_pred, R0

    def _active_coord_mse(self, R_pred, R0, node_mask):
        """Coordinate MSE over real nodes only, for a comparable monitored metric."""
        mask = node_mask.unsqueeze(-1)
        sq_err = ((R_pred - R0) ** 2 * mask).sum()
        return sq_err / (mask.sum() * 3.0 + 1e-6)

    def _skip_nonfinite_batch(self, batch_idx, total, coord_loss, node_loss, edge_loss):
        """Drop a batch whose loss is not finite, loudly.

        A single NaN or inf reaching backward fills every gradient with NaN, and
        AdamW then writes NaN into every parameter and both moment buffers; the
        model never recovers. Returning None makes Lightning skip backward and the
        optimizer step, leaving the weights untouched.

        Caveat: if the parameters are *already* non-finite, every subsequent batch
        is skipped too and the run will make no progress while still looking alive.
        `train_nonfinite_skips` climbing step-for-step is that failure.
        """
        self._nonfinite_skips += 1
        logger.warning(
            "Non-finite training loss at epoch %d, batch %d (global step %d): "
            "total=%s coord_mse=%s node_ce=%s edge_ce=%s. "
            "Skipping the optimizer step; %d batch(es) skipped so far.",
            self.current_epoch, batch_idx, self.global_step,
            total.item(), coord_loss.item(), node_loss.item(), edge_loss.item(),
            self._nonfinite_skips,
        )
        self._log_nonfinite_skips()
        return None

    def _log_nonfinite_skips(self):
        # Logged every step, not just on failure, so the wandb series is a
        # continuous line that steps up rather than a scatter of isolated points.
        self.log("train_nonfinite_skips", float(self._nonfinite_skips),
                 on_step=True, on_epoch=False, prog_bar=True)

    def training_step(self, batch, batch_idx):
        total, coord_loss, node_loss, edge_loss, *_ = self._shared_step(batch)

        if not torch.isfinite(total):
            return self._skip_nonfinite_batch(batch_idx, total, coord_loss, node_loss, edge_loss)

        self.log("train_loss", total, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_coord_mse", coord_loss, on_epoch=True, prog_bar=True)
        self.log("train_edge_ce", edge_loss, on_epoch=True)
        self.log("train_node_ce", node_loss, on_epoch=True)
        self._log_nonfinite_skips()
        return total

    def _eval_step(self, batch, prefix):
        # A fixed mid-chain timestep keeps the monitored metric comparable across epochs.
        B = batch["x"].shape[0]
        t_int = torch.full((B, 1), self.T // 2, dtype=torch.long, device=self.device)

        total, coord_loss, node_loss, edge_loss, X_pred, R_pred, R0 = self._shared_step(batch, t_int=t_int)

        node_mask = batch["node_mask"]
        coord_mse = self._active_coord_mse(R_pred, R0, node_mask)
        # Plain accuracy is inflated by the Off-padding majority; report
        # precision/recall of the minority vertex class (class 0) instead.
        # Metric names kept for dashboard continuity: "real" == vertex.
        pred_active = X_pred.argmax(dim=-1) == 0
        true_active = batch["node_categories"].argmax(dim=-1) == 0
        tp = (pred_active & true_active).float().sum()
        real_precision = tp / pred_active.float().sum().clamp(min=1)
        real_recall = tp / true_active.float().sum().clamp(min=1)

        self.log(f"{prefix}_coord_mse", coord_mse, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_real_precision", real_precision, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_real_recall", real_recall, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_edge_ce", edge_loss, on_epoch=True)
        self.log(f"{prefix}_loss", total, on_epoch=True)
        return coord_mse

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, "test")

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
        """Run the reverse chain from the limit distribution.

        Returns:
            pos (Tensor): [B, n_max, 3] coordinates, in scaled units. Multiply by
                `coord_scale` for metres.
            node_labels (Tensor): [B, n_max] long, sampled node class labels.
            edge_labels (Tensor): [B, n_max, n_max] long, sampled edge class labels.
        """
        self.eval()
        N = self.n_max
        node_mask = torch.ones((batch_size, N), device=self.device)

        pos, X_t, E_t = self.noise.sample_limit_dist(batch_size, N, self.device)

        for s_int in reversed(range(0, self.T)):
            t_int = torch.full((batch_size, 1), s_int + 1, dtype=torch.long, device=self.device)
            s_arr = torch.full((batch_size, 1), s_int, dtype=torch.long, device=self.device)

            R_pred, E_pred, X_pred = self(X_t, pos, E_t, t_int, node_mask=node_mask)
            pos, X_t, E_t = self.noise.sample_zs_from_zt_and_pred(
                pos_t=pos, X_t=X_t, E_t=E_t,
                pred_pos=R_pred, pred_X=X_pred, pred_E=E_pred,
                t_int=t_int, s_int=s_arr, node_mask=node_mask,
            )

        # The chain's final state is the sample; do not re-read the network head.
        node_labels = X_t.argmax(dim=-1)
        edge_labels = E_t.argmax(dim=-1)
        return pos, node_labels, edge_labels

    # ------------------------------------------------------------------
    # CityJSON generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_cityjson(self, batch_size=1):
        """
        Generates buildings via diffusion sampling and exports each as a CityJSON dict.
        """
        from src.dataset.dataset import VERTEX
        from src.post_process.post_process import graph_to_cityjson

        pos, node_labels, edge_labels = self.sample(batch_size=batch_size)

        results = []
        for i in range(batch_size):
            n_vertices = int((node_labels[i] == VERTEX).sum())
            if n_vertices < 3:
                logger.warning(f"Building {i} has only {n_vertices} vertex nodes, skipping.")
                continue

            # The chain runs in scaled units; CityJSON is metres. se2 keeps
            # absolute heights: restore the z offset the targets subtracted.
            coords = (pos[i] * self.coord_scale).cpu().numpy()
            coords[:, 2] += self.z_shift
            cj = graph_to_cityjson(
                coords,
                node_labels[i].cpu().numpy(),
                edge_labels[i].cpu().numpy(),
                building_id=f"generated_building_{i}",
            )
            if cj:
                results.append(cj)

        return results
