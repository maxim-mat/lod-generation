"""Lightning module for non-autoregressive whole-mesh generation.

Holds the four swappable pieces together and owns nothing else: the dataset
decides the representation, the denoiser the network, the process the
schedule, and this module only routes between them and logs.

Loss is measured on a single corruption step, which is cheap enough to run
every batch. Geometry is measured by `MeshSetEvalCallback`, which runs the full
reverse trajectory and is therefore gated to every N epochs -- the same split
`MeshEvalCallback` uses on the autoregressive branch, and for the same reason.
"""
import logging

import torch
import torch.nn as nn
import lightning as L
from omegaconf import OmegaConf

from src.models.mesh_processes import create_process
from src.models.mesh_set_losses import (
    discrete_ce_loss,
    hungarian_loss,
    masked_mse_loss,
)
from src.models.mesh_set_unet import ConditionalMeshUNet
from src.utils.config import validate_combination

logger = logging.getLogger(__name__)


def create_denoiser(cfg, in_ch=10, out_bins=None):
    """Build the denoiser named by ``cfg.mesh_diffusion.denoiser``.

    Args:
        in_ch: input channels. 10 for every state but ``onehot``, which is
            ``9 * num_bins + 1`` (Task 11). Parameterised from the start so
            adding that arm is a call-site change, not a signature change.
        out_bins: set to ``num_bins`` for a categorical readout, ``None`` for a
            regression one.
    """
    d = cfg.mesh_diffusion
    if d.denoiser == "unet":
        return ConditionalMeshUNet(
            in_ch=in_ch, base=d.d_model // 4, cond_dim=d.d_model,
            time_dim=d.time_dim, n_head=d.n_head, dropout=d.dropout,
            out_bins=out_bins)
    if d.denoiser == "transformer":
        from src.models.mesh_set_transformer import MeshSetTransformer
        return MeshSetTransformer(
            in_ch=in_ch, d_model=d.d_model, n_head=d.n_head,
            num_layers=d.num_layers, dropout=d.dropout,
            pos_embed=d.pos_embed, time_dim=d.time_dim, out_bins=out_bins)
    raise ValueError(f"Unknown denoiser: {d.denoiser!r}.")


class MeshDiffusionModule(L.LightningModule):
    """Conditional face-set diffusion: LOD1 in, LOD2 out, one trajectory.

    Args:
        cfg: the resolved root `Config`. Validated here rather than in
            `train_mesh_diffusion` so a module built from a checkpoint in a
            notebook gets the same gate a training run does.
    """

    def __init__(self, cfg):
        super().__init__()
        validate_combination(cfg)
        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True))
        d = cfg.mesh_diffusion
        self.cfg_d = d
        self.num_bins = cfg.mesh_data.num_bins
        self.lr = cfg.training.lr
        self.lr_scheduler = cfg.training.lr_scheduler
        self.cond_dropout = d.cond_dropout
        self.guidance = d.guidance
        self.eval_steps = d.eval_steps
        self.loss_name = d.loss
        self.presence_weight = d.presence_weight
        self.match_presence_weight = d.match_presence_weight

        # The readout decides the head, not the process: a Gaussian process
        # with loss: ce also needs bin logits (Task 11, plan D10).
        self.state = d.state
        self.x0_clamp = d.x0_clamp
        self.categorical = d.loss == "ce"
        in_ch = 9 * self.num_bins + 1 if d.state == "onehot" else 10
        self.denoiser = create_denoiser(
            cfg, in_ch=in_ch, out_bins=self.num_bins if self.categorical else None)
        self.process = create_process(cfg)

    # -- plumbing ---------------------------------------------------------

    def _denoise(self, x, t, cond, mask, cond_mask):
        return self.denoiser(x, t, cond, mask, cond_mask)

    def _loss(self, pred, target, batch, x0=None):
        mask = batch["x_mask"]
        if self.loss_name == "mse":
            return masked_mse_loss(pred, target, mask, self.presence_weight)
        if self.loss_name == "hungarian":
            return hungarian_loss(pred, target, mask, self.presence_weight,
                                  self.match_presence_weight)
        if self.loss_name == "ce":
            logits, presence_logits = pred
            return discrete_ce_loss(
                logits, batch["x_bins"], mask, presence_logits,
                mask.float(), self.presence_weight)
        raise ValueError(f"Unknown loss: {self.loss_name!r}.")

    def _shared_step(self, batch, stage):
        x0 = batch["x"]
        cond = batch["cond"]
        cond_mask = batch["cond_mask"]
        # Classifier-free guidance needs an unconditional branch to exist, and
        # it only exists if training sometimes saw no condition. Dropped per
        # *batch*, not per sample, which is what the reference implementations
        # do and what keeps the cross-attention path a single code path.
        if stage == "train" and torch.rand(1).item() < self.cond_dropout:
            cond, cond_mask = None, None

        t = self.process.sample_t(x0.shape[0], x0.device)
        x_t, aux = self.process.corrupt(x0, t)
        pred = self._denoise(x_t, t, cond, batch["x_mask"], cond_mask)
        target = self.process.target_for(x0, x_t, t, aux)
        loss = self._loss(pred, target, batch, x0=x0)

        self.log(f"{stage}_loss", loss, on_step=(stage == "train"),
                 on_epoch=True, prog_bar=True, batch_size=x0.shape[0])
        # Coordinate error in bins, logged separately from the objective:
        # under target=noise the loss is in epsilon units and is not comparable
        # across timesteps, so it says nothing about geometry on its own.
        if self.loss_name != "ce":
            with torch.no_grad():
                x0_hat = self.process.to_x0(x_t, float(t[0]), pred) \
                    if hasattr(self.process, "to_x0") else pred
                err = (x0_hat[:, :9] - x0[:, :9]).abs()
                m = batch["x_mask"][:, None, :].float()
                bins = (err * m).sum() / m.sum().clamp(min=1) * (self.num_bins - 1)
            self.log(f"{stage}_coord_err_bins", bins, on_epoch=True,
                     batch_size=x0.shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if self.lr_scheduler == "none":
            return opt
        if self.lr_scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.trainer.max_epochs if self.trainer else 100,
                eta_min=0.01 * self.lr)
            return {"optimizer": opt, "lr_scheduler": sched}
        raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler!r}.")

    # -- sampling ---------------------------------------------------------

    def _state_shape(self, batch):
        """Shape the process allocates its prior at.

        Not `batch["x"].shape`: the channel count follows `state`, and the
        `onehot` arm (Task 11) is 9 * num_bins + 1 channels wide while `bins`
        (Task 12) is 9. Centralised here so adding either is a one-line change
        rather than a hunt through the sampler.
        """
        b, _, f = batch["x"].shape
        if self.state == "bins":
            return (b, 9, f)
        if self.state == "onehot":
            return (b, 9 * self.num_bins + 1, f)
        return (b, 10, f)

    @torch.no_grad()
    def generate(self, batch, n_steps=None, scaffold=None):
        """Run the full reverse trajectory from the LOD1 condition.

        Args:
            batch: a collated batch; only `cond`, `cond_mask` and the shape of
                `x` are read. The target `x` is never looked at.
            n_steps: reverse steps. Defaults to `mesh_diffusion.eval_steps`.
            scaffold: optional ``(t, x) -> x`` projection, supplied by
                `MeshSetEvalCallback` when `scaffold.enabled`. Kept as a
                parameter rather than read from config so a caller can score
                the same checkpoint with and without it.

        Returns:
            Tensor: ``[B, 10, F]`` in the LOD1-normalized frame.
        """
        cond, cond_mask = batch["cond"], batch["cond_mask"]
        mask = batch["x_mask"]
        shape = self._state_shape(batch)

        def denoiser_fn(x_t, t):
            out = self._denoise(x_t, t, cond, mask, cond_mask)
            if self.guidance == 1.0:
                return out
            # Classifier-free guidance (Ho & Salimans, arXiv:2207.12598):
            # push away from the unconditional prediction. Two forward passes
            # per step, so it doubles the eval cost -- which is why it defaults
            # to 1.0 and is a per-experiment choice.
            uncond = self._denoise(x_t, t, None, mask, None)
            return uncond + self.guidance * (out - uncond)

        return self.process.sample(denoiser_fn, shape, batch["x"].device,
                                   n_steps=n_steps or self.eval_steps,
                                   callback=scaffold)
