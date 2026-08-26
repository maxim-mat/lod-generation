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


def _guide(cond_out, uncond_out, weight):
    """Classifier-free guidance, for a tensor prediction or a logits tuple."""
    if torch.is_tensor(cond_out):
        return uncond_out + weight * (cond_out - uncond_out)
    return tuple(u + weight * (c - u) for c, u in zip(cond_out, uncond_out))


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

    # -- state routing (plan D10) -----------------------------------------

    def _to_state(self, batch):
        """The tensor the *process* corrupts, for this arm's state.

        `continuous` and `quantized` corrupt the [B,10,F] float layout directly
        -- they differ only in whether the target was snapped, which happened in
        the dataset. `onehot` expands the bins here rather than in the
        dataloader; see `_pack`'s docstring for why.
        """
        if self.state in ("continuous", "quantized"):
            return batch["x"]
        if self.state == "bins":
            return batch["x_bins"]
        b, _, f = batch["x"].shape
        oh = torch.zeros(b, 9, self.num_bins, f, device=batch["x"].device)
        oh.scatter_(2, batch["x_bins"].unsqueeze(2), 1.0)
        # Centred to +-0.5, matching the coordinate and presence channels, so
        # one Gaussian noise level is correctly scaled for every channel.
        oh = oh - 0.5
        return torch.cat([oh.reshape(b, 9 * self.num_bins, f),
                          batch["x"][:, 9:10]], dim=1)

    def _clamp_x0(self, logits):
        """Predicted bin logits back to a state the next reverse step can take.

        Hard clamping (`argmax` to a one-hot) is Diffusion-LM's trick
        (arXiv:2205.14217 section 4.2) and is what keeps a *continuous* process
        over a *discrete* alphabet landing on grid points instead of drifting
        between them. `soft` keeps the softmax and is the ablation that shows
        whether the trick is load-bearing here.

        Returns:
            tuple: ``(state_x0, bins)``. For `onehot` the state is the
            (clamped) distribution itself, centred; for `quantized` it is the
            single coordinate that distribution collapses to.
        """
        probs = torch.softmax(logits, dim=2)              # [B,9,K,F]
        if self.x0_clamp == "hard":
            idx = probs.argmax(dim=2, keepdim=True)
            probs = torch.zeros_like(probs).scatter_(2, idx, 1.0)
        if self.state == "onehot":
            return probs - 0.5, probs.argmax(dim=2)
        # `quantized`: collapse the distribution to a single coordinate. Under
        # hard clamping this is the bin centre; under soft it is the posterior
        # mean, which is what makes the two visibly different at sample time.
        centres = torch.linspace(-0.5, 0.5, self.num_bins,
                                 device=probs.device).view(1, 1, -1, 1)
        return (probs * centres).sum(dim=2), probs.argmax(dim=2)

    def _from_state(self, coords, bins, presence_logits):
        """A categorical readout back to the ``[B,10,F]`` eval layout.

        `coords` wins when it is given -- under `x0_clamp: soft` the posterior
        mean is deliberately *not* a grid point, and rebuilding from `bins`
        would quietly undo the ablation being measured.
        """
        if coords is None:
            from src.dataset.mesh_dataset import dequantize
            coords = torch.from_numpy(
                dequantize(bins.detach().cpu().numpy(), self.num_bins)
            ).float().to(presence_logits.device)
        b, f = presence_logits.shape
        out = torch.zeros(b, 10, f, device=presence_logits.device)
        out[:, :9] = coords
        out[:, 9] = torch.where(presence_logits > 0, 0.5, -0.5)
        return out

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
        x0 = self._to_state(batch)
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
        if self.categorical:
            # The CE label is `x_bins`, which the loss reads off the batch --
            # there is no regression target to build here.
            loss = self._loss(pred, None, batch)
        else:
            loss = self._loss(pred, self.process.target_for(x0, x_t, t, aux),
                              batch, x0=x0)

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
            if self.guidance != 1.0:
                # Classifier-free guidance (Ho & Salimans, arXiv:2207.12598):
                # push away from the unconditional prediction. Two forward
                # passes per step, so it doubles the eval cost -- which is why
                # it defaults to 1.0 and is a per-experiment choice.
                uncond = self._denoise(x_t, t, None, mask, None)
                out = _guide(out, uncond, self.guidance)
            if not self.categorical or self.state == "bins":
                return out
            # Categorical readout over a Gaussian process: the process expects
            # an x0-shaped tensor, so project the logits back to the state.
            logits, presence_logits = out
            state_x0, self._last_bins = self._clamp_x0(logits)
            self._last_presence = presence_logits
            self._last_coords = state_x0 if self.state == "quantized" else None
            if self.state == "onehot":
                b, _, f = x_t.shape
                state_x0 = state_x0.reshape(b, 9 * self.num_bins, f)
            return torch.cat([state_x0, presence_logits.unsqueeze(1)], dim=1)

        out = self.process.sample(denoiser_fn, shape, batch["x"].device,
                                  n_steps=n_steps or self.eval_steps,
                                  callback=scaffold)
        if not self.categorical:
            return out
        # The trajectory's last x0 *prediction* is the answer, not the process's
        # own final state: that state is the same x0 re-noised by one schedule
        # rung, which would drift it straight back off the grid the clamp just
        # put it on -- the entire point of these arms.
        return self._from_state(self._last_coords, self._last_bins,
                                self._last_presence)
