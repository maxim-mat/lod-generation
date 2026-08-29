"""Lightning module for non-autoregressive whole-mesh generation.

Holds the four swappable pieces together and owns nothing else: the dataset
decides the representation, the denoiser the network, the process the
schedule, and this module only routes between them and logs.

Loss is measured on a single corruption step, which is cheap enough to run
every batch. Geometry is measured by `MeshSetEvalCallback`, which runs the full
reverse trajectory and is therefore gated to every N epochs -- the same split
`MeshEvalCallback` uses on the autoregressive branch, and for the same reason.
"""
import contextlib
import logging

import torch
import torch.nn as nn
import lightning as L
from omegaconf import OmegaConf
from torch.optim.swa_utils import AveragedModel

from src.models.mesh_processes import create_process
from src.models.mesh_set_losses import (
    discrete_ce_loss,
    hungarian_loss,
    masked_mean_per_sample,
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
        # The positional encoding is scaled by the corpus face cap, never by
        # the batch's padded width -- otherwise the same face is encoded
        # differently depending on which buildings share its batch.
        return MeshSetTransformer(
            in_ch=in_ch, d_model=d.d_model, n_head=d.n_head,
            num_layers=d.num_layers, dropout=d.dropout,
            pos_embed=d.pos_embed, time_dim=d.time_dim, out_bins=out_bins,
            pos_scale=cfg.mesh_data.max_faces or 200, time_cond=d.time_cond)
    raise ValueError(f"Unknown denoiser: {d.denoiser!r}.")


def _ramped_ema(decay):
    """EMA whose decay ramps in, rather than a constant from step one.

    A fresh `AveragedModel` is a copy of the *initialisation*, so a constant
    0.999 leaves it 90% initial noise after 100 updates and 37% after 1000 --
    measured. Since the early-stopping monitor samples from these weights, that
    is not a slow start, it is a monitor reporting on noise. `min(decay,
    (1+n)/(10+n))` tracks the live weights closely at first and tightens toward
    `decay` as the average earns its length.
    """
    @torch.no_grad()
    def ema_update(ema_params, current_params, num_averaged):
        n = float(num_averaged)
        d = min(decay, (1.0 + n) / (10.0 + n))
        torch._foreach_lerp_(list(ema_params), list(current_params), 1.0 - d)
    return ema_update


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
        self.warmup_steps = cfg.training.warmup_steps
        self.lr_decay_steps = cfg.training.lr_decay_steps
        self.lr_decay_rate = cfg.training.lr_decay_rate
        self.cond_dropout = d.cond_dropout
        self.guidance = d.guidance
        self.eval_steps = d.eval_steps
        self.loss_name = d.loss
        self.presence_weight = d.presence_weight
        self.match_presence_weight = d.match_presence_weight
        self.min_snr_gamma = d.min_snr_gamma

        # The readout decides the head, not the process: a Gaussian process
        # with loss: ce also needs bin logits (Task 11, plan D10).
        self.state = d.state
        self.x0_clamp = d.x0_clamp
        self.categorical = d.loss == "ce"
        in_ch = 9 * self.num_bins + 1 if d.state == "onehot" else 10
        self.denoiser = create_denoiser(
            cfg, in_ch=in_ch, out_bins=self.num_bins if self.categorical else None)
        self.process = create_process(cfg)

        # Sampling-time weights. Held on the module, not in a callback, so they
        # land in `state_dict` automatically -- checkpoints are selected on a
        # metric computed WITH them -- and so no callback ordering decides
        # whether eval sees them.
        self.ema_start_step = d.ema_start_step
        self.eval_weights = "ema"
        self.ema = None
        if d.ema_decay is not None:
            self.ema = AveragedModel(self.denoiser,
                                     multi_avg_fn=_ramped_ema(d.ema_decay))

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
            # [B,10,F]: nine coordinate-bin channels plus a two-state occupancy
            # channel, so presence rides its own categorical chain instead of
            # being a bare logit carried past the reverse loop.
            return torch.cat([batch["x_bins"],
                              batch["x_mask"].long().unsqueeze(1)], dim=1)
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

    def _model_input(self, x_t):
        """The process's state as something a denoiser can consume.

        Only `bins` needs this: its state is int64 indices, and the network
        reads the same [B, 10, F] float layout every other arm does. Presence
        is left at 0 -- at input time it is unknown, and the model predicts it.
        """
        if self.state != "bins":
            return x_t
        from src.dataset.mesh_dataset import dequantize
        coords = torch.from_numpy(
            dequantize(x_t[:, :9].detach().cpu().numpy(), self.num_bins)).float()
        out = torch.zeros(x_t.shape[0], 10, x_t.shape[-1], device=x_t.device)
        out[:, :9] = coords.to(x_t.device)
        # The network SEES the current occupancy state, mapped to the same
        # +-0.5 the continuous arms use. Leaving it at 0 would hide the very
        # variable the chain is there to refine.
        out[:, 9] = torch.where(x_t[:, 9] > 0, 0.5, -0.5)
        return out

    # -- plumbing ---------------------------------------------------------

    def _denoise(self, x, t, cond, mask, cond_mask):
        return self._eval_net()(x, t, cond, mask, cond_mask)

    def _eval_net(self):
        """The denoiser to run: the EMA copy outside training, once it has
        actually accumulated something. `n_averaged == 0` means it is still the
        initialisation, which would make every sampled metric meaningless."""
        if (self.ema is not None and not self.training
                and self.eval_weights == "ema" and int(self.ema.n_averaged) > 0):
            return self.ema.module
        return self.denoiser

    @contextlib.contextmanager
    def using_weights(self, which):
        """Temporarily sample from "ema" or "live" weights.

        EMA does not touch the gradients, so the live-weight metric from an
        EMA run IS the no-EMA result -- the training trajectories are
        identical. That makes the EMA/no-EMA comparison free inside every run,
        and leaves only the DECAY LENGTH needing separate runs.
        """
        prev = self.eval_weights
        self.eval_weights = which
        try:
            yield
        finally:
            self.eval_weights = prev

    @property
    def ema_active(self):
        """True once the EMA holds something other than the initialisation."""
        return self.ema is not None and int(self.ema.n_averaged) > 0

    def _loss(self, pred, target, batch, x0=None, weight=None):
        mask = batch["x_mask"]
        if self.loss_name == "mse":
            return masked_mse_loss(pred, target, mask, self.presence_weight,
                                   weight)
        if self.loss_name == "hungarian":
            return hungarian_loss(pred, target, mask, self.presence_weight,
                                  self.match_presence_weight, weight)
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
        # No target-axis mask. The surplus slots are the model's no-object
        # slots, not padding to be hidden: handing over `x_mask` told it
        # exactly where the mesh ended, so presence had nothing left to learn
        # and face count came from the label rather than the model. The mask
        # still gates the COORDINATE loss below -- that is supervision.
        pred = self._denoise(self._model_input(x_t), t, cond, None, cond_mask)
        # min-SNR: flatten the SNR weighting an epsilon objective applies for
        # free. `validate_combination` has already ruled out the arms where the
        # construction does not hold, so no capability check is needed here.
        w = (self.process.snr_weight(t, self.min_snr_gamma)
             if self.min_snr_gamma is not None else None)
        if self.categorical:
            # The CE label is `x_bins`, which the loss reads off the batch --
            # there is no regression target to build here.
            loss = self._loss(pred, None, batch)
        else:
            loss = self._loss(pred, self.process.target_for(x0, x_t, t, aux),
                              batch, x0=x0, weight=w)

        self.log(f"{stage}_loss", loss, on_step=(stage == "train"),
                 on_epoch=True, prog_bar=True, batch_size=x0.shape[0])
        if self.state == "bins":
            with torch.no_grad():
                logits = pred[0]
                correct = (logits.argmax(dim=2) == x0[:, :9]).float()
                m = batch["x_mask"][:, None, :].float()
                acc = (correct * m).sum() / (m.sum() * 9).clamp(min=1)
                # Bucketed by noise level, because the aggregate hides exactly
                # what D11 is about: heads that are accurate at low t and carry
                # nothing at high t still average to a healthy-looking number,
                # and the uniform-vs-gaussian transition A/B is precisely a
                # question about the high-noise buckets.
                per_sample = (correct * m).sum(dim=(1, 2)) / (
                    m.sum(dim=(1, 2)) * 9).clamp(min=1)
            self.log(f"{stage}_bin_acc", acc, on_epoch=True,
                     batch_size=x0.shape[0])
            self._log_by_t(stage, "bin_acc", per_sample, t)
        # Coordinate error in bins, logged separately from the objective:
        # under target=noise the loss is in epsilon units and is not comparable
        # across timesteps, so it says nothing about geometry on its own.
        if self.loss_name != "ce":
            with torch.no_grad():
                # `t`, not `float(t[0])`: every row draws its own time, and
                # converting the whole batch at row 0's noise level reads the
                # other rows off the wrong point of the schedule entirely.
                x0_hat = self.process.to_x0(x_t, t, pred) \
                    if hasattr(self.process, "to_x0") else pred
                err = (x0_hat[:, :9] - x0[:, :9]).abs()
                # Per SAMPLE, so the buckets below can split it by that row's
                # own t. `masked_mean_per_sample` divides by the nine coordinate
                # channels as well as the face slots; doing that by hand here
                # without the channel term is what made this metric read 9x
                # high -- 378 bins on a 128-bin grid.
                bins = masked_mean_per_sample(err, batch["x_mask"]) \
                    * (self.num_bins - 1)
            self.log(f"{stage}_coord_err_bins", bins.mean(), on_epoch=True,
                     batch_size=x0.shape[0])
            self._log_by_t(stage, "coord_err_bins", bins, t)
        return loss

    def _log_by_t(self, stage, name, per_sample, t):
        """Log `per_sample` split into four buckets of each row's OWN t.

        The aggregate is the one number that cannot answer the question this
        branch keeps hitting: an x0 estimate that is sharp at low noise and
        carries nothing at high noise averages to a healthy-looking figure,
        while the reverse trajectory is decided in the bucket that carries
        nothing. Four buckets is enough to see that and cheap enough to leave
        on for every arm.
        """
        bucket = (t.clamp(0.0, 0.999) * 4).long()
        for b in range(4):
            sel = bucket == b
            if sel.any():
                self.log(f"{stage}_{name}_t{b}", per_sample[sel].mean(),
                         on_epoch=True, batch_size=int(sel.sum()))

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Fold the live weights into the EMA, once they are worth averaging."""
        if self.ema is not None and self.global_step >= self.ema_start_step:
            self.ema.update_parameters(self.denoiser)

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        """Deliberately empty.

        The test tier is the free-running reverse trajectory and the geometric
        metrics it produces, run by `MeshSetEvalCallback.on_test_epoch_end`.
        A single-step loss here would be a different quantity reported under
        the same banner, and it is not what the branch is being judged on.
        Lightning still needs the hook to exist for `trainer.test()`.
        """
        return None

    def configure_optimizers(self):
        """AdamW, with optional linear warmup composed over any schedule.

        Warmup runs on the STEP interval, never the epoch one. Lightning
        defaults `interval` to "epoch", so a 500-step warmup left on the
        default would ramp over 500 *epochs* -- at 127 steps/epoch that is
        63,500 steps of ramp instead of 500. It trains, badly, silently, which
        is why every branch below sets the interval explicitly.
        """
        import torch.optim.lr_scheduler as S

        # `self.denoiser`, not `self.parameters()`: the latter also yields the
        # EMA copy, which is exactly as large. Their grads stay None so AdamW
        # skips them today, but handing an optimizer parameters it must never
        # update is the kind of thing a later change quietly turns into a bug.
        opt = torch.optim.AdamW(self.denoiser.parameters(), lr=self.lr)
        warmup = int(self.warmup_steps)

        if self.lr_scheduler not in ("none", "cosine", "step"):
            raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler!r}.")

        if self.lr_scheduler == "none":
            if warmup <= 0:
                return opt
            # Ramp, then flat. Constant-after-warmup is the diffusion
            # mainstream (DDPM, Improved DDPM, ADM, Imagen all do this); the
            # annealing habit comes from classifiers.
            sched = S.LambdaLR(opt, lambda step: min(1.0, (step + 1) / warmup))
            return {"optimizer": opt,
                    "lr_scheduler": {"scheduler": sched, "interval": "step"}}

        if self.lr_scheduler == "step":
            # Matches the AR branch's StepLR, on the epoch interval it assumes.
            sched = S.StepLR(opt, step_size=self.lr_decay_steps,
                             gamma=self.lr_decay_rate)
            if warmup <= 0:
                return {"optimizer": opt,
                        "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
            raise ValueError(
                "warmup_steps > 0 with lr_scheduler: step mixes a step-interval "
                "ramp with an epoch-interval decay. Use 'none' or 'cosine'.")

        # Cosine. T_max in STEPS, from Lightning's own estimate, which accounts
        # for max_epochs, accumulation and devices -- the previous version used
        # max_epochs on the epoch interval, which silently disagrees with a
        # step-interval warmup.
        total = int(self.trainer.estimated_stepping_batches) if self.trainer else 1000
        if warmup <= 0:
            sched = S.CosineAnnealingLR(opt, T_max=total, eta_min=0.01 * self.lr)
        else:
            sched = S.SequentialLR(
                opt,
                [S.LinearLR(opt, start_factor=1e-3, total_iters=warmup),
                 S.CosineAnnealingLR(opt, T_max=max(total - warmup, 1),
                                     eta_min=0.01 * self.lr)],
                milestones=[warmup])
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}

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
            return (b, 10, f)
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
        shape = self._state_shape(batch)

        def denoiser_fn(x_t, t):
            out = self._denoise(self._model_input(x_t), t, cond, None, cond_mask)
            if self.guidance != 1.0:
                # Classifier-free guidance (Ho & Salimans, arXiv:2207.12598):
                # push away from the unconditional prediction. Two forward
                # passes per step, so it doubles the eval cost -- which is why
                # it defaults to 1.0 and is a per-experiment choice.
                uncond = self._denoise(self._model_input(x_t), t, None, None, None)
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
        if self.state == "bins":
            # The chain carries occupancy in its own channel, so the final
            # state is the answer: bins are on the grid and presence is a bit.
            from src.dataset.mesh_dataset import dequantize
            coords = torch.from_numpy(
                dequantize(out[:, :9].detach().cpu().numpy(), self.num_bins)
            ).float().to(out.device)
            final = torch.zeros(out.shape[0], 10, out.shape[-1], device=out.device)
            final[:, :9] = coords
            final[:, 9] = torch.where(out[:, 9] > 0, 0.5, -0.5)
            return final
        # The trajectory's last x0 *prediction* is the answer, not the process's
        # own final state: that state is the same x0 re-noised by one schedule
        # rung, which would drift it straight back off the grid the clamp just
        # put it on -- the entire point of these arms.
        return self._from_state(self._last_coords, self._last_bins,
                                self._last_presence)
