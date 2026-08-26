"""Noising processes for the face-set branch.

One interface, three implementations, so the denoiser and the training loop
are written once and the process is a config switch. The split is the one that
worked in maxim-mat/trace-denoise-refactor (`src/diffusion/base_diffusion.py`):
the process owns the schedule and the reverse loop, the denoiser owns the
network, and neither knows what the other is.

Time is always a float in [0, 1] at the interface (plan D7). DDPM and D3PM
carry an integer index internally and convert; flow matching is natively
continuous and does not.
"""
import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseProcess(nn.Module, ABC):
    """Forward corruption and reverse sampling for one diffusion formulation.

    Subclasses implement four things: how to draw a training time, how to
    corrupt, what the regression target is, and how to take one reverse step.
    `sample` is shared and never overridden.

    `nn.Module` rather than a plain class so Lightning moves the schedule
    buffers to the accelerator with the model.
    """

    @abstractmethod
    def sample_t(self, n, device):
        """``[n]`` training times in ``[0, 1]``."""

    @abstractmethod
    def corrupt(self, x0, t):
        """``(x_t, aux)``. ``aux`` is whatever `target_for` needs -- the drawn
        noise, typically -- and is never inspected by the caller."""

    @abstractmethod
    def target_for(self, x0, x_t, t, aux):
        """What the denoiser is asked to regress at this ``t``."""

    @abstractmethod
    def step(self, x_t, t, t_prev, model_out):
        """One reverse step, ``x_t -> x_{t_prev}``. ``t`` are floats in [0,1]."""

    @abstractmethod
    def prior(self, shape, device):
        """A draw from the terminal distribution the reverse loop starts at."""

    def timesteps(self, n_steps):
        """``[(t, t_prev), ...]`` descending, ``n_steps`` long, ending at 0."""
        edges = torch.linspace(1.0, 0.0, n_steps + 1)
        return list(zip(edges[:-1].tolist(), edges[1:].tolist()))

    def sample(self, denoiser_fn, shape, device, n_steps=50, callback=None):
        """Run the reverse trajectory and return the final state.

        Args:
            denoiser_fn: ``(x_t [B,C,F], t [B] float) -> prediction``. The
                caller closes over the condition, the masks and any guidance,
                so this class never learns what a condition is.
            shape: ``(B, C, F)``.
            device: where to allocate.
            n_steps: reverse steps. Fewer than `noise_steps` is a strided
                (DDIM-style) trajectory, which is the only affordable setting
                for a per-epoch eval.
            callback: optional ``(t, x_t) -> x_t or None`` per step, for
                scaffold projection and trajectory logging. A callback that
                returns ``None`` is an observer and leaves the state alone --
                without that, a logging callback would silently blank the
                trajectory it was meant to watch.

        Returns:
            Tensor: ``[B, C, F]``.
        """
        x = self.prior(shape, device)
        for t, t_prev in self.timesteps(n_steps):
            t_batch = torch.full((shape[0],), t, device=device)
            with torch.no_grad():
                out = denoiser_fn(x, t_batch)
            x = self.step(x, t, t_prev, out)
            if callback is not None:
                replaced = callback(t_prev, x)
                if replaced is not None:
                    x = replaced
        return x


class GaussianProcess(BaseProcess):
    """DDPM forward process with a DDIM (deterministic, strided) reverse.

    Ho et al. arXiv:2006.11239 for the schedule, Song et al. arXiv:2010.02502
    for the reverse. DDIM rather than ancestral sampling because the eval
    budget is the binding constraint: a per-epoch callback over 16 buildings at
    1000 ancestral steps is 16,000 forward passes, and eta=0 lets 50 steps
    stand in with no retraining.

    Args:
        noise_steps: schedule resolution. Time in ``[0,1]`` is discretised onto
            this many rungs.
        beta_start, beta_end: linear schedule endpoints.
        target: "noise" (epsilon-prediction, the default and what nearly every
            reported result uses) or "original" (x0-prediction, which is better
            conditioned at low t and is what D3PM and most set models use).
    """

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02,
                 target="noise"):
        super().__init__()
        if target not in ("noise", "original"):
            raise ValueError(
                f"GaussianProcess target must be 'noise' or 'original', got {target!r}.")
        self.noise_steps = noise_steps
        self.target = target
        beta = torch.linspace(beta_start, beta_end, noise_steps)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha_hat", torch.cumprod(1.0 - beta, dim=0))

    def to_index(self, t):
        """Float time in ``[0,1]`` to an integer schedule rung."""
        idx = (t.clamp(0.0, 1.0) * (self.noise_steps - 1)).round().long()
        return idx.clamp(0, self.noise_steps - 1)

    def _abar(self, t, ndim=3):
        idx = self.to_index(t)
        a = self.alpha_hat[idx]
        return a.reshape(-1, *([1] * (ndim - 1)))

    def sample_t(self, n, device):
        return torch.rand(n, device=device)

    def corrupt(self, x0, t):
        a = self._abar(t, x0.dim())
        eps = torch.randn_like(x0)
        return a.sqrt() * x0 + (1 - a).sqrt() * eps, eps

    def target_for(self, x0, x_t, t, aux):
        return aux if self.target == "noise" else x0

    def to_x0(self, x_t, t, model_out):
        """Whatever the denoiser predicted, read as an ``x0`` estimate."""
        if self.target == "original":
            return model_out
        a = self._abar(torch.as_tensor(t, device=x_t.device).expand(x_t.shape[0]),
                       x_t.dim())
        return (x_t - (1 - a).sqrt() * model_out) / a.sqrt().clamp(min=1e-8)

    def step(self, x_t, t, t_prev, model_out):
        """Deterministic DDIM step (eta = 0)."""
        device = x_t.device
        n = x_t.shape[0]
        a_t = self._abar(torch.full((n,), t, device=device), x_t.dim())
        a_p = self._abar(torch.full((n,), t_prev, device=device), x_t.dim())
        x0 = self.to_x0(x_t, t, model_out)
        eps = ((x_t - a_t.sqrt() * x0) / (1 - a_t).sqrt().clamp(min=1e-8))
        return a_p.sqrt() * x0 + (1 - a_p).sqrt() * eps

    def prior(self, shape, device):
        return torch.randn(shape, device=device)


class FlowMatchingProcess(BaseProcess):
    """Conditional flow matching along the straight path (Lipman et al.,
    arXiv:2210.02747), with an Euler integrator.

    Same denoiser, different target: instead of a noise level indexed into a
    schedule, time is continuous and the network regresses the constant
    velocity ``x0 - z`` of the straight line from prior to data. There is no
    beta schedule to tune, and the trajectory is straight by construction, so
    few-step sampling degrades far more gracefully than a strided DDPM does.

    Time convention: ``t = 1`` is data, ``t = 0`` is the prior, matching
    `BaseProcess.sample`, which runs ``t`` from 1 down to 0. That is the
    reverse of the usual flow-matching paper convention and is chosen so all
    three processes share one loop.

    Args:
        sigma_min: floor on the prior's contribution at ``t = 1``. 0 gives the
            exact straight path; a small positive value keeps the target
            distribution absolutely continuous, which matters for likelihood
            evaluation and not at all for sampling.
    """

    def __init__(self, sigma_min=0.0):
        super().__init__()
        self.sigma_min = sigma_min
        # No buffers, but keep the nn.Module contract so `create_process`
        # returns something Lightning treats identically to GaussianProcess.
        self.register_buffer("_unused", torch.zeros(1), persistent=False)

    def _coef(self, t, ndim):
        t = t.reshape(-1, *([1] * (ndim - 1)))
        return t, (1.0 - (1.0 - self.sigma_min) * t)

    def sample_t(self, n, device):
        return torch.rand(n, device=device)

    def corrupt(self, x0, t):
        z = torch.randn_like(x0)
        a, b = self._coef(t, x0.dim())
        return a * x0 + b * z, z

    def target_for(self, x0, x_t, t, aux):
        return (1.0 - self.sigma_min) * x0 - (1.0 - self.sigma_min) * aux \
            if self.sigma_min else x0 - aux

    def to_x0(self, x_t, t, model_out):
        """Read a velocity as an ``x0`` estimate: ``x0 = x_t + (1 - t) v``.

        Exact on the straight path, and it is what makes the coordinate-error
        log in `MeshDiffusionModule` comparable across all three processes.
        """
        tt = torch.as_tensor(t, device=x_t.device, dtype=x_t.dtype)
        return x_t + (1.0 - tt) * model_out

    def step(self, x_t, t, t_prev, model_out):
        """One Euler step. ``t_prev < t`` in this convention, so the increment
        is negative and the integrator walks *back* toward the prior; the
        reverse loop's descending timesteps invert that into progress."""
        return x_t + (t_prev - t) * model_out

    def prior(self, shape, device):
        return torch.randn(shape, device=device)

    def timesteps(self, n_steps):
        """Ascending here, not descending: flow matching integrates from the
        prior at ``t = 0`` toward the data at ``t = 1``, which is the opposite
        direction to a denoising trajectory."""
        edges = torch.linspace(0.0, 1.0, n_steps + 1)
        return list(zip(edges[:-1].tolist(), edges[1:].tolist()))


def create_process(cfg):
    """Build the process named by ``cfg.mesh_diffusion.process``.

    A plain switch, matching `src.train.train`'s handling of `config_set`:
    there are three of these and they all live in this file.
    """
    d = cfg.mesh_diffusion
    if d.process == "ddpm":
        return GaussianProcess(d.noise_steps, d.beta_start, d.beta_end, d.target)
    if d.process == "flow":
        from src.models.mesh_processes import FlowMatchingProcess
        return FlowMatchingProcess()
    if d.process == "d3pm":
        from src.models.mesh_processes import DiscreteProcess
        return DiscreteProcess(d.noise_steps, cfg.mesh_data.num_bins,
                               transition=d.transition,
                               sigma_max=d.transition_sigma)
    raise ValueError(f"Unknown process: {d.process!r}.")
