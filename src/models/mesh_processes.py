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

import numpy as np
import torch
import torch.nn as nn

from src.models.noise import cosine_beta_schedule_discrete

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
        x0_clip: clamp the x0 estimate to ``+-x0_clip`` before it is used to
            build the next state; ``None`` disables. Static thresholding, and
            not cosmetic: an untrained epsilon model predicts ~0, which makes
            the step a pure rescaling whose factor telescopes to x157 across
            the trajectory. Every continuous state lives in [-0.5, 0.5], so
            that is the natural bound.
    """

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02,
                 target="noise", x0_clip=0.5):
        super().__init__()
        if target not in ("noise", "original"):
            raise ValueError(
                f"GaussianProcess target must be 'noise' or 'original', got {target!r}.")
        if x0_clip is not None and x0_clip <= 0:
            raise ValueError(
                f"x0_clip must be positive or None, got {x0_clip!r}.")
        self.noise_steps = noise_steps
        self.target = target
        self.x0_clip = x0_clip
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
            return self._clip(model_out)
        a = self._abar(torch.as_tensor(t, device=x_t.device).expand(x_t.shape[0]),
                       x_t.dim())
        return self._clip((x_t - (1 - a).sqrt() * model_out) / a.sqrt().clamp(min=1e-8))

    def _clip(self, x0):
        """Static thresholding. See the class docstring for why this is on."""
        return x0 if self.x0_clip is None else x0.clamp(-self.x0_clip, self.x0_clip)

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


class DiscreteProcess(BaseProcess):
    """D3PM over coordinate bins (Austin et al., arXiv:2107.03006), x0-parameterised.

    Nine independent categorical chains per face -- one per coordinate channel
    -- plus a two-state chain for presence.

    Two transitions, and the choice is the arm (plan D11):

      * ``uniform`` -- a corrupted bin is redrawn uniformly. The standard
        text/graph setting, and the right one when the alphabet is *nominal*.
      * ``gaussian`` -- a corrupted bin moves to a nearby one, with a
        discretized-Gaussian kernel whose width grows along the schedule
        (their section 3.2, ``D3PM-gauss``). Coordinate bins are *ordinal*:
        bin 40 and bin 41 are half a centimetre apart, and a uniform transition
        discards that structure at every step. This is the better-matched
        process for quantized geometry and is the default here.

    Why this arm exists at all: bins never leave the grid, so two faces naming
    the same corner emit byte-identical coordinates and `weld` merges them with
    no snap. It shares that property with the ``state: onehot`` arm in Task 11;
    what it does *not* share is the corruption, which is the thing under test.

    Args:
        noise_steps: schedule resolution.
        num_bins: alphabet size per coordinate channel.
        transition: "uniform" or "gaussian".
        sigma_max: discretized-Gaussian width at t = 1, in bins. Only read
            under ``transition="gaussian"``. 16 on a 128-bin grid is an eighth
            of the range, which reaches near-uniform by the end of the schedule
            while staying local for most of it.
    """

    def __init__(self, noise_steps=1000, num_bins=128, transition="gaussian",
                 sigma_max=16.0):
        super().__init__()
        if transition not in ("uniform", "gaussian"):
            raise ValueError(
                f"transition must be 'uniform' or 'gaussian', got {transition!r}.")
        self.noise_steps = noise_steps
        self.num_bins = num_bins
        self.transition = transition
        self.sigma_max = sigma_max
        # One nu per chain, all 1.0: the per-feature exponent exists for the
        # Levi branch, where nodes and edges corrupt at different rates. Here
        # every chain is the same kind of variable.
        betas = cosine_beta_schedule_discrete(noise_steps, [1.0])[:, 0]
        alpha_bar = np.cumprod(1.0 - betas)
        self.register_buffer("alpha_bar", torch.tensor(alpha_bar, dtype=torch.float32))
        # Offsets used by the Gaussian transition, precomputed once.
        self.register_buffer(
            "_offsets", torch.arange(-num_bins + 1, num_bins, dtype=torch.float32))

    def _gaussian_offsets(self, t, shape, device):
        """Signed bin displacements drawn from the discretized Gaussian at t.

        Width interpolates from ~0 at t = 0 to `sigma_max` at t = 1, so a
        coordinate wanders locally early and reaches the whole grid late. The
        draw is truncated by the reflect-free clamp in `corrupt`, which is the
        one place the boundary is handled.
        """
        a = self.alpha_bar[self.to_index(t)]
        sigma = ((1.0 - a) * self.sigma_max).reshape(-1, *([1] * (len(shape) - 1)))
        return torch.round(torch.randn(shape, device=device) * sigma).long()

    def to_index(self, t):
        idx = (t.clamp(0.0, 1.0) * (len(self.alpha_bar) - 1)).round().long()
        return idx.clamp(0, len(self.alpha_bar) - 1)

    def _keep_prob(self, t, shape):
        """P(a bin survives to time t) under the uniform transition."""
        a = self.alpha_bar[self.to_index(t)]
        return a.reshape(-1, *([1] * (len(shape) - 1)))

    def sample_t(self, n, device):
        return torch.rand(n, device=device)

    def _corrupt_presence(self, p0, t):
        """Two-state chain over occupancy: keep, else redraw from {0, 1}.

        Presence gets a chain of its own rather than riding through the reverse
        loop as a bare logit. Without it, face count in the categorical arms was
        decided by a single forward pass at the last step while every other arm
        refined it across the whole trajectory -- so b6/b7's face-count numbers
        were not comparable with b1-b5's, and plan D2's "presence is diffused"
        was false for exactly the arms whose alphabet has no natural two-state
        member.

        Always a uniform transition, whatever `self.transition` says: the
        discretized-Gaussian kernel exists to exploit *ordinality*, and {absent,
        present} has none. There is no sense in which absent is "near" present.
        """
        keep = self._keep_prob(t, p0.shape)
        redraw = torch.randint_like(p0, 0, 2)
        stay = torch.rand(p0.shape, device=p0.device) < keep
        return torch.where(stay, p0, redraw)

    def corrupt(self, x0, t):
        """``x0 [B,10,F]`` int64 -> ``(x_t [B,10,F], None)``.

        Channels 0-8 are coordinate bins in ``[0, num_bins)``; channel 9 is
        occupancy in ``{0, 1}``. The two are corrupted by different chains --
        see `_corrupt_presence` -- and carried in one tensor so the process
        keeps `BaseProcess`'s single-state contract and needs no `sample`
        override.

        Sampled from the marginal q(x_t | x0) directly rather than by walking
        the chain. Under `uniform` the marginal is exactly "keep with
        probability alpha_bar, else redraw uniformly", which is O(1) in t.
        Under `gaussian` it is a single displacement drawn at the width the
        schedule has reached, which is the marginal of the composed kernel up
        to the boundary clamp -- exact in the interior, and the interior is
        where every coordinate that matters lives.
        """
        coords, presence = x0[:, :9], x0[:, 9]
        out_p = self._corrupt_presence(presence, t)

        if self.transition == "uniform":
            keep = self._keep_prob(t, coords.shape)
            redraw = torch.randint_like(coords, 0, self.num_bins)
            stay = torch.rand(coords.shape, device=coords.device) < keep
            return torch.cat([torch.where(stay, coords, redraw),
                              out_p.unsqueeze(1)], dim=1), None
        x0 = coords

        # Three-way mixture, and the weights are not arbitrary: the terminal
        # distribution has to BE the sampler's prior, or the reverse loop
        # starts somewhere the forward process never reaches. `p_uniform` is
        # (1 - alpha_bar)^2 so it goes to 1 exactly as alpha_bar goes to 0,
        # which makes the t = 1 marginal uniform and `prior` correct for both
        # transitions. The local component peaks in the middle of the
        # trajectory -- which is where ordinal structure is worth having, since
        # that is the stretch on which the model learns to refine a coordinate
        # rather than to invent one.
        a = self._keep_prob(t, x0.shape)
        p_uniform = (1.0 - a) ** 2
        u = torch.rand(x0.shape, device=x0.device)
        shifted = (x0 + self._gaussian_offsets(t, x0.shape, x0.device))
        # Clamp, not wrap: bin 0 and bin 127 are opposite ends of a building,
        # not neighbours, and a wrapping kernel would teach the model that the
        # floor is adjacent to the roof.
        shifted = shifted.clamp(0, self.num_bins - 1)
        uniform = torch.randint_like(x0, 0, self.num_bins)
        out = torch.where(u < a, x0,
                          torch.where(u < a + p_uniform, uniform, shifted))
        return torch.cat([out, out_p.unsqueeze(1)], dim=1), None

    def target_for(self, x0, x_t, t, aux):
        return x0

    def step(self, x_t, t, t_prev, model_out):
        """Ancestral step: sample x0 from the predicted posterior, re-noise to
        ``t_prev``.

        Presence is sampled from its own Bernoulli and re-noised on its own
        chain, so occupancy is refined across the trajectory exactly as the
        coordinates are.

        The exact reverse posterior for a uniform chain would weight q(s|t,x0)
        against the predicted p(x0); this samples x0 first and re-noises, which
        is the same expectation with one extra sampling step and none of the
        [B, F, K, K] intermediate that the exact form needs at K = 128. ponytail:
        exact posterior via `compute_batched_over0_posterior_distribution` if
        the sample quality turns out to be limited by this and not by the model.
        """
        logits, presence_logits = model_out
        b, c, k, f = logits.shape
        probs = torch.softmax(logits, dim=2).permute(0, 1, 3, 2).reshape(-1, k)
        coords = torch.multinomial(probs, 1).reshape(b, c, f)
        presence = torch.bernoulli(torch.sigmoid(presence_logits)).long()
        x0 = torch.cat([coords, presence.unsqueeze(1)], dim=1)
        if t_prev <= 0.0:
            return x0
        t_batch = torch.full((b,), t_prev, device=logits.device)
        x_prev, _ = self.corrupt(x0, t_batch)
        return x_prev

    def prior(self, shape, device):
        """``[B, 10, F]``: coordinate bins uniform over the alphabet, occupancy
        uniform over {0, 1} -- the terminal marginal of both chains."""
        b, c, f = shape
        coords = torch.randint(0, self.num_bins, (b, c - 1, f), device=device)
        presence = torch.randint(0, 2, (b, 1, f), device=device)
        return torch.cat([coords, presence], dim=1)


def create_process(cfg):
    """Build the process named by ``cfg.mesh_diffusion.process``.

    A plain switch, matching `src.train.train`'s handling of `config_set`:
    there are three of these and they all live in this file.
    """
    d = cfg.mesh_diffusion
    if d.process == "ddpm":
        return GaussianProcess(d.noise_steps, d.beta_start, d.beta_end, d.target,
                               x0_clip=d.x0_clip)
    if d.process == "flow":
        from src.models.mesh_processes import FlowMatchingProcess
        return FlowMatchingProcess()
    if d.process == "d3pm":
        from src.models.mesh_processes import DiscreteProcess
        return DiscreteProcess(d.noise_steps, cfg.mesh_data.num_bins,
                               transition=d.transition,
                               sigma_max=d.transition_sigma)
    raise ValueError(f"Unknown process: {d.process!r}.")
