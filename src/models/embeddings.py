"""Input featurization for the rEGNN denoiser: timestep and pairwise-distance lifts.

A bare scalar into a ReLU MLP is hard to learn a high-frequency function of,
because such MLPs are biased toward low frequencies (Tancik et al. 2020,
"Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional
Domains", arXiv:2006.10739). These modules lift the scalar timestep and the
scalar pairwise distance onto richer bases so the downstream network can resolve
nearby steps and multiple length scales. `build_dist_embed` selects the distance
lift from a config string.

`SinusoidalPositionEmbedding` is the same idea applied to *sequence position*
rather than a scalar, and is shared by both mesh-transformer backbones.
"""
import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Fourier lift of the scalar timestep t/T in [0, 1] to a `dim`-vector.

    Lifting t onto a bank of sines/cosines gives the downstream MLP the basis to
    resolve nearby steps (Tancik et al. 2020, arXiv:2006.10739).
    """

    def __init__(self, dim, max_freq=1000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"time embedding dim must be even, got {dim}.")
        self.dim = dim
        # ponytail: max_freq ~ T sets the finest step the lift can resolve
        # (needs max_freq * (1/T) >~ pi). Raise it if T grows past ~500.
        self.register_buffer(
            "freqs",
            torch.exp(torch.linspace(0.0, math.log(max_freq), dim // 2)),
            persistent=False,
        )

    def forward(self, t_norm):
        args = t_norm * self.freqs  # [B, 1] * [dim/2] -> [B, dim/2]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class SinusoidalDistanceEmbedding(nn.Module):
    """Fourier lift of a scalar pairwise distance to a `dim`-vector, the geometry
    analogue of `SinusoidalTimeEmbedding` (Tancik et al. 2020, arXiv:2006.10739).

    Unlike the timestep, distance has no natural [0, 1] range: `coord_scale`
    standardises coordinates to ~unit pooled variance, so distances are O(1).
    `max_freq` sets the finest length scale the lift can resolve on that scale.
    """

    def __init__(self, dim, max_freq=64.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"distance embedding dim must be even, got {dim}.")
        self.dim = dim
        # ponytail: geometric frequency bank up to max_freq. max_freq is scale-
        # dependent (tie it to the normalised-distance histogram, see the spec);
        # promote to a config knob if it needs per-dataset tuning.
        self.register_buffer(
            "freqs",
            torch.exp(torch.linspace(0.0, math.log(max_freq), dim // 2)),
            persistent=False,
        )

    def forward(self, d):
        args = d * self.freqs  # [..., 1] * [dim/2] -> [..., dim/2]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class SinusoidalPositionEmbedding(nn.Module):
    """Fixed sinusoidal absolute positions (Vaswani et al. 2017,
    arXiv:1706.03762 section 3.5).

    Unlike a learned `nn.Embedding(max_seq_len, d)` this has no row count, so it
    is defined at every position and the sequence-length ceiling disappears
    rather than moving. That is the reason it exists here: both mesh backbones
    were capped by a table sized from a default -- OPT's 2048 rides in from a
    hub config whose weights are not even loaded.

    No parameters and a non-persistent buffer, so a checkpoint saved with this
    option carries no position rows at all and can be reloaded at any length.

    Args:
        dim (int): output width; must be even.
        base (float): geometric wavelength range, 10000 in the paper.
    """

    def __init__(self, dim, base=10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"position embedding dim must be even, got {dim}.")
        self.dim = dim
        # Half the width is sin, half cos -- the same layout as the siblings
        # above, which is *not* the paper's interleaving. Nothing downstream
        # reads individual channels, so the permutation is free.
        self.register_buffer(
            "inv_freq",
            torch.exp(torch.arange(0, dim, 2, dtype=torch.float)
                      * (-math.log(base) / dim)),
            persistent=False,
        )

    def forward(self, pos):
        """`pos` of any shape -> that shape with a trailing `dim` axis."""
        args = pos.unsqueeze(-1).float() * self.inv_freq
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DistanceMLP(nn.Module):
    """Small nonlinear lift of a scalar distance to `dim` features -- the pairwise
    analogue of `PositionsMLP`'s MLP-of-a-norm, replacing the single Linear."""

    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, d):
        return self.mlp(d)


class BesselBasis(nn.Module):
    """Radial Bessel basis with a smooth polynomial cutoff (DimeNet;
    Gasteiger et al. 2020, arXiv:2003.03123).

    e_n(d) = sqrt(2/r_max) * sin(n*pi*d/r_max) / d, n = 1..dim, times a polynomial
    envelope that decays to zero at d = r_max. `r_max` is the cutoff in the model's
    normalised coordinate units, calibrated once from the train split
    (`CityJSONDataModule.compute_dist_r_max`).
    """

    def __init__(self, dim, r_max, envelope_p=6):
        super().__init__()
        if r_max is None or r_max <= 0:
            raise ValueError(f"BesselBasis needs a positive r_max, got {r_max}.")
        self.r_max = float(r_max)
        self.p = envelope_p
        self.register_buffer(
            "freqs", math.pi * torch.arange(1, dim + 1, dtype=torch.float32), persistent=False
        )

    def _envelope(self, x):
        # DimeNet polynomial envelope u(x), x = d / r_max clamped to [0, 1]:
        # u(1) = u'(1) = u''(1) = 0, so the basis and its slope vanish at the cutoff.
        p = self.p
        x = x.clamp(max=1.0)
        return (1.0
                - (p + 1) * (p + 2) / 2 * x ** p
                + p * (p + 2) * x ** (p + 1)
                - p * (p + 1) / 2 * x ** (p + 2))

    def forward(self, d):
        # sin(n*pi*d/r_max)/d; numerator is 0 at d=0 so eps stays finite there
        # (the self-distance diagonal is masked downstream anyway).
        radial = torch.sin(d * self.freqs / self.r_max) / (d + 1e-7)
        return math.sqrt(2.0 / self.r_max) * radial * self._envelope(d / self.r_max)


def build_dist_embed(dist_embed, dim, r_max=None):
    """Map a scalar pairwise distance to features. Returns (module, out_dim).

    'raw' keeps MiDi's single raw-distance channel (Identity, out_dim 1); the
    others lift it so `lin_dist1` sees multi-scale structure. See the distance
    featurization spec for why a linear map of raw distance is the bottleneck.
    """
    if dist_embed == "raw":
        return nn.Identity(), 1
    if dist_embed == "sinusoidal":
        return SinusoidalDistanceEmbedding(dim), dim
    if dist_embed == "mlp":
        return DistanceMLP(dim), dim
    if dist_embed == "bessel":
        return BesselBasis(dim, r_max), dim
    raise ValueError(
        f"dist_embed must be 'raw', 'sinusoidal', 'mlp' or 'bessel', got {dist_embed!r}."
    )
