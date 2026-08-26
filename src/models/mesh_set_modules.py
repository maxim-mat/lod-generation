"""Blocks shared by both face-set denoisers.

Ported from the conditional 1-D U-Net in maxim-mat/trace-denoise-refactor
(`src/modules/`), with two deliberate changes:

  * conditioning is cross-attention, never the elementwise sum that repo uses.
    That sum requires the condition and the target to have equal, aligned
    length; LOD1 is ~12 triangles against LOD2's ~100, so there is no alignment
    to exploit. Cross-attention is also MeshWeaver's fix (ii) for exactly the
    architecture this branch is competing with.
  * the padding-mask convention is True = real (matching this repo's face-set
    collate), and it is inverted once at each `nn.MultiheadAttention` call
    rather than at every call site.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    """Fourier features for a diffusion timestep (Vaswani, arXiv:1706.03762 3.5).

    Args:
        t: ``[B]`` float in ``[0, 1]``. Normalised, not an integer index --
            that is what lets one denoiser serve DDPM, D3PM and flow matching,
            whose native time variables are otherwise incomparable.
        dim: embedding width. Must be even.

    Returns:
        Tensor: ``[B, dim]``.
    """
    if dim % 2:
        raise ValueError(f"timestep_embedding dim must be even, got {dim}.")
    half = dim // 2
    # Scaled to the usual 0..1000 band so the frequencies land where the
    # standard schedules put them, whatever the process calls its own time.
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
    args = (t.float() * 1000.0)[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class SinusoidalFacePositions(nn.Module):
    """Fixed positional encoding over the face axis. No parameters, no ceiling.

    Only meaningful when the face order carries information -- i.e. under
    `order: morton`. `validate_combination` is what enforces that.
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, length, device):
        """``[length, d_model]``."""
        pos = torch.arange(length, device=device).float()
        return timestep_embedding(pos / max(length - 1, 1), self.d_model)


def _mha_mask(key_padding_mask):
    """True = real (ours) to True = ignore (torch's). ``None`` passes through."""
    return None if key_padding_mask is None else ~key_padding_mask


class SelfAttention1d(nn.Module):
    """Pre-norm multi-head self-attention over the face axis, channels-first."""

    def __init__(self, channels, n_head=8):
        super().__init__()
        n_head = min(n_head, max(1, channels // 8))
        self.mha = nn.MultiheadAttention(channels, n_head, batch_first=True)
        self.ln = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels), nn.Linear(channels, channels),
            nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x, key_padding_mask=None):
        """``[B, C, F] -> [B, C, F]``. ``key_padding_mask`` is [B, F], True = real."""
        h = x.permute(0, 2, 1)
        hn = self.ln(h)
        attn, _ = self.mha(hn, hn, hn, key_padding_mask=_mha_mask(key_padding_mask),
                           need_weights=False)
        h = h + attn
        h = h + self.ff(h)
        return h.permute(0, 2, 1)


class CrossAttention1d(nn.Module):
    """Face-axis cross-attention onto an encoded condition of any length.

    This is the conditioning mechanism, and it is the reason the condition need
    not be length-matched to the target. It is also MeshWeaver's fix (ii):
    local geometric context at every layer instead of one global prefix.
    """

    def __init__(self, channels, cond_dim, n_head=8):
        super().__init__()
        n_head = min(n_head, max(1, channels // 8))
        self.mha = nn.MultiheadAttention(channels, n_head, batch_first=True,
                                         kdim=cond_dim, vdim=cond_dim)
        self.ln_q = nn.LayerNorm(channels)
        self.ln_kv = nn.LayerNorm(cond_dim)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels), nn.Linear(channels, channels),
            nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x, cond, cond_mask=None):
        """``x [B,C,F]``, ``cond [B,D,Fc]`` -> ``[B,C,F]``. ``cond=None`` is a no-op."""
        if cond is None:
            return x
        h = x.permute(0, 2, 1)
        kv = self.ln_kv(cond.permute(0, 2, 1))
        attn, _ = self.mha(self.ln_q(h), kv, kv,
                           key_padding_mask=_mha_mask(cond_mask),
                           need_weights=False)
        h = h + attn
        h = h + self.ff(h)
        return h.permute(0, 2, 1)


class FaceEncoder(nn.Module):
    """The LOD1 condition as a sequence of per-face embeddings.

    A two-layer MLP applied per face, not a convolution: the condition is read
    only through cross-attention, which is permutation invariant, so there is
    nothing for a convolution's locality to buy here.
    """

    def __init__(self, in_ch=10, d_model=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, d_model, 1), nn.GELU(),
            nn.Conv1d(d_model, d_model, 1))

    def forward(self, cond):
        """``[B, 10, Fc] -> [B, d_model, Fc]``. ``None`` passes through."""
        return None if cond is None else self.net(cond)


class DoubleConv(nn.Module):
    """Two GroupNorm-GELU convolutions, optionally residual."""

    def __init__(self, in_ch, out_ch, mid_ch=None, residual=False):
        super().__init__()
        self.residual = residual
        mid_ch = mid_ch or out_ch
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, mid_ch), mid_ch), nn.GELU(),
            nn.Conv1d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch))

    def forward(self, x):
        return F.gelu(x + self.net(x)) if self.residual else self.net(x)


class Down(nn.Module):
    """Stride-2 downsample along the face axis, with an additive time embedding."""

    def __init__(self, in_ch, out_ch, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_ch, in_ch, residual=True),
            DoubleConv(in_ch, out_ch))
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch))

    def forward(self, x, t):
        x = self.net(x)
        return x + self.emb(t)[:, :, None]


class Up(nn.Module):
    """Nearest-neighbour upsample, skip concat, additive time embedding."""

    def __init__(self, in_ch, out_ch, emb_dim=128):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.net = nn.Sequential(
            DoubleConv(in_ch, in_ch, residual=True),
            DoubleConv(in_ch, out_ch, mid_ch=in_ch // 2))
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch))

    def forward(self, x, skip, t):
        x = torch.cat([skip, self.up(x)], dim=1)
        x = self.net(x)
        return x + self.emb(t)[:, :, None]


def downsample_mask(mask):
    """``[B, F] -> [B, F//2]``. A coarse face is real if either child was.

    ``any`` rather than ``all``: a half-padded pair still carries real geometry,
    and masking it out would delete that geometry from every coarser level.
    """
    return mask.reshape(mask.shape[0], -1, 2).any(dim=-1)
