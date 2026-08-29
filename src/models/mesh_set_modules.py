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

    Args:
        d_model: embedding width.
        scale: constant the face index is divided by before the Fourier lift.
            Must NOT be the padded length. It was, once: `pos / (F_pad - 1)`
            made face 5's encoding differ between a batch padded to 24 and one
            padded to 64, so the same building was encoded differently
            depending on which buildings it happened to share a batch with --
            a nuisance variable worth ~0.2 of a coordinate channel. A fixed
            constant (`mesh_data.max_faces`) makes position absolute, which is
            what a positional encoding is supposed to be.
    """

    def __init__(self, d_model, scale=200):
        super().__init__()
        self.d_model = d_model
        self.scale = max(int(scale) - 1, 1)

    def forward(self, length, device):
        """``[length, d_model]``. Depends only on ``length`` through the slice
        taken, never through the encoding of any individual position."""
        pos = torch.arange(length, device=device).float()
        return timestep_embedding(pos / self.scale, self.d_model)


def _mha_mask(key_padding_mask):
    """True = real (ours) to True = ignore (torch's). ``None`` passes through."""
    return None if key_padding_mask is None else ~key_padding_mask


def _modulate(h, shift, scale):
    """FiLM a normalised ``[B, F, C]`` activation (DiT, arXiv:2212.09748 sec 3.2).

    ``scale`` is the *residual* around 1, so a zero-initialised projection
    leaves the normalised activation exactly as it found it. Both terms are
    ``[B, C]`` -- one modulation per SAMPLE, broadcast across faces, which is
    what keeps a modulated block permutation equivariant.
    """
    return h * (1.0 + scale[:, None, :]) + shift[:, None, :]


def _run_ff(ff, h, mod=None):
    """Run a ``Sequential(LayerNorm, Linear, GELU, Linear)`` with optional FiLM.

    Indexes into the Sequential rather than splitting it into two attributes so
    the parameter names -- and therefore every existing checkpoint -- survive
    the addition of modulation.
    """
    hn = ff[0](h)
    if mod is not None:
        hn = _modulate(hn, *mod)
    for layer in ff[1:]:
        hn = layer(hn)
    return hn


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

    def forward(self, x, key_padding_mask=None, mod=None):
        """``[B, C, F] -> [B, C, F]``. ``key_padding_mask`` is [B, F], True = real.

        Args:
            mod: optional adaLN-Zero modulation, six ``[B, C]`` tensors
                ``(shift, scale, gate)`` for the attention branch then the same
                for the feed-forward one. ``None`` is the plain pre-norm block
                the U-Net uses, and is bit-identical to the unmodulated form.
        """
        h = x.permute(0, 2, 1)
        sh_a, sc_a, g_a, sh_f, sc_f, g_f = mod if mod is not None else (None,) * 6
        hn = self.ln(h)
        if mod is not None:
            hn = _modulate(hn, sh_a, sc_a)
        attn, _ = self.mha(hn, hn, hn, key_padding_mask=_mha_mask(key_padding_mask),
                           need_weights=False)
        h = h + (attn if mod is None else g_a[:, None, :] * attn)
        ff = _run_ff(self.ff, h, None if mod is None else (sh_f, sc_f))
        h = h + (ff if mod is None else g_f[:, None, :] * ff)
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

    def forward(self, x, cond, cond_mask=None, mod=None):
        """``x [B,C,F]``, ``cond [B,D,Fc]`` -> ``[B,C,F]``. ``cond=None`` is a no-op.

        Args:
            mod: optional adaLN-Zero modulation, as `SelfAttention1d`. Only the
                QUERY norm is modulated; `ln_kv` normalises the condition, which
                is not on the residual stream the timestep is steering.
        """
        if cond is None:
            return x
        h = x.permute(0, 2, 1)
        sh_a, sc_a, g_a, sh_f, sc_f, g_f = mod if mod is not None else (None,) * 6
        kv = self.ln_kv(cond.permute(0, 2, 1))
        q = self.ln_q(h)
        if mod is not None:
            q = _modulate(q, sh_a, sc_a)
        attn, _ = self.mha(q, kv, kv,
                           key_padding_mask=_mha_mask(cond_mask),
                           need_weights=False)
        h = h + (attn if mod is None else g_a[:, None, :] * attn)
        ff = _run_ff(self.ff, h, None if mod is None else (sh_f, sc_f))
        h = h + (ff if mod is None else g_f[:, None, :] * ff)
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


def zero_pad(x, mask):
    """Force padded faces to exactly 0. ``mask`` is ``[B, F]`` bool, True = real.

    Held as an invariant after every block in the U-Net. Masking the *norm*
    statistics is not enough on its own: a 3-kernel convolution still reads its
    neighbours, so whatever sits in a pad slot is mixed into the real face next
    to it. Pinning pads to a constant 0 makes that contribution identical no
    matter how much padding the batch happened to carry, which is the property
    that actually matters -- the boundary face sees the same zeros at F_pad 24
    as at F_pad 64.
    """
    return x if mask is None else x * mask[:, None, :].to(x.dtype)


class MaskedGroupNorm(nn.Module):
    """GroupNorm whose mean and variance ignore padded faces.

    `nn.GroupNorm` pools statistics over (channel group x face axis). On a
    padded face set that folds pad slots into the very mean and variance every
    *real* face is then scaled by, so a building's activations depend on how
    much padding its batch happened to carry. Measured before this existed: a
    mean drift of ~23 coordinate bins on the same building batched two ways,
    reaching face 0 at 256 slots from the pad boundary -- far outside any
    convolution's receptive field.

    Same parameterisation and same eps as `nn.GroupNorm`, so this is a drop-in
    replacement that preserves the reference architecture's inductive bias; only
    the support of the statistics changes.
    """

    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        if num_channels % num_groups:
            raise ValueError(
                f"num_channels {num_channels} is not divisible by num_groups {num_groups}.")
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x, mask=None):
        """``[B, C, F] -> [B, C, F]``. ``mask`` is ``[B, F]`` bool, True = real."""
        b, c, f = x.shape
        g = self.num_groups
        xg = x.reshape(b, g, c // g, f)
        if mask is None:
            m = torch.ones(b, 1, 1, f, dtype=x.dtype, device=x.device)
        else:
            m = mask[:, None, None, :].to(x.dtype)
        n = (m.sum(dim=(2, 3), keepdim=True) * (c // g)).clamp(min=1.0)
        mean = (xg * m).sum(dim=(2, 3), keepdim=True) / n
        var = (((xg - mean) * m) ** 2).sum(dim=(2, 3), keepdim=True) / n
        xg = (xg - mean) / (var + self.eps).sqrt()
        out = xg.reshape(b, c, f)
        return out * self.weight[None, :, None] + self.bias[None, :, None]


class DoubleConv(nn.Module):
    """Two masked-GroupNorm/GELU convolutions, optionally residual.

    `forward` takes the face mask because the normalisation does; padded slots
    are returned at exactly 0 so the next convolution cannot read anything else
    out of them.
    """

    def __init__(self, in_ch, out_ch, mid_ch=None, residual=False):
        super().__init__()
        self.residual = residual
        mid_ch = mid_ch or out_ch
        self.conv1 = nn.Conv1d(in_ch, mid_ch, 3, padding=1, bias=False)
        self.norm1 = MaskedGroupNorm(min(8, mid_ch), mid_ch)
        self.conv2 = nn.Conv1d(mid_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = MaskedGroupNorm(min(8, out_ch), out_ch)

    def forward(self, x, mask=None):
        h = F.gelu(self.norm1(self.conv1(x), mask))
        # Re-zeroed BETWEEN the two convolutions, not only at the end. The norm
        # adds its learned bias at every position including pads, so the
        # intermediate is non-zero there even when the input was clean -- and
        # `conv2` reads its neighbours. Leaving this out was worth ~0.2 of a
        # coordinate channel at the deepest level, where the padded width
        # changes how many pad slots sit next to a real one.
        h = zero_pad(h, mask)
        h = self.norm2(self.conv2(h), mask)
        h = F.gelu(x + h) if self.residual else h
        return zero_pad(h, mask)


class Down(nn.Module):
    """Stride-2 downsample along the face axis, with an additive time embedding.

    ``mask`` is the mask at the *output* resolution -- the caller has already
    halved it, since the U-Net needs those coarse masks for its attention
    blocks anyway and deriving them twice invites the two copies to disagree.
    """

    def __init__(self, in_ch, out_ch, emb_dim=128):
        super().__init__()
        self.pool = nn.MaxPool1d(2)
        self.conv1 = DoubleConv(in_ch, in_ch, residual=True)
        self.conv2 = DoubleConv(in_ch, out_ch)
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch))

    def forward(self, x, t, mask=None):
        x = self.conv2(self.conv1(self.pool(x), mask), mask)
        return zero_pad(x + self.emb(t)[:, :, None], mask)


class Up(nn.Module):
    """Nearest-neighbour upsample, skip concat, additive time embedding.

    ``mask`` is the mask at the skip's (i.e. the output's) resolution.
    """

    def __init__(self, in_ch, out_ch, emb_dim=128):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = DoubleConv(in_ch, in_ch, residual=True)
        self.conv2 = DoubleConv(in_ch, out_ch, mid_ch=in_ch // 2)
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch))

    def forward(self, x, skip, t, mask=None):
        x = torch.cat([skip, self.up(x)], dim=1)
        x = self.conv2(self.conv1(x, mask), mask)
        return zero_pad(x + self.emb(t)[:, :, None], mask)


def downsample_mask(mask):
    """``[B, F] -> [B, F//2]``. A coarse face is real if either child was.

    ``any`` rather than ``all``: a half-padded pair still carries real geometry,
    and masking it out would delete that geometry from every coarser level.
    """
    return mask.reshape(mask.shape[0], -1, 2).any(dim=-1)
