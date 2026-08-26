"""Attention-augmented 1-D U-Net over a spatially sorted face set.

The face axis is treated as a sequence, which is only legitimate because
`MeshSetDataset` sorts faces by the Morton code of their centroid -- adjacent
slots are then spatially adjacent, and a stride-2 convolution pools a
neighbourhood rather than an arbitrary subset. `validate_combination` refuses
this denoiser under `order: none` for exactly that reason.
"""
import torch
import torch.nn as nn

from src.models.mesh_set_modules import (
    CrossAttention1d,
    DoubleConv,
    Down,
    FaceEncoder,
    SelfAttention1d,
    Up,
    downsample_mask,
    timestep_embedding,
)


class ConditionalMeshUNet(nn.Module):
    """Denoiser: ``(x_t, t, LOD1) -> prediction`` over a padded face set.

    Args:
        in_ch, out_ch: channel counts. 10 = 9 coordinates + presence.
        base: channel width at full resolution; doubles twice going down.
        cond_dim: width of the encoded LOD1 condition.
        time_dim: timestep embedding width.
        n_head: attention heads (clamped down on narrow levels).
        dropout: applied to the bottleneck only.
        out_bins: when set, the head emits ``[B, 9, out_bins, F]`` bin logits
            plus ``[B, F]`` presence logits instead of ``[B, 10, F]`` -- the
            D3PM arm. ``None`` for every continuous process.
        cond_ch: channels of the LOD1 condition. Always 10, and deliberately
            *not* tied to `in_ch`: the ``onehot`` state widens the noised input
            to 9 * num_bins + 1, but the condition it cross-attends to is the
            same plain face set every other arm sees.
    """

    def __init__(self, in_ch=10, out_ch=10, base=64, cond_dim=256,
                 time_dim=128, n_head=8, dropout=0.1, out_bins=None,
                 cond_ch=10):
        super().__init__()
        self.time_dim = time_dim
        self.out_bins = out_bins
        self.cond_encoder = FaceEncoder(cond_ch, cond_dim)

        c1, c2, c3 = base, base * 2, base * 4
        self.inc = DoubleConv(in_ch, c1)
        self.down1, self.sa1 = Down(c1, c2, time_dim), SelfAttention1d(c2, n_head)
        self.ca1 = CrossAttention1d(c2, cond_dim, n_head)
        self.down2, self.sa2 = Down(c2, c3, time_dim), SelfAttention1d(c3, n_head)
        self.ca2 = CrossAttention1d(c3, cond_dim, n_head)
        self.down3, self.sa3 = Down(c3, c3, time_dim), SelfAttention1d(c3, n_head)
        self.ca3 = CrossAttention1d(c3, cond_dim, n_head)

        self.bot = nn.Sequential(
            DoubleConv(c3, c3 * 2), nn.Dropout(dropout), DoubleConv(c3 * 2, c3))

        self.up1, self.sa4 = Up(c3 + c3, c2, time_dim), SelfAttention1d(c2, n_head)
        self.ca4 = CrossAttention1d(c2, cond_dim, n_head)
        self.up2, self.sa5 = Up(c2 + c2, c1, time_dim), SelfAttention1d(c1, n_head)
        self.ca5 = CrossAttention1d(c1, cond_dim, n_head)
        self.up3, self.sa6 = Up(c1 + c1, c1, time_dim), SelfAttention1d(c1, n_head)

        head_ch = out_ch if out_bins is None else 9 * out_bins + 1
        self.outc = nn.Conv1d(c1, head_ch, 1)
        # Zero-init the head so the model starts as the identity on x_t rather
        # than injecting noise of its own on step 0 (Nichol & Dhariwal's
        # zero-module trick, arXiv:2102.09672).
        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)

    def forward(self, x, t, cond=None, mask=None, cond_mask=None):
        """``x [B,10,F]``, ``t [B]`` in ``[0,1]``, ``cond [B,10,Fc]`` or None."""
        f = x.shape[-1]
        if f % 8:
            raise ValueError(
                f"Face-set length {f} is not divisible by 8; the U-Net "
                "downsamples three times. mesh_set_collate_fn pads to a "
                "multiple of 8 -- a caller bypassed it.")
        if mask is None:
            mask = torch.ones(x.shape[0], f, dtype=torch.bool, device=x.device)

        temb = timestep_embedding(t, self.time_dim)
        c = self.cond_encoder(cond)
        m1 = mask
        m2 = downsample_mask(m1)
        m3 = downsample_mask(m2)
        m4 = downsample_mask(m3)

        x1 = self.inc(x)
        x2 = self.ca1(self.sa1(self.down1(x1, temb), m2), c, cond_mask)
        x3 = self.ca2(self.sa2(self.down2(x2, temb), m3), c, cond_mask)
        x4 = self.ca3(self.sa3(self.down3(x3, temb), m4), c, cond_mask)

        x4 = self.bot(x4)

        h = self.ca4(self.sa4(self.up1(x4, x3, temb), m3), c, cond_mask)
        h = self.ca5(self.sa5(self.up2(h, x2, temb), m2), c, cond_mask)
        h = self.sa6(self.up3(h, x1, temb), m1)
        out = self.outc(h)

        if self.out_bins is None:
            return out
        logits = out[:, : 9 * self.out_bins].reshape(
            x.shape[0], 9, self.out_bins, f)
        return logits, out[:, -1]
