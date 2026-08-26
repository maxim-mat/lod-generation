"""Permutation-equivariant transformer denoiser over a face set.

With `pos_embed: none` this network cannot tell one output slot from another,
which is the point: a mesh is a *set* of faces, and any model that reads a
canonical order is reading a convention rather than geometry. The price is that
a slot-to-slot loss becomes meaningless -- there is no slot -- so this
configuration is only legal with `loss: hungarian`, which `validate_combination`
enforces.

No causal mask anywhere. The whole mesh is processed and emitted at once, so
none of the autoregressive machinery -- KV cache, beam search, exposure bias --
applies or is needed.
"""
import torch
import torch.nn as nn

from src.models.mesh_set_modules import (
    CrossAttention1d,
    FaceEncoder,
    SelfAttention1d,
    SinusoidalFacePositions,
    timestep_embedding,
)


class _Block(nn.Module):
    """Self-attention over faces, cross-attention to LOD1, additive time.

    The time embedding is added to every token rather than modulating the norms
    (DiT's adaLN-Zero). ponytail: additive is the smaller thing that works and
    matches the conv path's `Down`/`Up`; upgrade to adaLN-Zero if the time
    signal turns out to be too weak to steer the late steps.
    """

    def __init__(self, d_model, n_head, time_dim, dropout):
        super().__init__()
        self.sa = SelfAttention1d(d_model, n_head)
        self.ca = CrossAttention1d(d_model, d_model, n_head)
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, temb, cond, mask, cond_mask):
        x = x + self.emb(temb)[:, :, None]
        x = self.sa(x, mask)
        x = self.ca(x, cond, cond_mask)
        return self.drop(x)


class MeshSetTransformer(nn.Module):
    """Denoiser: ``(x_t, t, LOD1) -> prediction``, no convolution, no order.

    Args:
        in_ch, out_ch: 10 = 9 coordinates + presence.
        d_model, n_head, num_layers, dropout: the usual transformer knobs.
        pos_embed: "sinusoidal" makes the model order-aware, which is only
            meaningful under `order: morton`; "none" makes it permutation
            equivariant, which requires `loss: hungarian`.
        time_dim: timestep embedding width.
        pos_scale: constant the face index is divided by in the positional
            encoding. Pass `mesh_data.max_faces`; never the padded length.
        out_bins: when set, emits ``[B,9,K,F]`` bin logits and ``[B,F]``
            presence logits for the D3PM arm instead of ``[B,10,F]``.
        cond_ch: channels of the LOD1 condition. Always 10, and deliberately
            *not* tied to `in_ch` -- see `ConditionalMeshUNet`.
    """

    def __init__(self, in_ch=10, out_ch=10, d_model=256, n_head=8,
                 num_layers=8, dropout=0.1, pos_embed="sinusoidal",
                 time_dim=128, out_bins=None, cond_ch=10, pos_scale=200):
        super().__init__()
        if pos_embed not in ("sinusoidal", "none"):
            raise ValueError(
                f"pos_embed must be 'sinusoidal' or 'none', got {pos_embed!r}.")
        self.time_dim = time_dim
        self.out_bins = out_bins
        self.pos = (SinusoidalFacePositions(d_model, scale=pos_scale)
                    if pos_embed == "sinusoidal" else None)

        self.inp = nn.Conv1d(in_ch, d_model, 1)
        self.cond_encoder = FaceEncoder(cond_ch, d_model)
        self.blocks = nn.ModuleList(
            [_Block(d_model, n_head, time_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

        head_ch = out_ch if out_bins is None else 9 * out_bins + 1
        self.outc = nn.Conv1d(d_model, head_ch, 1)
        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)

    def forward(self, x, t, cond=None, mask=None, cond_mask=None):
        """``x [B,10,F]``, ``t [B]`` in ``[0,1]``, ``cond [B,10,Fc]`` or None."""
        b, _, f = x.shape
        if mask is None:
            mask = torch.ones(b, f, dtype=torch.bool, device=x.device)

        h = self.inp(x)
        if self.pos is not None:
            h = h + self.pos(f, x.device).T[None]
        temb = timestep_embedding(t, self.time_dim)
        c = self.cond_encoder(cond)

        for block in self.blocks:
            h = block(h, temb, c, mask, cond_mask)

        h = self.norm(h.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.outc(h)
        if self.out_bins is None:
            return out
        logits = out[:, : 9 * self.out_bins].reshape(b, 9, self.out_bins, f)
        return logits, out[:, -1]
