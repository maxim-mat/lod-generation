"""Sequence layout of the aligned image prefix: position ids and the attention mask.

Spec: docs/superpowers/specs/2026-11-02-aligned-prefix-design.md, §3.1 and §3.5.

The sequence handed to OPT is ``[image (P) | LOD1 condition (C) | target (T)]``.

* Positions (§3.1): mesh tokens count only mesh tokens, so they land on exactly
  the ids an image-free (v5) model gives them; image tokens get `NO_POSITION`,
  which `OPTSinusoidalPositions` embeds as the zero vector. They carry only the
  encoder's own 2-D (u, v) code.
* Attention (§3.5): causal, plus -- when bidirectional -- image queries see all
  image and all LOD1 keys. LOD1 rows see every image token by causality alone.
  Pad and masked-image keys are never attended, except that every image query
  keeps its own diagonal key: a query with no key at all is NaN under SDPA, and
  the NaN would spread. Nothing reads a masked image row, because every other
  row has its key masked.
"""
import torch

# Distinct from the -1 that OPT's cumsum gives pad slots, which is clamped to 0.
NO_POSITION = -2


def mesh_position_ids(n_img, mesh_keep):
    """``[B, n_img + M]`` OPT position ids: `NO_POSITION` for the image slots,
    then a cumsum over the mesh-only keep mask (pads come out as -1).

    Args:
        n_img (int): image prefix length P.
        mesh_keep (BoolTensor): ``[B, M]`` True at real condition/target tokens.
    """
    keep = mesh_keep.long()
    mesh = torch.cumsum(keep, dim=1) * keep - 1
    img = mesh.new_full((mesh.shape[0], n_img), NO_POSITION)
    return torch.cat([img, mesh], dim=1)


def prefix_attention_mask(img_keep, cond_keep, tgt_keep, bidirectional):
    """``[B, 1, L, L]`` bool keep-mask (True = attend), ``L = P + C + T``.

    Args:
        img_keep (BoolTensor): ``[B, P]``; False for a masked (invalid) image.
        cond_keep (BoolTensor): ``[B, C]``; False at condition pads.
        tgt_keep (BoolTensor): ``[B, T]``; False at target pads.
        bidirectional (bool): image queries also see later image and all LOD1 keys.
    """
    p, c = img_keep.shape[1], cond_keep.shape[1]
    keep = torch.cat([img_keep, cond_keep, tgt_keep], dim=1)          # [B, L]
    n = keep.shape[1]
    q = torch.arange(n, device=keep.device)[:, None]
    k = torch.arange(n, device=keep.device)[None, :]
    allowed = k <= q
    if bidirectional:
        allowed = allowed | ((q < p) & (k < p + c))
    mask = allowed[None] & keep[:, None, :]
    mask = mask | ((q == k) & (q < p))[None]
    return mask[:, None]
