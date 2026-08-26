"""LOD1 as a legal region for LOD2 vertices, enforced during sampling.

MeshWeaver (arXiv:2606.04688) section 3.2 fix (3): mask the logits of empty
voxels so every predicted vertex is anchored near the input surface. The
2026-08-24 research scan flags this as the item most likely to transfer here,
because LOD1 already bounds where an LOD2 vertex can legally sit.

In an autoregressive model it is a logit mask applied once per token. In
diffusion it is a projection applied once per reverse step, which is both
cheaper and stronger: it acts on the whole mesh at every noise level rather
than on one coordinate at a time with no chance to revise.

The dilation is not a tuning knob to leave at zero. LOD2 is *not* contained in
LOD1 -- the ridge rises above what is only the median roof height, which is the
same fact `margin_hi` exists for -- so an undilated scaffold would project every
ridge vertex back down onto the LOD1 roof and flatten exactly the geometry the
model is being asked to invent.
"""
import logging

import numpy as np
import torch

from src.dataset.mesh_dataset import dequantize, quantize

logger = logging.getLogger(__name__)


def _rasterize(tri, grid):
    """``[F,3,3]`` in ``[-0.5,0.5]`` to a ``[G,G,G]`` bool occupancy grid.

    Marks the voxel of every corner and of the edge midpoints. Not a
    conservative triangle rasteriser: the scaffold is dilated anyway, and a
    dilation of 3 voxels covers a triangle whose edges are shorter than 6
    voxels, which every LOD1 prism face is at G = 32.
    """
    occ = np.zeros((grid, grid, grid), dtype=bool)
    if len(tri) == 0:
        return occ
    pts = [tri[:, i] for i in range(3)]
    pts += [(tri[:, i] + tri[:, (i + 1) % 3]) / 2 for i in range(3)]
    pts += [tri.mean(axis=1)]
    p = np.concatenate(pts, axis=0)
    idx = np.clip(np.rint((p + 0.5) * (grid - 1)).astype(int), 0, grid - 1)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ


def _fill_axis(occ, axis):
    """Occupied span along one axis: everything between the first and last hit.

    Lines that touch nothing stay empty (``lo`` ends above ``hi``).
    """
    shape = [-1 if k == axis else 1 for k in range(3)]
    idx = np.broadcast_to(np.arange(occ.shape[axis]).reshape(shape), occ.shape)
    lo = np.where(occ, idx, occ.shape[axis] + 1).min(axis=axis, keepdims=True)
    hi = np.where(occ, idx, -1).max(axis=axis, keepdims=True)
    return (idx >= lo) & (idx <= hi)


def _solidify(occ):
    """The LOD1 shell's interior, as the union of its three axis spans.

    A surface rasterisation is a hollow shell, and a shell is not a legal
    *region*: it forbids the inside of the building, and it forbids everything
    over a footprint whose roof and floor the LOD1 mesh does not spell out --
    which is most of the volume a ridge has to rise through.

    `_raster` in `src/eval/mesh_metrics.py` is the exact solid fill, but it is
    a ray-parity fill along +z and so needs a watertight shell with non-vertical
    faces; a prism of four vertical walls contributes no xy footprint at all and
    fills to nothing. The union of the three axis spans degrades gracefully
    instead: exact for a convex closed solid, and merely *permissive* for a
    non-convex or open one. Permissive is the safe error here -- the scaffold
    says where a vertex MAY sit, so over-filling weakens the constraint while
    under-filling deletes the geometry the model was asked to invent.
    """
    return occ | _fill_axis(occ, 0) | _fill_axis(occ, 1) | _fill_axis(occ, 2)


def _dilate(occ, r):
    """Cubic dilation by ``r`` voxels, via repeated 6-neighbour ``or``."""
    out = occ.copy()
    for _ in range(r):
        acc = out.copy()
        for axis in range(3):
            acc |= np.roll(out, 1, axis=axis)
            acc |= np.roll(out, -1, axis=axis)
        out = acc
    return out


def lod1_scaffold(cond, cond_mask, voxel=1.0 / 32, dilate=3):
    """Occupancy grid of the dilated LOD1 surface, per batch item.

    Args:
        cond: ``[B, 10, Fc]`` LOD1 face set in the normalized box.
        cond_mask: ``[B, Fc]`` bool, True at real faces.
        voxel: grid spacing in normalized box units; the grid is ``1/voxel``
            cells per axis.
        dilate: growth in voxels. See the module docstring -- 0 is wrong.

    Returns:
        BoolTensor: ``[B, G, G, G]`` on ``cond``'s device.
    """
    b, _, fc = cond.shape
    grid = int(round(1.0 / voxel))
    out = np.zeros((b, grid, grid, grid), dtype=bool)
    c = cond.detach().cpu().numpy()
    m = cond_mask.detach().cpu().numpy()
    for i in range(b):
        # Mixing the integer index with the boolean mask puts the selected
        # faces first, so this is already [n_real, 9] -- no transpose.
        tri = c[i, :9, m[i]].reshape(-1, 3, 3) if m[i].any() else np.zeros((0, 3, 3))
        out[i] = _dilate(_solidify(_rasterize(tri, grid)), dilate)
    return torch.from_numpy(out).to(cond.device)


def _legal_points(scaffold_i, grid):
    """``[N, 3]`` centre coordinates of the occupied voxels, in ``[-0.5,0.5]``."""
    idx = torch.nonzero(scaffold_i, as_tuple=False).float()
    return idx / (grid - 1) - 0.5


def make_projector(scaffold, apply_below_t=0.5):
    """Continuous-state projection callback for `BaseProcess.sample`.

    Args:
        scaffold: ``[B, G, G, G]`` bool from `lod1_scaffold`.
        apply_below_t: only project once ``t`` is below this. Above it the
            state is nearly pure noise and every coordinate is out of bounds,
            so projecting would overwrite the denoiser rather than guide it.

    Returns:
        callable: ``(t, x [B,10,F]) -> x``. Presence (channel 9) is never
        touched -- it is not a coordinate and has no legal region.
    """
    grid = scaffold.shape[-1]
    legal = [_legal_points(scaffold[i], grid) for i in range(scaffold.shape[0])]

    def project(t, x):
        if t >= apply_below_t:
            return x
        out = x.clone()
        for i, pts in enumerate(legal):
            if len(pts) == 0:
                continue
            v = out[i, :9].T.reshape(-1, 3)            # [F*3, 3]
            d = torch.cdist(v, pts.to(v.device))       # [F*3, N]
            nearest = pts.to(v.device)[d.argmin(dim=1)]
            # Only move what is actually outside: a legal vertex snapped to a
            # voxel centre would be quantized twice, once here and once in
            # faces_to_mesh, for no reason.
            outside = d.min(dim=1).values > (1.5 / (grid - 1))
            v = torch.where(outside[:, None], nearest, v)
            out[i, :9] = v.reshape(-1, 9).T
        return out

    return project


def make_bin_masker(scaffold, num_bins=128, apply_below_t=0.5):
    """Discrete-state analogue: move illegal bins to the nearest legal one.

    The logit mask MeshWeaver describes would be the exact form, but the
    denoiser's nine heads are independent per axis while legality is a joint
    property of the triple -- masking each axis separately admits combinations
    the scaffold forbids. Projecting the sampled triple is the honest version.

    Args:
        scaffold: ``[B, G, G, G]`` bool.
        num_bins: coordinate alphabet size.
        apply_below_t: as `make_projector`.

    Returns:
        callable: ``(t, bins [B,9,F]) -> bins``.
    """
    grid = scaffold.shape[-1]
    legal = [_legal_points(scaffold[i], grid) for i in range(scaffold.shape[0])]

    def project(t, bins):
        if t >= apply_below_t:
            return bins
        out = bins.clone()
        for i, pts in enumerate(legal):
            if len(pts) == 0:
                continue
            coords = torch.from_numpy(
                dequantize(out[i].detach().cpu().numpy(), num_bins)).float()
            v = coords.T.reshape(-1, 3)
            d = torch.cdist(v, pts.cpu())
            outside = d.min(dim=1).values > (1.5 / (grid - 1))
            v = torch.where(outside[:, None], pts.cpu()[d.argmin(dim=1)], v)
            snapped = quantize(v.reshape(-1, 9).numpy(), num_bins)
            out[i] = torch.from_numpy(snapped).to(out.device).T
        return out

    return project
