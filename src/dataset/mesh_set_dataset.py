"""LOD2 as a padded face set, for the non-autoregressive diffusion branch.

`MeshDataset` spells a mesh out as a token sequence for an autoregressive
model. This spells the same pair out as a fixed-width array of faces, which is
what a diffusion denoiser consumes: `[F, 10]`, nine coordinate channels plus a
presence channel, in the same LOD1-normalized frame `MeshDataset` already uses.

Vertices are repeated per face and connectivity is not represented. Shared
vertices come back post-hoc, in `src/models/mesh_set_postprocess.py`.
"""
import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset.mesh_dataset import (
    NUM_BINS,
    _scan_lod_dir,
    dequantize,
    normalize_to_unit_box,
    quantize,
)

logger = logging.getLogger(__name__)

# Presence channel values. Symmetric around 0 so the sign is the decision rule
# and the channel has the same scale as a coordinate -- one Gaussian noise
# level then fits all ten channels without a per-channel weight.
PRESENT, ABSENT = 0.5, -0.5
N_CHANNELS = 10


def faces_to_array(verts, faces):
    """``[V,3]`` vertices + ``[F,3]`` indices to ``[F,3,3]`` explicit corners."""
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        return np.zeros((0, 3, 3), dtype=float)
    return verts[faces]


def _grid_key(tri, num_bins):
    """``[F,3]`` lexicographic (z,y,x) rank of each corner, on the bin grid.

    Decided on the quantized grid rather than on the floats so the comparison
    is exact and total: two corners in the same bin are the same corner as far
    as the model can ever tell, and float lexicographic order on near-equal
    coordinates is not stable across platforms.
    """
    q = quantize(np.asarray(tri, dtype=float).reshape(-1, 3), num_bins)
    q = q.reshape(-1, 3, 3)
    return (q[:, :, 2] * num_bins + q[:, :, 1]) * num_bins + q[:, :, 0]


def rotate_faces_canonical(tri, num_bins=NUM_BINS):
    """Cyclically rotate each face so its ``(z,y,x)``-smallest corner leads.

    A triangle has three spellings with identical geometry and identical
    winding. Without this the coordinate loss punishes a face for being spelled
    from a different corner, which is not an error the model can fix -- it is
    an error in the target. Rotation, not sorting: a sort would reverse winding
    on half the faces, which `fix_winding` would then have to undo.

    Args:
        tri: ``[F,3,3]`` face corners.
        num_bins: grid used to break ties exactly. Must match the dataset's.

    Returns:
        np.ndarray: ``[F,3,3]``, same dtype semantics as the input.
    """
    tri = np.asarray(tri, dtype=float)
    if len(tri) == 0:
        return tri
    start = _grid_key(tri, num_bins).argmin(axis=1)
    idx = (start[:, None] + np.arange(3)[None, :]) % 3
    return np.take_along_axis(tri, idx[:, :, None], axis=1)


def morton_order(tri, num_bins=NUM_BINS):
    """Permutation sorting faces by the Morton code of their centroid.

    Z-order interleaving gives a 1-D sequence in which adjacency implies
    spatial proximity, which is the property a stride-2 convolution along the
    face axis needs and a file-order face list does not have. Ties are broken
    by the canonical corner key so the order is total and reproducible.

    Args:
        tri: ``[F,3,3]`` face corners in ``[-0.5, 0.5]``.
        num_bins: quantization grid; the code uses ``log2(num_bins)`` bits/axis.

    Returns:
        np.ndarray: ``[F]`` int64 permutation.
    """
    tri = np.asarray(tri, dtype=float)
    if len(tri) == 0:
        return np.zeros(0, dtype=np.int64)
    bits = int(np.log2(num_bins))
    if 2 ** bits != num_bins:
        raise ValueError(f"num_bins must be a power of two for Morton codes, got {num_bins}.")

    q = quantize(tri.mean(axis=1), num_bins).astype(np.int64)   # [F,3] centroids
    code = np.zeros(len(q), dtype=np.int64)
    for b in range(bits):
        for axis in range(3):
            code |= ((q[:, axis] >> b) & 1) << (3 * b + axis)
    # Secondary key: the face's own smallest corner, so co-centroid faces
    # (a fold, two triangles of one quad) get a deterministic order.
    tie = _grid_key(tri, num_bins).min(axis=1)
    return np.lexsort((tie, code)).astype(np.int64)


def _pack(tri, num_bins, order, state="continuous"):
    """``[F,3,3]`` corners to the ``[F,10]`` channel layout (+ optional bins).

    `state` decides whether the float channels are snapped to the grid and
    whether integer bins come along (plan D10). The one-hot expansion is NOT
    done here: it is 9 x num_bins channels per face, which would multiply the
    collate buffer and the worker-to-main-process transfer by ~128 for a tensor
    `MeshDiffusionModule` can rebuild from `x_bins` on the accelerator with one
    `scatter_`.
    """
    tri = rotate_faces_canonical(tri, num_bins)
    if order == "morton":
        tri = tri[morton_order(tri, num_bins)]
    elif order != "none":
        raise ValueError(f"Unknown order: {order!r}. Expected 'morton' or 'none'.")

    x = np.zeros((len(tri), N_CHANNELS), dtype=np.float32)
    x[:, :9] = tri.reshape(len(tri), 9)
    x[:, 9] = PRESENT
    if state == "continuous":
        return x, None
    if state not in ("quantized", "onehot", "bins"):
        raise ValueError(
            f"Unknown state: {state!r}. Expected 'continuous', 'quantized', "
            "'onehot' or 'bins'.")
    bins = quantize(tri.reshape(-1, 3), num_bins).reshape(len(tri), 9).astype(np.int64)
    # Snap the float channels onto the same grid so a regression readout and a
    # categorical one are scored against an identical target -- which is what
    # makes the `quantized`/`mse` control arm (b3) interpretable at all.
    x[:, :9] = dequantize(bins, num_bins).astype(np.float32)
    return x, bins


class MeshSetDataset(Dataset):
    """Paired (LOD1, LOD2) buildings as padded face-set arrays.

    Shares `MeshDataset`'s scan, `max_faces` filter and normalization frame
    exactly, so a diffusion run and an autoregressive run on the same
    `dataset_dir` see the same corpus and the same coordinate frame.

    Args:
        dataset_dir: root holding the two LOD directories.
        lod_in, lod_out: directory names, not LOD numbers.
        num_bins: quantization grid, for rotation ties, Morton codes and the
            quantized states. Must match `mesh_data.num_bins` for comparability.
        margin_lo, margin_hi: per-axis headroom, passed to
            `normalize_to_unit_box`. Same values as the AR branch.
        max_faces: drop pairs where either side exceeds this triangle count.
        max_files: read only the first N files per LOD. Smoke tests only.
        order: "morton" (spatially sorted) or "none" (file order).
        state: "continuous" | "quantized" | "onehot" | "bins" (plan D10).
            Anything but "continuous" snaps the float channels to the grid and
            emits `x_bins`. "onehot" is expanded in the model, not here.
    """

    def __init__(self, dataset_dir, lod_in="LOD1", lod_out="LOD2",
                 num_bins=NUM_BINS, margin_lo=(0.0, 0.0, 0.0),
                 margin_hi=(0.0, 0.0, 0.1), max_faces=200, max_files=None,
                 order="morton", state="continuous"):
        self.num_bins = num_bins
        self.margin_lo = np.asarray(margin_lo, dtype=float)
        self.margin_hi = np.asarray(margin_hi, dtype=float)
        self.order = order
        self.state = state

        meshes_in = _scan_lod_dir(dataset_dir, lod_in, max_files)
        meshes_out = _scan_lod_dir(dataset_dir, lod_out, max_files)
        ids = sorted(set(meshes_in) & set(meshes_out))
        if max_faces is not None:
            kept = [i for i in ids
                    if len(meshes_out[i][1]) <= max_faces
                    and len(meshes_in[i][1]) <= max_faces]
            logger.info("Dropped %d/%d buildings over max_faces=%d",
                        len(ids) - len(kept), len(ids), max_faces)
            ids = kept
        self.ids = ids
        self.pairs = [(meshes_in[i], meshes_out[i]) for i in ids]
        if not self.ids:
            logger.warning("No buildings shared between %s and %s under %s",
                           lod_in, lod_out, dataset_dir)
        self.max_faces_seen = max(
            (max(len(f_out), len(f_in))
             for (_, f_in), (_, f_out) in self.pairs), default=0)
        logger.info("MeshSetDataset: %d pairs, longest face set %d",
                    len(self.ids), self.max_faces_seen)

    def __len__(self):
        return len(self.ids)

    def mesh_pair(self, index):
        """Raw ``((verts, faces), (verts, faces))`` in metres, for .obj dumps."""
        return self.pairs[index]

    def __getitem__(self, index):
        (v_in, f_in), (v_out, f_out) = self.pairs[index]
        v_in_n, center, scale = normalize_to_unit_box(
            v_in, margin_lo=self.margin_lo, margin_hi=self.margin_hi)
        v_out_n, _, _ = normalize_to_unit_box(
            v_out, ref=v_in, margin_lo=self.margin_lo, margin_hi=self.margin_hi)

        x, bins = _pack(faces_to_array(v_out_n, f_out),
                        self.num_bins, self.order, self.state)
        # The condition is always Morton-sorted: it is read by cross-attention,
        # which is permutation invariant, so the order costs nothing -- but a
        # sorted condition makes the U-Net's condition encoder see the same
        # locality its main path does.
        cond, _ = _pack(faces_to_array(v_in_n, f_in),
                        self.num_bins, "morton", "continuous")

        item = {
            "x": torch.from_numpy(x),
            "cond": torch.from_numpy(cond),
            "id": self.ids[index],
            "center": torch.tensor(center, dtype=torch.float32),
            "scale": torch.tensor(scale, dtype=torch.float32),
        }
        if bins is not None:
            item["x_bins"] = torch.from_numpy(bins)
        return item


def _pad_stack(seqs, width, fill_coord=0.0):
    """``list[[F,10]]`` to ``([B,10,width], [B,width] bool)``, True = real."""
    out = torch.zeros((len(seqs), width, N_CHANNELS), dtype=torch.float32)
    out[:, :, :9] = fill_coord
    out[:, :, 9] = ABSENT
    mask = torch.zeros((len(seqs), width), dtype=torch.bool)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
        mask[i, : len(s)] = True
    return out.permute(0, 2, 1).contiguous(), mask


def mesh_set_collate_fn(batch, multiple_of=8, width=None, jitter_to=None):
    """Right-pad a batch of face sets to a common, U-Net-divisible width.

    The slots past a building's face count are NOT masked out of the model --
    they are the slots it must learn to mark absent, DETR's no-object class
    (arXiv:2005.12872 section 3.1). `x_mask` is still returned, because the
    coordinate loss needs to know which slots carry a real target, but it is
    supervision and never a model input. Handing it to the denoiser is what
    made face count readable straight off the input rather than predicted.

    That makes the slot budget a real hyperparameter rather than a packing
    detail:

    * training uses the batch maximum, optionally jittered up toward
      `jitter_to`, so the model meets a range of budgets and does not overfit
      to one;
    * eval pins `width` to a constant so a score is reproducible and two
      epochs are comparable.

    Args:
        batch: list of `MeshSetDataset` items.
        multiple_of: pad width is rounded up to this. 8 for the 3-level U-Net.
        width: exact slot budget. Must be >= the batch's own requirement.
        jitter_to: sample a budget uniformly in ``[base, jitter_to]``. Ignored
            when `width` is given.

    Returns:
        dict: channels-first tensors plus boolean masks (True = real face).
    """
    def base_width(key):
        longest = max(len(item[key]) for item in batch)
        return int(np.ceil(max(longest, 1) / multiple_of) * multiple_of)

    base = base_width("x")
    if width is not None:
        target = int(width)
        if target < base:
            raise ValueError(
                f"slot budget {target} is below this batch's requirement {base}. "
                "mesh_data.max_faces must not exceed mesh_diffusion.slot_budget.")
    elif jitter_to is not None and int(jitter_to) > base:
        steps = (int(jitter_to) - base) // multiple_of + 1
        target = base + multiple_of * int(torch.randint(steps, (1,)).item())
    else:
        target = base

    x, x_mask = _pad_stack([item["x"] for item in batch], target)
    # The condition keeps its own tight width: it is an input the model reads
    # through cross-attention, never something it has to decide the extent of,
    # so its padding stays masked and costs nothing to keep minimal.
    cond, cond_mask = _pad_stack([item["cond"] for item in batch], base_width("cond"))
    out = {"x": x, "x_mask": x_mask, "cond": cond, "cond_mask": cond_mask,
           "ids": [item["id"] for item in batch]}
    for key in ("center", "scale"):
        out[key] = torch.stack([item[key] for item in batch])
    if "x_bins" in batch[0]:
        w = x.shape[-1]
        bins = torch.zeros((len(batch), w, 9), dtype=torch.long)
        for i, item in enumerate(batch):
            bins[i, : len(item["x_bins"])] = item["x_bins"]
        out["x_bins"] = bins.permute(0, 2, 1).contiguous()
    return out
