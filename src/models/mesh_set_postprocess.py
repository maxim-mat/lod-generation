"""A sampled face set back to an indexed triangle mesh.

The autoregressive branch gets vertex sharing free: its tokens *are* bin
indices, so two faces that name the same corner produce byte-identical
coordinates and `weld` merges them. A continuous sampler produces two floats
that differ in the eighth decimal, `weld` merges nothing, and every edge of the
mesh is a crack. The snap onto the `num_bins` grid is what restores the
autoregressive branch's guarantee -- and it costs nothing in accuracy, since
half a bin is already the tokenizer's own error floor.

Order matters and is the reference's (see `src/eval/mesh_postprocess`): weld
before dropping duplicates, because two faces are only duplicates once their
corners are the same vertex.
"""
import logging

import numpy as np
import torch

from src.dataset.mesh_dataset import NUM_BINS, dequantize, fix_winding, quantize
from src.eval.mesh_postprocess import drop_duplicate_faces, weld

logger = logging.getLogger(__name__)


def _as_face_major(x):
    """``[10, F]`` or ``[F, 10]`` to ``[F, 10]`` numpy."""
    a = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    if a.ndim != 2:
        raise ValueError(f"Expected a 2-D sample, got shape {a.shape}.")
    if a.shape[0] == 10 and a.shape[1] != 10:
        return a.T
    if a.shape[1] == 10:
        return a
    raise ValueError(
        f"Neither axis of {a.shape} is the 10-channel axis; a sample must be "
        "[10, F] or [F, 10].")


def _triangle_areas(tri):
    """``[F,3,3] -> [F]`` via half the cross-product norm."""
    if len(tri) == 0:
        return np.zeros(0)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)


def faces_to_mesh(x, num_bins=NUM_BINS, snap=True, min_area=1e-6):
    """One sampled face set to ``(verts, faces, stats)``.

    Args:
        x: ``[10, F]`` or ``[F, 10]``. Channels 0-8 are the three corners in
            the normalized box; channel 9 is presence, and a face survives when
            it is positive.
        num_bins: grid to snap onto. Must match the dataset's, or the snap
            moves geometry rather than de-duplicating it.
        snap: round coordinates onto the bin grid before welding. Turning this
            off is a diagnostic, not a mode: see the module docstring.
        min_area: faces below this (normalized box units squared) are dropped.
            A diffusion sample can land all three corners in one bin; such a
            face contributes no surface and makes `fix_winding` pick a normal
            from a zero-length cross product.

    Returns:
        tuple: ``(verts [V,3], faces [F,3] int64, stats dict)``. `stats` carries
        the four drop counts and the final vertex/face counts, which is what
        makes a bad sample legible -- a mesh with the right chamfer and 400
        vertices where 60 belong is a different failure from a wrong shape.
    """
    a = _as_face_major(x)
    stats = {"n_slots": len(a)}

    present = a[:, 9] > 0.0
    tri = a[present, :9].reshape(-1, 3, 3).astype(float)
    stats["n_dropped_absent"] = int((~present).sum())

    if snap and len(tri):
        tri = dequantize(quantize(tri.reshape(-1, 3), num_bins), num_bins)
        tri = tri.reshape(-1, 3, 3)

    keep = _triangle_areas(tri) > min_area
    stats["n_dropped_degenerate"] = int((~keep).sum())
    tri = tri[keep]

    if len(tri) == 0:
        stats.update(n_dropped_duplicate=0, n_verts=0, n_faces=0)
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), stats

    verts = tri.reshape(-1, 3)
    faces = np.arange(len(verts), dtype=np.int64).reshape(-1, 3)

    # Counted across the weld, not after it: `weld` is `canonicalize`, which
    # already drops duplicate faces itself, so a count taken afterwards is
    # structurally always zero. The two drop counts stay disjoint because the
    # only faces the vertex merge can collapse are ones with two exactly equal
    # corners -- exactly-zero area, and so already gone at `min_area`.
    before = len(faces)
    verts, faces = weld(verts, faces)
    faces = drop_duplicate_faces(faces)
    stats["n_dropped_duplicate"] = before - len(faces)

    faces = fix_winding(verts, faces)
    stats.update(n_verts=len(verts), n_faces=len(faces))
    return verts, faces, stats
