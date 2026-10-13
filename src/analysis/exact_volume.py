#!/usr/bin/env python3
"""Exact symmetric-difference volume via manifold3d CSG.

vol(A \\ B) + vol(B \\ A), computed on the real solids rather than a
discretization, so a threshold read off the resulting histogram means what it
says -- no resolution floor, and a reported zero means the two solids are the
same solid.

Only one boolean per pair is needed. Inclusion-exclusion gives

    |A △ B| = |A| + |B| - 2|A ∩ B|
    |A ∪ B| = |A| + |B| -  |A ∩ B|

and both identities were measured exact to 1.5e-14 on this corpus, so the
intersection is computed and the rest is arithmetic -- four times cheaper than
evaluating A-B, B-A, A+B and A^B separately.

The precision trap
------------------
`manifold3d.Mesh` stores vertices as float32, and manifold3d derives its
vertex-merge tolerance from the bounding box -- so coordinate magnitude sets
it. This corpus is EPSG:28992 with y ~ 450,000, where float32 gives a **5.4e-2 m
tolerance**: it welds vertices 5 cm apart. Measured over 200 real pairs, the
symmetric difference then drifts by a median 2.8e-3 and up to **4.4e-2**
relative. Note that is ~100x worse than the drift on a single volume (4.6e-5),
because a symmetric difference is a small difference of two large near-equal
numbers and cancellation amplifies the error.

**`Mesh64` (float64) is what fixes this**, taking the tolerance to 4.5e-7 m and
the drift to a median 4.2e-11 / max 1.6e-8 --
`test_world_coordinates_do_not_coarsen_the_tolerance` pins it, and is the one
test in the suite that fails if the backend is swapped back to `Mesh`.

Pairs are ALSO shifted to a shared local origin. Measured, that is insurance
rather than a requirement: at float64 the results are the same without it. It is
kept because it costs nothing and makes the conditioning independent of the
CRS. Shared, not per-solid -- translating each solid by its own centre would
destroy their relative position, which `test_both_solids_share_one_origin` pins.
"""
import logging

import numpy as np

try:
    import manifold3d as m3
except ImportError as exc:                       # pragma: no cover
    raise ImportError(
        "exact_volume needs manifold3d (pip install manifold3d); it is listed "
        "in requirements.txt."
    ) from exc

logger = logging.getLogger(__name__)

# A solid whose volume is below this is treated as nothing at all. Squarely
# below any real building and above the noise of a degenerate sliver.
EMPTY_M3 = 1e-9


class ManifoldError(ValueError):
    """A mesh manifold3d could not turn into a solid."""


def to_manifold(verts, faces, origin):
    """Build a `Manifold` from a triangle mesh, shifted to `origin`.

    Args:
        verts: [V, 3] coordinates in metres.
        faces: [F, 3] triangle indices.
        origin: [3] point subtracted before the float conversion. Pass the same
            origin for both solids of a pair, or their relative position moves.

    Returns:
        manifold3d.Manifold

    Raises:
        ManifoldError: the mesh is empty, not a closed orientable surface, or
            inside-out. The sign check lives here rather than at the call sites
            so every path degrades the same way: `fix_winding` is known to
            randomise on open meshes, and a negative volume would otherwise
            flow into the arithmetic below and produce a consistent, entirely
            meaningless answer.
    """
    verts = np.asarray(verts, dtype=float).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(verts) == 0 or len(faces) == 0:
        raise ManifoldError("empty mesh")

    mesh = m3.Mesh64(
        vert_properties=np.ascontiguousarray(verts - np.asarray(origin, dtype=float),
                                             dtype=np.float64),
        tri_verts=np.ascontiguousarray(faces, dtype=np.uint64))
    man = m3.Manifold(mesh)
    if man.is_empty():
        raise ManifoldError(f"not a closed solid: status {man.status()!r}")
    if man.volume() < 0:
        raise ManifoldError("inverted winding (negative volume)")
    return man


def _blank(vol_a=0.0, vol_b=0.0, symdiff=0.0, union=0.0, inter=0.0,
           ok=False, note=""):
    rel = symdiff / union if union > EMPTY_M3 else (0.0 if symdiff <= EMPTY_M3 else 1.0)
    return {"vol_lod1_m3": vol_a, "vol_lod2_m3": vol_b, "symdiff_m3": symdiff,
            "union_m3": union, "intersection_m3": inter,
            "iou": (inter / union) if union > EMPTY_M3 else 1.0,
            "rel_symdiff": rel, "ok": ok, "note": note}


def exact_pair_volumes(cond, target):
    """Exact volumetric distance between one building's LOD1 and its LOD2.

    Args:
        cond: ``(verts, faces)`` of the LOD1 solid.
        target: ``(verts, faces)`` of the LOD2 solid.

    Returns:
        dict: ``symdiff_m3``, ``rel_symdiff`` (scale-free, in [0, 1]), ``iou``,
        ``vol_lod1_m3``, ``vol_lod2_m3``, ``union_m3``, ``intersection_m3``,
        ``ok`` and ``note``.

        ``ok`` is False when the answer is not trustworthy -- an empty or
        non-closed mesh, or a negative volume from inverted winding, which
        `fix_winding` is known to produce on open meshes. Such rows still carry
        finite numbers so a corpus pass never crashes, but they must be
        filtered out of any distribution rather than averaged into it.
    """
    (cv, cf), (tv, tf) = cond, target
    cv = np.asarray(cv, dtype=float).reshape(-1, 3)
    tv = np.asarray(tv, dtype=float).reshape(-1, 3)

    have = [v for v in (cv, tv) if len(v)]
    if not have:
        return _blank(ok=True, note="both empty")
    # ONE origin for the pair: a per-solid origin would stack two buildings that
    # are actually 50 m apart.
    origin = np.vstack(have).min(axis=0)

    try:
        a = to_manifold(cv, cond[1], origin)
    except ManifoldError as exc:
        try:
            b = to_manifold(tv, target[1], origin)
            vb = float(b.volume())
        except ManifoldError:
            return _blank(note=f"both unusable: {exc}")
        return _blank(vol_b=vb, symdiff=vb, union=vb, note=f"lod1 unusable: {exc}")

    try:
        b = to_manifold(tv, target[1], origin)
    except ManifoldError as exc:
        va = float(a.volume())
        return _blank(vol_a=va, symdiff=va, union=va, note=f"lod2 unusable: {exc}")

    # Both volumes are non-negative here: `to_manifold` rejects inside-out.
    va, vb = float(a.volume()), float(b.volume())
    inter = float((a ^ b).volume())
    # Clamp: an intersection marginally outside [0, min(va, vb)] is float noise
    # at the 1e-14 level, and a negative would put rel_symdiff above 1.
    inter = float(np.clip(inter, 0.0, min(va, vb)))
    symdiff = va + vb - 2.0 * inter
    union = va + vb - inter
    return _blank(vol_a=va, vol_b=vb, symdiff=max(symdiff, 0.0), union=union,
                  inter=inter, ok=True)
