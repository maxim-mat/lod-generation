"""Paired geometric metrics: one generated mesh against its ground-truth twin.

Everything in `src/eval/` before this compares *distributions* -- MMD and
Wasserstein over feature vectors, novelty against the whole train set -- because
unconditional diffusion has no paired reference to compare against. The mesh
transformer is the first model here where every LOD1 condition has exactly one
true LOD2, so per-sample accuracy is finally definable.

Conventions worth knowing before reading a number off a chart:

  * Every function takes ``(verts [V, 3] metres, faces [F, 3])`` tuples and
    returns ``nan`` on empty or degenerate input rather than raising, so an
    aggregator can ``nanmean`` over a batch where some generations decoded to
    nothing.
  * Chamfer is **not** squared and is averaged over both directions, so its
    unit is metres and it reads as "average surface error". Callers log it as
    ``*_chamfer_m`` to keep that unambiguous.
  * Chamfer, F-score and p95 Hausdorff are three reductions over the *same*
    two distance arrays, so `surface_distances` computes them once and the
    three take arrays rather than meshes.

numpy + scipy only, no torch and no lightning, so this stays importable and
testable on its own -- same rule as `building_features.py`.
"""
import importlib.util
import logging

import numpy as np
import trimesh

from src.analysis.cityobject_analysis import is_watertight

logger = logging.getLogger(__name__)

# A triangle counts as part of the roof when its normal has this much upward
# tilt -- the same "upward-facing" notion `convert_to_lod1` uses to pick the
# LOD1 height, so `roof_height_error` measures the quantity that model is
# actually asked to improve on.
_UP = 0.1


# ----------------------------------------------------------------------
# Surface sampling and point-set distances
# ----------------------------------------------------------------------

def sample_surface(verts, faces, n=4096, seed=0):
    """``[n, 3]`` points spread uniformly over the triangle areas.

    Area-weighted, so a mesh whose triangles differ wildly in size (every LOD2
    roof, after fan triangulation) is still sampled evenly in space rather than
    evenly per triangle.

    Returns:
        np.ndarray: [n, 3], or [0, 3] if the mesh has no positive area.
    """
    v = np.asarray(verts, dtype=float)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(f) == 0 or len(v) == 0:
        return np.zeros((0, 3))

    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = float(areas.sum())
    if not np.isfinite(total) or total <= 0:
        return np.zeros((0, 3))

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(f), size=n, p=areas / total)
    # Reflect the unit square into the unit triangle: uniform in barycentrics.
    u, w = rng.random((n, 1)), rng.random((n, 1))
    flip = (u + w > 1).ravel()
    u[flip], w[flip] = 1.0 - u[flip], 1.0 - w[flip]
    return a[idx] + u * (b - a)[idx] + w * (c - a)[idx]


def _closest_surface_distance(mesh, points):
    """Exact distance from each point to the mesh *surface*, not to a sample."""
    # trimesh's only hard dependency is numpy; `rtree` -- which `closest_point`
    # needs, because it walks `mesh.triangles_tree` -- is under its `easy`
    # extra. So `pip install trimesh` yields a package that imports cleanly and
    # then dies inside the first eval. Checked here so the message names the
    # package instead of surfacing as an AttributeError from a library.
    if importlib.util.find_spec("rtree") is None:
        raise ImportError(
            "rtree is not installed, so trimesh cannot build the spatial index "
            "`closest_point` needs for point-to-surface distances. "
            "`pip install rtree` (it is in requirements.txt). trimesh declares "
            "it only as an extra (`trimesh[easy]`), so a plain install of "
            "trimesh leaves it out.")
    v = np.asarray(mesh[0], dtype=float)
    f = np.asarray(mesh[1], dtype=np.int64).reshape(-1, 3)
    tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
    return np.asarray(trimesh.proximity.closest_point(tm, points)[1], dtype=float)


def surface_distances(gen, gt, n=4096, seed=0):
    """Point-to-surface distances both ways between two meshes.

    Only the *query* side is sampled; the target is the triangles themselves,
    via `trimesh.proximity.closest_point`. That is the mesh-to-mesh convention
    (CGAL, MeshLab), as opposed to sampling both sides and doing
    point-cloud-to-point-cloud, which is what PyTorch3D's `chamfer_distance`
    computes because it takes point clouds by definition.

    The distinction is not cosmetic here. The point-cloud version this replaced
    answered every query with the nearest *sample*, and a target's samples sit
    ~sqrt(area / n) apart, so it carried a noise floor: 0.129 m at n=4096 on
    real buildings, scaling as 1/sqrt(n) and confirmed by a mesh scoring 0.129
    against *itself*. Measured on real LOD1/LOD2 pairs it overstated chamfer by
    126% (0.2011 m vs 0.0827 m), which is most of the LOD1 identity baseline.
    Now a mesh scores exactly 0 against itself at any sample count.

    Note two other conventions that differ across implementations and make
    cross-paper numbers incomparable: distances here are **unsquared** metres
    (PyTorch3D squares by default), and `chamfer_distance` **averages** the two
    directions rather than summing them.

    Returns:
        tuple: (d_gen->gt [n], d_gt->gen [n]), both empty if either mesh is
        degenerate or the proximity query fails.
    """
    p = sample_surface(*gen, n=n, seed=seed)
    q = sample_surface(*gt, n=n, seed=seed + 1)
    if len(p) == 0 or len(q) == 0:
        return np.zeros(0), np.zeros(0)
    try:
        return _closest_surface_distance(gt, p), _closest_surface_distance(gen, q)
    except (ValueError, IndexError, MemoryError) as exc:
        # A mesh degenerate enough to break the BVH is a real failure, but it
        # must not take the whole eval down mid-epoch.
        logger.warning("proximity query failed, reporting no distances: %s", exc)
        return np.zeros(0), np.zeros(0)


def chamfer_distance(d_ab, d_ba):
    """Mean bidirectional surface error in metres (not squared)."""
    if len(d_ab) == 0 or len(d_ba) == 0:
        return float("nan")
    return 0.5 * (float(d_ab.mean()) + float(d_ba.mean()))


def chamfer_floor(lod1, ref, n_points, seed=0):
    """Chamfer in metres of the do-nothing prediction: LOD1 handed back as-is.

    The reference line every logged chamfer needs. A distance in metres cannot
    be read on its own -- 0.2 m is excellent on a cathedral and worse than
    useless on a shed -- and the honest comparison is not the corpus mean in
    `src/eval/lod1_baseline.py` but this: the *same* buildings, scored against
    the *same* reference at the *same* point budget as the model's own number.
    A run that sits above its floor has not yet learned anything its input did
    not already say.

    Args:
        lod1: ``(verts [V,3] metres, faces)`` -- the condition, decoded exactly
            the way the model's own output is decoded, so the two numbers are
            the same quantity.
        ref: the mesh `lod1` is scored against. Must be whatever the generated
            mesh is scored against, or the ratio compares two different things.
        n_points, seed: passed to `surface_distances`. Must match the call that
            produced the number this is a floor for.

    Returns:
        float: ``nan`` when either side has no surface, matching
        `chamfer_distance`.
    """
    if not len(np.asarray(lod1[1])) or not len(np.asarray(ref[1])):
        return float("nan")
    d_ab, d_ba = surface_distances(lod1, ref, n=n_points, seed=seed)
    return float(chamfer_distance(d_ab, d_ba))


def chamfer_ratio(value, floor, min_floor=1e-3):
    """``value / floor``, or nan where the ratio is not defined.

    Split out because every call site has the same two traps: a nan on either
    side, and a floor at zero -- LOD1 and LOD2 being the same mesh, so doing
    nothing is already a perfect answer. Either one, left in the series,
    poisons the epoch mean for every other building in the split.

    `min_floor` is a length in metres, not an epsilon guarding division. The
    zero case does not arrive as an exact zero: point-sampled distances between
    two copies of one mesh come out around 1e-16, which divides to 1e16 rather
    than raising. Below a millimetre of surface difference the condition simply
    IS the target -- the building has no LOD2 detail to predict -- and the
    ratio carries noise instead of information, so the sample is dropped from
    this series. It still appears in the two raw series it was computed from.
    """
    if not np.isfinite(value) or not np.isfinite(floor) or floor < min_floor:
        return float("nan")
    return float(value / floor)


def f_score(d_ab, d_ba, tau):
    """Precision / recall / F1 of surface points falling within ``tau`` metres.

    More readable than chamfer because the threshold is in metres: "what
    fraction of the surface is within 25 cm" survives being quoted out of
    context, where a mean distance does not.
    """
    if len(d_ab) == 0 or len(d_ba) == 0:
        return float("nan"), float("nan"), float("nan")
    precision = float((d_ab < tau).mean())
    recall = float((d_ba < tau).mean())
    denom = precision + recall
    return precision, recall, (0.0 if denom == 0 else 2 * precision * recall / denom)


def hausdorff_p95(d_ab, d_ba):
    """Worst-case deviation, 95th percentile rather than the max.

    One stray vertex -- and a partially decoded generation has several --
    dominates a true Hausdorff distance completely, which makes the metric a
    report on the single worst outlier instead of on the mesh.
    """
    if len(d_ab) == 0 or len(d_ba) == 0:
        return float("nan")
    return float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95)))


# ----------------------------------------------------------------------
# Voxel grid: volumetric IoU and footprint IoU
# ----------------------------------------------------------------------

def _grid(verts_a, verts_b, voxel_m=0.25, max_dim=128):
    """Shared voxel grid over the union bounding box of two meshes.

    Fixed *voxel size*, not fixed grid dimension: a fixed dimension would make
    the resolution a function of building size, so IoU on a 40 m warehouse and
    on an 8 m house would not be the same measurement. `max_dim` only caps the
    worst case (128^3 booleans) by coarsening the voxel when a building is too
    large for the nominal size.

    Returns:
        tuple: (gx, gy, gz cell-centre coordinates, voxel edge in metres).
    """
    lo = np.minimum(verts_a.min(axis=0), verts_b.min(axis=0))
    hi = np.maximum(verts_a.max(axis=0), verts_b.max(axis=0))
    voxel = max(float(voxel_m), float((hi - lo).max()) / max_dim)

    lo, hi = lo - voxel, hi + voxel          # one voxel of padding each side
    dims = np.maximum(np.ceil((hi - lo) / voxel).astype(int), 1)
    axes = [lo[k] + (np.arange(dims[k]) + 0.5) * voxel for k in range(3)]
    return axes[0], axes[1], axes[2], voxel


def _raster(verts, faces, gx, gy, gz):
    """One pass over triangles yielding both the solid fill and its shadow.

    ``inside`` is a ray-parity fill along +z: a cell is inside iff an odd
    number of surface crossings lie above its centre. That is only meaningful
    for a watertight shell, which is why `volumetric_iou` gates on it.
    ``shadow`` is the xy projection, which needs no such guarantee.

    Returns:
        tuple: (inside [Nx, Ny, Nz] bool, shadow [Nx, Ny] bool).
    """
    v = np.asarray(verts, dtype=float)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    nx, ny, nz = len(gx), len(gy), len(gz)

    shadow = np.zeros((nx, ny), dtype=bool)
    # Difference array: a crossing at z adds 1 to every cell centre below it,
    # so one +1 at the bottom and one -1 at the crossing beats writing a range.
    diff = np.zeros((nx * ny, nz + 1), dtype=np.int32)

    for tri in f:
        a, b, c = v[tri]
        det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(det) < 1e-12:
            continue                          # vertical wall: no xy footprint

        i0 = np.searchsorted(gx, min(a[0], b[0], c[0]), side="left")
        i1 = np.searchsorted(gx, max(a[0], b[0], c[0]), side="right")
        j0 = np.searchsorted(gy, min(a[1], b[1], c[1]), side="left")
        j1 = np.searchsorted(gy, max(a[1], b[1], c[1]), side="right")
        if i0 >= i1 or j0 >= j1:
            continue

        X, Y = np.meshgrid(gx[i0:i1], gy[j0:j1], indexing="ij")
        l1 = ((b[1] - c[1]) * (X - c[0]) + (c[0] - b[0]) * (Y - c[1])) / det
        l2 = ((c[1] - a[1]) * (X - c[0]) + (a[0] - c[0]) * (Y - c[1])) / det
        l3 = 1.0 - l1 - l2
        hit = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if not hit.any():
            continue

        shadow[i0:i1, j0:j1] |= hit

        # Barycentric interpolation gives the crossing height exactly.
        z = l1 * a[2] + l2 * b[2] + l3 * c[2]
        ii, jj = np.nonzero(hit)
        col = (ii + i0) * ny + (jj + j0)
        top = np.searchsorted(gz, z[hit], side="left")
        np.add.at(diff, (col, 0), 1)
        np.add.at(diff, (col, top), -1)

    inside = (diff.cumsum(axis=1)[:, :nz] % 2) == 1
    return inside.reshape(nx, ny, nz), shadow


def is_watertight_mesh(faces):
    """Closed and consistently oriented, via the shared shell-defect check."""
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(f) == 0:
        return False
    # `is_watertight` speaks CityJSON: a list of faces, each a list of rings.
    return bool(is_watertight([[t.tolist()] for t in f]))


def _iou(mask_a, mask_b):
    union = np.count_nonzero(mask_a | mask_b)
    return float("nan") if union == 0 else np.count_nonzero(mask_a & mask_b) / union


def volumetric_iou(gen, gt, voxel_m=0.25, max_dim=128):
    """3D IoU on a shared voxel grid; ``nan`` unless both meshes are watertight.

    A ray-parity fill of an open shell leaks, so the alternative to skipping is
    a number that is confidently wrong. Callers must report the skip rate --
    otherwise the surviving average silently describes an unknown subset.
    """
    if not (is_watertight_mesh(gen[1]) and is_watertight_mesh(gt[1])):
        return float("nan")
    gx, gy, gz, _ = _grid(np.asarray(gen[0], float), np.asarray(gt[0], float),
                          voxel_m, max_dim)
    return _iou(_raster(*gen, gx, gy, gz)[0], _raster(*gt, gx, gy, gz)[0])


def footprint_iou(gen, gt, voxel_m=0.25, max_dim=128):
    """2D IoU of the xy shadows -- defined even when the mesh is not closed.

    For this model it should sit near 1.0 by construction, since the condition
    *is* the footprint, so it doubles as a drift alarm rather than a headline
    score. Caveat: a walls-only mesh projects to zero area, so at least one
    non-vertical face (ground or roof) has to survive for this to mean anything.
    """
    gx, gy, gz, _ = _grid(np.asarray(gen[0], float), np.asarray(gt[0], float),
                          voxel_m, max_dim)
    return _iou(_raster(*gen, gx, gy, gz)[1], _raster(*gt, gx, gy, gz)[1])


# ----------------------------------------------------------------------
# Domain metrics
# ----------------------------------------------------------------------

def _mean_roof_z(verts, faces):
    """Area-weighted mean height of the upward-facing triangles."""
    v = np.asarray(verts, dtype=float)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(f) == 0:
        return float("nan")

    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        nz = np.where(area > 0, cross[:, 2] / (2 * area), 0.0)

    up = (nz > _UP) & (area > 0)
    if not up.any():
        return float("nan")
    return float(np.average((a[up, 2] + b[up, 2] + c[up, 2]) / 3.0, weights=area[up]))


def roof_height_error(gen, gt):
    """``(max_z_error, mean_roof_z_error)`` in metres.

    The LOD1 condition already fixes the footprint; the height of the roof is
    most of what the model has to add, so these two are the task stated as
    numbers you can read without a renderer.
    """
    v_gen, v_gt = np.asarray(gen[0], float), np.asarray(gt[0], float)
    if len(v_gen) == 0 or len(v_gt) == 0:
        return float("nan"), float("nan")

    max_err = abs(float(v_gen[:, 2].max()) - float(v_gt[:, 2].max()))
    mean_err = abs(_mean_roof_z(*gen) - _mean_roof_z(*gt))
    return max_err, mean_err


def mesh_metrics(gen, gt, taus=(0.25, 0.5), n_points=4096, seed=0, voxel_m=0.25):
    """Every paired metric for one (generated, ground-truth) pair.

    Shares the sampled distances across chamfer / F-score / Hausdorff and the
    voxel grid across the two IoUs, so the whole set costs about what the most
    expensive member costs alone.

    Args:
        gen, gt: ``(verts [V, 3] metres, faces [F, 3])``.
        taus: F-score thresholds in metres.

    Returns:
        dict: str -> float, ``nan`` where a metric is undefined for this pair.
        ``watertight_gen`` / ``watertight_gt`` are 0/1 so a caller can average
        them into the rate that makes ``vol_iou`` readable.
    """
    d_ab, d_ba = surface_distances(gen, gt, n=n_points, seed=seed)

    out = {
        "chamfer_m": chamfer_distance(d_ab, d_ba),
        "hausdorff_p95_m": hausdorff_p95(d_ab, d_ba),
        "watertight_gen": float(is_watertight_mesh(gen[1])),
        "watertight_gt": float(is_watertight_mesh(gt[1])),
    }
    for tau in taus:
        out[f"fscore_{int(round(tau * 100))}cm"] = f_score(d_ab, d_ba, tau)[2]

    if len(np.asarray(gen[0])) and len(np.asarray(gt[0])):
        out["vol_iou"] = volumetric_iou(gen, gt, voxel_m=voxel_m)
        out["footprint_iou"] = footprint_iou(gen, gt, voxel_m=voxel_m)
    else:
        out["vol_iou"] = out["footprint_iou"] = float("nan")

    out["roof_max_z_err_m"], out["roof_mean_z_err_m"] = roof_height_error(gen, gt)
    return out
