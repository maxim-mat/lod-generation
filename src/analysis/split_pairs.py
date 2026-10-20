"""Each sampled test building and its nearest training building, under several splits.

Feeds `src/split_explorer.py`. For every split it rebuilds train/val/test exactly
as `MeshDataModule` does, samples test buildings, and shortlists the K nearest
*training* buildings around each by footprint centroid. Every shortlisted pair is
scored three ways, and the best candidate per criterion becomes a "match":

  * distance:  footprint-centroid distance in metres;
  * volume:    exact LOD2 volume symmetric difference (manifold3d CSG);
  * chamfer:   LOD2 mesh-to-mesh chamfer distance.

Shape metrics compare the two LOD2 meshes after moving each to a common origin
(footprint centroid at x = y = 0, lowest vertex at z = 0) without rotation, so
they measure shape, not position. The shortlist is local on purpose: the
question is whether a split leaves a near-copy of a test building *nearby* in
training, not whether one exists anywhere in the city.

Usage:
    python -m src.analysis.split_pairs --config configs/mesh-v5-full-opt-coord.yaml \\
        --out outputs/split_pairs
    streamlit run src/split_explorer.py
"""
import logging

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.analysis.exact_volume import exact_pair_volumes
from src.dataset.splits import (buffer_train, footprint, grid_labels, random_split_indices,
                                tile_split_indices)
from src.eval.mesh_metrics import chamfer_distance, surface_distances

logger = logging.getLogger(__name__)

# name -> (scheme, block_m, buffer_m); names are `splits.split_name` of the tuple.
SPLITS = {
    "random": ("random", None, 0.0),
    "tile": ("tile", None, 0.0),
    "tile-buffer50m": ("tile", None, 50.0),
    "grid500m-buffer50m": ("grid", 500.0, 50.0),
    "grid1000m-buffer50m": ("grid", 1000.0, 50.0),
}
# criterion -> the column it minimises
CRITERIA = {"distance": "distance_m", "volume": "symdiff_m3", "chamfer": "chamfer_m"}


def split_parts(dataset, split, ratios, seed, block_m=None, buffer_m=0.0):
    """``[train, val, test]`` indices, exactly as `MeshDataModule.setup` builds them.

    Composes the same `splits.py` functions in the same order; pinned equal to
    the datamodule by `test_split_parts_matches_the_datamodule`. Kept here rather
    than refactored out of `setup` because that method is on the training path.

    Args:
        dataset: has ``len()``, ``.files`` and ``.xy``; ``.xy`` is only read for
            the grid split or a buffer.
    """
    if split == "random":
        return random_split_indices(len(dataset), ratios, seed)
    groups = dataset.files if split == "tile" else grid_labels(dataset.xy, block_m)
    parts, _ = tile_split_indices(groups, ratios, seed)
    parts[0], _ = buffer_train(dataset.xy, parts[0], parts[1] + parts[2], buffer_m)
    return parts


def sample_test(test_idx, n, seed):
    """Seeded sample of `n` test indices, sorted; all of them when there are fewer."""
    test_idx = list(test_idx)
    if len(test_idx) <= n:
        return sorted(test_idx)
    pick = np.random.default_rng(seed).choice(len(test_idx), size=n, replace=False)
    return sorted(test_idx[i] for i in pick)


def shortlist(xy, train_idx, query_idx, k):
    """The `k` training buildings nearest each query, nearest first.

    Returns:
        tuple: (dataset indices ``[Q, k']``, distances in metres ``[Q, k']``),
        with ``k' = min(k, len(train_idx))``.
    """
    xy = np.asarray(xy, float)
    train_idx = np.asarray(train_idx)
    k = min(k, len(train_idx))
    dist, pos = cKDTree(xy[train_idx]).query(xy[list(query_idx)], k=k)
    dist, pos = np.asarray(dist).reshape(-1, k), np.asarray(pos).reshape(-1, k)
    return train_idx[pos], dist


def centred(mesh):
    """``(verts, faces)`` moved so the footprint centroid is at x = y = 0 and the lowest vertex at z = 0."""
    verts, faces = mesh
    verts = np.asarray(verts, float)
    _, centre = footprint(verts, faces)
    return verts - [centre[0], centre[1], verts[:, 2].min()], np.asarray(faces)


def pair_metrics(a, b, n_points):
    """Shape distance between two LOD2 meshes after `centred`.

    Returns:
        dict: ``symdiff_m3``, ``rel_symdiff`` (NaN unless ``vol_ok``), ``vol_ok``
        and ``chamfer_m`` (NaN if the surface query fails).
    """
    a, b = centred(a), centred(b)
    try:
        vol = exact_pair_volumes(a, b)
        ok = bool(vol["ok"])
    except Exception as exc:                 # noqa: BLE001 - CSG is C++; never trust it to be total
        logger.warning("volume failed: %s", exc)
        vol, ok = {}, False
    try:
        chamfer = chamfer_distance(*surface_distances(a, b, n=n_points, seed=0))
    except Exception as exc:                 # noqa: BLE001 - same for the BVH
        logger.warning("chamfer failed: %s", exc)
        chamfer = float("nan")
    return {"symdiff_m3": float(vol["symdiff_m3"]) if ok else float("nan"),
            "rel_symdiff": float(vol["rel_symdiff"]) if ok else float("nan"),
            "vol_ok": ok, "chamfer_m": chamfer}


def best_matches(candidates):
    """One row per (split, test_id, criterion): the candidate minimising that criterion.

    The volume criterion considers only ``vol_ok`` rows, so a building whose
    candidates are all unusable solids has no volume match rather than a NaN one.
    """
    out = []
    for criterion, column in CRITERIA.items():
        pool = candidates[candidates["vol_ok"]] if criterion == "volume" else candidates
        pool = pool[np.isfinite(pool[column])]
        if pool.empty:
            continue
        idx = pool.groupby(["split", "test_id"], sort=False)[column].idxmin()
        out.append(pool.loc[idx].assign(criterion=criterion))
    if not out:
        return candidates.iloc[0:0].assign(criterion=pd.Series(dtype=str))
    return pd.concat(out, ignore_index=True)
