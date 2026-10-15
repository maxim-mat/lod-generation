"""Train/val/test index splits for `MeshDataModule`.

"random" reproduces the split every run up to v5 used (`torch.utils.data.random_split`
with a seeded generator), index for index. "tile" assigns whole 3DBAG tiles (one
CityJSON file each) to a split: neighbouring buildings are correlated (see
`src.analysis.spatial_correlation`), so a building-level split puts close
relatives of every test building into train and scores optimistically. "grid"
does the same with square blocks of a chosen size, keyed on each building's
footprint centroid; `buffer_train` then drops training buildings near val/test
ones, which is what actually guarantees separation (spatial block
cross-validation, Roberts et al. 2017, Ecography, doi:10.1111/ecog.02881).
"""
import collections

import numpy as np
import torch
from scipy.spatial import cKDTree


def _sizes(n, ratios):
    sizes = [int(ratios[0] * n), int(ratios[1] * n)]
    return sizes + [n - sum(sizes)]


def random_split_indices(n, ratios, seed):
    """``[train, val, test]`` index lists, identical to `torch.utils.data.random_split`."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    out, start = [], 0
    for size in _sizes(n, ratios):
        out.append(perm[start:start + size])
        start += size
    return out


def tile_split_indices(groups, ratios, seed):
    """Whole groups per split, balanced by item count.

    Groups are shuffled with `seed`, then each goes to the split furthest below
    its target share -- so no group straddles two splits and every split lands
    within one group's size of its target.

    Args:
        groups: one group label (tile) per item.
        ratios: (train, val, test) shares, summing to 1.
        seed (int): seeds the group shuffle.

    Returns:
        tuple: ([train, val, test] index lists, {group: split index}).
    """
    sizes = collections.Counter(groups)
    names = sorted(sizes)
    np.random.default_rng(seed).shuffle(names)
    targets = [r * len(groups) for r in ratios]
    counts, assign = [0, 0, 0], {}
    for g in names:
        k = max(range(3), key=lambda s: (targets[s] - counts[s], -s))
        assign[g] = k
        counts[k] += sizes[g]
    out = [[], [], []]
    for i, g in enumerate(groups):
        out[assign[g]].append(i)
    return out, assign


def footprint(verts, faces, tol=1e-3):
    """Projected area and area-weighted centroid of the triangles at the mesh's lowest z.

    For an extruded LOD1 that base is the footprint. A mesh without one (no
    ground surface) has area 0 and falls back to its x/y bounding-box centre,
    so it still lands in the right grid block.
    """
    tri = verts[faces]
    flat = (np.abs(tri[:, :, 2] - verts[:, 2].min()) < tol).all(axis=1)
    t = tri[flat][:, :, :2]
    area = 0.5 * np.abs((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
                        - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1]))
    if area.sum() <= 0:
        return 0.0, (verts[:, :2].min(axis=0) + verts[:, :2].max(axis=0)) / 2
    return float(area.sum()), (t.mean(axis=1) * area[:, None]).sum(axis=0) / area.sum()


def grid_labels(xy, block_m):
    """Square-block label ``"i_j"`` per point: ``floor(x / block_m)``, ``floor(y / block_m)``."""
    ij = np.floor(np.asarray(xy, float) / block_m).astype(np.int64)
    return [f"{i}_{j}" for i, j in ij]


def buffer_train(xy, train_idx, eval_idx, buffer_m):
    """Drop training items closer than `buffer_m` to any eval item.

    Returns:
        tuple: (kept train indices, in their original order; number dropped).
    """
    train_idx = list(train_idx)
    if buffer_m <= 0 or not len(eval_idx) or not train_idx:
        return train_idx, 0
    xy = np.asarray(xy, float)
    dist, _ = cKDTree(xy[list(eval_idx)]).query(xy[train_idx])
    kept = [i for i, d in zip(train_idx, dist) if d >= buffer_m]
    return kept, len(train_idx) - len(kept)


def split_name(split, block_m=None, buffer_m=0.0):
    """Unique name of a split scheme, e.g. ``grid250m-buffer50m``.

    Two runs share val/test buildings only if their names (and seeds) match;
    `MeshTransformerModule` records it as ``data_split``.
    """
    name = f"grid{block_m:g}m" if split == "grid" else split
    return f"{name}-buffer{buffer_m:g}m" if buffer_m else name
