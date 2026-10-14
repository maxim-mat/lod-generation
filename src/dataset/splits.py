"""Train/val/test index splits for `MeshDataModule`.

"random" reproduces the split every run up to v5 used (`torch.utils.data.random_split`
with a seeded generator), index for index. "tile" assigns whole 3DBAG tiles (one
CityJSON file each) to a split: neighbouring buildings are correlated (see
`src.analysis.spatial_correlation`), so a building-level split puts close
relatives of every test building into train and scores optimistically.
"""
import collections

import numpy as np
import torch


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
