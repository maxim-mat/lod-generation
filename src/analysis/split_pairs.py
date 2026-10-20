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
import argparse
import json
import logging
import multiprocessing as mp
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.analysis.exact_volume import exact_pair_volumes
from src.dataset.splits import (buffer_train, footprint, grid_labels, random_split_indices,
                                tile_split_indices)
from src.eval.mesh_metrics import chamfer_distance, surface_distances
from src.filter_cityjson import _git_commit

REPO = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)

# name -> (scheme, block_m, buffer_m); names are `splits.split_name` of the tuple.
SPLITS = {
    "random": ("random", None, 0.0),
    "tile": ("tile", None, 0.0),
    "tile-buffer50m": ("tile", None, 50.0),
    "grid500m-buffer50m": ("grid", 500.0, 50.0),
    "grid1000m-buffer50m": ("grid", 1000.0, 50.0),
}
COLUMNS = ["split", "test_id", "test_tile", "cand_id", "cand_tile", "rank", "distance_m",
           "symdiff_m3", "rel_symdiff", "vol_ok", "chamfer_m", "test_faces", "cand_faces"]
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


def _score(job):
    """Pool worker: ``(a, b, n_points)`` -> `pair_metrics` dict."""
    a, b, n_points = job
    return pair_metrics(a, b, n_points)


def candidate_table(dataset, splits, n_sample, k, n_points, seed, workers=1,
                    ratios=(0.8, 0.1, 0.1)):
    """Every shortlisted (test, training candidate) pair of every split, scored.

    Each unique pair is scored once however many splits share it -- the tile
    split and its buffered variant have identical test sets, for one.

    Args:
        dataset: has ``.ids``, ``.files``, ``.xy`` and ``.pairs[i][1]``, the raw
            LOD2 ``(verts, faces)`` in metres (as `MeshDataset` does).
        splits: names from `SPLITS`.
        n_sample: test buildings sampled per split (all, if the split has fewer).
        k: shortlist length.
        n_points: chamfer surface samples per side.
        seed: the split seed, also used for the test sample.
        workers: processes for the pair scoring; 1 runs inline.

    Returns:
        DataFrame: one row per split x test x candidate (columns: see the tests).
    """
    rows, jobs = [], {}
    for name in splits:
        scheme, block_m, buffer_m = SPLITS[name]
        train, _, test = split_parts(dataset, scheme, ratios, seed, block_m, buffer_m)
        tests = sample_test(test, n_sample, seed)
        idx, dist = shortlist(dataset.xy, train, tests, k)
        logger.info("%s: %d train, %d test, %d sampled", name, len(train), len(test), len(tests))
        for q, t in enumerate(tests):
            for rank, (c, d) in enumerate(zip(idx[q], dist[q])):
                rows.append((name, t, int(c), rank, float(d)))
                jobs.setdefault((t, int(c)), None)

    keys = list(jobs)
    work = ((dataset.pairs[t][1], dataset.pairs[c][1], n_points) for t, c in keys)
    logger.info("scoring %d unique pairs on %d worker(s)", len(keys), workers)
    started = time.perf_counter()
    if workers > 1:
        with mp.Pool(workers) as pool:
            results = []
            for i, r in enumerate(pool.imap(_score, work, chunksize=64), 1):
                results.append(r)
                if i % 5000 == 0:
                    logger.info("%d/%d pairs, %.0f s", i, len(keys), time.perf_counter() - started)
    else:
        results = [_score(job) for job in work]
    scores = dict(zip(keys, results))

    out = []
    for name, t, c, rank, d in rows:
        m = scores[(t, c)]
        out.append({"split": name, "test_id": dataset.ids[t], "test_tile": dataset.files[t],
                    "cand_id": dataset.ids[c], "cand_tile": dataset.files[c], "rank": rank,
                    "distance_m": d, "symdiff_m3": m["symdiff_m3"],
                    "rel_symdiff": m["rel_symdiff"], "vol_ok": m["vol_ok"],
                    "chamfer_m": m["chamfer_m"],
                    "test_faces": len(dataset.pairs[t][1][1]),
                    "cand_faces": len(dataset.pairs[c][1][1])})
    return pd.DataFrame(out, columns=COLUMNS)


def load_dataset(cfg):
    """`MeshDataset` built from a training config's `mesh_data`, as training builds it (no imagery)."""
    from src.dataset.mesh_dataset import MeshDataset

    m = cfg.mesh_data
    return MeshDataset(dataset_dir=m.dataset_dir, lod_in=m.lod_in, lod_out=m.lod_out,
                       num_bins=m.num_bins, margin_lo=list(m.margin_lo),
                       margin_hi=list(m.margin_hi), max_faces=m.max_faces,
                       max_files=m.max_files, tokenization=m.tokenization)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path,
                        default=REPO / "configs" / "mesh-v5-full-opt-coord.yaml",
                        help="training config whose mesh_data, seed and ratios define the splits")
    parser.add_argument("--out", type=Path, default=REPO / "outputs" / "split_pairs")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--n-sample", type=int, default=1000)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--n-points", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from src.utils.initialization import load_config

    cfg = load_config(args.config, [])
    ratios = tuple(cfg.training.train_val_test_split)
    started = time.perf_counter()
    dataset = load_dataset(cfg)
    cands = candidate_table(dataset, args.splits, args.n_sample, args.k, args.n_points,
                            cfg.seed, args.workers, ratios)
    pairs = best_matches(cands)

    args.out.mkdir(parents=True, exist_ok=True)
    cands.to_csv(args.out / "candidates.csv", index=False)
    pairs.to_csv(args.out / "pairs.csv", index=False)
    m = cfg.mesh_data
    run = {"created": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit(),
           "config": str(args.config), "seed": cfg.seed, "ratios": list(ratios),
           "n_sample": args.n_sample, "k": args.k, "n_points": args.n_points,
           "chamfer_seed": 0, "workers": args.workers,
           "splits": {name: list(SPLITS[name]) for name in args.splits},
           "counts": {name: {"test_sampled": int(g["test_id"].nunique()), "rows": len(g)}
                      for name, g in cands.groupby("split")},
           "data": {"dataset_dir": str(m.dataset_dir), "lod_in": m.lod_in,
                    "lod_out": m.lod_out, "max_faces": m.max_faces,
                    "n_buildings": len(dataset)},
           "seconds": round(time.perf_counter() - started, 1)}
    (args.out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    logger.info("wrote %s (%d candidate rows, %d matches)", args.out, len(cands), len(pairs))


if __name__ == "__main__":
    main()
