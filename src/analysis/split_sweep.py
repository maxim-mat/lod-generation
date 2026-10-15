"""Which grid block size and buffer separate train from val/test? A sweep.

For every grid block size x buffer (plus the random and tile splits as
references, over several seeds) it splits the buildings exactly as
`MeshDataModule` would and records, for the val/test buildings:
  * the distance to the nearest training building (median, p10, p90, min,
    and the share within 25/50/100 m);
  * how similar that nearest training building is (correlation and median
    absolute difference) on five targets;
  * the training data the buffer throws away and the number of val/test
    blocks, which bound how fine the blocks can usefully be.

Targets: log footprint, log LOD2 volume and log LOD2 face count, plus two that
factor out what the model already gets from LOD1 -- `roof_ratio`
(log LOD2 / LOD1 volume) and `faces_resid` (log face count minus its
least-squares fit on log footprint and log LOD1 volume). Similarity in the
first three is partly given away by the LOD1 input; similarity in the last two
is what a neighbour could leak.

It also writes a fine-grained correlogram of every target, which sets the
buffer: the distance where the correlation has fallen to its long-range level.
Spatial block cross-validation with a buffer follows Roberts et al. 2017
(Ecography, doi:10.1111/ecog.02881).

Usage:
    python -m src.analysis.split_sweep "data/The Hague/cleaner" --out outputs/split_sweep
    python -m src.analysis.split_sweep --table outputs/split_sweep/buildings.csv \\
        --out outputs/split_sweep_b
    python -m src.analysis.plot_split_sweep outputs/split_sweep
"""
import argparse
import json
import logging
import multiprocessing as mp
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.analysis.spatial_correlation import building_table, correlogram
from src.dataset.splits import (buffer_train, grid_labels, random_split_indices,
                                tile_split_indices)
from src.filter_cityjson import _git_commit

logger = logging.getLogger(__name__)
TARGETS = ("log_footprint", "log_volume", "log_faces", "roof_ratio", "faces_resid")
SIZES = [50, 100, 250, 500, 1000, 2000]
BUFFERS = [0, 25, 50, 100]
SEEDS = [42, 0, 1]
EDGES_M = [0, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000]


def add_targets(table):
    """`table` plus the five log-scale targets; see the module docstring."""
    t = table.copy()
    t["log_footprint"] = np.log(t["footprint_m2"])
    t["log_volume"] = np.log(t["volume_m3"])
    t["log_faces"] = np.log(t["n_faces"])
    log_v1 = np.log(t["lod1_volume_m3"])
    t["roof_ratio"] = t["log_volume"] - log_v1
    x = np.column_stack([np.ones(len(t)), t["log_footprint"], log_v1])
    coef, *_ = np.linalg.lstsq(x, t["log_faces"].to_numpy(), rcond=None)
    t["faces_resid"] = t["log_faces"] - x @ coef
    return t


def _valid(table):
    keep = ((table["footprint_m2"] > 0) & (table["volume_m3"] > 0) & (table["n_faces"] > 0)
            & (table["lod1_volume_m3"] > 0))
    return table[keep].reset_index(drop=True)


def _score(xy, targets, train, ev):
    """Nearest-train distance and similarity for the eval buildings."""
    train, ev = np.asarray(train), np.asarray(ev)
    dist, j = cKDTree(xy[train]).query(xy[ev])
    row = {"dist_median_m": float(np.median(dist)),
           "dist_p10_m": float(np.percentile(dist, 10)),
           "dist_p90_m": float(np.percentile(dist, 90)),
           "dist_min_m": float(dist.min())}
    for d in (25, 50, 100):
        row[f"within_{d}m"] = float((dist < d).mean())
    for name in TARGETS:
        values = targets[name].to_numpy()
        mine, theirs = values[ev], values[train[j]]
        row[f"corr_{name}"] = float(np.corrcoef(mine, theirs)[0, 1])
        row[f"mad_{name}"] = float(np.median(np.abs(mine - theirs)))
    return row


def sweep(table, sizes=SIZES, buffers=BUFFERS, seeds=SEEDS, ratios=(0.8, 0.1, 0.1)):
    """One row per (scheme, block size, buffer, seed); see the module docstring.

    Splits exactly as `MeshDataModule` does: `random_split_indices` for random,
    `tile_split_indices` over tile or grid labels, then `buffer_train`.
    """
    if not len(sizes):
        raise ValueError("sizes must list at least one grid block size.")
    t = add_targets(_valid(table))
    xy = t[["x", "y"]].to_numpy()
    configs = ([("random", 0, 0)] + [("tile", 0, b) for b in buffers]
               + [("grid", s, b) for s in sizes for b in buffers])
    labels = {"tile": t["tile"].tolist()}
    labels.update({s: grid_labels(xy, s) for s in sizes})
    rows = []
    for seed in seeds:
        for scheme, block, buffer in configs:
            if scheme == "random":
                parts = random_split_indices(len(t), ratios, seed)
                n_groups = n_eval_groups = len(t)
            else:
                groups = labels["tile" if scheme == "tile" else block]
                parts, assign = tile_split_indices(groups, ratios, seed)
                n_groups = len(assign)
                n_eval_groups = sum(k > 0 for k in assign.values())
            ev = parts[1] + parts[2]
            train, dropped = buffer_train(xy, parts[0], ev, buffer)
            rows.append({"scheme": scheme, "block_m": block, "buffer_m": buffer, "seed": seed,
                         "n_groups": n_groups, "n_eval_groups": n_eval_groups,
                         "n_train": len(train), "n_eval": len(ev),
                         "train_lost_frac": dropped / max(len(parts[0]), 1),
                         **_score(xy, t, train, ev)})
        logger.info("seed %s: %d configurations", seed, len(configs))
    return pd.DataFrame(rows)


def correlograms(table, edges=EDGES_M, n_anchors=5000, seed=0):
    """Long-format correlogram of every target: target, lo, hi, n_pairs, corr."""
    t = add_targets(_valid(table))
    xy = t[["x", "y"]].to_numpy()
    return pd.concat([correlogram(xy, t[name].to_numpy(), edges, n_anchors, seed)
                      .assign(target=name) for name in TARGETS], ignore_index=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subset", type=Path, nargs="?",
                        help="dataset subset with LOD1/ and LOD2/ (or give --table)")
    parser.add_argument("--table", type=Path,
                        help="buildings.csv from an earlier run, instead of parsing the subset")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizes", type=float, nargs="+", default=SIZES)
    parser.add_argument("--buffers", type=float, nargs="+", default=BUFFERS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    parser.add_argument("--n-anchors", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=min(8, mp.cpu_count()))
    args = parser.parse_args(argv)
    if (args.subset is None) == (args.table is None):
        parser.error("give exactly one of a subset directory or --table")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.table is not None:
        try:
            table = pd.read_csv(args.table, dtype={"id": str, "tile": str})
        except OSError as e:
            raise SystemExit(f"cannot read --table {args.table}: {e}")
    else:
        table = building_table(args.subset, args.workers)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "buildings.csv", index=False)

    rows = sweep(table, args.sizes, args.buffers, args.seeds, tuple(args.ratios))
    rows.to_csv(args.out / "split_sweep.csv", index=False)
    correlograms(table, n_anchors=args.n_anchors, seed=args.seeds[0]).to_csv(
        args.out / "correlogram.csv", index=False)
    run = {"created": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit(),
           "n_buildings": len(table), "n_scored": len(_valid(table)),
           "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}
    (args.out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
