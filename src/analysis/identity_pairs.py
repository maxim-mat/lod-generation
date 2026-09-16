#!/usr/bin/env python3
"""How often is the LOD2 answer just the LOD1 input?

For every paired building this measures vol(LOD1 \\ LOD2) + vol(LOD2 \\ LOD1)
exactly, via manifold3d CSG (`src.analysis.exact_volume`), so the "the model
just copied its input" observation becomes a distribution rather than an
impression from a sample of ten. Pairs near zero are ones where copying is the
*correct* answer, and they are also the pairs that teach the model to copy.

One scan of the corpus, O(n) -- each building is compared only against its own
LOD1, never against other buildings. Parallel over tile pairs, which match 1:1
between the LOD folders.

Rows carry an `ok` flag. A building whose LOD1 or LOD2 is not a closed,
correctly-wound solid still produces finite numbers so a corpus pass never
dies, but those rows are not measurements and must be filtered out of any
distribution rather than averaged into it. `note` says what went wrong.

Usage:
    python -m src.analysis.identity_pairs --dataset-dir "data/The Hague/cleaner"
"""
import argparse
import csv
import json
import logging
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.analysis.exact_volume import exact_pair_volumes
from src.dataset.mesh_dataset import parse_cityjson_file_to_meshes

logger = logging.getLogger(__name__)


def tile_pairs(dataset_dir, lod_in="LOD1", lod_out="LOD2"):
    """``(relative_path, in_path, out_path)`` for tiles present at both LODs.

    Matching is on the path *below* the LOD folder, which was verified 1:1 on
    this corpus. A tile present at only one LOD is skipped -- the LOD folders
    do not always share an object set.

    Raises:
        FileNotFoundError: either LOD folder is missing.
    """
    dataset_dir = Path(dataset_dir)
    roots = {}
    for lod in (lod_in, lod_out):
        root = dataset_dir / lod
        if not root.is_dir():
            raise FileNotFoundError(f"No LOD folder {root!s} under {dataset_dir!r}")
        roots[lod] = {
            str(p.relative_to(root)): p
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.name.lower().endswith((".json", ".city.json"))
        }
    shared = sorted(set(roots[lod_in]) & set(roots[lod_out]))
    return [(rel, roots[lod_in][rel], roots[lod_out][rel]) for rel in shared]


def scan_tile(pair):
    """One tile pair to a list of rows. The multiprocessing unit of work."""
    rel, in_path, out_path = pair
    try:
        cond = parse_cityjson_file_to_meshes(in_path)
        target = parse_cityjson_file_to_meshes(out_path)
    except (OSError, ValueError) as exc:
        # One corrupt tile must not kill a corpus pass.
        logger.warning("skipping %s: %s", rel, exc)
        return []

    rows = []
    for oid in sorted(set(cond) & set(target)):
        try:
            d = exact_pair_volumes(cond[oid], target[oid])
        except Exception as exc:                 # noqa: BLE001 - CSG is C++; never trust it to be total
            logger.warning("skipping %s in %s: %s", oid, rel, exc)
            continue
        rows.append({"object_id": oid, "tile": str(rel), **d})
    return rows


def scan(dataset_dir, lod_in="LOD1", lod_out="LOD2", workers=None, limit=None):
    """Every paired building's exact LOD1-vs-LOD2 distance.

    Returns:
        tuple: (rows, stats). `stats` records corpus totals and run parameters
        so the output is self-describing.
    """
    pairs = tile_pairs(dataset_dir, lod_in, lod_out)
    if limit is not None:
        pairs = pairs[:limit]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)

    rows = []
    if workers > 1:
        with Pool(workers) as pool:
            for i, out in enumerate(pool.imap_unordered(scan_tile, pairs, chunksize=2), 1):
                rows.extend(out)
                if i % 25 == 0:
                    logger.info("%d/%d tiles, %d buildings", i, len(pairs), len(rows))
    else:
        for pair in pairs:
            rows.extend(scan_tile(pair))

    rows.sort(key=lambda r: r["object_id"])
    n_ok = sum(1 for r in rows if r["ok"])
    stats = {"dataset_dir": str(dataset_dir), "lod_in": lod_in, "lod_out": lod_out,
             "backend": "manifold3d", "n_tiles": len(pairs),
             "n_buildings": len(rows), "n_ok": n_ok,
             "n_flagged": len(rows) - n_ok}
    return rows, stats


COLUMNS = ["object_id", "tile", "symdiff_m3", "rel_symdiff", "iou",
           "vol_lod1_m3", "vol_lod2_m3", "union_m3", "intersection_m3",
           "ok", "note"]


def write_outputs(rows, stats, out_dir):
    """`pairs.csv` (one row per building) and `stats.json`.

    CSV rather than JSON because the explorer only ever needs the scalar
    columns plus the object name -- it loads geometry lazily, on selection.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pairs.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1)
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset-dir", default="data/The Hague/cleaner")
    ap.add_argument("--lod-in", default="LOD1")
    ap.add_argument("--lod-out", default="LOD2")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N tiles only")
    ap.add_argument("--out-dir", default="outputs/identity_pairs/exact")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rows, stats = scan(args.dataset_dir, args.lod_in, args.lod_out,
                       workers=args.workers, limit=args.limit)
    out = Path(args.out_dir)
    write_outputs(rows, stats, out)

    logger.info("%d buildings (%d ok, %d flagged) -> %s",
                stats["n_buildings"], stats["n_ok"], stats["n_flagged"], out)
    rel = np.array([r["rel_symdiff"] for r in rows if r["ok"]])
    if len(rel):
        logger.info("rel_symdiff over %d ok rows: median %.4f  p05 %.4f  p25 %.4f",
                    len(rel), np.median(rel), *np.percentile(rel, [5, 25]))
        for t in (0.001, 0.01, 0.02, 0.05, 0.10):
            logger.info("  <= %.3f : %6d buildings (%.2f%%)",
                        t, int((rel <= t).sum()), 100 * float((rel <= t).mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
