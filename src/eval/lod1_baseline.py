#!/usr/bin/env python3
"""The do-nothing floor: score the LOD1 condition as if it were the prediction.

Every paired metric the mesh transformer reports is a distance between a
generated LOD2 and the real one. None of them is interpretable without knowing
what the *identity* scores -- a model that ignores its input entirely and echoes
the LOD1 box back gets some chamfer, some IoU, and a run has to beat that before
it has learned anything at all. `val_gen_footprint_iou` in particular is near 1.0
by construction, because the condition already is the footprint.

The companion number is the tokenizer ceiling: LOD2 put through the same
`num_bins` discretization and scored against itself. That is the best any model
on this vocabulary could reach. A run's real position is where it sits between
the two, not its absolute chamfer.

Sampling note: use a *uniform* subset (`sample_mini_dataset --alpha 0`). The
default inverse-size weighting draws small buildings, and every distance metric
here scales with building size, so a size-biased subset reports a flattering
floor.

Usage:
    python -m src.eval.lod1_baseline "data/The Hague/mini_cleaner" \
        --out outputs/lod1_baseline.json
"""
import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

from src.dataset.mesh_dataset import (NUM_BINS, dequantize, normalize_to_unit_box,
                                      parse_cityjson_file_to_meshes, quantize, tokenize,
                                      detokenize)
from src.eval.mesh_metrics import mesh_metrics
from src.filter_cityjson import _git_commit

logger = logging.getLogger(__name__)

KEYS = ("chamfer_m", "hausdorff_p95_m", "vol_iou", "footprint_iou",
        "fscore_25cm", "fscore_50cm", "roof_mean_z_err_m", "roof_max_z_err_m",
        "watertight_gen", "watertight_gt")


def round_trip(mesh, num_bins):
    """The mesh as the tokenizer can best represent it, back in metres."""
    v, f = mesh
    vn, centre, scale = normalize_to_unit_box(v)
    v_rt, f_rt = detokenize(tokenize(vn, f, num_bins), num_bins)
    return v_rt * scale + centre, f_rt


def _job(args):
    tile, root, taus, n_points, voxel_m, num_bins, do_ceiling = args
    root = Path(root)
    try:
        lod1 = parse_cityjson_file_to_meshes(root / "LOD1" / tile)
        lod2 = parse_cityjson_file_to_meshes(root / "LOD2" / tile)
    except (OSError, ValueError) as exc:
        logger.error("skipping %s: %s", tile, exc)
        return [], []

    floor, ceiling = [], []
    for oid in sorted(set(lod1) & set(lod2)):
        gt = lod2[oid]
        if len(gt[1]) == 0 or len(lod1[oid][1]) == 0:
            continue
        floor.append(mesh_metrics(lod1[oid], gt, taus=taus, n_points=n_points,
                                  voxel_m=voxel_m))
        if do_ceiling:
            ceiling.append(mesh_metrics(round_trip(gt, num_bins), gt, taus=taus,
                                        n_points=n_points, voxel_m=voxel_m))
    return floor, ceiling


def summarise(rows):
    """Percentiles per metric, ignoring the nans each one defines away."""
    out = {}
    cols = defaultdict(list)
    for r in rows:
        for k, v in r.items():
            cols[k].append(v)
    for k, vals in cols.items():
        a = np.asarray(vals, dtype=float)
        ok = a[np.isfinite(a)]
        out[k] = {
            "n": int(len(a)), "defined": int(len(ok)),
            "defined_rate": float(len(ok) / max(len(a), 1)),
            "mean": float(ok.mean()) if len(ok) else float("nan"),
            **{f"p{p}": (float(np.percentile(ok, p)) if len(ok) else float("nan"))
               for p in (5, 25, 50, 75, 95)},
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="dataset holding LOD1/ and LOD2/")
    ap.add_argument("--out", type=Path, default=Path("outputs/lod1_baseline.json"))
    ap.add_argument("--n-points", type=int, default=4096)
    ap.add_argument("--voxel-m", type=float, default=0.25)
    ap.add_argument("--num-bins", type=int, default=NUM_BINS)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.25, 0.5])
    ap.add_argument("--no-ceiling", action="store_true",
                    help="skip the tokenizer round-trip companion")
    ap.add_argument("--workers", type=int, default=min(10, cpu_count()))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    lod2 = args.root / "LOD2"
    tiles = sorted(p.relative_to(lod2) for p in lod2.rglob("*.json") if p.is_file())
    logger.info("%d tiles under %s", len(tiles), args.root)

    jobs = [(str(t), str(args.root), tuple(args.taus), args.n_points,
             args.voxel_m, args.num_bins, not args.no_ceiling) for t in tiles]
    floor, ceiling = [], []
    with Pool(processes=args.workers) as pool:
        for i, (f, c) in enumerate(pool.imap_unordered(_job, jobs), 1):
            floor.extend(f)
            ceiling.extend(c)
            if i % 50 == 0 or i == len(tiles):
                logger.info("  %d/%d tiles, %d pairs", i, len(tiles), len(floor))

    report = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset": str(args.root.resolve()),
        "pairs": len(floor),
        "params": {"n_points": args.n_points, "voxel_m": args.voxel_m,
                   "num_bins": args.num_bins, "taus": args.taus},
        "lod1_identity_floor": summarise(floor),
        "tokenizer_ceiling": summarise(ceiling) if ceiling else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    hdr = f"{'metric':22s} {'floor p50':>10s} {'floor mean':>11s} {'defined':>8s}"
    if ceiling:
        hdr += f" | {'ceiling p50':>12s}"
    print(f"\n{len(floor)} LOD1/LOD2 pairs from {args.root}\n")
    print(hdr)
    print("-" * len(hdr))
    for k in KEYS:
        if k not in report["lod1_identity_floor"]:
            continue
        s = report["lod1_identity_floor"][k]
        line = (f"{k:22s} {s['p50']:10.4f} {s['mean']:11.4f} "
                f"{100 * s['defined_rate']:7.1f}%")
        if ceiling and k in report["tokenizer_ceiling"]:
            line += f" | {report['tokenizer_ceiling'][k]['p50']:12.4f}"
        print(line)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
