#!/usr/bin/env python3
"""Carve a small, fast-to-train subset out of the cleaned dataset.

Sampling is weighted by ``1 / n_vertices ** alpha``, so large buildings are
drawn less often. That is deliberate rather than incidental: the dense edge
tensor is ``[B, n_max, n_max, edge_dim]``, so cost grows with the square of the
largest graph in the split, and a handful of 4,000-node outliers set ``n_max``
for everything. Down-weighting them shrinks ``n_max`` far faster than it shrinks
the object count.

The subset is therefore *not* distribution-faithful, by design. That needs no
special handling during training: ``train`` recomputes every train-split
statistic -- class marginals, ``coord_scale``, ``dist_r_max`` -- from whichever
datamodule it builds, so a run on this set is internally consistent.

What does not carry over is comparison *between* runs. At 2% the scale is
3.56 m against the full corpus's 8.00 m, and Off is 0.832 of node slots against
0.737, so scaled-unit metrics like ``val_coord_mse`` and the class-balance
weights sit on a different footing. Use it for pipeline shakedowns, sweeps and
overfitting checks; do not read its numbers as an estimate of full-corpus
performance.

Objects are chosen once, on a reference folder (LOD2 by default), and the same
ids are then carried into every other LOD folder present. That keeps
``LOD2``/``LOD1_synth`` exactly paired; ``LOD1`` is an independent extraction,
so it contributes whichever of the chosen ids it happens to hold.

Usage:
    python -m src.sample_mini_dataset "data/The Hague/clean" \
        --out "data/The Hague/mini" --fraction 0.05
"""
import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.filter_cityjson import _git_commit, _write, compact_vertices

logger = logging.getLogger(__name__)

DEFAULT_FOLDERS = ("LOD2", "LOD1", "LOD1_synth")
REFERENCE = "LOD2"


def object_vertex_counts(cj):
    """``{object_id: distinct outer-ring vertices}``.

    Outer rings only, matching what the parser turns into vertex nodes -- inner
    rings never reach the graph, so counting them would misjudge graph size.
    """
    counts = {}
    for oid, obj in (cj.get("CityObjects") or {}).items():
        ids = set()
        for geom in obj.get("geometry") or []:
            boundaries = geom.get("boundaries") or []
            if geom.get("type") == "Solid":
                faces = [f for shell in boundaries for f in shell]
            elif geom.get("type") in ("MultiSurface", "CompositeSurface"):
                faces = boundaries
            else:
                continue
            for face in faces:
                if face and len(face[0]) >= 3:
                    ids.update(face[0])
        if ids:
            counts[oid] = len(ids)
    return counts


def weighted_sample(counts, fraction, seed, alpha=1.0):
    """Draw ``fraction`` of ``counts`` without replacement, favouring small objects.

    Uses the exponential race (Efraimidis-Spirakis): with keys ``Exp(1) / w_i``
    the ``k`` smallest are an exact weighted sample without replacement, in
    ``O(n log n)``. ``numpy.random.choice(replace=False, p=...)`` is exact too
    but quadratic, and this runs over ~400k objects.
    """
    keys = list(counts)
    k = int(round(fraction * len(keys)))
    k = max(0, min(k, len(keys)))
    if k == 0:
        return set()
    if k == len(keys):
        return set(keys)

    n = np.array([counts[key] for key in keys], dtype=float)
    weights = np.power(np.maximum(n, 1.0), -float(alpha))
    rng = np.random.default_rng(seed)
    race = rng.exponential(size=len(keys)) / weights
    return {keys[i] for i in np.argpartition(race, k - 1)[:k]}


def write_subset(cj, keep_ids):
    """Copy of ``cj`` holding only ``keep_ids``, with vertices compacted."""
    kept = {oid: obj for oid, obj in (cj.get("CityObjects") or {}).items()
            if oid in keep_ids}
    if not kept:
        return None
    return compact_vertices({**cj, "CityObjects": kept})


def _load(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cj = json.load(fh)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return cj if cj.get("type") == "CityJSON" else None


def sample_dataset(root, out_dir, fraction, seed, alpha=1.0,
                   folders=DEFAULT_FOLDERS, reference=REFERENCE):
    """Write a subsample of ``root`` into ``out_dir``. Returns a stats dict."""
    root, out_dir = Path(root), Path(out_dir)
    present = [f for f in folders if (root / f).is_dir()]
    if reference not in present:
        raise FileNotFoundError(f"Reference folder {reference} not found under {root}")

    # 1. size every object in the reference folder
    counts, total = {}, 0
    for src in sorted((root / reference).rglob("*.json")):
        cj = _load(src)
        if cj is None:
            continue
        rel = src.relative_to(root / reference)
        for oid, n in object_vertex_counts(cj).items():
            counts[(str(rel), oid)] = n
            total += 1
    if not counts:
        raise ValueError(f"No CityObjects found under {root / reference}")

    # 2. one draw, reused by every folder so the LODs stay aligned
    chosen = weighted_sample(counts, fraction, seed, alpha)
    by_file = {}
    for rel, oid in chosen:
        by_file.setdefault(rel, set()).add(oid)

    stats = {"population": total, "sampled": len(chosen), "files": 0,
             "per_folder": {}}
    sizes = np.array([counts[key] for key in chosen], dtype=float)
    allsizes = np.array(list(counts.values()), dtype=float)
    stats["vertices"] = {
        "population_median": float(np.median(allsizes)),
        "population_max": float(allsizes.max()),
        "sample_median": float(np.median(sizes)),
        "sample_max": float(sizes.max()),
    }

    # 3. write every folder, restricted to the chosen ids
    for name in present:
        written = objects = 0
        for src in sorted((root / name).rglob("*.json")):
            rel = str(src.relative_to(root / name))
            keep = by_file.get(rel)
            if not keep:
                continue
            cj = _load(src)
            if cj is None:
                continue
            subset = write_subset(cj, keep)
            if subset is None:
                continue
            _write(out_dir / name / rel, subset)
            written += 1
            objects += len(subset["CityObjects"])
        stats["per_folder"][name] = {"files": written, "objects": objects}
        stats["files"] += written
        logger.info("%s: %d files, %d objects", name, written, objects)

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "source": str(root),
        "args": {"fraction": fraction, "seed": seed, "alpha": alpha,
                 "reference": reference, "folders": list(present)},
        "stats": stats,
        "warning": "Size-biased subset: not distribution-faithful. Train-split "
                   "statistics are recomputed per datamodule, so a run on this "
                   "set is self-consistent; its scaled-unit metrics are simply "
                   "not comparable with a full-corpus run's.",
    }
    (out_dir / "sample_manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "sample_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                  encoding="utf-8")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="Cleaned dataset root holding the LOD folders.")
    ap.add_argument("--out", required=True, help="Where to write the mini dataset.")
    ap.add_argument("--fraction", type=float, default=0.05,
                    help="Share of objects to keep (default: 0.05).")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Exponent on 1/n_vertices. 0 = uniform, 1 = inverse size "
                         "(default), >1 = harsher against large buildings.")
    ap.add_argument("--folders", nargs="+", default=list(DEFAULT_FOLDERS),
                    help=f"LOD folders to carry (default: {' '.join(DEFAULT_FOLDERS)}).")
    ap.add_argument("--reference", default=REFERENCE,
                    help="Folder the draw is made on (default: LOD2).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root, out_dir = Path(args.root).resolve(), Path(args.out).resolve()
    if not root.is_dir():
        sys.exit(f"Dataset root not found: {root}")
    if not 0.0 <= args.fraction <= 1.0:
        sys.exit(f"--fraction must be in [0, 1], got {args.fraction}")
    for name in args.folders:
        d = out_dir / name
        if d.exists() and any(d.iterdir()):
            sys.exit(f"Output folder already exists and is not empty: {d}")

    try:
        stats = sample_dataset(root, out_dir, args.fraction, args.seed, args.alpha,
                               tuple(args.folders), args.reference)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(str(exc))

    v = stats["vertices"]
    logger.info("sampled %d of %d objects (%.2f%%) into %d files",
                stats["sampled"], stats["population"],
                100 * stats["sampled"] / max(stats["population"], 1), stats["files"])
    logger.info("vertices/object: population median %.0f max %.0f -> "
                "sample median %.0f max %.0f",
                v["population_median"], v["population_max"],
                v["sample_median"], v["sample_max"])


if __name__ == "__main__":
    main()
