#!/usr/bin/env python3
"""Select the trainable subset of a cleaned CityJSON dataset.

Runs over ``transform_cityjson``'s output and writes the subset the mesh
transformer should actually train on. Unlike ``filter_cityjson`` (which decides
what geometry *is*) and ``transform_cityjson`` (which repairs what it can),
this script only ever *drops whole objects* -- no geometry is rewritten, so
every survivor is byte-identical to its input apart from vertex compaction.

Three filters, applied in this order:

  1. **real-LOD1 pairing.** Keep only objects that have a real ``LOD1/``
     counterpart. Measured on The Hague, ``LOD1/`` is Source B only (246 files,
     265,845 objects, zero Source A), because Source A ships no lod-1 geometry
     at all -- so this filter also silently selects Source B. ``LOD1_synth/``
     is deliberately *not* written: it is derived from LOD2 and pinned to it,
     and the point of this selection is to train against the real pair. Re-run
     ``transform_cityjson --regenerate-synth`` against the output if it is
     wanted later.

  2. **geometry defects**, every class ``cityobject_analysis.object_defects``
     reports except ``disconnected``. Disconnected objects are kept on purpose:
     they are overwhelmingly buildings with a detached dormer or installation,
     which is real geometry rather than breakage.

  3. **vertex-count outliers**, Tukey fences at ``k`` IQRs.

On the fence transform: vertex counts are positive, heavy-tailed and spiked at
8 (the minimal box). Raw Tukey puts the upper fence at 78 vertices -- the 95th
percentile -- so it would call one ordinary building in twenty an outlier.
That is a skew artifact, not an outlier population. On ``log10`` the fences
land near p99.9 and catch only genuine mega-objects. ``log`` is therefore the
default; ``--fence-transform raw`` is kept because the QQ plot this writes is
the thing that settles it, and it should be possible to see both.

Usage:
    python -m src.select_cityjson "data/The Hague/clean" \
        --out "data/The Hague/cleaner"
"""
import argparse
import csv
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

from src.analysis.cityobject_analysis import _faces_and_types, object_defects
from src.filter_cityjson import _git_commit, _write, compact_vertices, world_vertices

logger = logging.getLogger(__name__)

# Everything object_defects can report except `disconnected`, which is kept.
DROP_DEFECTS = (
    "open_shell", "non_manifold_edge", "non_positive_volume",
    "repeated_ring_vertex", "roof_normal_down", "non_planar_face",
    "wall_not_vertical", "collinear_face", "degenerate_ring",
    "ground_normal_up", "missing_ground", "missing_roof", "missing_wall",
    "unknown_semantic_label", "null_semantic_label",
)
KEEP_DEFECTS = ("disconnected",)
CENSUS_FIELDS = ("source", "tile", "id", "n_verts", "n_faces", "defects")


# ----------------------------------------------------------------------
# Pure selection logic
# ----------------------------------------------------------------------

def _fence_bounds(values, k, transform):
    """``(y, lo, hi)`` all in transform space, so comparisons stay exact."""
    x = np.asarray(values, dtype=float)
    if transform not in ("log", "raw"):
        raise ValueError(f"Unknown fence transform: {transform!r}")
    if x.size == 0:
        raise ValueError("no values to compute fences from")
    y = np.log10(x) if transform == "log" else x
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    return y, q1 - k * iqr, q3 + k * iqr


def tukey_fences(values, k=1.5, transform="log"):
    """Tukey fences on ``values``, converted back to the original units.

    For *reporting* -- the manifest and the QQ plot want a vertex count, not a
    logarithm. Do not filter by comparing against these: ``10 ** log10(v)``
    does not round-trip, so a value sitting exactly on the fence lands on the
    wrong side of it. Use `tukey_inliers`, which compares in transform space.

    Args:
        values: 1-D positive counts.
        k (float): IQR multiplier. 1.5 is the standard outlier fence.
        transform (str): ``"log"`` (log10 first, the default -- see the module
            docstring) or ``"raw"``.

    Returns:
        tuple: ``(lo, hi)`` in the original units. A lower fence below the
        data's minimum simply drops nothing, which is the expected outcome
        here: there is no such thing as a too-simple building below the
        8-vertex box.
    """
    _, lo, hi = _fence_bounds(values, k, transform)
    return (float(10 ** lo), float(10 ** hi)) if transform == "log" else (float(lo), float(hi))


def tukey_inliers(values, k=1.5, transform="log"):
    """Boolean mask: True where the value is within the fences.

    Compared in transform space. A degenerate IQR of 0 -- every object the same
    size -- collapses the fences onto that value and marks all of them inliers,
    which is right.
    """
    y, lo, hi = _fence_bounds(values, k, transform)
    return (y >= lo) & (y <= hi)


def select(rows, lod1_ids, k=1.5, transform="log"):
    """Ids to keep, and a count for every step of the ladder.

    Args:
        rows: per-object dicts with ``id``, ``n_verts`` and ``defects``.
        lod1_ids (set): object ids that have a real LOD1 counterpart.

    Returns:
        tuple: ``(keep_ids set, stats dict)``.
    """
    stats = {"lod2_objects": len(rows)}

    paired = [r for r in rows if r["id"] in lod1_ids]
    stats["after_lod1_pairing"] = len(paired)

    def is_bad(r):
        found = str(r["defects"]).split("|") if r["defects"] else []
        return any(d in DROP_DEFECTS for d in found)

    clean = [r for r in paired if not is_bad(r)]
    stats["after_defect_drop"] = len(clean)
    stats["defect_breakdown"] = dict(Counter(
        d for r in paired for d in str(r["defects"]).split("|")
        if d in DROP_DEFECTS))

    if not clean:
        return set(), stats

    verts = np.array([int(r["n_verts"]) for r in clean], dtype=float)
    inlier = tukey_inliers(verts, k=k, transform=transform)
    lo, hi = tukey_fences(verts, k=k, transform=transform)
    kept = [r for r, ok in zip(clean, inlier) if ok]
    stats.update({
        "fence_transform": transform, "fence_k": k,
        "fence_lo_verts": lo, "fence_hi_verts": hi,
        "dropped_below_fence": int((~inlier & (verts < np.median(verts))).sum()),
        "dropped_above_fence": int((~inlier & (verts >= np.median(verts))).sum()),
        "after_vertex_fence": len(kept),
    })
    stats["kept_disconnected"] = sum(
        1 for r in kept if "disconnected" in str(r["defects"]).split("|"))
    return {r["id"] for r in kept}, stats


# ----------------------------------------------------------------------
# Scanning
# ----------------------------------------------------------------------

def _scan_tile(args):
    path, source = args
    out = []
    try:
        with open(path, encoding="utf-8") as fh:
            cj = json.load(fh)
        verts = world_vertices(cj)
        for oid, obj in cj.get("CityObjects", {}).items():
            faces = _faces_and_types(obj)
            if not faces:
                continue
            active = {v for r, _ in faces for v in r}
            out.append({
                "source": source, "tile": Path(path).name, "id": oid,
                "n_verts": len(active), "n_faces": len(faces),
                "defects": "|".join(object_defects(obj, verts)),
            })
    except (OSError, ValueError, KeyError) as exc:
        logger.error("scan failed on %s: %s", Path(path).name, exc)
    return out


def _tiles(root):
    for path in sorted(p for p in root.rglob("*.json") if p.is_file()):
        rel = path.relative_to(root).parts
        yield str(path), (rel[0] if len(rel) > 1 else ".")


def scan_lod2(root, workers):
    """Per-object census of the LOD2 folder: counts plus the defect list."""
    jobs = list(_tiles(root))
    logger.info("scanning %d LOD2 tiles with %d workers", len(jobs), workers)
    rows = []
    with Pool(processes=workers) as pool:
        for i, part in enumerate(pool.imap_unordered(_scan_tile, jobs), 1):
            rows.extend(part)
            if i % 25 == 0 or i == len(jobs):
                logger.info("  %d/%d tiles, %d objects", i, len(jobs), len(rows))
    return rows


def _lod1_ids_in(path):
    try:
        with open(path, encoding="utf-8") as fh:
            cj = json.load(fh)
    except (OSError, ValueError):
        logger.error("could not read %s", Path(path).name)
        return set()
    return {oid for oid, obj in cj.get("CityObjects", {}).items() if obj.get("geometry")}


def lod1_ids(root, workers):
    """Every object id carrying real lod-1 geometry, and the sources they sit in."""
    jobs = list(_tiles(root))
    ids, by_source = set(), Counter()
    with Pool(processes=workers) as pool:
        for (_, source), got in zip(jobs, pool.imap(_lod1_ids_in, [j[0] for j in jobs])):
            ids |= got
            by_source[source] += len(got)
    logger.info("LOD1: %d objects across %d tiles, per source %s",
                len(ids), len(jobs), dict(by_source))
    return ids, dict(by_source)


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------

def _write_tile(args):
    src, rel, out_root, keep = args
    try:
        with open(src, encoding="utf-8") as fh:
            cj = json.load(fh)
        kept = {oid: obj for oid, obj in cj.get("CityObjects", {}).items()
                if oid in keep and obj.get("geometry")}
        if not kept:
            return 0
        _write(Path(out_root) / rel, compact_vertices({**cj, "CityObjects": kept}))
        return len(kept)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        logger.error("write failed on %s: %s", Path(src).name, exc)
        return 0


def write_subset(in_root, out_root, keep_ids, workers):
    """Rewrite every tile of ``in_root`` keeping only ``keep_ids``."""
    jobs = [(path, Path(path).relative_to(in_root), str(out_root), keep_ids)
            for path, _ in _tiles(in_root)]
    with Pool(processes=workers) as pool:
        written = list(pool.imap_unordered(_write_tile, jobs))
    return sum(written), sum(1 for n in written if n)


def qq_plot(verts, k, out_path):
    """Normal QQ plot of the vertex count, raw and log10, with both fences."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sps

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, transform in zip(axes, ("raw", "log")):
        y = np.log10(verts) if transform == "log" else verts.astype(float)
        sps.probplot(y, dist="norm", plot=ax)
        lo, hi = tukey_fences(verts, k=k, transform=transform)
        for bound, label in ((lo, "lo"), (hi, "hi")):
            val = np.log10(bound) if transform == "log" else bound
            if y.min() <= val <= y.max():
                ax.axhline(val, color="crimson", ls="--", lw=1)
                ax.text(ax.get_xlim()[0], val, f" {label} fence = {bound:.1f} verts",
                        color="crimson", va="bottom", fontsize=8)
        drop = ((verts < lo) | (verts > hi)).sum()
        ax.set_title(f"{transform}: fences [{lo:.1f}, {hi:.1f}] "
                     f"-> {drop} outliers ({100 * drop / len(verts):.2f}%)")
        ax.set_ylabel("log10(n_verts)" if transform == "log" else "n_verts")
    fig.suptitle(f"LOD2 vertex count, normal QQ, Tukey k={k}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="cleaned dataset (holds LOD1/ and LOD2/)")
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    ap.add_argument("--fence-k", type=float, default=1.5, help="Tukey IQR multiplier")
    ap.add_argument("--fence-transform", choices=("log", "raw"), default="log")
    ap.add_argument("--workers", type=int, default=min(10, cpu_count()))
    ap.add_argument("--reuse-census", type=Path,
                    help="skip the LOD2 scan and read this census CSV instead")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the ladder and write the QQ plot, but no tiles")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    lod1_root, lod2_root = args.root / "LOD1", args.root / "LOD2"
    for path in (lod1_root, lod2_root):
        if not path.is_dir():
            ap.error(f"missing {path}")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.reuse_census:
        with open(args.reuse_census, newline="", encoding="utf-8") as fh:
            rows = [{k: r.get(k, "") for k in CENSUS_FIELDS}
                    for r in csv.DictReader(fh)]
        logger.info("reusing census: %d objects from %s", len(rows), args.reuse_census)
    else:
        rows = scan_lod2(lod2_root, args.workers)
        with open(args.out / "selection_census.csv", "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CENSUS_FIELDS)
            w.writeheader()
            w.writerows(rows)

    ids, lod1_by_source = lod1_ids(lod1_root, args.workers)
    keep, stats = select(rows, ids, k=args.fence_k, transform=args.fence_transform)
    stats["lod1_objects_by_source"] = lod1_by_source
    stats["lod1_sources"] = sorted(lod1_by_source)

    verts = np.array([int(r["n_verts"]) for r in rows
                      if r["id"] in ids and not any(
                          d in DROP_DEFECTS for d in str(r["defects"]).split("|"))])
    if len(verts):
        qq_plot(verts, args.fence_k, args.out / "vertex_qq.png")

    print("\nselection ladder")
    for key in ("lod2_objects", "after_lod1_pairing", "after_defect_drop",
                "after_vertex_fence"):
        print(f"  {key:22s} {stats.get(key, 0):8d}")
    print(f"  fences ({stats.get('fence_transform')}) : "
          f"[{stats.get('fence_lo_verts', 0):.1f}, {stats.get('fence_hi_verts', 0):.1f}] verts "
          f"-> dropped {stats.get('dropped_below_fence', 0)} below / "
          f"{stats.get('dropped_above_fence', 0)} above")
    print(f"  kept with `disconnected`: {stats.get('kept_disconnected', 0)}")
    kept_frac = 100 * stats.get("after_vertex_fence", 0) / max(stats["lod2_objects"], 1)
    print(f"  kept {kept_frac:.2f}% of LOD2, "
          f"{100 * stats.get('after_vertex_fence', 0) / max(stats.get('after_lod1_pairing', 1), 1):.2f}% "
          "of the LOD1-paired population")

    if args.dry_run:
        print("\n--dry-run: no tiles written")
        return

    for lod, root in (("LOD2", lod2_root), ("LOD1", lod1_root)):
        n_obj, n_files = write_subset(root, args.out / lod, keep, args.workers)
        stats[f"written_{lod}"] = {"objects": n_obj, "files": n_files}
        logger.info("%s: wrote %d objects across %d tiles", lod, n_obj, n_files)

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "source": str(args.root.resolve()),
        "args": {"out": str(args.out), "fence_k": args.fence_k,
                 "fence_transform": args.fence_transform},
        "drop_defects": list(DROP_DEFECTS),
        "keep_defects": list(KEEP_DEFECTS),
        "totals": stats,
    }
    with open(args.out / "selection_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
