"""Are neighbouring buildings alike? Evidence for (or against) a geographic split.

Three measurements on log footprint area (LOD1 base), log volume (LOD2) and log
face count (LOD2 triangles, the unit the model generates):
  * ICC(1) across 3DBAG tiles, with the within-tile / total SD ratio, against a
    null from shuffled tile labels;
  * a distance correlogram of building pairs;
  * nearest-train-neighbour distance and similarity for val/test buildings under
    the random split (every run up to v5) vs the tile split
    (`src.dataset.splits`). A random split that finds closer, more similar train
    neighbours is scoring on easier-than-real test data.

Usage:
    python -m src.analysis.spatial_correlation "data/The Hague/cleaner" \\
        --out outputs/spatial_correlation
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

from src.dataset.mesh_dataset import parse_cityjson_file_to_meshes, signed_volume
from src.dataset.splits import random_split_indices, tile_split_indices
from src.filter_cityjson import _git_commit

logger = logging.getLogger(__name__)
EDGES_M = [0, 25, 50, 100, 250, 500, 1000, 2000, 4000]


def _base(verts, faces, tol=1e-3):
    """Projected area and area-weighted centroid of the triangles at the mesh's lowest z."""
    tri = verts[faces]
    flat = (np.abs(tri[:, :, 2] - verts[:, 2].min()) < tol).all(axis=1)
    t = tri[flat][:, :, :2]
    area = 0.5 * np.abs((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
                        - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1]))
    centroid = (t.mean(axis=1) * area[:, None]).sum(axis=0) / max(area.sum(), 1e-12)
    return float(area.sum()), centroid


def building_metrics(lod1_path, lod2_path):
    """``{id: (x, y, footprint_m2, volume_m3, n_faces)}`` for buildings present in both files.

    Footprint and centroid come from the LOD1 base; volume and triangle count from LOD2.
    """
    lod1 = parse_cityjson_file_to_meshes(lod1_path)
    lod2 = parse_cityjson_file_to_meshes(lod2_path)
    out = {}
    for oid, (v2, f2) in lod2.items():
        if oid not in lod1:
            continue
        area, (x, y) = _base(*lod1[oid])
        out[oid] = (float(x), float(y), area, abs(float(signed_volume(v2, f2))), int(len(f2)))
    return out


def icc1(values, groups):
    """One-way ANOVA ICC(1) with unequal group sizes (n0 correction)."""
    values = np.asarray(values, float)
    codes, groups = np.unique(groups, return_inverse=True)
    k, n = len(codes), len(values)
    sizes = np.bincount(groups)
    means = np.bincount(groups, weights=values) / sizes
    grand = values.mean()
    msb = (sizes * (means - grand) ** 2).sum() / (k - 1)
    msw = ((values - means[groups]) ** 2).sum() / (n - k)
    n0 = (n - (sizes ** 2).sum() / n) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


def permutation_null(values, groups, n_perm=200, seed=0):
    """ICC(1) under shuffled group labels: what 'no spatial structure' looks like."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    return np.array([icc1(values, rng.permutation(groups)) for _ in range(n_perm)])


def correlogram(xy, values, edges=EDGES_M, n_anchors=5000, seed=0):
    """Pearson correlation of building pairs, binned by centroid distance.

    Pairs are drawn from `n_anchors` random anchors to every other building,
    one anchor at a time: cost is linear in the anchor count and memory stays
    a few arrays of N floats. A radius query instead materialises ~100k
    neighbours per anchor at city density, which ran out of memory on cleaner.
    """
    xy, z = np.asarray(xy, float), np.asarray(values, float)
    z = (z - z.mean()) / z.std()
    rng = np.random.default_rng(seed)
    anchors = rng.choice(len(xy), size=min(n_anchors, len(xy)), replace=False)
    n_bins = len(edges) - 1
    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)
    for a in anchors:
        b = np.digitize(np.hypot(*(xy - xy[a]).T), edges) - 1
        ok = (b >= 0) & (b < n_bins)
        ok[a] = False
        sums += np.bincount(b[ok], weights=z[a] * z[ok], minlength=n_bins)
        counts += np.bincount(b[ok], minlength=n_bins)
    return pd.DataFrame({"lo": edges[:-1], "hi": edges[1:], "n_pairs": counts,
                         "corr": sums / np.maximum(counts, 1)})


def nearest_train(xy, values, train_idx, eval_idx):
    """Distance to, and similarity with, each eval building's nearest train building."""
    xy, values = np.asarray(xy, float), np.asarray(values, float)
    train_idx, eval_idx = np.asarray(train_idx), np.asarray(eval_idx)
    dist, j = cKDTree(xy[train_idx]).query(xy[eval_idx])
    mine, theirs = values[eval_idx], values[train_idx[j]]
    return {"median_dist_m": float(np.median(dist)),
            "corr": float(np.corrcoef(mine, theirs)[0, 1]),
            "median_abs_diff": float(np.median(np.abs(mine - theirs)))}


def _job(args):
    root, rel = args
    return rel, building_metrics(root / "LOD1" / rel, root / "LOD2" / rel)


def building_table(subset_root, workers=8):
    """One row per building: id, tile, x, y, footprint_m2, volume_m3, n_faces."""
    root = Path(subset_root)
    rels = sorted(p.relative_to(root / "LOD2") for p in (root / "LOD2").rglob("*.json"))
    rows = []
    with mp.Pool(workers) as pool:
        for done, (rel, metrics) in enumerate(pool.imap_unordered(_job, [(root, r) for r in rels]), 1):
            rows += [(oid, rel.as_posix(), *m) for oid, m in metrics.items()]
            if done % 25 == 0:
                logger.info("%d/%d files", done, len(rels))
    return pd.DataFrame(rows, columns=["id", "tile", "x", "y", "footprint_m2", "volume_m3",
                                       "n_faces"])


def report(table, ratios=(0.8, 0.1, 0.1), seed=42, n_perm=200):
    """All three measurements, for log footprint, log LOD2 volume and log LOD2 face count."""
    table = table[(table["footprint_m2"] > 0) & (table["volume_m3"] > 0)
                  & (table["n_faces"] > 0)].reset_index(drop=True)
    xy = table[["x", "y"]].to_numpy()
    r_train, r_val, r_test = random_split_indices(len(table), ratios, seed)
    (t_train, t_val, t_test), _ = tile_split_indices(table["tile"].tolist(), ratios, seed)
    out = {"n_buildings": len(table), "n_tiles": int(table["tile"].nunique()), "metrics": {}}
    for name in ("footprint_m2", "volume_m3", "n_faces"):
        v = np.log(table[name].to_numpy())
        within = table.assign(v=v).groupby("tile")["v"].std().mean()
        null = permutation_null(v, table["tile"].to_numpy(), n_perm, seed)
        out["metrics"][name] = {
            "icc1": icc1(v, table["tile"].to_numpy()),
            "icc1_null_p99": float(np.percentile(null, 99)),
            "within_tile_sd_over_total_sd": float(within / v.std()),
            "correlogram": correlogram(xy, v, seed=seed).to_dict(orient="records"),
            "nearest_train": {"random": nearest_train(xy, v, r_train, r_val + r_test),
                              "tile": nearest_train(xy, v, t_train, t_val + t_test)},
        }
    return out


def to_markdown(rep):
    lines = [f"# Spatial correlation ({rep['n_buildings']} buildings, {rep['n_tiles']} tiles)", ""]
    for name, m in rep["metrics"].items():
        nr, nt = m["nearest_train"]["random"], m["nearest_train"]["tile"]
        lines += [f"## log {name}", "",
                  f"- ICC(1) across tiles: **{m['icc1']:.3f}** (shuffled-label 99th pct "
                  f"{m['icc1_null_p99']:.3f}); within-tile SD / total SD "
                  f"{m['within_tile_sd_over_total_sd']:.3f}",
                  f"- Nearest train building of a val/test building, random vs tile split: "
                  f"median {nr['median_dist_m']:.0f} m vs {nt['median_dist_m']:.0f} m; "
                  f"correlation {nr['corr']:.3f} vs {nt['corr']:.3f}",
                  "", "| distance (m) | pairs | correlation |", "|---|---|---|"]
        lines += [f"| {c['lo']}-{c['hi']} | {c['n_pairs']} | {c['corr']:.3f} |"
                  for c in m["correlogram"]]
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-perm", type=int, default=200)
    parser.add_argument("--workers", type=int, default=min(8, mp.cpu_count()))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    table = building_table(args.subset, args.workers)
    rep = report(table, seed=args.seed, n_perm=args.n_perm)
    rep.update(created=datetime.now(timezone.utc).isoformat(), git_commit=_git_commit(),
               args={k: str(v) for k, v in vars(args).items()})
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "buildings.csv", index=False)
    (args.out / "report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    (args.out / "report.md").write_text(to_markdown(rep), encoding="utf-8")
    logger.info("wrote %s", args.out / "report.md")


if __name__ == "__main__":
    main()
