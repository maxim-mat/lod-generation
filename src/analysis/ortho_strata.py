"""Post-hoc metrics of image-conditioned runs, split by what the image can and cannot show.

Strata: `b3_pw_datum` (point-cloud year vs the 2022 imagery -- 64.9% of `cleaner`
is 2023), LOD2 height (relief displacement grows with height), and all.
Reads the per-object `objects.csv` of one post-hoc run per image mode.

Usage:
    python -m src.analysis.ortho_strata --dataset-dir "data/The Hague/cleaner" \\
        --run true=<posthoc dir>/<run> --run shuffled=<...> --run none=<...> \\
        --metric chamfer_m_vs_lod2 --out outputs/ortho_strata
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.ortho_tiles import pw_datum_by_id
from src.visualize_cityjson import load_cityjson

logger = logging.getLogger(__name__)
HEIGHT_BINS = [(0, 10, "<10"), (10, 20, "10-20"), (20, 40, "20-40"), (40, np.inf, ">=40")]


def building_attributes(subset_root):
    """``{id: (pw_datum, height_m)}`` over the subset's LOD2 files."""
    out = {}
    for path in sorted((Path(subset_root) / "LOD2").rglob("*.json")):
        cj, verts = load_cityjson(path)
        datum = pw_datum_by_id(path)
        for oid, obj in cj["CityObjects"].items():
            idx = {i for g in obj.get("geometry", []) for i in _flat(g["boundaries"])}
            z = np.asarray(verts)[sorted(idx), 2]
            out[oid] = (datum[oid], float(z.max() - z.min()))
    return out


def to_markdown(table):
    """Plain markdown table (no `tabulate` dependency)."""
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in table.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v)
                                        for v in row) + " |")
    return "\n".join(lines) + "\n"


def _flat(nested):
    if isinstance(nested, int):
        return [nested]
    return [i for item in nested for i in _flat(item)]


def _height_bin(h):
    return next(label for lo, hi, label in HEIGHT_BINS if lo <= h < hi)


def _gap(datum, year):
    if datum == "unknown":
        return "unknown"
    gap = abs(int(datum) - int(year))
    return str(gap) if gap <= 2 else ">2"


def stratify(subset_root, runs, metric, assignment=None):
    """Per-stratum mean/median of `metric` for each run ({mode: objects DataFrame}).

    Every run is restricted to the ids present in all of them, so modes and
    models are compared on identical buildings.
    """
    attrs = building_attributes(subset_root)
    common = set(attrs).intersection(*(set(df["object_id"]) for df in runs.values()))
    strata = ["all", "pw_datum", "height_bin"] + (["imagery_gap"] if assignment else [])
    rows = []
    for mode, df in runs.items():
        df = df[df["object_id"].isin(common)].copy()
        df["pw_datum"] = df["object_id"].map(lambda o: attrs[o][0])
        df["height_bin"] = df["object_id"].map(lambda o: _height_bin(attrs[o][1]))
        if assignment:
            df["imagery_gap"] = df["object_id"].map(lambda o: _gap(attrs[o][0], assignment[o]))
        df["all"] = "all"
        for stratum in strata:
            for value, g in df.groupby(stratum):
                vals = g[metric].dropna()
                rows.append({"mode": mode, "stratum": stratum, "value": str(value),
                             "n": len(vals), "mean": float(vals.mean()),
                             "median": float(vals.median())})
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="mode=<run dir>")
    parser.add_argument("--metric", default="chamfer_m_vs_lod2")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    runs = {}
    for spec in args.run:
        mode, path = spec.split("=", 1)
        runs[mode] = pd.read_csv(Path(path) / "objects.csv")
    assignment_path = args.dataset_dir / "ortho" / "year_assignment.json"
    assignment = (json.loads(assignment_path.read_text(encoding="utf-8"))
                  if assignment_path.is_file() else None)
    table = stratify(args.dataset_dir, runs, args.metric, assignment)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "strata.csv", index=False)
    (args.out / "strata.md").write_text(to_markdown(table), encoding="utf-8")
    (args.out / "strata_args.json").write_text(json.dumps(vars(args), default=str, indent=2),
                                               encoding="utf-8")
    logger.info("wrote %s", args.out / "strata.csv")


if __name__ == "__main__":
    main()
