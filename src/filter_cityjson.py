#!/usr/bin/env python3
"""Filter a CityJSON dataset to exactly one Solid per building, per LOD folder.

The raw The Hague dataset mixes several representations inside each LOD folder,
which the graph parser silently unions together:

  * 3DBAG (Source B) parent ``Building`` objects carry only a lod-0 footprint
    MultiSurface; the volumes live on their ``BuildingPart`` children. Parsed
    as-is, every parent becomes a spurious 5-node graph (footprint ring + one
    face node).
  * Those ``BuildingPart`` objects hold three alternative Solids (LOD 1.2, 1.3
    and 2.2). They are *views* of one building, not parts of it, so unioning
    them yields three interpenetrating copies.
  * A few parts are degenerate slivers (e.g. a 0.6 x 0.6 m footprint extruded
    12.5 m) left over from party-wall strips.

Only the LOD2 folder is read. The dataset's LOD1 folder is derived data --
``convert_to_lod1`` applied to the LOD2 folder, once per geometry, which is why
its files carry all three variants with every ``lod`` tag rewritten to "1".
Filtering it independently would pair each LOD2 sample with an LOD1 solid
derived from a *different* source variant, so the LOD1 output is regenerated
here from the already-filtered LOD2 instead.

Output mirrors the input tree, so ``CityJsonLodDataset`` reads it unchanged.
Buildings split across several ``BuildingPart``s stay separate objects; 3DBAG
already suffixes their ids (``<Pand>-0``, ``-1``), so they remain unique.

Usage:
    python -m src.filter_cityjson "data/The Hague" --out "data/The Hague filtered"
"""
import argparse
import json
import logging
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.geometry.geometry import convert_to_lod1

logger = logging.getLogger(__name__)

SOLID_TYPES = ("Solid",)
DEFAULT_MAX_SLENDERNESS = 5.0   # height / min horizontal extent; corpus p99.9 is 2.15
DEFAULT_MIN_EXTENT = 1.0        # metres; corpus p1 is 1.96


def world_vertices(cj):
    """Vertex array in world coordinates, applying ``transform`` when present."""
    v = np.asarray(cj["vertices"], dtype=float)
    t = cj.get("transform")
    if t:
        v = v * np.asarray(t["scale"], dtype=float) + np.asarray(t["translate"], dtype=float)
    return v


def _faces(solid):
    """Outer rings of every surface in a Solid, flattened across shells."""
    return [ring[0] for shell in solid["boundaries"] for ring in shell]


def select_solid(obj, lod):
    """The single Solid representing this object at ``lod``, or None.

    Candidates are Solids whose ``lod`` tag truncates to ``lod``, so 2.2 counts
    as LOD2 while 1.2 and 1.3 do not. If several survive -- possible when an
    exporter has flattened the tags -- the finest wins, i.e. the most faces.
    """
    candidates = []
    for geom in obj.get("geometry", []):
        if geom.get("type") not in SOLID_TYPES:
            continue
        try:
            tag = int(float(geom["lod"]))
        except (KeyError, TypeError, ValueError):
            continue
        if tag == lod:
            candidates.append(geom)
    if not candidates:
        return None
    return max(candidates, key=lambda g: len(_faces(g)))


def _ring_area(points):
    """Shoelace area of a ring projected to the xy plane."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def is_sliver(vertices, solid, max_slenderness=DEFAULT_MAX_SLENDERNESS,
              min_extent=DEFAULT_MIN_EXTENT):
    """True for degenerate geometry: a pillar-like extrusion or a zero-area base.

    Measured over 28125 lod-2 Solids from both sources, slenderness sits at
    p99.9 = 2.15 and minimum horizontal extent at p1 = 1.96 m, so these
    thresholds isolate genuine artifacts rather than small buildings.
    """
    rings = _faces(solid)
    points = vertices[sorted({i for ring in rings for i in ring})]
    dx, dy, dz = points.max(axis=0) - points.min(axis=0)
    width = min(dx, dy)
    if width < min_extent or (width > 0 and dz / width > max_slenderness):
        return True

    sem = solid.get("semantics") or {}
    surfaces = sem.get("surfaces") or []
    values = [v for shell in (sem.get("values") or []) for v in shell]
    ground = [rings[i] for i, v in enumerate(values)
              if i < len(rings) and v is not None and v < len(surfaces)
              and surfaces[v].get("type") == "GroundSurface"]
    # Only meaningful when the file actually labels a ground surface.
    return bool(ground) and sum(_ring_area(vertices[r]) for r in ground) < 1e-6


def filter_city_objects(cj, lod, max_slenderness=DEFAULT_MAX_SLENDERNESS,
                        min_extent=DEFAULT_MIN_EXTENT):
    """Keep one Solid per object at ``lod``; drop footprint parents and slivers.

    Returns ``(city_objects, stats)``. ``parents``/``children`` are stripped
    because the objects they reference are gone.
    """
    vertices = world_vertices(cj)
    kept, stats = {}, {"kept": 0, "no_lod_solid": 0, "slivers": 0, "sliver_ids": []}

    for oid, obj in cj.get("CityObjects", {}).items():
        solid = select_solid(obj, lod)
        if solid is None:
            stats["no_lod_solid"] += 1
            continue
        if max_slenderness is not None and is_sliver(vertices, solid, max_slenderness, min_extent):
            stats["slivers"] += 1
            stats["sliver_ids"].append(oid)
            continue
        kept[oid] = {**{k: v for k, v in obj.items() if k not in ("geometry", "parents", "children")},
                     "geometry": [solid]}
        stats["kept"] += 1

    return kept, stats


def _remap(boundaries, mapping):
    if isinstance(boundaries, list):
        return [_remap(b, mapping) for b in boundaries]
    return mapping[boundaries]


def compact_vertices(cj):
    """Drop vertices no surviving geometry references, reindexing boundaries."""
    used = sorted({i for obj in cj["CityObjects"].values()
                   for geom in obj["geometry"] for i in _remap_ids(geom["boundaries"])})
    mapping = {old: new for new, old in enumerate(used)}
    objects = {
        oid: {**obj, "geometry": [{**geom, "boundaries": _remap(geom["boundaries"], mapping)}
                                  for geom in obj["geometry"]]}
        for oid, obj in cj["CityObjects"].items()
    }
    return {**cj, "CityObjects": objects, "vertices": [cj["vertices"][i] for i in used]}


def _remap_ids(boundaries):
    for b in boundaries:
        if isinstance(b, list):
            yield from _remap_ids(b)
        else:
            yield b


def filter_cityjson(cj, lod, max_slenderness=DEFAULT_MAX_SLENDERNESS,
                    min_extent=DEFAULT_MIN_EXTENT):
    """Filtered copy of a CityJSON dict, or None when no object survives."""
    kept, _ = filter_city_objects(cj, lod, max_slenderness, min_extent)
    return compact_vertices({**cj, "CityObjects": kept}) if kept else None


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _find_lod_dir(dataset_dir, lod):
    for d in sorted(dataset_dir.iterdir()):
        if d.is_dir() and d.name.lower() == f"lod{lod}":
            return d
    return None


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def process_dataset(dataset_dir, out_dir, max_slenderness, min_extent, lod=2):
    """Filter the LODn folder into ``out_dir``, regenerating LOD1 beside it."""
    src_dir = _find_lod_dir(dataset_dir, lod)
    if src_dir is None:
        raise FileNotFoundError(f"No LOD{lod} folder under {dataset_dir}")
    out_lod2 = out_dir / src_dir.name
    out_lod1 = out_dir / ((_find_lod_dir(dataset_dir, 1) or Path("LOD1")).name)

    totals = {f"lod{lod}": {"files": 0, "kept": 0, "no_lod_solid": 0, "slivers": 0, "sliver_ids": []},
              "lod1_generated": {"files": 0}, "errors": []}

    for src in sorted(p for p in src_dir.rglob("*") if p.is_file()):
        rel = src.relative_to(src_dir)

        try:
            with open(src, "r", encoding="utf-8") as fh:
                cj = json.load(fh)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            cj = None                        # description.txt and other non-JSON assets

        if cj is None or cj.get("type") != "CityJSON":
            for dst in (out_lod2 / rel, out_lod1 / rel):
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            continue

        kept, stats = filter_city_objects(cj, lod, max_slenderness, min_extent)
        if not kept:
            logger.warning("no LOD%d objects survived in %s", lod, src.name)
            continue
        filtered = compact_vertices({**cj, "CityObjects": kept})

        try:
            _write(out_lod2 / rel, filtered)
            # convert_to_lod1 mutates its argument, so hand it a copy; the
            # LOD1 pair must come from the same solid the LOD2 file keeps.
            lod1 = compact_vertices(convert_to_lod1(deepcopy(filtered)))
            _write(out_lod1 / rel, lod1)
        except (OSError, KeyError, ValueError) as exc:
            logger.error("failed on %s: %s", src.name, exc)
            totals["errors"].append(f"{rel}: {exc}")
            continue

        bucket = totals[f"lod{lod}"]
        bucket["files"] += 1
        for key in ("kept", "no_lod_solid", "slivers"):
            bucket[key] += stats[key]
        bucket["sliver_ids"] += [f"{src.name}:{i}" for i in stats["sliver_ids"]]
        totals["lod1_generated"]["files"] += 1
        logger.info("%s -> kept %d, dropped %d (%d slivers)",
                    src.name, stats["kept"], stats["no_lod_solid"], stats["slivers"])

    return totals


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", help="Dataset root containing the LOD2 folder.")
    ap.add_argument("--out", required=True, help="Output folder (must not already exist).")
    ap.add_argument("--lod", type=int, default=2, help="Source LOD folder to filter (default: 2).")
    ap.add_argument("--max-slenderness", type=float, default=DEFAULT_MAX_SLENDERNESS,
                    help="Drop solids taller than this multiple of their narrowest span.")
    ap.add_argument("--min-extent", type=float, default=DEFAULT_MIN_EXTENT,
                    help="Drop solids whose narrowest horizontal span is below this (metres).")
    ap.add_argument("--keep-slivers", action="store_true", help="Disable the sliver screen.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.is_dir():
        sys.exit(f"Dataset folder not found: {dataset_dir}")
    out_dir = Path(args.out).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"Output folder already exists and is not empty: {out_dir}")
    if out_dir == dataset_dir:
        sys.exit("Output folder must differ from the dataset folder.")
    out_dir.mkdir(parents=True, exist_ok=True)

    max_slenderness = None if args.keep_slivers else args.max_slenderness
    try:
        totals = process_dataset(dataset_dir, out_dir, max_slenderness, args.min_extent, args.lod)
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "source": str(dataset_dir),
        "args": vars(args),
        "totals": totals,
    }
    (out_dir / "filter_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if totals["errors"]:
        logger.warning("%d files failed: %s", len(totals["errors"]), totals["errors"][:5])
    bucket = totals[f"lod{args.lod}"]
    logger.info("lod%d: %d files, %d objects kept, %d dropped, %d slivers",
                args.lod, bucket["files"], bucket["kept"], bucket["no_lod_solid"], bucket["slivers"])
    logger.info("lod1: %d files regenerated from the filtered output",
                totals["lod1_generated"]["files"])
    logger.info("manifest written to %s", out_dir / "filter_manifest.json")


if __name__ == "__main__":
    main()
