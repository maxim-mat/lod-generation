#!/usr/bin/env python3
"""Split a raw CityJSON dataset into folders of exactly one LOD each.

A raw folder holds whatever the supplier shipped, with sources in subfolders.
Its files mix several representations inside one object, which the graph parser
silently unions together:

  * 3DBAG parent ``Building`` objects carry only a lod-0 footprint MultiSurface;
    the volumes live on their ``BuildingPart`` children. Parsed as-is, every
    parent becomes a spurious 5-node graph (footprint ring + one face node).
  * Those ``BuildingPart`` objects hold three alternative Solids (LOD 1.2, 1.3
    and 2.2). They are *views* of one building, not parts of it, so unioning
    them yields three interpenetrating copies.
  * A few parts are degenerate slivers (e.g. a 0.6 x 0.6 m footprint extruded
    12.5 m) left over from party-wall strips.

Each output folder is one pure LOD extracted from those mixed geometries:

  ``LOD1/``        real lod-1 Solids, where the supplier shipped any
  ``LOD2/``        real lod-2 Solids
  ``LOD1_synth/``  lod-1 *derived* from the LOD2 output by ``convert_to_lod1``,
                   written only with ``--convert-on-the-fly``

LOD1 and LOD2 are independent extractions -- a source with no lod-1 geometry
simply contributes nothing to ``LOD1/``. ``LOD1_synth/`` exists because real
lod-1 data usually has no matching lod-2 description of the same building,
whereas a converted solid is paired with its LOD2 original by construction:
same filename, same object ids. The height it extrudes to is a heuristic, so
prefer real ``LOD1/`` geometry when a source provides it.

Every folder mirrors the raw subfolder layout, so ``CityJsonLodDataset`` reads
them unchanged. Buildings split across several ``BuildingPart``s stay separate
objects; 3DBAG already suffixes their ids (``<Pand>-0``, ``-1``), so they
remain unique.

Usage:
    python -m src.filter_cityjson "data/The Hague/raw" --out "data/The Hague"
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

from src.geometry.geometry import DEFAULT_ROOF_PERCENTILE, convert_to_lod1

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
    as LOD2 while 1.2 and 1.3 do not. Several routinely survive at lod 1, where
    3DBAG ships both 1.2 and 1.3; the finest wins, i.e. the most faces, which
    picks 1.3 wherever it actually refines 1.2 and 1.2 where the two are the
    same shape. The result is always the most detailed solid at that LOD.
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
    because the objects they reference are gone; a part that carries no
    attributes of its own inherits the dropped parent's instead.
    """
    vertices = world_vertices(cj)
    objects = cj.get("CityObjects", {})
    kept, stats = {}, {"kept": 0, "no_lod_solid": 0, "slivers": 0, "sliver_ids": []}

    for oid, obj in objects.items():
        solid = select_solid(obj, lod)
        if solid is None:
            stats["no_lod_solid"] += 1
            continue
        if max_slenderness is not None and is_sliver(vertices, solid, max_slenderness, min_extent):
            stats["slivers"] += 1
            stats["sliver_ids"].append(oid)
            continue
        entry = {k: v for k, v in obj.items() if k not in ("geometry", "parents", "children")}
        # 3DBAG hangs every BAG attribute on the parent Building and leaves the
        # BuildingPart's null, so dropping the parent would discard them all.
        if not entry.get("attributes"):
            entry["attributes"] = next(
                (a for pid in obj.get("parents", [])
                 if (a := objects.get(pid, {}).get("attributes"))), entry.get("attributes"))
        kept[oid] = {**entry, "geometry": [solid]}
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


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


SYNTH_DIR = "LOD1_synth"


def process_dataset(raw_dir, out_dir, max_slenderness, min_extent,
                    convert_on_the_fly=False, lods=(1, 2),
                    roof_percentile=DEFAULT_ROOF_PERCENTILE):
    """Split ``raw_dir`` into one folder per LOD under ``out_dir``.

    ``raw_dir`` is the supplier's tree, source subfolders and all; every output
    folder mirrors its layout. With ``convert_on_the_fly`` the LOD2 output is
    additionally converted down into ``LOD1_synth/``, keeping the filenames and
    object ids of its LOD2 original so the two form matched pairs.
    """
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw folder not found: {raw_dir}")

    names = {lod: f"LOD{lod}" for lod in lods}
    totals = {names[lod]: {"files": 0, "kept": 0, "no_lod_solid": 0, "slivers": 0, "sliver_ids": []}
              for lod in lods}
    totals[SYNTH_DIR] = {"files": 0}
    totals["errors"] = []

    roots = [out_dir / n for n in names.values()] + ([out_dir / SYNTH_DIR] if convert_on_the_fly else [])

    for src in sorted(p for p in raw_dir.rglob("*") if p.is_file()):
        rel = src.relative_to(raw_dir)

        try:
            with open(src, "r", encoding="utf-8") as fh:
                cj = json.load(fh)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            cj = None                        # description.txt and other non-JSON assets

        if cj is None or cj.get("type") != "CityJSON":
            for root in roots:
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, root / rel)
            continue

        for lod in lods:
            kept, stats = filter_city_objects(cj, lod, max_slenderness, min_extent)
            if not kept:
                logger.info("no lod-%d geometry in %s", lod, src.name)
                continue
            filtered = compact_vertices({**cj, "CityObjects": kept})

            try:
                _write(out_dir / names[lod] / rel, filtered)
                if convert_on_the_fly and lod == 2:
                    # convert_to_lod1 mutates its argument, so hand it a copy;
                    # the synthetic pair must come from the solid LOD2 kept.
                    _write(out_dir / SYNTH_DIR / rel,
                           compact_vertices(convert_to_lod1(deepcopy(filtered), roof_percentile)))
                    totals[SYNTH_DIR]["files"] += 1
            except (OSError, KeyError, IndexError, ValueError) as exc:
                logger.error("failed on %s at lod %d: %s", src.name, lod, exc)
                totals["errors"].append(f"{rel} (lod{lod}): {exc}")
                continue

            bucket = totals[names[lod]]
            bucket["files"] += 1
            for key in ("kept", "no_lod_solid", "slivers"):
                bucket[key] += stats[key]
            bucket["sliver_ids"] += [f"{src.name}:{i}" for i in stats["sliver_ids"]]
            logger.info("%s lod%d -> kept %d, dropped %d (%d slivers)",
                        src.name, lod, stats["kept"], stats["no_lod_solid"], stats["slivers"])

    return totals


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir", help="Raw dataset folder, source subfolders and all.")
    ap.add_argument("--out", required=True,
                    help="Dataset root to write LOD1/, LOD2/ and LOD1_synth/ into.")
    ap.add_argument("--lods", type=int, nargs="+", default=[1, 2],
                    help="LODs to extract into their own folders (default: 1 2).")
    ap.add_argument("--convert-on-the-fly", action="store_true",
                    help="Also derive LOD1_synth/ from the LOD2 output, paired by file and object id.")
    ap.add_argument("--roof-percentile", type=float, default=DEFAULT_ROOF_PERCENTILE,
                    help="Area-weighted roof-height quantile the synthetic LOD1 cap sits at "
                         f"(default: {DEFAULT_ROOF_PERCENTILE}, matching 3DBAG lod1.2:lod2.2 volume).")
    ap.add_argument("--max-slenderness", type=float, default=DEFAULT_MAX_SLENDERNESS,
                    help="Drop solids taller than this multiple of their narrowest span.")
    ap.add_argument("--min-extent", type=float, default=DEFAULT_MIN_EXTENT,
                    help="Drop solids whose narrowest horizontal span is below this (metres).")
    ap.add_argument("--keep-slivers", action="store_true", help="Disable the sliver screen.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    raw_dir = Path(args.raw_dir).resolve()
    if not raw_dir.is_dir():
        sys.exit(f"Raw folder not found: {raw_dir}")
    out_dir = Path(args.out).resolve()
    written = [out_dir / f"LOD{l}" for l in args.lods] + [out_dir / SYNTH_DIR]
    for d in written:
        if d.exists() and any(d.iterdir()):
            sys.exit(f"Output folder already exists and is not empty: {d}")
    if raw_dir in written or raw_dir == out_dir:
        sys.exit("Raw folder must sit outside the folders being written.")
    out_dir.mkdir(parents=True, exist_ok=True)

    max_slenderness = None if args.keep_slivers else args.max_slenderness
    try:
        totals = process_dataset(raw_dir, out_dir, max_slenderness, args.min_extent,
                                 args.convert_on_the_fly, tuple(args.lods), args.roof_percentile)
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "source": str(raw_dir),
        "args": vars(args),
        "totals": totals,
    }
    (out_dir / "filter_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if totals["errors"]:
        logger.warning("%d failures: %s", len(totals["errors"]), totals["errors"][:5])
    for lod in args.lods:
        b = totals[f"LOD{lod}"]
        logger.info("LOD%d: %d files, %d objects kept, %d without a lod-%d solid, %d slivers",
                    lod, b["files"], b["kept"], b["no_lod_solid"], lod, b["slivers"])
    if args.convert_on_the_fly:
        logger.info("%s: %d files derived from the LOD2 output", SYNTH_DIR, totals[SYNTH_DIR]["files"])
    logger.info("manifest written to %s", out_dir / "filter_manifest.json")


if __name__ == "__main__":
    main()
