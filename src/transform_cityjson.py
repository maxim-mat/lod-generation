#!/usr/bin/env python3
"""Repair the fixable geometry defects in a split CityJSON dataset.

Runs over the output of ``filter_cityjson`` -- ``LOD2/``, ``LOD1/`` and
``LOD1_synth/`` -- and writes a cleaned copy. It fixes what can be fixed
without inventing geometry, and drops objects that cannot describe a building.

Measured over 402,242 LOD2 objects of The Hague, the defects fall into three
groups, and only the first is safely repairable:

  * **degenerate faces** -- 4,598 objects carry a face with exactly zero area
    (collinear ring), and 779 a ring that visits a vertex twice. Both break the
    Levi face-node contract: a zero-area face has no normal, so it has no
    surface class, and a repeated index makes a self-loop in ``EDGE_VV``.
    Removing a zero-area face cannot change the shape, and 99.2% of them
    contribute no vertex that another face does not already provide.
  * **wrongly wound Ground/Roof rings** -- reversible, but only when reversing
    is *verifiably* right. The test is closure: a ring is reversed only if that
    strictly reduces the shell's unpaired and same-direction edges. This is what
    separates the three real footprints wound backwards (unpaired 81 -> 3) from
    the 237 mislabelled soffits whose reversal would break a watertight solid.
  * **open shells, non-manifold edges, non-positive volume** -- not repaired.
    Capping a hole invents a facet the model would then learn as real, and the
    non-manifold cases are not winding errors at all: four faces meet on the
    offending edge, so there is nothing to rewind. Non-positive volume is a
    symptom of those two (566 of 569), not a defect of its own.

Objects missing GroundSurface, RoofSurface or WallSurface are dropped, since a
building the model can learn from needs all three. Before that test runs, a
top-most ``OuterFloorSurface`` on an object with no roof at all is relabelled
``RoofSurface``: that is a Source A labelling convention rather than broken
geometry, and it recovers ~103 of the 106 otherwise-doomed objects.

``LOD1_synth/`` is pinned to ``LOD2/``: the drop set is computed on LOD2 and the
same object ids are removed from the synthetic pair, so the two folders keep
identical id sets. ``LOD1/`` is an independent extraction and is cleaned on its
own terms.

Usage:
    python -m src.transform_cityjson "data/The Hague" --out "data/The Hague/clean"
"""
import argparse
import json
import logging
import shutil
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from src.filter_cityjson import SYNTH_DIR, _git_commit, _write, compact_vertices, world_vertices
from src.geometry.geometry import (DEFAULT_ROOF_PERCENTILE, convert_to_lod1, face_normal,
                                   shell_edge_defects)

logger = logging.getLogger(__name__)

REQUIRED_SEMANTICS = ("GroundSurface", "RoofSurface", "WallSurface")
GROUND_NZ_MAX = -0.999          # matches the analysis fence
ROOF_RELABEL_FROM = "OuterFloorSurface"
DEFAULT_FOLDERS = ("LOD2", "LOD1", SYNTH_DIR)


# ==============================================================================
# Face list <-> CityJSON Solid
# ==============================================================================

def _read_faces(solid):
    """``[[shell_index, rings, surface_index], ...]`` flattened across shells."""
    sem = solid.get("semantics") or {}
    values = sem.get("values") or []
    out = []
    for si, shell in enumerate(solid.get("boundaries") or []):
        shell_values = values[si] if si < len(values) else []
        for fi, face in enumerate(shell):
            out.append([si, face, shell_values[fi] if fi < len(shell_values) else None])
    return out


def _write_faces(solid, faces):
    """Rebuild boundaries and semantics.values from a flat face list.

    Shells that lost every face disappear. The ``surfaces`` table is left as-is;
    entries no longer referenced are harmless and pruning them would only
    renumber every value for no gain.
    """
    shells, values = [], []
    for si in sorted({f[0] for f in faces}):
        group = [f for f in faces if f[0] == si]
        shells.append([f[1] for f in group])
        values.append([f[2] for f in group])
    solid["boundaries"] = shells
    if solid.get("semantics") is not None:
        solid["semantics"]["values"] = values


def _surface_type(solid, index):
    surfaces = (solid.get("semantics") or {}).get("surfaces") or []
    if index is None or not isinstance(index, int) or not 0 <= index < len(surfaces):
        return None
    entry = surfaces[index]
    return entry.get("type") if entry else None


def _retype(solid, index, new_type):
    """Point one face at a fresh surface entry carrying ``new_type``.

    A new entry rather than an edit in place: several faces routinely share one
    ``surfaces`` index, and mutating it would relabel all of them.
    """
    surfaces = solid["semantics"]["surfaces"]
    old = surfaces[index] if isinstance(index, int) and 0 <= index < len(surfaces) else {}
    surfaces.append({**(old or {}), "type": new_type})
    return len(surfaces) - 1


# ==============================================================================
# Repairs
# ==============================================================================

def _dedupe(ring):
    """Drop consecutive duplicate indices, including across the wrap."""
    out = [v for i, v in enumerate(ring) if v != ring[i - 1]]
    return out


def _closure_score(faces):
    """Total edge defects of a shell; lower is better, 0 is watertight."""
    unpaired, reused = shell_edge_defects([f[1] for f in faces])
    return len(unpaired) + len(reused)


def _repair_rings(faces, fixes):
    """Dedupe consecutive duplicate indices; drop rings too short to be a face.

    A ring that revisits a vertex *non*-consecutively is a pinch point -- two
    wings meeting at a corner, or a courtyard touching the outline -- and is
    common in real footprints. The Levi wireframe represents it fine, since
    ``EDGE_VV`` is a set and the shared vertex simply gains degree. Dropping
    such a face would take the whole building with it through the missing-ground
    test, and the objects affected include the largest in the corpus (a 36,305
    m2 footprint), so it is deliberately left alone.
    """
    kept = []
    for entry in faces:
        rings = []
        for ring in entry[1]:
            fixed = _dedupe(ring)
            if len(fixed) != len(ring):
                fixes["deduped_ring_vertex"] += 1
            rings.append(fixed)
        if not rings or len(set(rings[0])) < 3:
            fixes["dropped_degenerate_ring"] += 1
            continue
        holes = [h for h in rings[1:] if len(set(h)) >= 3]
        if len(holes) != len(rings) - 1:
            fixes["dropped_degenerate_hole"] += len(rings) - 1 - len(holes)
        entry[1] = [rings[0]] + holes
        kept.append(entry)
    return kept


def _drop_zero_area(faces, verts, fixes):
    kept = []
    for entry in faces:
        if face_normal([verts[i] for i in entry[1][0]]) is None:
            fixes["dropped_zero_area_face"] += 1
            continue
        kept.append(entry)
    return kept


def _relabel_roof(solid, faces, verts, fixes):
    """Promote a top-most OuterFloorSurface when the object has no roof."""
    types = [_surface_type(solid, e[2]) for e in faces]
    if "RoofSurface" in types or ROOF_RELABEL_FROM not in types:
        return
    best, best_z = None, None
    for entry, stype in zip(faces, types):
        if stype != ROOF_RELABEL_FROM:
            continue
        pts = [verts[i] for i in entry[1][0]]
        n = face_normal(pts)
        if n is None or n[2] <= 0:
            continue
        z = sum(p[2] for p in pts) / len(pts)
        if best_z is None or z > best_z:
            best, best_z = entry, z
    if best is not None:
        best[2] = _retype(solid, best[2], "RoofSurface")
        fixes["relabelled_roof"] += 1


def _fix_orientation(solid, faces, verts, fixes):
    """Reverse a wrongly-wound Ground/Roof ring only when closure improves.

    An upward ground that reversal cannot fix is a stray sliver rather than a
    winding error, so it is dropped instead; a downward roof is left alone,
    because most are mislabelled soffits whose reversal would open the solid.
    """
    dropped = set()
    for idx, entry in enumerate(faces):
        stype = _surface_type(solid, entry[2])
        n = face_normal([verts[i] for i in entry[1][0]])
        if n is None or not ((stype == "GroundSurface" and n[2] > GROUND_NZ_MAX)
                             or (stype == "RoofSurface" and n[2] <= 0)):
            continue

        # score against the faces still standing, not the original list
        live = [f for j, f in enumerate(faces) if j not in dropped]
        flipped = [r[::-1] for r in entry[1]]
        trial = [[e[0], flipped, e[2]] if e is entry else e for e in live]
        if _closure_score(trial) < _closure_score(live):
            entry[1] = flipped
            fixes["reversed_ring"] += 1
        elif stype == "GroundSurface":
            fixes["dropped_unfixable_ground"] += 1
            dropped.add(idx)
        else:
            fixes["left_downward_roof"] += 1
    return [f for j, f in enumerate(faces) if j not in dropped]


def transform_object(obj, vertices):
    """Repair one CityObject in place. Returns ``(keep, fixes)``.

    ``keep`` is False when the object has no Solid, loses every face, or ends up
    without one of :data:`REQUIRED_SEMANTICS`.
    """
    fixes = Counter()
    solid = next((g for g in obj.get("geometry") or []
                  if g.get("type") == "Solid"), None)
    if solid is None:
        return False, fixes

    faces = _read_faces(solid)
    before = _closure_score(faces)
    faces = _repair_rings(faces, fixes)
    faces = _drop_zero_area(faces, vertices, fixes)
    if not faces:
        return False, fixes
    _relabel_roof(solid, faces, vertices, fixes)
    faces = _fix_orientation(solid, faces, vertices, fixes)
    if not faces:
        return False, fixes
    _write_faces(solid, faces)

    # Removing a zero-area face can open a shell it was quietly pairing edges
    # for. That is accepted -- the Levi graph has no watertightness to lose and
    # a normal-less face node is worse -- but it must never be silent.
    after = _closure_score(faces)
    if after > before:
        fixes["closure_worsened"] += 1
        if before == 0:
            fixes["closure_broke_watertight"] += 1
    elif after < before:
        fixes["closure_improved"] += 1

    present = {_surface_type(solid, e[2]) for e in faces}
    missing = [t for t in REQUIRED_SEMANTICS if t not in present]
    if missing:
        for t in missing:
            fixes[f"missing_{t}"] += 1
        return False, fixes
    return True, fixes


def transform_cityjson(cj, drop_ids=frozenset()):
    """Cleaned copy of a CityJSON dict.

    Returns ``(cityjson_or_None, stats, dropped_ids)``. ``drop_ids`` removes
    objects regardless of their own geometry, which is how ``LOD1_synth`` is
    held to the LOD2 drop set.
    """
    vertices = world_vertices(cj)
    stats, kept, dropped = Counter(), {}, set()

    for oid, obj in (cj.get("CityObjects") or {}).items():
        if oid in drop_ids:
            dropped.add(oid)
            stats["dropped_by_pair"] += 1
            continue
        keep, fixes = transform_object(obj, vertices)
        stats.update(fixes)
        if keep:
            kept[oid] = obj
        else:
            dropped.add(oid)
            stats["dropped_objects"] += 1
    stats["kept_objects"] += len(kept)
    if not kept:
        return None, stats, dropped
    return compact_vertices({**cj, "CityObjects": kept}), stats, dropped


# ==============================================================================
# Dataset walk
# ==============================================================================

def _transform_folder(src_root, out_root, drop_map=None, record_drops=False):
    """Clean one LOD folder, mirroring its layout. Returns ``(stats, drops)``."""
    stats, drops = Counter(), {}
    for src in sorted(p for p in src_root.rglob("*") if p.is_file()):
        rel = src.relative_to(src_root)
        try:
            with open(src, "r", encoding="utf-8") as fh:
                cj = json.load(fh)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            cj = None
        if cj is None or cj.get("type") != "CityJSON":
            (out_root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out_root / rel)
            continue

        pinned = (drop_map or {}).get(str(rel), frozenset())
        try:
            out, file_stats, dropped = transform_cityjson(cj, pinned)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.error("failed on %s: %s", rel, exc)
            stats["errors"] += 1
            continue
        stats.update(file_stats)
        if record_drops and dropped:
            drops[str(rel)] = dropped
        if out is None:
            logger.info("no object survived in %s", rel)
            stats["files_emptied"] += 1
            continue
        _write(out_root / rel, out)
        stats["files"] += 1
    return stats, drops


def regenerate_synth(lod2_dir, out_dir, roof_percentile=DEFAULT_ROOF_PERCENTILE):
    """Derive ``LOD1_synth`` from an already-cleaned ``LOD2`` folder.

    Ordering matters. ``convert_to_lod1`` returns an object untouched when it
    finds no ``RoofSurface``, so running it before the repair left 103 objects
    in ``LOD1_synth`` still tagged lod 2 and byte-identical to their LOD2
    original -- a "pair" that was the identity function. Converting from the
    cleaned LOD2, where those roofs are labelled, converts them properly.

    The conversion is cleaned in the same pass, because ``convert_to_lod1``
    emits a degenerate ``[a, b, b, a]`` wall wherever a footprint edge
    collapses; without this the synthetic folder reintroduces defects the LOD2
    folder no longer has.
    """
    stats = Counter()
    for src in sorted(p for p in lod2_dir.rglob("*") if p.is_file()):
        rel = src.relative_to(lod2_dir)
        try:
            with open(src, "r", encoding="utf-8") as fh:
                cj = json.load(fh)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            cj = None
        if cj is None or cj.get("type") != "CityJSON":
            (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out_dir / rel)
            continue

        try:
            synth = convert_to_lod1(deepcopy(cj), roof_percentile)
            cleaned, file_stats, dropped = transform_cityjson(synth)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.error("synth conversion failed on %s: %s", rel, exc)
            stats["errors"] += 1
            continue
        stats.update(file_stats)
        if dropped:
            # Would desynchronise the pair, so it is surfaced rather than logged
            # at debug and forgotten.
            logger.warning("%s: %d synthetic objects dropped, pairing broken: %s",
                           rel, len(dropped), sorted(dropped)[:5])
        if cleaned is None:
            stats["files_emptied"] += 1
            continue
        _write(out_dir / rel, cleaned)
        stats["files"] += 1
    return stats


def process_dataset(root, out_dir, folders=DEFAULT_FOLDERS):
    """Clean each LOD folder under ``root`` into ``out_dir``.

    LOD2 is processed first so ``LOD1_synth`` can be pinned to its drop set.
    """
    present = [f for f in folders if (root / f).is_dir()]
    if not present:
        raise FileNotFoundError(f"No LOD folders found under {root}")
    ordered = ([f for f in present if f == "LOD2"]
               + [f for f in present if f != "LOD2"])

    totals, drop_map = {}, {}
    for name in ordered:
        pin = drop_map if name == SYNTH_DIR else None
        stats, drops = _transform_folder(root / name, out_dir / name, pin,
                                         record_drops=(name == "LOD2"))
        totals[name] = dict(stats)
        if name == "LOD2":
            drop_map = drops
        logger.info("%s: %d files, %d objects kept, %d dropped",
                    name, stats["files"], stats["kept_objects"],
                    stats["dropped_objects"] + stats["dropped_by_pair"])
    return totals


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="Dataset root holding LOD1/, LOD2/ and LOD1_synth/.")
    ap.add_argument("--out", required=True, help="Where to write the cleaned copy.")
    ap.add_argument("--folders", nargs="*", default=list(DEFAULT_FOLDERS),
                    help=f"LOD folders to clean (default: {' '.join(DEFAULT_FOLDERS)}). "
                         "Pass with no values to only regenerate the synthetic pair.")
    ap.add_argument("--regenerate-synth", action="store_true",
                    help=f"Derive {SYNTH_DIR}/ from the cleaned LOD2 instead of cleaning the "
                         "input's copy, and overwrite whatever is there. Required for the "
                         "roofless objects convert_to_lod1 would otherwise pass through.")
    ap.add_argument("--roof-percentile", type=float, default=DEFAULT_ROOF_PERCENTILE,
                    help=f"Cap quantile for the derived LOD1 (default: {DEFAULT_ROOF_PERCENTILE}).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root, out_dir = Path(args.root).resolve(), Path(args.out).resolve()
    if not root.is_dir():
        sys.exit(f"Dataset root not found: {root}")

    folders = [f for f in args.folders if not (args.regenerate_synth and f == SYNTH_DIR)]
    for name in folders:
        d = out_dir / name
        if d.exists() and any(d.iterdir()):
            sys.exit(f"Output folder already exists and is not empty: {d}")
        if (root / name) == d:
            sys.exit("Refusing to overwrite the input in place; choose another --out.")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        totals = process_dataset(root, out_dir, tuple(folders)) if folders else {}
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    if args.regenerate_synth:
        lod2 = out_dir / "LOD2"
        if not lod2.is_dir():
            sys.exit(f"Cannot derive {SYNTH_DIR}/ without a cleaned LOD2 at {lod2}")
        logger.info("deriving %s/ from the cleaned LOD2 (overwriting)", SYNTH_DIR)
        stats = regenerate_synth(lod2, out_dir / SYNTH_DIR, args.roof_percentile)
        totals[SYNTH_DIR] = dict(stats)
        logger.info("%s: %d files, %d objects, %d dropped",
                    SYNTH_DIR, stats["files"], stats["kept_objects"],
                    stats["dropped_objects"])

    # An incremental run (e.g. --folders with no values) must not erase the
    # provenance of the folders it left alone.
    path = out_dir / "transform_manifest.json"
    previous = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8")).get("totals", {})
        except (OSError, json.JSONDecodeError):
            logger.warning("could not read the existing manifest; it will be replaced")
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "source": str(root),
        "args": vars(args),
        "totals": {**previous, **totals},
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("manifest written to %s", path)


if __name__ == "__main__":
    main()
