#!/usr/bin/env python3
"""Groups of LOD1 buildings the mesh transformer cannot tell apart.

Why this exists
---------------
`mesh-v5-full-opt-coord` generates at roughly its do-nothing floor
(`gen_chamfer` 0.091 m against a `gen_chamfer_floor` of 0.103 m on test), and
the open question is whether that is a model failure or the task's own ceiling:
LOD1 is lossy, so a given LOD1 may be consistent with many different LOD2s.

That ceiling is measurable without a model. Find LOD1s that are *identical as
far as the model is concerned*, then look at how much their LOD2s disagree.
Whatever that disagreement is, no model conditioned on LOD1 alone can do better
-- it is the conditional entropy of the task, in the model's own units.

The equivalence relation
------------------------
Two objects are the same condition iff their LOD1 token sequences agree. That
is a statement about `normalize_to_unit_box` + `quantize`, not about metres, so
this module reuses those functions rather than reimplementing a similarity
measure. Three consequences, all deliberate and all pinned by tests:

  * Translation is erased -- the frame centres on the object's own box.
  * *Uniform* scale is erased -- the frame divides by the longest box axis, so
    a 10 m and a 20 m building of the same proportions are one condition.
  * `num_bins` IS the tolerance. Two buildings agreeing to within a bin are
    near-identical by construction (a bin is 1/127 of the longest axis, ~0.16 m
    on a 20 m building). Loosening the tolerance means coarsening the grid, not
    adding an epsilon and a nearest-neighbour search.

Wireframes only. Nothing here triangulates: the descriptor needs the vertex
*set*, and earcutting every surface of 259k objects to throw the faces away
would dominate the runtime.

Rejected alternatives: `trimesh.Trimesh.identifier` is built for exactly this,
but it is rotation- and scale-invariant by construction and needs a
well-formed mesh, which a good share of this corpus is not. Shapely's
`hausdorff_distance` works on footprint polygons, but LOD1 here is a solid, the
descriptor needs no boolean geometry, and shapely is not in requirements.txt.

Usage:
    python -m src.analysis.lod1_duplicates --dataset-dir "data/The Hague/cleaner"
"""
import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import directed_hausdorff

from src.dataset.dataset import _iter_surfaces
from src.dataset.mesh_dataset import NUM_BINS, dequantize, normalize_to_unit_box, quantize
from src.filter_cityjson import world_vertices

logger = logging.getLogger(__name__)

# The training frame, from configs/mesh-v5-full-coord.yaml. Only +z is lifted:
# the LOD2 ridge rises above the LOD1 box, which is a median roof height.
MARGIN_LO = (0.0, 0.0, 0.0)
MARGIN_HI = (0.0, 0.0, 0.1)


# ----------------------------------------------------------------------
# Wireframe extraction
# ----------------------------------------------------------------------

def object_vertices(city_obj, vertices):
    """The unique world-coordinate points one CityObject references.

    Every ring of every surface contributes, interiors included -- a courtyard
    has corners and they are part of the shape. Faces are not built: this is
    the vertex set, which is all the descriptor needs.

    Args:
        city_obj (dict): a CityObject, as stored under ``CityObjects``.
        vertices (np.ndarray): [V, 3] file vertices already in world
            coordinates (see :func:`src.filter_cityjson.world_vertices`).

    Returns:
        np.ndarray: [U, 3] unique points, empty [0, 3] if the object has no
        readable surface.
    """
    idx = set()
    for geom in city_obj.get("geometry", []) or []:
        for rings, _ in _iter_surfaces(geom):
            for ring in rings:
                idx.update(int(i) for i in ring)
    if not idx:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(vertices, dtype=float)[sorted(idx)]


def iter_file_vertices(path):
    """``{object_id: vertices [U, 3]}`` for one CityJSON file.

    Raises:
        OSError: the file cannot be read.
        ValueError: the file is not parseable CityJSON.
    """
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cj = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    try:
        verts = world_vertices(cj)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} has no usable 'vertices' array: {exc}") from exc

    out = {}
    for oid, obj in cj.get("CityObjects", {}).items():
        pts = object_vertices(obj, verts)
        if len(pts):
            out[oid] = pts
    return out


def scan_lod_vertices(dataset_dir, lod_dir, max_files=None):
    """``{object_id: (vertices, source_file)}`` for every object under a LOD.

    Mirrors `src.dataset.mesh_dataset._scan_lod_dir`'s walk order and its
    tolerance for a single corrupt tile, but keeps the file each object came
    from -- the explorer has to re-open it to draw the object.
    """
    root = Path(dataset_dir) / lod_dir
    if not root.is_dir():
        raise FileNotFoundError(f"No LOD folder {root!s} under {dataset_dir!r}")

    out, seen = {}, 0
    for parent, _, files in sorted(os.walk(root)):
        for name in sorted(files):
            if not name.lower().endswith((".json", ".city.json")):
                continue
            if max_files is not None and seen >= max_files:
                return out
            seen += 1
            path = Path(parent) / name
            try:
                for oid, pts in iter_file_vertices(path).items():
                    out[oid] = (pts, str(path.relative_to(dataset_dir)))
            except (OSError, ValueError) as exc:
                logger.warning("skipping %s: %s", name, exc)
    return out


# ----------------------------------------------------------------------
# The descriptor
# ----------------------------------------------------------------------

def shape_key(vertices, num_bins=NUM_BINS, ref=None,
              margin_lo=MARGIN_LO, margin_hi=MARGIN_HI):
    """``(key, quantized)`` -- the model's own view of a vertex set.

    Args:
        vertices (np.ndarray): [V, 3] points in metres.
        num_bins (int): grid resolution. This is the tolerance knob: coarser
            bins merge buildings that a finer grid separates.
        ref (np.ndarray, optional): points whose box defines the frame. None
            uses `vertices` itself, which is right for LOD1. An LOD2 target
            must pass its own LOD1 here, exactly as `MeshDataset` does --
            rescaling the target by its own box would erase the height
            difference that distinguishes one roof from another.
        margin_lo, margin_hi: per-axis headroom, as in `normalize_to_unit_box`.

    Returns:
        tuple: (key bytes, quantized [U, 3] int64 sorted lexicographically).
        Equal keys mean the two objects produce the same condition tokens.
    """
    verts = np.asarray(vertices, dtype=float)
    if verts.size == 0:
        return b"", np.zeros((0, 3), dtype=np.int64)

    normed, _, _ = normalize_to_unit_box(
        verts, ref=verts if ref is None else ref,
        margin_lo=margin_lo, margin_hi=margin_hi)
    q = quantize(normed, num_bins)
    # Unique + lexsort makes the key independent of how the file happened to
    # order its vertices, and of a corner arriving once per touching face.
    q = np.unique(q, axis=0)
    return q.astype(np.int64).tobytes(), q.astype(np.int64)


def frame_scale(vertices, margin_lo=MARGIN_LO, margin_hi=MARGIN_HI):
    """The per-axis metres-per-unit of a vertex set's own normalization frame."""
    _, _, scale = normalize_to_unit_box(
        np.asarray(vertices, dtype=float), margin_lo=margin_lo, margin_hi=margin_hi)
    return np.broadcast_to(np.asarray(scale, dtype=float), (3,)).copy()


# ----------------------------------------------------------------------
# Grouping
# ----------------------------------------------------------------------

def group_identical(keys):
    """Object ids sharing a key, singletons dropped.

    Args:
        keys (dict): ``{object_id: key}`` from :func:`shape_key`.

    Returns:
        list: groups as sorted id lists, largest first, ties broken by the
        first id so a re-run produces the same file.
    """
    buckets = defaultdict(list)
    for oid, key in keys.items():
        buckets[key].append(oid)
    groups = [sorted(ids) for ids in buckets.values() if len(ids) > 1]
    return sorted(groups, key=lambda g: (-len(g), g[0]))


def hausdorff_bins(a, b):
    """Symmetric Hausdorff distance between two quantized point sets, in bins."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0]))


def group_spread(members, targets, scales, num_bins=NUM_BINS):
    """How much the LOD2s of one LOD1-identical group disagree.

    This is the quantity the whole module exists to produce. Every member has
    the same condition, so any disagreement here is irreducible: it is what a
    model conditioned on LOD1 alone must get wrong.

    Bins are the primary unit because they are the model's -- directly
    comparable to `val_bin_mae` and to the 128-bin grid. Metres are reported
    too, per pair, from the mean of the two members' frame scales; the members
    of a group can be different physical sizes (uniform scale is erased by the
    frame), so a group has no single metre scale.

    Args:
        members (list): object ids in the group.
        targets (dict): ``{object_id: quantized LOD2 [U, 3]}``, each quantized
            in its own LOD1 frame.
        scales (dict): ``{object_id: [3] metres per normalized unit}``.
        num_bins (int): the grid `targets` was quantized on.

    Returns:
        dict: ``max_bins``, ``mean_bins``, ``max_m``, ``mean_m``, ``n_members``,
        ``n_pairs``. The distances are nan for a group of one -- there is no
        disagreement to measure, and zero would read as agreement.
    """
    n = len(members)
    if n < 2:
        return {"max_bins": float("nan"), "mean_bins": float("nan"),
                "max_m": float("nan"), "mean_m": float("nan"),
                "n_members": n, "n_pairs": 0}

    bins, metres = [], []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = targets[members[i]], targets[members[j]]
            bins.append(hausdorff_bins(a, b))
            s = (np.asarray(scales[members[i]], dtype=float)
                 + np.asarray(scales[members[j]], dtype=float)) / 2.0
            metres.append(hausdorff_bins(dequantize(a, num_bins) * s,
                                         dequantize(b, num_bins) * s))
    return {"max_bins": float(np.max(bins)), "mean_bins": float(np.mean(bins)),
            "max_m": float(np.max(metres)), "mean_m": float(np.mean(metres)),
            "n_members": n, "n_pairs": len(bins)}


# ----------------------------------------------------------------------
# Corpus pass
# ----------------------------------------------------------------------

def find_groups(dataset_dir, lod_in="LOD1", lod_out="LOD2", num_bins=NUM_BINS,
                margin_lo=MARGIN_LO, margin_hi=MARGIN_HI, max_files=None):
    """Every LOD1-identical group in a corpus, with its LOD2 spread.

    Only objects present at *both* LODs are considered, matching
    `MeshDataset`'s ``set(meshes_in) & set(meshes_out)``.

    Returns:
        tuple: (groups, stats). Each group is a dict with ``key`` (hex digest),
        ``members``, ``files`` and the :func:`group_spread` fields. `stats`
        records corpus totals so a run is self-describing.
    """
    lod1 = scan_lod_vertices(dataset_dir, lod_in, max_files)
    lod2 = scan_lod_vertices(dataset_dir, lod_out, max_files)
    ids = sorted(set(lod1) & set(lod2))
    logger.info("%d objects at %s, %d at %s, %d paired",
                len(lod1), lod_in, len(lod2), lod_out, len(ids))

    keys, targets, scales, files = {}, {}, {}, {}
    for oid in ids:
        cond, cond_file = lod1[oid]
        key, _ = shape_key(cond, num_bins, margin_lo=margin_lo, margin_hi=margin_hi)
        if not key:
            continue
        keys[oid] = key
        # The target is quantized in the CONDITION's frame, as in training.
        _, targets[oid] = shape_key(lod2[oid][0], num_bins, ref=cond,
                                    margin_lo=margin_lo, margin_hi=margin_hi)
        scales[oid] = frame_scale(cond, margin_lo, margin_hi)
        files[oid] = {"lod1": cond_file, "lod2": lod2[oid][1]}

    groups = []
    for members in group_identical(keys):
        spread = group_spread(members, targets, scales, num_bins)
        groups.append({"key": _digest(keys[members[0]]), "members": members,
                       "files": {m: files[m] for m in members}, **spread})
    groups.sort(key=lambda g: (-g["max_bins"], -g["n_members"], g["members"][0]))

    grouped = sum(g["n_members"] for g in groups)
    stats = {"dataset_dir": str(dataset_dir), "lod_in": lod_in, "lod_out": lod_out,
             "num_bins": num_bins, "margin_lo": list(margin_lo),
             "margin_hi": list(margin_hi), "n_paired": len(ids),
             "n_keyed": len(keys), "n_groups": len(groups),
             "n_objects_in_groups": grouped,
             "frac_in_groups": grouped / len(keys) if keys else 0.0}
    return groups, stats


def _digest(key):
    """A short stable name for a key -- the raw bytes are not printable."""
    import hashlib
    return hashlib.sha1(key).hexdigest()[:12]


def write_outputs(groups, stats, out_dir):
    """`groups.json` (full, for the explorer) and `summary.csv` (one row each)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "groups.json", "w", encoding="utf-8") as fh:
        json.dump({"stats": stats, "groups": groups}, fh, indent=1)

    cols = ["key", "n_members", "n_pairs", "max_bins", "mean_bins", "max_m",
            "mean_m", "members"]
    with open(out_dir / "summary.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for g in groups:
            row = {c: g[c] for c in cols if c != "members"}
            writer.writerow({**row, "members": " ".join(g["members"])})
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset-dir", default="data/The Hague/cleaner")
    ap.add_argument("--lod-in", default="LOD1")
    ap.add_argument("--lod-out", default="LOD2")
    ap.add_argument("--num-bins", type=int, default=NUM_BINS,
                    help="grid resolution; coarser merges more (the tolerance knob)")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--out-dir", default=None,
                    help="default: outputs/lod1_duplicates/bins<N>")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    groups, stats = find_groups(args.dataset_dir, args.lod_in, args.lod_out,
                                args.num_bins, max_files=args.max_files)
    out = Path(args.out_dir or f"outputs/lod1_duplicates/bins{args.num_bins}")
    write_outputs(groups, stats, out)
    logger.info("%d groups covering %d/%d objects (%.2f%%) -> %s",
                stats["n_groups"], stats["n_objects_in_groups"], stats["n_keyed"],
                100 * stats["frac_in_groups"], out)
    if groups:
        finite = [g["max_m"] for g in groups if np.isfinite(g["max_m"])]
        logger.info("LOD2 spread: median %.3f m, p95 %.3f m, max %.3f m",
                    float(np.median(finite)), float(np.percentile(finite, 95)),
                    float(np.max(finite)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
