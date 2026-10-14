"""Build orthophoto mosaics per imagery year and CityJSON file of a dataset subset.

Each building is assigned the imagery year in ``--years`` closest to its 3DBAG
point-cloud year (`src.ortho_tiles.assign_year`), recorded in
``<out>/year_assignment.json``. Each ``<subset>/<lod_out>/<Source>/<name>.city.json``
then gets one ``<out>/<year>/<Source>/<name>.tif`` per year its buildings use:
that year's raw 1 km tiles pasted together over the union of those buildings'
crop windows (`src.dataset.ortho.crop_bounds` at ``--beta-max``), at the raw
tiles' native resolution and pixel grid, so no pixel is resampled. Only the
256 px blocks that intersect a window are written (GDAL sparse files), which
keeps a sparse subset such as ``mini_cleaner`` small.

The windows use the same LOD1 frame the dataset uses, so ``--lod-in`` and the
margins must match the training config; they are recorded in
``ortho_manifest.json`` and `MeshDataset` checks them before reading.

Usage:
    python -m src.ortho_parse "data/The Hague/mini_cleaner" --years 2022 2023 \\
        --raw "data/The Hague/raw/ortho" --out "data/The Hague/mini_cleaner/ortho"
"""
import argparse
import collections
import json
import logging
import math
import multiprocessing as mp
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.windows import Window

from src.dataset.ortho import crop_bounds
from src.filter_cityjson import _git_commit
from src.ortho_tiles import (GSD_M, TILE_M, assign_year, building_frames, pw_datum_by_id,
                             readme_text, tile_name, tiles_for_bounds)

logger = logging.getLogger(__name__)

OVERVIEWS = [2, 4, 8, 16]
BETAS_REPORTED = (0.05, 0.1, 0.25)


def _coverage(window, tile_dir, year):
    """Fraction of a window's area that falls on raw tiles present on disk."""
    xmin, ymin, xmax, ymax = window
    covered = 0.0
    for tx, ty in tiles_for_bounds(window):
        if (tile_dir / tile_name(tx, ty, year)).is_file():
            covered += ((min(xmax, tx + TILE_M) - max(xmin, tx))
                        * (min(ymax, ty) - max(ymin, ty - TILE_M)))
    return covered / ((xmax - xmin) * (ymax - ymin))


def mosaic_file(windows, tile_dir, out_path, year, gsd=GSD_M, block=256, chunk=4096,
                quality=90):
    """Paste raw tiles into one sparse, tiled YCbCr-JPEG GeoTIFF over `windows`.

    All arithmetic is done in integer centimetres on the raw tiles' pixel grid,
    so tile pixels are copied, never resampled, and no float drift accumulates.

    Args:
        windows: {id: (xmin, ymin, xmax, ymax)} in metres.
        tile_dir: folder of raw tiles for `year`.
        out_path: GeoTIFF to write.
        gsd: raw ground sampling distance, metres (0.08 for the real tiles).
        block, chunk: output block size and read chunk, pixels; chunk must be a
            multiple of block.
        quality: JPEG quality of the output and its overviews.

    Returns:
        dict: coverage {id: fraction on present tiles}, missing_tiles, bytes,
        width, height.
    """
    g = round(gsd * 100)
    if TILE_M * 100 % g or chunk % block:
        raise ValueError(f"gsd {gsd} must divide {TILE_M} m and chunk must be a multiple of block.")
    tile_dir, out_path = Path(tile_dir), Path(out_path)

    boxes = list(windows.values())
    needed = set().union(*(tiles_for_bounds(b) for b in boxes))
    x0 = math.floor(min(b[0] for b in boxes) * 100 / g) * g
    y0 = math.floor(min(b[1] for b in boxes) * 100 / g) * g
    x1 = math.ceil(max(b[2] for b in boxes) * 100 / g) * g
    y1 = math.ceil(max(b[3] for b in boxes) * 100 / g) * g
    width, height = (x1 - x0) // g, (y1 - y0) // g

    # Which output blocks any window touches; everything else stays unallocated.
    mask = np.zeros((math.ceil(height / block), math.ceil(width / block)), bool)
    for xmin, ymin, xmax, ymax in boxes:
        c0, c1 = math.floor((xmin * 100 - x0) / g), math.ceil((xmax * 100 - x0) / g)
        r0, r1 = math.floor((y1 - ymax * 100) / g), math.ceil((y1 - ymin * 100) / g)
        mask[r0 // block:(r1 - 1) // block + 1, c0 // block:(c1 - 1) // block + 1] = True

    profile = dict(driver="GTiff", width=width, height=height, count=3, dtype="uint8",
                   crs="EPSG:28992", transform=from_origin(x0 / 100, y1 / 100, gsd, gsd),
                   tiled=True, blockxsize=block, blockysize=block, compress="JPEG",
                   photometric="YCBCR", jpeg_quality=quality, interleave="pixel",
                   sparse_ok=True, bigtiff="IF_SAFER")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    missing, sources = set(), {}
    try:
        with rasterio.open(out_path, "w", **profile) as dst:
            for cr in range(0, height, chunk):
                for cc in range(0, width, chunk):
                    h, w = min(chunk, height - cr), min(chunk, width - cc)
                    blocks = mask[cr // block:(cr + h - 1) // block + 1,
                                  cc // block:(cc + w - 1) // block + 1]
                    if not blocks.any():
                        continue
                    cx0, cy1 = x0 + cc * g, y1 - cr * g
                    cx1, cy0 = cx0 + w * g, cy1 - h * g
                    arr = np.zeros((3, h, w), np.uint8)
                    for tx, ty in sorted(tiles_for_bounds((cx0 / 100, cy0 / 100,
                                                           cx1 / 100, cy1 / 100))):
                        name = tile_name(tx, ty, year)
                        if name not in sources:
                            path = tile_dir / name
                            if not path.is_file():
                                # Read chunks span the windows' bounding box and
                                # can touch tiles no window needs.
                                if (tx, ty) in needed:
                                    missing.add(name)
                                continue
                            src = rasterio.open(path)
                            if (round(src.transform.a * 100) != g or
                                    (round(src.transform.c), round(src.transform.f)) != (tx, ty)):
                                src.close()
                                raise ValueError(f"{path} is not a {gsd} m tile anchored at "
                                                 f"({tx}, {ty}): transform {src.transform}")
                            sources[name] = src
                        src = sources.get(name)
                        if src is None:
                            continue
                        ix0, ix1 = max(cx0, tx * 100), min(cx1, (tx + TILE_M) * 100)
                        iy0, iy1 = max(cy0, (ty - TILE_M) * 100), min(cy1, ty * 100)
                        if ix1 <= ix0 or iy1 <= iy0:
                            continue
                        ww, hh = (ix1 - ix0) // g, (iy1 - iy0) // g
                        data = src.read((1, 2, 3), window=Window(
                            (ix0 - tx * 100) // g, (ty * 100 - iy1) // g, ww, hh))
                        dc0, dr0 = (ix0 - cx0) // g, (cy1 - iy1) // g
                        arr[:, dr0:dr0 + hh, dc0:dc0 + ww] = data
                    for bi, bj in np.argwhere(blocks):
                        r, c = bi * block, bj * block
                        bh, bw = min(block, h - r), min(block, w - c)
                        dst.write(arr[:, r:r + bh, c:c + bw],
                                  window=Window(cc + c, cr + r, bw, bh))
    finally:
        for src in sources.values():
            src.close()

    # Overviews let a large building's crop decode a few blocks instead of
    # tens of megapixels. Without these options GDAL writes them uncompressed RGB.
    with rasterio.Env(COMPRESS_OVERVIEW="JPEG", PHOTOMETRIC_OVERVIEW="YCBCR",
                      INTERLEAVE_OVERVIEW="PIXEL", JPEG_QUALITY_OVERVIEW=quality):
        with rasterio.open(out_path, "r+") as dst:
            dst.build_overviews(OVERVIEWS, Resampling.average)

    return {"coverage": {oid: _coverage(b, tile_dir, year) for oid, b in windows.items()},
            "missing_tiles": sorted(missing), "bytes": out_path.stat().st_size,
            "width": width, "height": height}


def _parse_one(job):
    """Mosaic one CityJSON file: one sparse GeoTIFF per imagery year it uses.

    Top-level so a spawned Pool can pickle it.
    """
    rel, lod1_path, lod2_path, out_dir, cfg = job
    try:
        frames, without_lod1 = building_frames(lod1_path, lod2_path,
                                               cfg["margin_lo"], cfg["margin_hi"])
        datum = pw_datum_by_id(lod2_path)
        assignment = {oid: assign_year(datum[oid], cfg["years"]) for oid in frames}
        result = {"rel": rel, "assignment": assignment, "without_lod1": without_lod1,
                  "need_beta": [f.need_beta for f in frames.values()],
                  "pw_datum": dict(collections.Counter(datum[oid] for oid in frames)),
                  "written": [], "bytes": 0, "missing_tiles": [], "partial": {}}
        name = Path(rel).name.split(".")[0] + ".tif"
        for year in sorted(set(assignment.values())):
            windows = {oid: crop_bounds(f.center, f.scale, cfg["beta_max"])
                       for oid, f in frames.items() if assignment[oid] == year}
            out_path = out_dir / str(year) / Path(rel).with_name(name)
            stats = mosaic_file(windows, cfg["raw_dir"] / str(year), out_path, year,
                                gsd=cfg["gsd"])
            result["written"].append(year)
            result["bytes"] += stats["bytes"]
            result["missing_tiles"] += stats["missing_tiles"]
            result["partial"].update({k: v for k, v in stats["coverage"].items()
                                      if v < 1 - 1e-9})
        return result
    except (OSError, KeyError, IndexError, ValueError, rasterio.errors.RasterioError) as exc:
        return {"rel": rel, "error": f"{type(exc).__name__}: {exc}"}


def parse_subset(subset_root, raw_dir, out_dir, years=(2022, 2023), beta_max=0.25,
                 lod_in="LOD1", lod_out="LOD2", margin_lo=(0.0, 0.0, 0.0),
                 margin_hi=(0.0, 0.0, 0.1), gsd=GSD_M, workers=1):
    """Write per-year mosaics for every `lod_out` CityJSON file of a subset.

    Also writes ``<out_dir>/year_assignment.json`` ({object id: imagery year}),
    which `MeshDataset` reads to find each building's mosaic.

    Returns:
        dict: totals -- files, buildings (overall and per year), mosaics written
        per year, errors, partial coverage per id, missing tiles, ids duplicated
        across files, bytes written, the ``b3_pw_datum`` distribution, and how
        far LOD2 reaches past the unit box (``need_beta`` percentiles and the
        fraction of buildings needing more than each reported beta).
    """
    subset_root, out_dir = Path(subset_root), Path(out_dir)
    years = sorted(int(y) for y in years)
    cfg = {"margin_lo": list(margin_lo), "margin_hi": list(margin_hi), "beta_max": beta_max,
           "raw_dir": Path(raw_dir), "years": years, "gsd": gsd}
    jobs = []
    for lod2_path in sorted((subset_root / lod_out).rglob("*.json")):
        rel = lod2_path.relative_to(subset_root / lod_out)
        jobs.append((rel.as_posix(), subset_root / lod_in / rel, lod2_path, out_dir, cfg))
    if not jobs:
        raise FileNotFoundError(f"No CityJSON under {subset_root / lod_out}")

    totals = {"files": len(jobs), "files_written": {str(y): 0 for y in years}, "buildings": 0,
              "buildings_per_year": {str(y): 0 for y in years}, "without_lod1": 0,
              "errors": {}, "partial_coverage": {}, "missing_tiles": set(),
              "bytes_written": 0, "pw_datum": collections.Counter()}
    assignment, duplicates, need = {}, set(), []
    pool = mp.Pool(workers) if workers > 1 else None
    try:
        results = pool.imap_unordered(_parse_one, jobs) if pool else map(_parse_one, jobs)
        for done, res in enumerate(results, 1):
            if "error" in res:
                logger.error("failed on %s: %s", res["rel"], res["error"])
                totals["errors"][res["rel"]] = res["error"]
                continue
            for oid, year in res["assignment"].items():
                if oid in assignment:
                    duplicates.add(oid)
                assignment[oid] = year
                totals["buildings_per_year"][str(year)] += 1
            totals["buildings"] += len(res["assignment"])
            totals["without_lod1"] += res["without_lod1"]
            totals["pw_datum"].update(res["pw_datum"])
            need += res["need_beta"]
            for year in res["written"]:
                totals["files_written"][str(year)] += 1
            totals["bytes_written"] += res["bytes"]
            totals["missing_tiles"].update(res["missing_tiles"])
            totals["partial_coverage"].update(res["partial"])
            if done % 25 == 0:
                logger.info("%d/%d files", done, len(jobs))
    finally:
        if pool:
            pool.close()
            pool.join()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "year_assignment.json").write_text(json.dumps(assignment), encoding="utf-8")
    need = np.asarray(need or [0.0])
    totals.update(
        missing_tiles=sorted(totals["missing_tiles"]),
        pw_datum=dict(totals["pw_datum"]),
        duplicate_ids=sorted(duplicates),
        need_beta_pct={p: float(np.percentile(need, p)) for p in (50, 90, 99)},
        need_beta_max=float(need.max()),
        need_beta_frac_over={str(b): float((need > b).mean()) for b in BETAS_REPORTED},
    )
    return totals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subset", type=Path, help="subset root holding LOD1/ and LOD2/")
    parser.add_argument("--raw", type=Path, required=True, help="raw ortho root (year subfolders)")
    parser.add_argument("--out", type=Path, required=True, help="output folder, must be empty")
    parser.add_argument("--years", type=int, nargs="+", default=[2022, 2023],
                        help="imagery years to choose from per building")
    parser.add_argument("--beta-max", type=float, default=0.25,
                        help="largest crop margin the dataset may request, normalized units")
    parser.add_argument("--lod-in", default="LOD1", help="folder whose meshes define the frame")
    parser.add_argument("--lod-out", default="LOD2")
    parser.add_argument("--margin-lo", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--margin-hi", type=float, nargs=3, default=[0.0, 0.0, 0.1])
    parser.add_argument("--gsd", type=float, default=GSD_M, help="raw tile resolution, metres")
    parser.add_argument("--workers", type=int, default=min(8, mp.cpu_count()))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.subset.is_dir():
        sys.exit(f"subset not found: {args.subset}")
    args.years = sorted(set(args.years))
    missing_years = [y for y in args.years if not (args.raw / str(y)).is_dir()]
    if missing_years:
        sys.exit(f"raw tiles not found for years {missing_years} under {args.raw}")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"output folder is not empty: {args.out}")

    try:
        totals = parse_subset(args.subset, args.raw, args.out, args.years, args.beta_max,
                              args.lod_in, args.lod_out, args.margin_lo, args.margin_hi,
                              args.gsd, args.workers)
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    manifest = {"created": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit(),
                "source": str(args.subset.resolve()), "raw": str(args.raw.resolve()),
                "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "totals": totals}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ortho_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    body = [
        "## This folder",
        "",
        f"`<year>/` holds one mosaic per CityJSON file of `{args.subset.name}/{args.lod_out}`",
        "that has buildings assigned that year: same relative path and name with `.tif`.",
        "`year_assignment.json` maps every building id to its year. Tiled 256 px YCbCr JPEG",
        "GeoTIFF at the raw resolution, overviews 2/4/8/16. Only blocks under a crop",
        "window of that year's buildings are stored; everything else reads as 0.",
        "",
        f"Crop windows: frame of `{args.lod_in}` (margins lo {args.margin_lo}, hi "
        f"{args.margin_hi}) plus beta_max = {args.beta_max}. The dataset refuses a larger",
        "`image_margin` or a different `lod_in`/margins (see `ortho_manifest.json`).",
        "",
        f"Mosaics written per year: {totals['files_written']} (of {totals['files']} files); "
        f"buildings per year: {totals['buildings_per_year']}; partially covered: {len(totals['partial_coverage'])}; "
        f"missing raw tiles: {len(totals['missing_tiles'])}; errors: {len(totals['errors'])}.",
        f"Produced by `python -m src.ortho_parse` at commit {manifest['git_commit']}.",
    ]
    (args.out / "README.md").write_text(
        readme_text(f"Orthophoto mosaics: {args.subset.name}", args.years, totals["pw_datum"],
                    body),
        encoding="utf-8")
    logger.info("wrote %d mosaics (%.1f MB) to %s", sum(totals["files_written"].values()),
                totals["bytes_written"] / 1e6, args.out)
    return totals


if __name__ == "__main__":
    main()
