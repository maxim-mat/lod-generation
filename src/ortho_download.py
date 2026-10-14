"""Download the raw 8 cm orthophoto tiles one or more dataset subsets need.

For every building with both LOD1 and LOD2 in the given subsets, the crop window
the dataset can request (`src.dataset.ortho.crop_bounds` at ``--beta-max``, in
the ``--lod-in`` frame) is mapped to the 1 x 1 km tiles it touches; only those
tiles are fetched. Their download paths need the flight block each tile was
delivered in, looked up in the portal's sheet index (``--sheet-index``; its
asset name carries a build hash, so it may need updating).

Downloads resume: a tile whose size on disk matches the server's
Content-Length is skipped, partial files are written to ``.part`` and renamed
only after the GeoTIFF opens with the expected size and anchor.

Usage:
    python -m src.ortho_download "data/The Hague/cleaner" "data/The Hague/mini_cleaner" \\
        --out "data/The Hague/raw/ortho" --check      # one HEAD request, then exit
    ... --dry-run                                    # tile count and size estimate
    ...                                              # the download itself
"""
import argparse
import collections
import hashlib
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import rasterio

from src.dataset.ortho import crop_bounds
from src.filter_cityjson import _git_commit
from src.ortho_tiles import (building_frames, load_sheet_index, pw_datum_by_id, readme_text,
                             sheet_for_tile, tile_name, tile_url, tiles_for_bounds)

logger = logging.getLogger(__name__)

DEFAULT_SHEET_INDEX = ("https://basisdata.nl/hwh-ortho/AUX/bladwijzer/assets/"
                       "BM2022-CzCy-9pa.geojson")
# Mean of 78 HEAD-sampled 1 km tiles (2021 flight, 2026-10-14). Only used for
# the --dry-run size estimate.
MEAN_TILE_MB = 38.4
TILE_PX = 12500
READ_CHUNK = 1 << 20


def _plan_one(job):
    """Tiles and point-cloud years of one file. Top-level for a spawned Pool."""
    lod1, lod2, beta, margin_lo, margin_hi = job
    frames, _ = building_frames(lod1, lod2, margin_lo, margin_hi)
    tiles = set()
    for f in frames.values():
        tiles |= tiles_for_bounds(crop_bounds(f.center, f.scale, beta))
    datum = pw_datum_by_id(lod2)
    return tiles, {oid: datum[oid] for oid in frames}


def plan_downloads(subset_roots, beta_max, lod_in, lod_out, margin_lo, margin_hi,
                   workers=1, max_files=None):
    """Tiles every crop window of the given subsets touches.

    Returns:
        tuple: (set of (xmin, ymax) tile corners, Counter of ``b3_pw_datum`` over
        the unique buildings planned).
    """
    jobs = []
    for root in map(Path, subset_roots):
        files = sorted((root / lod_out).rglob("*.json"))[:max_files]
        jobs += [(root / lod_in / p.relative_to(root / lod_out), p, beta_max,
                  list(margin_lo), list(margin_hi)) for p in files]
    tiles, datum = set(), {}
    pool = mp.Pool(workers) if workers > 1 else None
    try:
        results = pool.imap_unordered(_plan_one, jobs) if pool else map(_plan_one, jobs)
        for done, (t, d) in enumerate(results, 1):
            tiles |= t
            datum.update(d)
            if done % 25 == 0:
                logger.info("planned %d/%d files", done, len(jobs))
    finally:
        if pool:
            pool.close()
            pool.join()
    return tiles, collections.Counter(datum.values())


def _validate(path, origin, tile_px):
    """Raise ValueError unless `path` is a 3-band tile of `tile_px`² anchored at `origin`."""
    with rasterio.open(path) as src:
        anchor = (round(src.transform.c), round(src.transform.f))
        if (src.width, src.height, src.count) != (tile_px, tile_px, 3) or anchor != tuple(origin):
            raise ValueError(f"unexpected tile: {src.width}x{src.height}x{src.count} at {anchor}, "
                             f"expected {tile_px}x{tile_px}x3 at {tuple(origin)}")


def download_tile(url, dest, origin, opener=urllib.request.urlopen, sleep=time.sleep,
                  retries=4, backoff=5.0, tile_px=TILE_PX, timeout=120):
    """Fetch one tile to `dest`, resumably.

    Returns:
        str: "skipped" if `dest` already has the server's size, else "downloaded".

    Raises:
        RuntimeError: still failing after `retries` attempts (network error,
            truncated body, or a body that is not the expected GeoTIFF).
    """
    dest = Path(dest)
    part = dest.with_name(dest.name + ".part")
    last = None
    for attempt in range(retries):
        try:
            with opener(urllib.request.Request(url, method="HEAD"), timeout=timeout) as resp:
                size = int(resp.headers["Content-Length"])
            if dest.is_file() and dest.stat().st_size == size:
                return "skipped"
            dest.parent.mkdir(parents=True, exist_ok=True)
            with opener(urllib.request.Request(url, method="GET"), timeout=timeout) as resp, \
                    open(part, "wb") as f:
                while chunk := resp.read(READ_CHUNK):
                    f.write(chunk)
            if part.stat().st_size != size:
                raise OSError(f"truncated: {part.stat().st_size} of {size} bytes")
            _validate(part, origin, tile_px)
            os.replace(part, dest)
            return "downloaded"
        except (OSError, ValueError, KeyError, rasterio.errors.RasterioError) as exc:
            last = exc
            part.unlink(missing_ok=True)
            if attempt + 1 < retries:
                sleep(backoff * 2 ** attempt)
    raise RuntimeError(f"{url}: {type(last).__name__}: {last}")


def download_all(tiles, index, out_dir, year, opener=urllib.request.urlopen, sleep=time.sleep,
                 workers=4, tile_px=TILE_PX):
    """Download every planned tile; returns per-outcome counts and failures."""
    stats = {"tiles_planned": len(tiles), "downloaded": 0, "skipped": 0,
             "not_in_index": [], "failed": {}}
    jobs = {}
    for tx, ty in sorted(tiles):
        name = tile_name(tx, ty, year)
        sheet = sheet_for_tile(tx, ty, index)
        if sheet is None:
            stats["not_in_index"].append(name)
            continue
        jobs[name] = (tile_url(tx, ty, *sheet, year), Path(out_dir) / str(year) / name, (tx, ty))
    with ThreadPoolExecutor(workers) as ex:
        futures = {ex.submit(download_tile, url, dest, origin, opener=opener, sleep=sleep,
                             tile_px=tile_px): name for name, (url, dest, origin) in jobs.items()}
        for done, fut in enumerate(as_completed(futures), 1):
            name = futures[fut]
            try:
                stats[fut.result()] += 1
            except RuntimeError as exc:
                logger.error("failed: %s", exc)
                stats["failed"][name] = str(exc)
            if done % 25 == 0:
                logger.info("%d/%d tiles (%d downloaded, %d skipped, %d failed)", done, len(jobs),
                            stats["downloaded"], stats["skipped"], len(stats["failed"]))
    return stats


def _read_index(source, opener):
    """Raw bytes of the sheet index from a URL or a local path."""
    if str(source).startswith(("http://", "https://")):
        with opener(urllib.request.Request(str(source), method="GET"), timeout=120) as resp:
            return resp.read()
    return Path(source).read_bytes()


def main(argv=None, opener=urllib.request.urlopen, sleep=time.sleep):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subsets", type=Path, nargs="+", help="subset roots holding LOD1/ and LOD2/")
    parser.add_argument("--out", type=Path, required=True, help="raw ortho root (resumable)")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--sheet-index", default=DEFAULT_SHEET_INDEX,
                        help="URL or path of the portal's flight-block GeoJSON")
    parser.add_argument("--beta-max", type=float, default=0.25)
    parser.add_argument("--lod-in", default="LOD1")
    parser.add_argument("--lod-out", default="LOD2")
    parser.add_argument("--margin-lo", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--margin-hi", type=float, nargs=3, default=[0.0, 0.0, 0.1])
    parser.add_argument("--plan-workers", type=int, default=min(8, mp.cpu_count()))
    parser.add_argument("--workers", type=int, default=4, help="parallel downloads")
    parser.add_argument("--tile-px", type=int, default=TILE_PX, help="expected tile side, px")
    parser.add_argument("--check", action="store_true",
                        help="HEAD the first planned tile and exit")
    parser.add_argument("--dry-run", action="store_true", help="plan only, download nothing")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    for root in args.subsets:
        if not (root / args.lod_out).is_dir():
            sys.exit(f"no {args.lod_out}/ under {root}")
    index_bytes = _read_index(args.sheet_index, opener)
    index = load_sheet_index(json.loads(index_bytes))
    plan_kw = dict(beta_max=args.beta_max, lod_in=args.lod_in, lod_out=args.lod_out,
                   margin_lo=args.margin_lo, margin_hi=args.margin_hi, workers=args.plan_workers)

    if args.check:
        tiles, _ = plan_downloads(args.subsets[:1], max_files=1, **plan_kw)
        tx, ty = min(tiles)
        sheet = sheet_for_tile(tx, ty, index)
        if sheet is None:
            sys.exit(f"tile {tile_name(tx, ty, args.year)} is not in the sheet index")
        url = tile_url(tx, ty, *sheet, args.year)
        with opener(urllib.request.Request(url, method="HEAD"), timeout=120) as resp:
            size = int(resp.headers["Content-Length"])
        logger.info("OK %s -> %.1f MB", url, size / 1e6)
        return {"check": url, "bytes": size}

    tiles, pw = plan_downloads(args.subsets, **plan_kw)
    if args.dry_run:
        totals = {"tiles_planned": len(tiles),
                  "not_in_index": sum(sheet_for_tile(x, y, index) is None for x, y in tiles),
                  "estimated_gb": round(len(tiles) * MEAN_TILE_MB / 1e3, 1)}
        logger.info("dry run: %s", totals)
        return totals

    totals = download_all(tiles, index, args.out, args.year, opener=opener, sleep=sleep,
                          workers=args.workers, tile_px=args.tile_px)
    index_copy = args.out / f"sheet_index_{args.year}.geojson"
    index_copy.write_bytes(index_bytes)
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit(),
        "sources": [str(p.resolve()) for p in args.subsets],
        "sheet_index": {"source": str(args.sheet_index), "copy": index_copy.name,
                        "sha256": hashlib.sha256(index_bytes).hexdigest()},
        "args": {k: (str(v) if isinstance(v, Path) else
                     [str(x) for x in v] if k == "subsets" else v) for k, v in vars(args).items()},
        "totals": totals, "pw_datum": dict(pw),
    }
    (args.out / "download_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                     encoding="utf-8")
    body = [
        "## This folder",
        "",
        f"`{args.year}/` holds the original 1 x 1 km tiles, unmodified, named",
        f"`{args.year}_<xmin>_<ymax>_RGB_hrl.tif` after their upper-left corner (RD metres).",
        f"Only tiles under a building crop window of these subsets were fetched: "
        f"{', '.join(p.name for p in args.subsets)}.",
        f"`{index_copy.name}` is the portal sheet index used to build the URLs.",
        "",
        f"Tiles planned: {totals['tiles_planned']}; downloaded this run: {totals['downloaded']};"
        f" already present: {totals['skipped']}; not in sheet index: "
        f"{len(totals['not_in_index'])}; failed: {len(totals['failed'])}.",
        f"Produced by `python -m src.ortho_download` at commit {manifest['git_commit']};",
        "see `download_manifest.json`.",
    ]
    (args.out / "README.md").write_text(
        readme_text("Raw orthophoto tiles", args.year, pw, body), encoding="utf-8")
    logger.info("done: %s", {k: (len(v) if isinstance(v, (list, dict)) else v)
                             for k, v in totals.items()})
    return totals


if __name__ == "__main__":
    main()
