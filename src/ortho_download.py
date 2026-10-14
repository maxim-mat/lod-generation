"""Download the raw 8 cm orthophoto tiles one or more dataset subsets need.

Each building with both LOD1 and LOD2 in the given subsets is assigned the
imagery year in ``--years`` closest to its 3DBAG point-cloud year
(`src.ortho_tiles.assign_year`). Its crop window (`src.dataset.ortho.crop_bounds`
at ``--beta-max``, in the ``--lod-in`` frame) is mapped to the 1 x 1 km tiles it
touches, and only those tiles are fetched -- from that year only. Download
paths need the flight block a tile was delivered in, looked up in each year's
sheet index (``--sheet-index YEAR=URL_OR_PATH``; the asset names carry a build
hash, so the defaults may go stale).

Downloads resume: a tile whose size on disk matches the server's
Content-Length is skipped, partial files are written to ``.part`` and renamed
only after the GeoTIFF opens with the expected size and anchor. A 403/404 is
final for that URL and the next candidate block is tried (tiles on a block
boundary are not always delivered under the centre's block).

Usage:
    python -m src.ortho_download "data/The Hague/cleaner" --out "data/The Hague/raw/ortho" \\
        --years 2022 2023 --check        # one HEAD request per year, then exit
    ... --dry-run                         # tile count and size estimate per year
    ...                                   # the download itself
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
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import rasterio

from src.dataset.ortho import crop_bounds
from src.filter_cityjson import _git_commit
from src.ortho_tiles import (SHEET_INDEX_URLS, assign_year, building_frames, load_sheet_index,
                             pw_datum_by_id, readme_text, sheet_for_tile, sheets_for_tile,
                             tile_name, tile_url, tiles_for_bounds)

logger = logging.getLogger(__name__)

# Mean tile size from HEAD samples (2021: 31.1 MB, 2023: 32.8 MB, 2022 on disk:
# 32.3 MB; 2026-10-14). Only used for the --dry-run size estimate.
MEAN_TILE_MB = 32.5
TILE_PX = 12500
READ_CHUNK = 1 << 20


def _plan_one(job):
    """Per-year tiles and point-cloud years of one file. Top-level for a spawned Pool."""
    lod1, lod2, beta, margin_lo, margin_hi, years = job
    frames, _ = building_frames(lod1, lod2, margin_lo, margin_hi)
    datum = pw_datum_by_id(lod2)
    tiles = collections.defaultdict(set)
    for oid, f in frames.items():
        tiles[assign_year(datum[oid], years)] |= tiles_for_bounds(
            crop_bounds(f.center, f.scale, beta))
    return dict(tiles), {oid: datum[oid] for oid in frames}


def plan_downloads(subset_roots, beta_max, lod_in, lod_out, margin_lo, margin_hi, years,
                   workers=1, max_files=None):
    """Tiles each imagery year needs for the crop windows of the given subsets.

    Returns:
        tuple: ({year: set of (xmin, ymax) tile corners}, Counter of
        ``b3_pw_datum`` over the unique buildings planned).
    """
    jobs = []
    for root in map(Path, subset_roots):
        files = sorted((root / lod_out).rglob("*.json"))[:max_files]
        jobs += [(root / lod_in / p.relative_to(root / lod_out), p, beta_max,
                  list(margin_lo), list(margin_hi), list(years)) for p in files]
    tiles, datum = collections.defaultdict(set), {}
    pool = mp.Pool(workers) if workers > 1 else None
    try:
        results = pool.imap_unordered(_plan_one, jobs) if pool else map(_plan_one, jobs)
        for done, (t, d) in enumerate(results, 1):
            for year, s in t.items():
                tiles[year] |= s
            datum.update(d)
            if done % 25 == 0:
                logger.info("planned %d/%d files", done, len(jobs))
    finally:
        if pool:
            pool.close()
            pool.join()
    return dict(tiles), collections.Counter(datum.values())


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
            truncated body, or a body that is not the expected GeoTIFF), or at
            once on HTTP 403/404, which retrying cannot fix.
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
        except urllib.error.HTTPError as exc:     # before OSError: HTTPError subclasses it
            last = exc
            part.unlink(missing_ok=True)
            if exc.code in (403, 404):
                break
            if attempt + 1 < retries:
                sleep(backoff * 2 ** attempt)
        except (OSError, ValueError, KeyError, rasterio.errors.RasterioError) as exc:
            last = exc
            part.unlink(missing_ok=True)
            if attempt + 1 < retries:
                sleep(backoff * 2 ** attempt)
    raise RuntimeError(f"{url}: {type(last).__name__}: {last}")


def download_all(tiles, index, out_dir, year, opener=urllib.request.urlopen, sleep=time.sleep,
                 workers=4, tile_px=TILE_PX):
    """Download one year's planned tiles, trying each candidate flight block in turn."""
    stats = {"tiles_planned": len(tiles), "downloaded": 0, "skipped": 0,
             "not_in_index": [], "failed": {}}
    jobs = {}
    for tx, ty in sorted(tiles):
        name = tile_name(tx, ty, year)
        sheets = sheets_for_tile(tx, ty, index)
        if not sheets:
            stats["not_in_index"].append(name)
            continue
        jobs[name] = ([tile_url(tx, ty, *s, year) for s in sheets],
                      Path(out_dir) / str(year) / name, (tx, ty))

    def fetch(urls, dest, origin):
        errors = []
        for url in urls:
            try:
                return download_tile(url, dest, origin, opener=opener, sleep=sleep,
                                     tile_px=tile_px)
            except RuntimeError as exc:
                errors.append(str(exc))
        raise RuntimeError(" | ".join(errors))

    with ThreadPoolExecutor(workers) as ex:
        futures = {ex.submit(fetch, *job): name for name, job in jobs.items()}
        for done, fut in enumerate(as_completed(futures), 1):
            name = futures[fut]
            try:
                stats[fut.result()] += 1
            except RuntimeError as exc:
                logger.error("failed: %s", exc)
                stats["failed"][name] = str(exc)
            if done % 25 == 0:
                logger.info("%d %d/%d tiles (%d downloaded, %d skipped, %d failed)", year, done,
                            len(jobs), stats["downloaded"], stats["skipped"], len(stats["failed"]))
    return stats


def _read_index(source, opener):
    """Raw bytes of a sheet index from a URL or a local path."""
    if str(source).startswith(("http://", "https://")):
        with opener(urllib.request.Request(str(source), method="GET"), timeout=120) as resp:
            return resp.read()
    return Path(source).read_bytes()


def _sheet_sources(years, overrides):
    """{year: URL or path} from the defaults and ``YEAR=SOURCE`` overrides."""
    sources = {y: SHEET_INDEX_URLS.get(y) for y in years}
    for spec in overrides:
        year, _, source = spec.partition("=")
        if not source:
            sys.exit(f"--sheet-index expects YEAR=URL_OR_PATH, got {spec!r}")
        sources[int(year)] = source
    missing = [y for y, s in sources.items() if not s]
    if missing:
        sys.exit(f"no sheet index known for {missing}; pass --sheet-index YEAR=URL_OR_PATH")
    return sources


def main(argv=None, opener=urllib.request.urlopen, sleep=time.sleep):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subsets", type=Path, nargs="+", help="subset roots holding LOD1/ and LOD2/")
    parser.add_argument("--out", type=Path, required=True, help="raw ortho root (resumable)")
    parser.add_argument("--years", type=int, nargs="+", default=[2022, 2023],
                        help="imagery years to choose from per building")
    parser.add_argument("--sheet-index", action="append", default=[],
                        help="YEAR=URL_OR_PATH of a flight-block GeoJSON (overrides the default)")
    parser.add_argument("--beta-max", type=float, default=0.25)
    parser.add_argument("--lod-in", default="LOD1")
    parser.add_argument("--lod-out", default="LOD2")
    parser.add_argument("--margin-lo", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--margin-hi", type=float, nargs=3, default=[0.0, 0.0, 0.1])
    parser.add_argument("--plan-workers", type=int, default=min(8, mp.cpu_count()))
    parser.add_argument("--workers", type=int, default=4, help="parallel downloads")
    parser.add_argument("--tile-px", type=int, default=TILE_PX, help="expected tile side, px")
    parser.add_argument("--check", action="store_true",
                        help="HEAD the first planned tile of each year and exit")
    parser.add_argument("--dry-run", action="store_true", help="plan only, download nothing")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    years = sorted(set(args.years))

    for root in args.subsets:
        if not (root / args.lod_out).is_dir():
            sys.exit(f"no {args.lod_out}/ under {root}")
    sources = _sheet_sources(years, args.sheet_index)
    index_bytes = {y: _read_index(sources[y], opener) for y in years}
    indexes = {y: load_sheet_index(json.loads(b)) for y, b in index_bytes.items()}
    plan_kw = dict(beta_max=args.beta_max, lod_in=args.lod_in, lod_out=args.lod_out,
                   margin_lo=args.margin_lo, margin_hi=args.margin_hi, years=years,
                   workers=args.plan_workers)

    if args.check:
        tiles, _ = plan_downloads(args.subsets[:1], max_files=1, **plan_kw)
        checked = {}
        for year, planned in sorted(tiles.items()):
            tx, ty = min(planned)
            sheet = sheet_for_tile(tx, ty, indexes[year])
            if sheet is None:
                sys.exit(f"tile {tile_name(tx, ty, year)} is not in the {year} sheet index")
            url = tile_url(tx, ty, *sheet, year)
            with opener(urllib.request.Request(url, method="HEAD"), timeout=120) as resp:
                checked[str(year)] = int(resp.headers["Content-Length"])
            logger.info("OK %s -> %.1f MB", url, checked[str(year)] / 1e6)
        return {"check": checked}

    tiles, pw = plan_downloads(args.subsets, **plan_kw)
    if args.dry_run:
        totals = {"tiles_planned": {str(y): len(tiles.get(y, ())) for y in years},
                  "not_in_index": {str(y): sum(sheet_for_tile(x, t, indexes[y]) is None
                                               for x, t in tiles.get(y, ())) for y in years},
                  "estimated_gb": {str(y): round(len(tiles.get(y, ())) * MEAN_TILE_MB / 1e3, 1)
                                   for y in years}}
        logger.info("dry run: %s", totals)
        return totals

    per_year = {}
    for year in years:
        per_year[str(year)] = download_all(tiles.get(year, set()), indexes[year], args.out, year,
                                           opener=opener, sleep=sleep, workers=args.workers,
                                           tile_px=args.tile_px)
    totals = {
        "tiles_planned": sum(s["tiles_planned"] for s in per_year.values()),
        "downloaded": sum(s["downloaded"] for s in per_year.values()),
        "skipped": sum(s["skipped"] for s in per_year.values()),
        "not_in_index": [n for s in per_year.values() for n in s["not_in_index"]],
        "failed": {n: e for s in per_year.values() for n, e in s["failed"].items()},
        "per_year": per_year,
    }
    sheet_meta = {}
    for year in years:
        copy = args.out / f"sheet_index_{year}.geojson"
        copy.write_bytes(index_bytes[year])
        sheet_meta[str(year)] = {"source": str(sources[year]), "copy": copy.name,
                                 "sha256": hashlib.sha256(index_bytes[year]).hexdigest()}
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit(),
        "sources": [str(p.resolve()) for p in args.subsets],
        "sheet_index": sheet_meta,
        "args": {k: (str(v) if isinstance(v, Path) else
                     [str(x) for x in v] if k == "subsets" else v) for k, v in vars(args).items()}
                | {"years": years},
        "totals": totals, "pw_datum": dict(pw),
    }
    (args.out / "download_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                     encoding="utf-8")
    body = [
        "## This folder",
        "",
        "`<year>/` holds the original 1 x 1 km tiles, unmodified, named",
        "`<year>_<xmin>_<ymax>_RGB_hrl.tif` after their upper-left corner (RD metres).",
        "A year folder holds only the tiles under the crop windows of the buildings",
        f"assigned that year, for these subsets: {', '.join(p.name for p in args.subsets)}.",
        "Tiles from earlier runs that are no longer needed are left in place.",
        "`sheet_index_<year>.geojson` are the portal sheet indexes used to build the URLs.",
        "",
        *[f"- {y}: planned {s['tiles_planned']}, downloaded this run {s['downloaded']}, already "
          f"present {s['skipped']}, not in sheet index {len(s['not_in_index'])}, failed "
          f"{len(s['failed'])}" for y, s in per_year.items()],
        "",
        f"Produced by `python -m src.ortho_download` at commit {manifest['git_commit']};",
        "see `download_manifest.json`.",
    ]
    (args.out / "README.md").write_text(
        readme_text("Raw orthophoto tiles", years, pw, body), encoding="utf-8")
    logger.info("done: %s", {k: (len(v) if isinstance(v, (list, dict)) else v)
                             for k, v in totals.items() if k != "per_year"})
    return totals


if __name__ == "__main__":
    main()
