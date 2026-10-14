"""Shared helpers for the orthophoto pipeline (`src.ortho_download`, `src.ortho_parse`).

Beeldmateriaal Nederland publishes its 8 cm orthophotos as 1 x 1 km GeoTIFF tiles
in EPSG:28992 (RD New). A tile is named after its **upper-left** corner:
``2022_{xmin:06d}_{ymax:06d}_RGB_hrl.tif`` covers x in [xmin, xmin + 1000] and
y in [ymax - 1000, ymax]. Its download path also needs the flight block it was
delivered in (perceel, blok), which comes from the portal's sheet index -- a
GeoJSON of block polygons.

Per-building frames come from the same parser and normalisation the mesh
dataset uses, so the windows planned here are the windows read at training time.
"""
import collections
import json
import math
from pathlib import Path

import numpy as np
from matplotlib.path import Path as PolygonPath

from src.dataset.mesh_dataset import normalize_to_unit_box, parse_cityjson_file_to_meshes

TILE_M = 1000
GSD_M = 0.08
BASE_URL = "https://basisdata.nl/hwh-ortho"
# Per-year layout of the portal's tile store, from its tile-index viewer
# (basisdata.nl/hwh-ortho/AUX/bladwijzer). Only years verified against a real
# tile are listed; add others only after a HEAD request confirms them.
URL_TEMPLATES = {
    2021: "{base}/2021/{perceel}/{blok:02d}/beelden_tif_tegels/{name}",
    2022: "{base}/2022/Ortho/{perceel}/{blok:02d}/beelden_tif_tegels/{name}",
}

Frame = collections.namedtuple("Frame", "center scale need_beta")


def tile_name(xmin, ymax, year):
    """File name of the tile whose upper-left corner is (xmin, ymax)."""
    return f"{year}_{int(xmin):06d}_{int(ymax):06d}_RGB_hrl.tif"


def parse_tile_name(name):
    """``(xmin, ymin, xmax, ymax)`` in metres from a tile file name."""
    _, x, y = Path(name).name.split("_")[:3]
    xmin, ymax = int(x), int(y)
    return xmin, ymax - TILE_M, xmin + TILE_M, ymax


def tiles_for_bounds(bounds):
    """Set of ``(xmin, ymax)`` tile corners a world box touches.

    Half-open on the upper side, so a box ending exactly on a kilometre line
    does not pull in the neighbouring tile.
    """
    xmin, ymin, xmax, ymax = bounds
    cols = range(math.floor(xmin / TILE_M), math.ceil(xmax / TILE_M))
    rows = range(math.floor(ymin / TILE_M), math.ceil(ymax / TILE_M))
    return {(c * TILE_M, (r + 1) * TILE_M) for c in cols for r in rows}


def load_sheet_index(source):
    """Flight blocks as ``[(exterior, holes, perceel, blok)]`` from the portal GeoJSON.

    Args:
        source: a path to the GeoJSON, or the already-parsed FeatureCollection.

    ``PERCEEL`` is a label like ``"Perceel 1"``; only its number enters the URL.
    """
    blocks = []
    fc = source if isinstance(source, dict) else json.loads(Path(source).read_text(encoding="utf-8"))
    for feature in fc["features"]:
        props = feature["properties"]
        perceel = int(str(props["PERCEEL"]).split()[-1])
        blok = int(props["BLOK"])
        geom = feature["geometry"]
        polygons = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for rings in polygons:
            blocks.append((PolygonPath(rings[0]), [PolygonPath(r) for r in rings[1:]],
                           perceel, blok))
    return blocks


def sheet_for_tile(xmin, ymax, index):
    """``(perceel, blok)`` of the block containing the tile centre, or None."""
    centre = (xmin + TILE_M / 2, ymax - TILE_M / 2)
    for exterior, holes, perceel, blok in index:
        if exterior.contains_point(centre) and not any(h.contains_point(centre) for h in holes):
            return perceel, blok
    return None


def tile_url(xmin, ymax, perceel, blok, year):
    """Download URL of one tile (redirects to the portal's object store)."""
    if year not in URL_TEMPLATES:
        raise ValueError(f"No verified URL layout for {year}; known: {sorted(URL_TEMPLATES)}.")
    return URL_TEMPLATES[year].format(base=BASE_URL, perceel=perceel, blok=blok,
                                      name=tile_name(xmin, ymax, year))


def building_frames(lod1_path, lod2_path, margin_lo, margin_hi):
    """Per-building normalisation frame for every id present in both files.

    The frame is built from LOD1 alone, exactly as `MeshDataset.__getitem__`
    builds it, so a crop planned from it is the crop read at training time.

    Returns:
        tuple: ``({id: Frame(center [3], scale [3], need_beta)}, n_without_lod1)``.
        ``need_beta`` is how far the building's LOD2 reaches past the unit box in
        x/y (normalized units, 0 when inside) -- the margin a crop needs to show
        all of it.
    """
    lod1 = parse_cityjson_file_to_meshes(lod1_path)
    lod2 = parse_cityjson_file_to_meshes(lod2_path)
    frames = {}
    for oid, (v2, _) in lod2.items():
        if oid not in lod1:
            continue
        _, center, scale = normalize_to_unit_box(lod1[oid][0], margin_lo=margin_lo,
                                                 margin_hi=margin_hi)
        uv = np.abs((v2[:, :2] - center[:2]) / scale[:2])
        frames[oid] = Frame(center, scale, max(0.0, float(uv.max()) - 0.5))
    return frames, sum(oid not in lod1 for oid in lod2)


def readme_text(title, year, pw_counts, body):
    """README for an orthophoto data folder: source, licence, CRS and the year remark.

    Args:
        title: first heading.
        year: the imagery year used for every building.
        pw_counts: {point-cloud year: n buildings} over the data the folder serves.
        body: extra markdown lines specific to the folder.
    """
    total = sum(pw_counts.values()) or 1
    dist = [f"- {k}: {n} ({100 * n / total:.1f}%)" for k, n in
            sorted(pw_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    return "\n".join([
        f"# {title}",
        "",
        f"Source: Beeldmateriaal Nederland, 8 cm RGB orthophoto, {year} flight.",
        "Licence: CC-BY 4.0 -- attribute \"Beeldmateriaal Nederland\" in any derived figure.",
        "CRS: EPSG:28992 (RD New), metres, rasters stored north-up.",
        "",
        "## Year remark",
        "",
        f"Every building uses the **{year}** imagery. The 3DBAG LOD2 geometry was",
        "reconstructed from AHN point clouds of different years, recorded per building",
        f"in the CityObject attribute `b3_pw_datum`. Where that year is not {year} the photo",
        "can show a building that has since been built, demolished or extended.",
        "Point-cloud years of the buildings this data covers:",
        "",
        *dist,
        "",
        "## Relief displacement",
        "",
        "Whether this product is a true orthophoto is unverified. If it is not, the roofs",
        "of tall buildings are shifted away from their footprints by roughly",
        "height x tan(off-nadir angle), and facades may be visible.",
        "",
        *body,
        "",
    ])


def pw_datum_by_id(lod2_path):
    """{object id: 3DBAG ``b3_pw_datum`` (point-cloud year) as str, or "unknown"}."""
    cj = json.loads(Path(lod2_path).read_text(encoding="utf-8"))
    out = {}
    for oid, obj in cj.get("CityObjects", {}).items():
        datum = (obj.get("attributes") or {}).get("b3_pw_datum")
        out[oid] = "unknown" if datum is None else str(datum)
    return out


def pw_datum_counts(lod2_path):
    """Count of ``b3_pw_datum`` values over a file's objects."""
    return collections.Counter(pw_datum_by_id(lod2_path).values())
