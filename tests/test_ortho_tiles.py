"""Tile naming, sheet lookup and per-building frames for the orthophoto pipeline."""
import json

import numpy as np
import pytest

from src.dataset.mesh_dataset import normalize_to_unit_box, parse_cityjson_file_to_meshes
from src.ortho_tiles import (assign_year, building_frames, load_sheet_index, parse_tile_name,
                             pw_datum_counts, readme_text, sheet_for_tile, sheets_for_tile,
                             tile_name, tile_url, tiles_for_bounds)

MARGIN_LO, MARGIN_HI = [0.0, 0.0, 0.0], [0.0, 0.0, 0.1]
QUADS = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]


def write_tile(path, boxes, datum=None):
    """CityJSON tile with one box Solid per id: {id: (x0, y0, x1, y1, z1)}."""
    verts, objects = [], {}
    for oid, (x0, y0, x1, y1, z1) in boxes.items():
        base = len(verts)
        verts += [[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]
        obj = {"type": "Building", "geometry": [{
            "type": "Solid", "lod": "2.2",
            "boundaries": [[[[base + i for i in q]] for q in QUADS]]}]}
        if datum is not None:
            obj["attributes"] = {"b3_pw_datum": datum.get(oid)}
        objects[oid] = obj
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "CityJSON", "version": "2.0",
                                "CityObjects": objects, "vertices": verts}), encoding="utf-8")
    return path


def test_tile_name_encodes_the_upper_left_corner():
    name = tile_name(81000, 455000, 2022)
    assert name == "2022_081000_455000_RGB_hrl.tif"
    assert parse_tile_name(name) == (81000, 454000, 82000, 455000)


def test_tiles_for_bounds_is_half_open():
    # A window ending exactly on a km line must not pull in the neighbour.
    assert tiles_for_bounds((81200.0, 454100.0, 82000.0, 455000.0)) == {(81000, 455000)}
    assert tiles_for_bounds((81900.0, 454900.0, 82100.0, 455100.0)) == {
        (81000, 455000), (82000, 455000), (81000, 456000), (82000, 456000)}


def test_sheet_lookup_uses_the_tile_centre(tmp_path):
    # Two blocks side by side; the boundary at x = 81 600 cuts tile 81 000 off-centre.
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"BLOK": "3", "PERCEEL": "Perceel 1"},
         "geometry": {"type": "MultiPolygon", "coordinates": [[[
             [80000, 450000], [81600, 450000], [81600, 460000], [80000, 460000], [80000, 450000]]]]}},
        {"type": "Feature", "properties": {"BLOK": "12", "PERCEEL": "Perceel 2"},
         "geometry": {"type": "MultiPolygon", "coordinates": [[[
             [81600, 450000], [90000, 450000], [90000, 460000], [81600, 460000], [81600, 450000]]]]}},
    ]}
    path = tmp_path / "BM2022.geojson"
    path.write_text(json.dumps(fc), encoding="utf-8")
    index = load_sheet_index(path)

    assert sheet_for_tile(81000, 455000, index) == (1, 3)      # centre 81 500 -> block 3
    assert sheet_for_tile(82000, 455000, index) == (2, 12)
    assert sheet_for_tile(10000, 455000, index) is None


def test_tile_url_follows_the_2022_portal_layout():
    assert tile_url(81000, 455000, perceel=1, blok=3, year=2022) == (
        "https://basisdata.nl/hwh-ortho/2022/Ortho/1/03/beelden_tif_tegels/"
        "2022_081000_455000_RGB_hrl.tif")


def test_building_frames_use_lod1_only_and_match_the_dataset_frame(tmp_path):
    lod1 = write_tile(tmp_path / "LOD1" / "Source B" / "t.city.json",
                      {"a": (10, 20, 25, 40, 6), "b": (50, 50, 60, 60, 3)})
    # LOD2 of "a" is taller and wider than its LOD1; "c" has no LOD1 at all.
    lod2 = write_tile(tmp_path / "LOD2" / "Source B" / "t.city.json",
                      {"a": (8, 20, 25, 40, 9), "b": (50, 50, 60, 60, 3),
                       "c": (0, 0, 5, 5, 3)})

    frames, missing = building_frames(lod1, lod2, MARGIN_LO, MARGIN_HI)

    assert sorted(frames) == ["a", "b"] and missing == 1
    verts_a = parse_cityjson_file_to_meshes(lod1)["a"][0]
    _, center, scale = normalize_to_unit_box(verts_a, margin_lo=MARGIN_LO, margin_hi=MARGIN_HI)
    np.testing.assert_allclose(frames["a"].center, center)
    np.testing.assert_allclose(frames["a"].scale, scale)
    # LOD2 "a" reaches 2 m west of LOD1: (8 - 17.5) / 20 = -0.475 -> inside, need 0.
    assert frames["a"].need_beta == pytest.approx(0.0)
    assert frames["b"].need_beta == pytest.approx(0.0)


def test_need_beta_reports_lod2_outside_the_unit_box(tmp_path):
    lod1 = write_tile(tmp_path / "LOD1" / "t.city.json", {"a": (0, 0, 10, 10, 2)})
    lod2 = write_tile(tmp_path / "LOD2" / "t.city.json", {"a": (-3, 0, 10, 10, 2)})
    frames, _ = building_frames(lod1, lod2, MARGIN_LO, MARGIN_HI)
    # centre x 5, scale 10: x = -3 -> u = -0.8 -> 0.3 beyond the box
    assert frames["a"].need_beta == pytest.approx(0.3)


def test_pw_datum_counts(tmp_path):
    lod2 = write_tile(tmp_path / "t.city.json",
                      {"a": (0, 0, 1, 1, 1), "b": (2, 2, 3, 3, 1), "c": (4, 4, 5, 5, 1)},
                      datum={"a": 2022, "b": 2022, "c": None})
    assert pw_datum_counts(lod2) == {"2022": 2, "unknown": 1}


def test_tile_url_follows_the_2023_portal_layout():
    assert tile_url(81000, 455000, perceel=1, blok=3, year=2023) == (
        "https://basisdata.nl/hwh-ortho/2023/Ortho/1/03/beelden_tif_tegels/"
        "2023_081000_455000_RGB_hrl.tif")


@pytest.mark.parametrize("datum, years, expected", [
    ("2023", [2022, 2023], 2023),
    ("2022", [2022, 2023], 2022),
    ("2020", [2022, 2023], 2022),
    ("2014", [2023, 2022], 2022),         # order of `years` does not matter
    ("2025", [2022, 2023], 2023),
    ("2022", [2021, 2023], 2023),         # tie -> the later year
    ("unknown", [2022, 2023], 2023),      # no point-cloud year -> most recent imagery
])
def test_assign_year_picks_the_closest_imagery_year(datum, years, expected):
    assert assign_year(datum, years) == expected


def test_sheets_for_tile_lists_every_block_touching_the_tile():
    index = load_sheet_index({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"BLOK": "3", "PERCEEL": "Perceel 1"},
         "geometry": {"type": "MultiPolygon", "coordinates": [[[
             [80000, 450000], [81600, 450000], [81600, 460000], [80000, 460000], [80000, 450000]]]]}},
        {"type": "Feature", "properties": {"BLOK": "12", "PERCEEL": "Perceel 2"},
         "geometry": {"type": "MultiPolygon", "coordinates": [[[
             [81600, 450000], [90000, 450000], [90000, 460000], [81600, 460000], [81600, 450000]]]]}},
    ]})
    assert sheets_for_tile(81000, 455000, index) == [(1, 3), (2, 12)]   # centre block first
    assert sheets_for_tile(83000, 455000, index) == [(2, 12)]
    assert sheets_for_tile(10000, 455000, index) == []
    assert sheet_for_tile(81000, 455000, index) == (1, 3)


def test_readme_states_the_year_rule_and_each_point_cloud_years_imagery():
    text = readme_text("t", [2022, 2023], {"2023": 3, "2020": 1}, ["body"])
    for needle in ("2022", "2023", "CC-BY 4.0", "b3_pw_datum", "closest",
                   "2023: 3 (75.0%) -> 2023 imagery", "2020: 1 (25.0%) -> 2022 imagery (2 y apart)"):
        assert needle in text, needle
