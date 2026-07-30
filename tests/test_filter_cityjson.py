"""Unit tests for the CityJSON dataset filter.

Small hand-built CityJSON dicts only -- never the real dataset.
"""
import json
from pathlib import Path

import numpy as np

from src.filter_cityjson import (
    filter_city_objects, is_sliver, select_solid, world_vertices,
)


def box(x0, y0, w, d, h, base=0.0):
    """Axis-aligned box as (vertices, Solid boundaries, semantics)."""
    v = [(x0, y0, base), (x0 + w, y0, base), (x0 + w, y0 + d, base), (x0, y0 + d, base),
         (x0, y0, base + h), (x0 + w, y0, base + h), (x0 + w, y0 + d, base + h), (x0, y0 + d, base + h)]
    faces = [[[0, 3, 2, 1]], [[4, 5, 6, 7]], [[0, 1, 5, 4]],
             [[1, 2, 6, 5]], [[2, 3, 7, 6]], [[3, 0, 4, 7]]]
    sem = {"surfaces": [{"type": "GroundSurface"}, {"type": "RoofSurface"}, {"type": "WallSurface"}],
           "values": [[0, 1, 2, 2, 2, 2]]}
    return [list(p) for p in v], [faces], sem


def solid(lod, n_extra_faces=0):
    """A Solid geometry with a controllable face count (for tie-break tests)."""
    _, boundaries, sem = box(0, 0, 4, 5, 3)
    boundaries[0] = boundaries[0] + [[[0, 1, 2]]] * n_extra_faces
    sem = {"surfaces": sem["surfaces"], "values": [sem["values"][0] + [2] * n_extra_faces]}
    return {"type": "Solid", "lod": lod, "boundaries": boundaries, "semantics": sem}


def cityjson(objects, vertices):
    return {"type": "CityJSON", "version": "1.1", "CityObjects": objects, "vertices": vertices}


# --- geometry selection ---------------------------------------------------

def test_selects_tagged_lod2_solid_over_lod1_variants():
    obj = {"type": "BuildingPart", "geometry": [solid("1.2"), solid("1.3"), solid("2.2", 4)]}
    assert select_solid(obj, 2)["lod"] == "2.2"


def test_selects_finest_when_lod2_tags_are_flattened():
    obj = {"type": "BuildingPart", "geometry": [solid("2", 0), solid("2", 8)]}
    assert len(select_solid(obj, 2)["boundaries"][0]) == 14


def test_rejects_lod0_footprint_parent():
    obj = {"type": "Building", "geometry": [{"type": "MultiSurface", "lod": "0",
                                             "boundaries": [[[0, 1, 2, 3]]]}]}
    assert select_solid(obj, 2) is None


def test_rejects_wrong_integer_lod():
    obj = {"type": "BuildingPart", "geometry": [solid("1.2"), solid("1.3")]}
    assert select_solid(obj, 2) is None


# --- object filtering -----------------------------------------------------

def test_drops_parent_keeps_part_and_strips_dangling_refs():
    verts, boundaries, sem = box(0, 0, 4, 5, 3)
    cj = cityjson({
        "pand": {"type": "Building", "children": ["pand-0"], "attributes": {"year": 1900},
                 "geometry": [{"type": "MultiSurface", "lod": "0", "boundaries": [[[0, 1, 2, 3]]]}]},
        "pand-0": {"type": "BuildingPart", "parents": ["pand"],
                   "geometry": [{"type": "Solid", "lod": "2.2", "boundaries": boundaries, "semantics": sem}]},
    }, verts)
    kept, _ = filter_city_objects(cj, lod=2)
    assert set(kept) == {"pand-0"}
    assert "parents" not in kept["pand-0"]
    assert len(kept["pand-0"]["geometry"]) == 1


def test_part_inherits_attributes_from_the_dropped_parent():
    """3DBAG puts every BAG attribute on the parent; the part we keep has none."""
    verts, boundaries, sem = box(0, 0, 4, 5, 3)
    cj = cityjson({
        "pand": {"type": "Building", "children": ["pand-0"],
                 "attributes": {"b3_dak_type": "slanted", "bouwjaar": 1900},
                 "geometry": [{"type": "MultiSurface", "lod": "0", "boundaries": [[[0, 1, 2, 3]]]}]},
        "pand-0": {"type": "BuildingPart", "parents": ["pand"], "attributes": None,
                   "geometry": [{"type": "Solid", "lod": "2.2", "boundaries": boundaries, "semantics": sem}]},
    }, verts)
    kept, _ = filter_city_objects(cj, lod=2)
    assert kept["pand-0"]["attributes"] == {"b3_dak_type": "slanted", "bouwjaar": 1900}


def test_part_keeps_its_own_attributes_over_the_parents():
    verts, boundaries, sem = box(0, 0, 4, 5, 3)
    cj = cityjson({
        "pand": {"type": "Building", "children": ["pand-0"], "attributes": {"src": "parent"},
                 "geometry": [{"type": "MultiSurface", "lod": "0", "boundaries": [[[0, 1, 2, 3]]]}]},
        "pand-0": {"type": "BuildingPart", "parents": ["pand"], "attributes": {"src": "child"},
                   "geometry": [{"type": "Solid", "lod": "2.2", "boundaries": boundaries, "semantics": sem}]},
    }, verts)
    kept, _ = filter_city_objects(cj, lod=2)
    assert kept["pand-0"]["attributes"] == {"src": "child"}


def test_compacts_vertices_and_preserves_world_coordinates():
    verts_a, bounds_a, sem = box(0, 0, 4, 5, 3)
    verts_b, bounds_b, _ = box(50, 50, 4, 5, 3)
    shift = len(verts_a)
    bounds_b = [[[[i + shift for i in r] for r in f] for f in sh] for sh in bounds_b]
    cj = cityjson({
        "keep": {"type": "Building", "geometry": [{"type": "Solid", "lod": "2", "boundaries": bounds_b, "semantics": sem}]},
        "drop": {"type": "Building", "geometry": [{"type": "Solid", "lod": "1", "boundaries": bounds_a, "semantics": sem}]},
    }, verts_a + verts_b)
    kept, _ = filter_city_objects(cj, lod=2)
    out = {**cj, "CityObjects": kept}
    from src.filter_cityjson import compact_vertices
    out = compact_vertices(out)

    assert len(out["vertices"]) == 8
    before = world_vertices(cj)[sorted({i for sh in bounds_b for f in sh for i in f[0]})]
    after = world_vertices(out)[sorted({i for sh in out["CityObjects"]["keep"]["geometry"][0]["boundaries"]
                                        for f in sh for i in f[0]})]
    np.testing.assert_allclose(np.sort(before, axis=0), np.sort(after, axis=0))


def test_applies_transform_when_present():
    cj = {"type": "CityJSON", "CityObjects": {}, "vertices": [[0, 0, 0], [1000, 2000, 3000]],
          "transform": {"scale": [0.001, 0.001, 0.001], "translate": [100.0, 200.0, 0.0]}}
    np.testing.assert_allclose(world_vertices(cj)[1], [101.0, 202.0, 3.0])


# --- sliver screen --------------------------------------------------------

def test_flags_thin_tall_pillar():
    verts, boundaries, sem = box(0, 0, 0.6, 0.6, 12.5)
    v = world_vertices(cityjson({}, verts))
    assert is_sliver(v, {"boundaries": boundaries, "semantics": sem}, max_slenderness=5.0, min_extent=1.0)


def test_keeps_ordinary_building():
    verts, boundaries, sem = box(0, 0, 8, 12, 6)
    v = world_vertices(cityjson({}, verts))
    assert not is_sliver(v, {"boundaries": boundaries, "semantics": sem}, max_slenderness=5.0, min_extent=1.0)


def test_sliver_screen_removes_object():
    verts, boundaries, sem = box(0, 0, 0.6, 0.6, 12.5)
    cj = cityjson({"pillar": {"type": "BuildingPart",
                              "geometry": [{"type": "Solid", "lod": "2.2", "boundaries": boundaries, "semantics": sem}]}},
                  verts)
    kept, stats = filter_city_objects(cj, lod=2, max_slenderness=5.0, min_extent=1.0)
    assert kept == {}
    assert stats["slivers"] == 1


# --- end to end -----------------------------------------------------------

def test_filtered_fixture_still_parses_to_a_levi_graph(tmp_path):
    from src.dataset.dataset import parse_cityjson_file_to_graphs
    from src.filter_cityjson import filter_cityjson

    fixture = Path(__file__).parent / "fixtures" / "synthetic_lod2_building.city.json"
    cj = json.loads(fixture.read_text(encoding="utf-8"))
    out = filter_cityjson(cj, lod=2)
    assert out is not None
    path = tmp_path / "filtered.city.json"
    path.write_text(json.dumps(out), encoding="utf-8")

    # The fixture is already a single clean LOD2 Solid, so filtering must not
    # change the graph it produces.
    before = next(iter(parse_cityjson_file_to_graphs(fixture).values()))
    after = next(iter(parse_cityjson_file_to_graphs(path).values()))
    assert after["x"].shape == before["x"].shape
    np.testing.assert_allclose(np.sort(after["x"].numpy(), axis=0),
                               np.sort(before["x"].numpy(), axis=0), atol=1e-6)
    assert sorted(after["node_labels"].tolist()) == sorted(before["node_labels"].tolist())
    assert after["edge_index"].shape == before["edge_index"].shape


def _raw_tree(tmp_path):
    """A raw folder with one source subfolder, a tile and a non-JSON asset."""
    fixture = Path(__file__).parent / "fixtures" / "synthetic_lod2_building.city.json"
    raw = tmp_path / "raw" / "Source A"
    raw.mkdir(parents=True)
    (raw / "tile.city.json").write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    (raw / "description.txt").write_text("notes", encoding="utf-8")
    return tmp_path / "raw", tmp_path / "out"


def test_synthetic_lod1_is_opt_in(tmp_path):
    from src.filter_cityjson import process_dataset

    raw, out = _raw_tree(tmp_path)
    totals = process_dataset(raw, out, max_slenderness=None, min_extent=0.0)

    assert not (out / "LOD1_synth").exists()
    assert totals["LOD1_synth"]["files"] == 0
    assert (out / "LOD2" / "Source A" / "tile.city.json").exists()
    # the fixture is pure lod2, so there is no real lod-1 geometry to extract
    assert not (out / "LOD1" / "Source A" / "tile.city.json").exists()


def test_convert_on_the_fly_pairs_synth_lod1_with_lod2(tmp_path):
    """The synthetic pair must keep the LOD2 filename and object ids."""
    from src.dataset.dataset import parse_cityjson_file_to_graphs
    from src.filter_cityjson import process_dataset

    raw, out = _raw_tree(tmp_path)
    totals = process_dataset(raw, out, max_slenderness=None, min_extent=0.0,
                             convert_on_the_fly=True)

    assert totals["LOD2"]["files"] == 1
    assert totals["LOD1_synth"]["files"] == 1
    assert (out / "LOD1_synth" / "Source A" / "description.txt").exists()

    lod2 = parse_cityjson_file_to_graphs(out / "LOD2" / "Source A" / "tile.city.json")
    synth = parse_cityjson_file_to_graphs(out / "LOD1_synth" / "Source A" / "tile.city.json")
    assert set(synth) == set(lod2)
    g1, g2 = next(iter(synth.values())), next(iter(lod2.values()))
    assert g1["x"].shape[0] < g2["x"].shape[0]        # LOD1 is the coarser of the pair


def test_real_lod1_and_lod2_are_split_into_their_own_folders(tmp_path):
    """A 3DBAG-shaped part carrying 1.2 and 2.2 lands in both LOD folders."""
    import json as _json
    from src.filter_cityjson import process_dataset

    verts, boundaries, sem = box(0, 0, 8, 12, 6)
    raw = tmp_path / "raw" / "Source B"
    raw.mkdir(parents=True)
    cj = cityjson({
        "pand": {"type": "Building", "children": ["pand-0"], "attributes": {"year": 1900},
                 "geometry": [{"type": "MultiSurface", "lod": "0", "boundaries": [[[0, 1, 2, 3]]]}]},
        "pand-0": {"type": "BuildingPart", "parents": ["pand"],
                   "geometry": [{"type": "Solid", "lod": "1.2", "boundaries": boundaries, "semantics": sem},
                                {"type": "Solid", "lod": "2.2", "boundaries": boundaries, "semantics": sem}]},
    }, verts)
    (raw / "tile.city.json").write_text(_json.dumps(cj), encoding="utf-8")

    out = tmp_path / "out"
    totals = process_dataset(tmp_path / "raw", out, max_slenderness=None, min_extent=0.0)

    for lod in ("LOD1", "LOD2"):
        written = _json.loads((out / lod / "Source B" / "tile.city.json").read_text(encoding="utf-8"))
        assert set(written["CityObjects"]) == {"pand-0"}, f"{lod} kept the wrong objects"
        assert len(written["CityObjects"]["pand-0"]["geometry"]) == 1
        assert totals[lod]["kept"] == 1
    assert _json.loads((out / "LOD1" / "Source B" / "tile.city.json")
                       .read_text(encoding="utf-8"))["CityObjects"]["pand-0"]["geometry"][0]["lod"] == "1.2"


def test_returns_none_when_nothing_survives():
    from src.filter_cityjson import filter_cityjson
    cj = cityjson({"pand": {"type": "Building",
                            "geometry": [{"type": "MultiSurface", "lod": "0", "boundaries": [[[0, 1, 2]]]}]}},
                  [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    assert filter_cityjson(cj, lod=2) is None
