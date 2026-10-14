"""Unit tests for the LOD1-vs-LOD2 identity scan.

Synthetic CityJSON in a temp directory only -- never the real dataset. The
volumetric measure itself is pinned in `test_exact_volume`; what is tested here
is the corpus plumbing around it: tile pairing, object matching, and that one
bad tile cannot take down a 250-tile pass.
"""
import json

import pytest

from src.analysis.identity_pairs import scan_tile, tile_pairs

# The same cube geometry as tests/test_cityobject_analysis.py, which is
# outward-oriented -- exact CSG rejects an inside-out solid, so this matters.
CUBE_VERTS = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
              [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
CUBE_FACES = [[[0, 3, 2, 1]], [[4, 5, 6, 7]], [[0, 1, 5, 4]],
              [[1, 2, 6, 5]], [[2, 3, 7, 6]], [[3, 0, 4, 7]]]


def write_tile(path, objects, scale=(10.0, 10.0, 10.0)):
    """A CityJSON tile holding one scaled cube per entry of `objects`.

    `objects` maps object id to a z-scale multiplier, so two tiles can hold the
    same building at different heights.
    """
    verts, city = [], {}
    for oid, zmul in objects.items():
        base = len(verts)
        verts += [[x * scale[0], y * scale[1], z * scale[2] * zmul]
                  for x, y, z in CUBE_VERTS]
        city[oid] = {"type": "Building", "geometry": [
            {"type": "Solid", "lod": "1.2",
             "boundaries": [[[[i + base for i in ring] for ring in face]
                             for face in CUBE_FACES]]}]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "CityJSON", "version": "2.0",
                                "vertices": verts, "CityObjects": city}))


def make_corpus(tmp_path, lod1, lod2, rel="srcA/t1.json"):
    write_tile(tmp_path / "LOD1" / rel, lod1)
    write_tile(tmp_path / "LOD2" / rel, lod2)
    return tile_pairs(tmp_path)[0]


# --- tile pairing ---------------------------------------------------------

def test_tile_pairs_matches_on_the_path_below_the_lod_folder(tmp_path):
    write_tile(tmp_path / "LOD1" / "srcA" / "t1.json", {"a": 1.0})
    write_tile(tmp_path / "LOD2" / "srcA" / "t1.json", {"a": 1.0})
    got = tile_pairs(tmp_path)
    assert len(got) == 1
    rel, a, b = got[0]
    assert rel.replace("\\", "/") == "srcA/t1.json"
    assert a.name == b.name == "t1.json"


def test_tile_pairs_ignores_a_tile_present_at_only_one_lod(tmp_path):
    """The LOD folders do not always share a tile set."""
    write_tile(tmp_path / "LOD1" / "t1.json", {"a": 1.0})
    write_tile(tmp_path / "LOD1" / "orphan.json", {"b": 1.0})
    write_tile(tmp_path / "LOD2" / "t1.json", {"a": 1.0})
    assert len(tile_pairs(tmp_path)) == 1


def test_tile_pairs_is_sorted_so_a_rerun_reproduces(tmp_path):
    for name in ("c", "a", "b"):
        write_tile(tmp_path / "LOD1" / f"{name}.json", {"a": 1.0})
        write_tile(tmp_path / "LOD2" / f"{name}.json", {"a": 1.0})
    rels = [r for r, _, _ in tile_pairs(tmp_path)]
    assert rels == sorted(rels)


def test_tile_pairs_raises_on_a_missing_lod_folder(tmp_path):
    write_tile(tmp_path / "LOD1" / "t1.json", {"a": 1.0})
    with pytest.raises(FileNotFoundError):
        tile_pairs(tmp_path)


# --- scanning a tile ------------------------------------------------------

def test_an_lod2_identical_to_its_lod1_scores_zero(tmp_path):
    """The case the whole scan exists to count."""
    pair = make_corpus(tmp_path, {"a": 1.0}, {"a": 1.0})
    rows = scan_tile(pair)
    assert len(rows) == 1
    assert rows[0]["object_id"] == "a"
    assert rows[0]["ok"] is True
    assert rows[0]["symdiff_m3"] == pytest.approx(0.0, abs=1e-9)
    assert rows[0]["rel_symdiff"] == pytest.approx(0.0, abs=1e-12)


def test_a_taller_lod2_registers_the_exact_extra_volume(tmp_path):
    """A 10x10x10 LOD1 against a 10x10x15 LOD2 differs by exactly 500 m^3."""
    pair = make_corpus(tmp_path, {"a": 1.0}, {"a": 1.5})
    row = scan_tile(pair)[0]
    assert row["symdiff_m3"] == pytest.approx(500.0, rel=1e-9)
    assert row["vol_lod1_m3"] == pytest.approx(1000.0, rel=1e-9)
    assert row["vol_lod2_m3"] == pytest.approx(1500.0, rel=1e-9)


def test_only_objects_present_at_both_lods_are_scanned(tmp_path):
    pair = make_corpus(tmp_path, {"a": 1.0, "only1": 1.0}, {"a": 1.0, "only2": 1.0})
    assert [r["object_id"] for r in scan_tile(pair)] == ["a"]


def test_rows_carry_the_tile_so_the_explorer_can_load_lazily(tmp_path):
    """The explorer stores names only and re-opens the tile on selection."""
    pair = make_corpus(tmp_path, {"a": 1.0}, {"a": 1.0})
    assert scan_tile(pair)[0]["tile"].replace("\\", "/") == "srcA/t1.json"


def test_a_corrupt_tile_yields_no_rows_instead_of_raising(tmp_path):
    """One bad tile must not kill a 250-tile pass."""
    write_tile(tmp_path / "LOD1" / "t1.json", {"a": 1.0})
    write_tile(tmp_path / "LOD2" / "t1.json", {"a": 1.0})
    (tmp_path / "LOD2" / "t1.json").write_text("{ this is not json")
    assert scan_tile(tile_pairs(tmp_path)[0]) == []
