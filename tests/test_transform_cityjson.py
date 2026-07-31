"""Unit tests for the geometry repair pass.

Hand-built solids only. Every repair is pinned by a test that fails if the
repair stops firing *or* starts firing on clean geometry -- the second is the
dangerous direction, since a wrong "fix" silently corrupts training data.
"""
import pytest

from src.transform_cityjson import (
    REQUIRED_SEMANTICS, transform_cityjson, transform_object,
)

# --- fixtures -------------------------------------------------------------

CUBE_VERTS = [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
]
CUBE_FACES = [[[0, 3, 2, 1]], [[4, 5, 6, 7]], [[0, 1, 5, 4]],
              [[1, 2, 6, 5]], [[2, 3, 7, 6]], [[3, 0, 4, 7]]]
CUBE_SEM = ["GroundSurface", "RoofSurface", "WallSurface",
            "WallSurface", "WallSurface", "WallSurface"]


def make_object(faces=None, sem=None, verts=None):
    faces = [[list(r) for r in f] for f in (faces or CUBE_FACES)]
    sem = list(sem or CUBE_SEM)
    return {
        "type": "Building",
        "geometry": [{
            "type": "Solid", "lod": "2.2", "boundaries": [faces],
            "semantics": {"surfaces": [{"type": t} for t in sem],
                          "values": [list(range(len(sem)))]},
        }],
    }


def make_cj(objects, verts=None):
    return {"type": "CityJSON", "version": "1.1",
            "CityObjects": objects, "vertices": verts or CUBE_VERTS}


def faces_of(obj):
    return obj["geometry"][0]["boundaries"][0]


def types_of(obj):
    geom = obj["geometry"][0]
    surfaces = geom["semantics"]["surfaces"]
    return [surfaces[v]["type"] if v is not None else None
            for v in geom["semantics"]["values"][0]]


# --- clean geometry must be left alone ------------------------------------

def test_a_clean_cube_is_kept_and_untouched():
    obj = make_object()
    keep, fixes = transform_object(obj, CUBE_VERTS)
    assert keep
    assert fixes == {}
    assert faces_of(obj) == CUBE_FACES
    assert types_of(obj) == CUBE_SEM


# --- degenerate faces -----------------------------------------------------

def test_zero_area_face_is_dropped():
    verts = CUBE_VERTS + [[2, 0, 0], [3, 0, 0], [4, 0, 0]]   # collinear
    obj = make_object(CUBE_FACES + [[[8, 9, 10]]], CUBE_SEM + ["WallSurface"], verts)
    keep, fixes = transform_object(obj, verts)
    assert keep
    assert fixes["dropped_zero_area_face"] == 1
    assert len(faces_of(obj)) == 6
    assert types_of(obj) == CUBE_SEM


def test_consecutive_repeat_in_a_ring_is_deduped():
    faces = [list(f) for f in CUBE_FACES]
    faces[1] = [[4, 5, 5, 6, 7]]
    obj = make_object(faces)
    keep, fixes = transform_object(obj, CUBE_VERTS)
    assert keep
    assert fixes["deduped_ring_vertex"] == 1
    assert faces_of(obj)[1] == [[4, 5, 6, 7]]


def test_non_consecutive_repeat_is_kept_as_a_pinch_point():
    """Pinched footprints are legitimate and the Levi wireframe handles them.

    Dropping them cascades into the missing-ground test and takes some of the
    largest buildings in the corpus with it.
    """
    faces = [list(f) for f in CUBE_FACES]
    faces[1] = [[4, 5, 6, 5, 7]]
    obj = make_object(faces)
    keep, fixes = transform_object(obj, CUBE_VERTS)
    assert keep
    assert len(faces_of(obj)) == 6
    assert faces_of(obj)[1] == [[4, 5, 6, 5, 7]]


def test_degenerate_hole_is_dropped_without_losing_the_face():
    faces = [list(f) for f in CUBE_FACES]
    faces[0] = [CUBE_FACES[0][0], [1, 1, 1]]        # outer ring plus a collapsed hole
    obj = make_object(faces)
    keep, fixes = transform_object(obj, CUBE_VERTS)
    assert keep
    assert fixes["dropped_degenerate_hole"] == 1
    assert faces_of(obj)[0] == [CUBE_FACES[0][0]]


def test_semantics_stay_aligned_with_boundaries_after_a_drop():
    verts = CUBE_VERTS + [[2, 0, 0], [3, 0, 0], [4, 0, 0]]
    # degenerate face in the *middle* so a misaligned rebuild shifts the labels
    faces = CUBE_FACES[:2] + [[[8, 9, 10]]] + CUBE_FACES[2:]
    sem = CUBE_SEM[:2] + ["RoofSurface"] + CUBE_SEM[2:]
    obj = make_object(faces, sem, verts)
    transform_object(obj, verts)
    assert types_of(obj) == CUBE_SEM
    assert faces_of(obj) == CUBE_FACES


# --- orientation ----------------------------------------------------------

def test_reversed_ground_is_reversed_back_when_that_repairs_closure():
    faces = [list(f) for f in CUBE_FACES]
    faces[0] = [CUBE_FACES[0][0][::-1]]           # ground now points up
    obj = make_object(faces)
    keep, fixes = transform_object(obj, CUBE_VERTS)
    assert keep
    assert fixes["reversed_ring"] == 1
    assert faces_of(obj)[0] == CUBE_FACES[0]


def test_ground_sliver_is_dropped_when_reversing_cannot_help():
    """A detached upward ground triangle: reversing changes no edge pairing."""
    verts = CUBE_VERTS + [[10, 10, 5], [11, 10, 5], [10, 11, 5]]
    obj = make_object(CUBE_FACES + [[[8, 9, 10]]],
                      CUBE_SEM + ["GroundSurface"], verts)
    keep, fixes = transform_object(obj, verts)
    assert keep                                    # the real ground survives
    assert fixes["dropped_unfixable_ground"] == 1
    assert len(faces_of(obj)) == 6


def test_downward_roof_is_left_alone_when_reversing_would_not_help():
    """Most downward roofs are mislabelled soffits, not winding errors."""
    verts = CUBE_VERTS + [[10, 10, 5], [10, 11, 5], [11, 10, 5]]   # detached, faces down
    obj = make_object(CUBE_FACES + [[[8, 9, 10]]],
                      CUBE_SEM + ["RoofSurface"], verts)
    keep, fixes = transform_object(obj, verts)
    assert keep
    assert fixes["left_downward_roof"] == 1
    assert "reversed_ring" not in fixes
    assert len(faces_of(obj)) == 7                                  # nothing removed


# --- semantic relabel -----------------------------------------------------

def test_outer_floor_surface_is_relabelled_when_no_roof_exists():
    sem = ["GroundSurface", "OuterFloorSurface"] + ["WallSurface"] * 4
    obj = make_object(sem=sem)
    keep, fixes = transform_object(obj, CUBE_VERTS)
    assert keep
    assert fixes["relabelled_roof"] == 1
    assert types_of(obj)[1] == "RoofSurface"


def test_outer_floor_surface_is_left_alone_when_a_roof_exists():
    verts = CUBE_VERTS + [[0, 0, 2], [1, 0, 2], [1, 1, 2]]
    sem = CUBE_SEM + ["OuterFloorSurface"]
    obj = make_object(CUBE_FACES + [[[8, 9, 10]]], sem, verts)
    transform_object(obj, verts)
    assert "OuterFloorSurface" in types_of(obj)


def test_relabel_does_not_disturb_faces_sharing_the_old_surface_entry():
    """Two faces on one semantics index: relabelling must split, not mutate."""
    geom = {
        "type": "Solid", "lod": "2.2",
        "boundaries": [[[list(r) for r in f] for f in CUBE_FACES]],
        "semantics": {
            "surfaces": [{"type": "GroundSurface"}, {"type": "OuterFloorSurface"},
                         {"type": "WallSurface"}],
            "values": [[0, 1, 2, 2, 2, 2]],
        },
    }
    obj = {"type": "Building", "geometry": [geom]}
    transform_object(obj, CUBE_VERTS)
    assert types_of(obj) == CUBE_SEM


# --- object-level drops ---------------------------------------------------

@pytest.mark.parametrize("missing", REQUIRED_SEMANTICS)
def test_object_missing_a_required_surface_is_dropped(missing):
    sem = ["WallSurface" if t == missing else t for t in CUBE_SEM]
    obj = make_object(sem=sem)
    keep, _ = transform_object(obj, CUBE_VERTS)
    assert keep is (missing == "WallSurface")


def test_object_whose_only_ground_was_a_sliver_is_dropped():
    verts = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
             [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
             [10, 10, 5], [11, 10, 5], [10, 11, 5]]
    faces = CUBE_FACES[1:] + [[[8, 9, 10]]]
    sem = CUBE_SEM[1:] + ["GroundSurface"]
    obj = make_object(faces, sem, verts)
    keep, fixes = transform_object(obj, verts)
    assert fixes["dropped_unfixable_ground"] == 1
    assert not keep


# --- file level -----------------------------------------------------------

def test_transform_cityjson_drops_objects_and_compacts_vertices():
    good = make_object()
    bad = make_object(sem=["GroundSurface"] * 6)          # no roof, no wall
    cj = make_cj({"good": good, "bad": bad})
    out, stats, dropped = transform_cityjson(cj)
    assert set(out["CityObjects"]) == {"good"}
    assert dropped == {"bad"}
    assert len(out["vertices"]) == 8
    assert stats["dropped_objects"] == 1


def test_transform_cityjson_honours_an_external_drop_set():
    cj = make_cj({"a": make_object(), "b": make_object()})
    out, _, _ = transform_cityjson(cj, drop_ids={"b"})
    assert set(out["CityObjects"]) == {"a"}


def test_regenerate_synth_pairs_with_the_cleaned_lod2(tmp_path):
    """Derived LOD1 must carry the LOD2 ids and be tagged lod 1, not lod 2."""
    import json

    from src.transform_cityjson import regenerate_synth

    lod2, synth = tmp_path / "LOD2", tmp_path / "LOD1_synth"
    (lod2 / "Source A").mkdir(parents=True)
    cj = make_cj({"a": make_object(), "b": make_object()})
    (lod2 / "Source A" / "t.json").write_text(json.dumps(cj), encoding="utf-8")

    stats = regenerate_synth(lod2, synth)
    out = json.loads((synth / "Source A" / "t.json").read_text(encoding="utf-8"))
    assert set(out["CityObjects"]) == {"a", "b"}
    assert stats["dropped_objects"] == 0
    for obj in out["CityObjects"].values():
        assert str(obj["geometry"][0]["lod"]).startswith("1")


def test_transform_cityjson_returns_none_when_nothing_survives():
    cj = make_cj({"bad": make_object(sem=["GroundSurface"] * 6)})
    out, _, dropped = transform_cityjson(cj)
    assert out is None
    assert dropped == {"bad"}
