"""Unit tests for near-identical LOD1 grouping.

Hand-built geometry only -- never the real dataset. The point of these is to
pin the *equivalence relation*: two buildings belong together iff the mesh
transformer cannot tell their conditions apart, which is a statement about
`normalize_to_unit_box` + `quantize`, not about metres.
"""
import numpy as np
import pytest

from src.analysis.lod1_duplicates import (
    group_identical, group_spread, hausdorff_bins, object_vertices, shape_key,
)


# --- fixtures -------------------------------------------------------------

def box(size=(10.0, 6.0, 4.0), at=(0.0, 0.0, 0.0)):
    """The 8 corners of an axis-aligned box -- an LOD1 building, wireframe only."""
    s, o = np.asarray(size, dtype=float), np.asarray(at, dtype=float)
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
                       dtype=float)
    return corners * s + o


def gabled(size=(10.0, 6.0, 4.0), ridge=6.0, at=(0.0, 0.0, 0.0)):
    """A box with a ridge line on top -- stands in for an LOD2 target."""
    s, o = np.asarray(size, dtype=float), np.asarray(at, dtype=float)
    walls = box(size, at)
    ridge_pts = np.array([[0.5, 0.0, 0.0], [0.5, 1.0, 0.0]], dtype=float) * s + o
    ridge_pts[:, 2] = ridge + o[2]
    return np.vstack([walls, ridge_pts])


def solid(rings_per_face):
    """A CityJSON Solid whose boundaries are the given per-face ring lists."""
    return {"type": "Solid", "lod": "1.2", "boundaries": [rings_per_face]}


# --- the descriptor -------------------------------------------------------

def test_translated_copies_share_a_key():
    """Position is not shape. The frame is centred on the object's own box."""
    a, _ = shape_key(box(at=(0, 0, 0)))
    b, _ = shape_key(box(at=(84000.0, 452000.0, 12.0)))   # RD New coordinates
    assert a == b


def test_uniformly_scaled_copy_shares_a_key():
    """A 2x building is the SAME condition to the model.

    `normalize_to_unit_box` divides by the longest box axis, so absolute size
    is erased before the model ever sees it. This is the invariance that makes
    the group's LOD2 spread an exact ceiling rather than a bound -- and it is
    also the most surprising property of the grouping, so it is pinned here.
    """
    a, _ = shape_key(box(size=(10.0, 6.0, 4.0)))
    b, _ = shape_key(box(size=(20.0, 12.0, 8.0)))
    assert a == b


def test_anisotropically_scaled_copy_does_not_share_a_key():
    """Only the *longest* axis is normalized away; proportions are shape."""
    a, _ = shape_key(box(size=(10.0, 6.0, 4.0)))
    b, _ = shape_key(box(size=(10.0, 10.0, 4.0)))
    assert a != b


def test_key_is_invariant_to_vertex_order():
    """A CityJSON file may list the same corners in any order."""
    verts = box()
    rng = np.random.default_rng(0)
    a, _ = shape_key(verts)
    b, _ = shape_key(verts[rng.permutation(len(verts))])
    assert a == b


def test_key_ignores_duplicate_vertices():
    """Rings share corners, so the same point arrives many times."""
    verts = box()
    a, _ = shape_key(verts)
    b, _ = shape_key(np.vstack([verts, verts[:3], verts[-1:]]))
    assert a == b


def test_different_shapes_get_different_keys():
    a, _ = shape_key(box())
    b, _ = shape_key(gabled())
    assert a != b


def test_coarser_bins_merge_what_the_model_grid_separates():
    """`num_bins` IS the tolerance knob -- no separate epsilon.

    Two boxes differing by 5% of their longest axis are distinct on the
    model's 128-bin grid and identical on a 16-bin one. The perturbation has
    to clear a whole bin to be visible at all: at 128 bins a 1% difference
    rounds to the same two bins, which is the tolerance working as intended.
    """
    a = box(size=(10.0, 6.0, 4.0))
    b = box(size=(10.0, 6.5, 4.0))
    assert shape_key(a, num_bins=128)[0] != shape_key(b, num_bins=128)[0]
    assert shape_key(a, num_bins=16)[0] == shape_key(b, num_bins=16)[0]


def test_quantized_output_is_on_the_grid():
    _, q = shape_key(box(), num_bins=128)
    assert q.dtype == np.int64
    assert q.min() >= 0 and q.max() <= 127
    # The longest axis spans the full grid; the margin only lifts +z.
    assert q[:, 0].max() - q[:, 0].min() == 127


def test_target_is_quantized_in_the_condition_frame():
    """LOD2 must use the LOD1 box, or the height difference is normalized away.

    Same contract as `MeshDataset`: `ref` fixes the frame. A ridge above the
    LOD1 box has to land above the box's top bin, not on it.
    """
    lod1 = box(size=(10.0, 6.0, 4.0))
    lod2 = gabled(size=(10.0, 6.0, 4.0), ridge=5.0)
    _, q = shape_key(lod2, ref=lod1)
    _, q1 = shape_key(lod1, ref=lod1)
    assert q[:, 2].max() > q1[:, 2].max()


# --- grouping -------------------------------------------------------------

def test_groups_drop_singletons():
    keys = {"a": b"x", "b": b"x", "c": b"y"}
    assert group_identical(keys) == [["a", "b"]]


def test_groups_are_deterministic_and_sorted():
    keys = {"z": b"x", "a": b"x", "m": b"y", "b": b"y", "q": b"y"}
    assert group_identical(keys) == [["b", "m", "q"], ["a", "z"]]


def test_empty_input_groups_to_nothing():
    assert group_identical({}) == []


# --- spread ---------------------------------------------------------------

def test_hausdorff_is_symmetric_and_zero_on_itself():
    q = shape_key(box())[1]
    assert hausdorff_bins(q, q) == 0.0
    other = shape_key(gabled())[1]
    assert hausdorff_bins(q, other) == pytest.approx(hausdorff_bins(other, q))


def test_spread_is_zero_when_every_target_is_the_same():
    """The floor case: identical LOD1s that also share an LOD2 cost nothing."""
    lod1 = box()
    q = shape_key(gabled(), ref=lod1)[1]
    out = group_spread(["a", "b", "c"], {"a": q, "b": q, "c": q},
                       {k: np.array([10.0, 6.0, 4.4]) for k in "abc"})
    assert out["max_bins"] == 0.0
    assert out["mean_bins"] == 0.0
    assert out["max_m"] == 0.0


def test_spread_grows_with_target_disagreement():
    """The whole point: same condition, different roofs, measurable spread."""
    lod1 = box(size=(10.0, 6.0, 4.0))
    flat = shape_key(box(size=(10.0, 6.0, 4.0)), ref=lod1)[1]
    low = shape_key(gabled(size=(10.0, 6.0, 4.0), ridge=4.5), ref=lod1)[1]
    high = shape_key(gabled(size=(10.0, 6.0, 4.0), ridge=5.0), ref=lod1)[1]
    scales = {k: np.array([10.0, 10.0, 11.0]) for k in ("a", "b", "c")}
    near = group_spread(["a", "b"], {"a": flat, "b": low}, scales)
    far = group_spread(["a", "c"], {"a": flat, "c": high}, scales)
    assert 0.0 < near["max_bins"] < far["max_bins"]


def test_spread_reports_metres_via_the_frame_scale():
    """Bins are the model's unit; metres are what a reviewer reads."""
    lod1 = box(size=(10.0, 6.0, 4.0))
    a = shape_key(box(size=(10.0, 6.0, 4.0)), ref=lod1)[1]
    b = shape_key(gabled(size=(10.0, 6.0, 4.0), ridge=5.0), ref=lod1)[1]
    scale = np.array([11.0, 11.0, 11.0])
    out = group_spread(["a", "b"], {"a": a, "b": b}, {"a": scale, "b": scale})
    assert out["max_m"] == pytest.approx(out["max_bins"] * 11.0 / 127.0)


def test_spread_of_a_single_member_is_undefined_not_zero():
    """A group of one has no disagreement to measure; zero would read as agreement."""
    q = shape_key(box())[1]
    out = group_spread(["a"], {"a": q}, {"a": np.ones(3)})
    assert np.isnan(out["max_bins"])


# --- CityJSON wireframe extraction ---------------------------------------

def test_object_vertices_gathers_every_ring_without_triangulating():
    """Interior rings are geometry too -- a courtyard has corners."""
    verts = np.arange(24, dtype=float).reshape(8, 3)
    obj = {"type": "Building",
           "geometry": [solid([[[0, 1, 2, 3], [4, 5, 6]], [[1, 2, 7]]])]}
    got = object_vertices(obj, verts)
    assert len(got) == 8
    assert np.array_equal(np.unique(got, axis=0), np.unique(verts, axis=0))


def test_object_vertices_returns_unique_points():
    """The same corner is referenced by every face that touches it."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    obj = {"type": "Building",
           "geometry": [solid([[[0, 1, 2]], [[0, 2, 3]], [[0, 1, 2, 3]]])]}
    assert len(object_vertices(obj, verts)) == 4


def test_object_vertices_is_empty_for_an_unusable_geometry():
    obj = {"type": "Building", "geometry": [{"type": "MultiPoint",
                                             "boundaries": [0, 1]}]}
    assert len(object_vertices(obj, np.zeros((2, 3)))) == 0


def test_object_vertices_skips_degenerate_rings():
    """A two-point ring is not a surface; `_iter_surfaces` already drops it."""
    verts = np.arange(15, dtype=float).reshape(5, 3)
    obj = {"type": "Building", "geometry": [solid([[[0, 1]], [[2, 3, 4]]])]}
    assert len(object_vertices(obj, verts)) == 3
