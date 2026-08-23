"""Correct triangulation of a CityJSON surface.

A surface is ``[exterior_ring, hole, hole, ...]`` (CityJSON 2.0.1: "the first
array being the exterior boundary of the surface, and the others the interior
boundaries"). The previous fan-from-vertex-0 was only correct for a convex
single-ring face; these pin the two cases it got wrong -- concave rings and
holes -- plus the winding contract the fan used to give for free.
"""
import numpy as np
import pytest

from src.geometry.geometry import triangulate_face


def area(verts, tris):
    v = np.asarray(verts, dtype=float)
    return float(sum(0.5 * np.linalg.norm(np.cross(v[b] - v[a], v[c] - v[a]))
                     for a, b, c in tris))


def newell(points):
    p = np.asarray(points, dtype=float)
    return np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)


# A 10x10 square with a 3x3 hole, on z=0.
SQUARE = np.array([[0., 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],
                   [3, 3, 0], [3, 6, 0], [6, 6, 0], [6, 3, 0]])
# A U-shape, area 28. Deliberately not star-shaped from vertex 0: the notch
# hides part of the polygon from (0,0), so a fan rooted there spills across it.
# An L-shape will not do -- vertex 0 sees every other vertex and the fan is
# accidentally correct, which is a fixture that proves nothing.
U_SHAPE = np.array([[0., 0, 0], [6, 0, 0], [6, 6, 0], [4, 6, 0],
                    [4, 2, 0], [2, 2, 0], [2, 6, 0], [0, 6, 0]])


def test_convex_quad_is_two_triangles():
    tris = triangulate_face([[0, 1, 2, 3]], SQUARE)
    assert len(tris) == 2
    assert area(SQUARE, tris) == pytest.approx(100.0)


def test_bare_triangle_survives():
    tris = triangulate_face([[0, 1, 2]], SQUARE)
    assert len(tris) == 1
    assert area(SQUARE, tris) == pytest.approx(50.0)


def test_concave_ring_does_not_overshoot():
    """The regression: a fan from vertex 0 covers more than the polygon."""
    ring = list(range(len(U_SHAPE)))
    tris = triangulate_face([ring], U_SHAPE)
    assert area(U_SHAPE, tris) == pytest.approx(28.0)

    fan = [(0, k, k + 1) for k in range(1, len(ring) - 1)]
    assert area(U_SHAPE, fan) == pytest.approx(44.0)                # the old bug
    assert area(U_SHAPE, tris) < area(U_SHAPE, fan)


def test_concave_ring_keeps_the_same_triangle_count():
    """n-2 either way: earcut adds no Steiner points, so tokens do not move."""
    ring = list(range(len(U_SHAPE)))
    assert len(triangulate_face([ring], U_SHAPE)) == len(ring) - 2


def test_hole_is_not_paved_over():
    tris = triangulate_face([[0, 1, 2, 3], [4, 5, 6, 7]], SQUARE)
    assert area(SQUARE, tris) == pytest.approx(100.0 - 9.0)


def test_hole_contains_no_triangle_centroid():
    """Area alone could be right with a triangle straddling the hole."""
    tris = triangulate_face([[0, 1, 2, 3], [4, 5, 6, 7]], SQUARE)
    for a, b, c in tris:
        cx, cy, _ = (SQUARE[a] + SQUARE[b] + SQUARE[c]) / 3.0
        assert not (3 < cx < 6 and 3 < cy < 6), "triangle sits inside the hole"


def test_triangles_are_wound_with_the_exterior_ring():
    for ring in ([0, 1, 2, 3], [3, 2, 1, 0]):
        tris = triangulate_face([ring], SQUARE)
        want = newell(SQUARE[ring])
        want = want / np.linalg.norm(want)
        for a, b, c in tris:
            got = np.cross(SQUARE[b] - SQUARE[a], SQUARE[c] - SQUARE[a])
            assert float(got @ want) > 0, "triangle wound against the ring"


def test_works_on_a_tilted_plane():
    """Projection must use the face normal, not a fixed axis."""
    theta = 0.7
    rot = np.array([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])
    tilted = SQUARE @ rot.T
    tris = triangulate_face([[0, 1, 2, 3], [4, 5, 6, 7]], tilted)
    assert area(tilted, tris) == pytest.approx(91.0)


def test_degenerate_rings_yield_nothing():
    assert triangulate_face([[0, 1]], SQUARE) == []
    assert triangulate_face([], SQUARE) == []


def test_zero_area_ring_yields_nothing():
    collinear = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    assert triangulate_face([[0, 1, 2, 3]], collinear) == []


def test_indices_are_the_callers_indices():
    """Triangles must come back in the input's vertex numbering, not a local one."""
    tris = triangulate_face([[0, 1, 2, 3], [4, 5, 6, 7]], SQUARE)
    used = {i for t in tris for i in t}
    assert used <= set(range(len(SQUARE)))
    assert used & {4, 5, 6, 7}, "hole vertices should appear in the triangulation"


def test_missing_engine_raises_instead_of_returning_empty(monkeypatch):
    """An absent backend is an environment fault, not a degenerate face.

    cjio declares mapbox-earcut only as the `export` extra and then calls it
    unguarded, so without this the failure is a bare NameError from inside a
    library -- or worse, silently empty meshes if a caller treats [] as "skip".
    """
    from cjio import geom_help
    monkeypatch.setattr(geom_help, "MODULE_EARCUT_AVAILABLE", False)
    with pytest.raises(ImportError, match="mapbox-earcut"):
        triangulate_face([[0, 1, 2, 3]], SQUARE)


def test_degenerate_face_still_returns_empty_when_the_engine_is_present():
    """The other half of that contract: [] means the face, not the install."""
    assert triangulate_face([[0, 1]], SQUARE) == []
