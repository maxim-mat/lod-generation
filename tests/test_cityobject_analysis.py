"""Unit tests for the CityObject analysis statistics.

Hand-built solids only -- never the real dataset. Pins the geometry maths the
analysis draws its conclusions from; the plotting layer is exempt.
"""
import numpy as np
import pytest

from src.analysis.cityobject_analysis import (
    connected_components, count_coincident, face_area, face_normal,
    iter_faces, is_watertight, newell_vector, object_defects, object_metrics,
    planarity_ratio, signed_volume, vertex_degrees, wireframe_edges,
)


# --- fixtures -------------------------------------------------------------

def unit_cube():
    """Vertices and outward-oriented faces of the unit cube."""
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    faces = [[[0, 3, 2, 1]], [[4, 5, 6, 7]], [[0, 1, 5, 4]],
             [[1, 2, 6, 5]], [[2, 3, 7, 6]], [[3, 0, 4, 7]]]
    return verts, faces


DEFAULT_SEM = ("GroundSurface", "RoofSurface", "WallSurface",
               "WallSurface", "WallSurface", "WallSurface")


def cube_solid(sem_types=DEFAULT_SEM):
    """A CityJSON Solid for the unit cube with per-face semantic types."""
    _, faces = unit_cube()
    surfaces, values = [], []
    for t in sem_types:
        surfaces.append({"type": t})
        values.append(len(surfaces) - 1)
    return {"type": "Solid", "lod": "2.2", "boundaries": [faces],
            "semantics": {"surfaces": surfaces, "values": [values]}}


def cube_object(sem_types=DEFAULT_SEM):
    """The Solid wrapped in a CityObject, which is what the analysis consumes."""
    return {"type": "Building", "geometry": [cube_solid(sem_types)]}


# --- normals, areas -------------------------------------------------------

def test_newell_and_normal_of_an_upward_square():
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    np.testing.assert_allclose(newell_vector(pts), [0, 0, 2])
    np.testing.assert_allclose(face_normal(pts), [0, 0, 1])
    assert face_area(pts) == pytest.approx(1.0)


def test_reversing_a_ring_flips_the_normal():
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    np.testing.assert_allclose(face_normal(pts[::-1]), [0, 0, -1])


def test_area_of_a_slanted_face_uses_true_3d_extent():
    # 1 x 1 footprint rising 1 in z -> sqrt(2) area, not 1
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 1], [0, 1, 1]], dtype=float)
    assert face_area(pts) == pytest.approx(np.sqrt(2))


def test_degenerate_ring_has_no_normal():
    collinear = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    assert face_normal(collinear) is None


# --- planarity ------------------------------------------------------------

def test_planar_quad_has_zero_out_of_plane_deviation():
    pts = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]], dtype=float)
    assert planarity_ratio(pts) == pytest.approx(0.0, abs=1e-12)


def test_warped_quad_is_detected_and_scales_with_the_warp():
    def warped(h):
        return np.array([[0, 0, 0], [2, 0, 0], [2, 2, h], [0, 2, 0]], dtype=float)
    small, large = planarity_ratio(warped(0.1)), planarity_ratio(warped(1.0))
    assert 0 < small < large
    assert large > 0.05          # comfortably above the 2% defect fence


def test_triangles_are_always_planar():
    pts = np.array([[0, 0, 0], [3, 1, 7], [-2, 5, 1]], dtype=float)
    assert planarity_ratio(pts) == pytest.approx(0.0, abs=1e-12)


# --- face iteration (must mirror dataset._iter_faces) ---------------------

def test_solid_faces_are_flattened_across_shells():
    got = list(iter_faces(cube_solid()))
    assert len(got) == 6
    assert [t for _, t in got][:2] == ["GroundSurface", "RoofSurface"]


def test_multisurface_boundaries_are_used_directly():
    geom = {"type": "MultiSurface", "lod": "2",
            "boundaries": [[[0, 1, 2]], [[0, 1, 3]]]}
    assert [r for r, _ in iter_faces(geom)] == [[0, 1, 2], [0, 1, 3]]


def test_unsupported_geometry_types_yield_nothing():
    for gtype in ("MultiSolid", "CompositeSolid", "MultiPoint", "GeometryInstance"):
        geom = {"type": gtype, "lod": "2", "boundaries": [[[[0, 1, 2]]]]}
        assert list(iter_faces(geom)) == []


def test_rings_with_fewer_than_three_vertices_are_skipped():
    geom = {"type": "MultiSurface", "lod": "2", "boundaries": [[[0, 1]], [[0, 1, 2]]]}
    assert [r for r, _ in iter_faces(geom)] == [[0, 1, 2]]


def test_only_the_outer_ring_is_taken():
    geom = {"type": "MultiSurface", "lod": "2",
            "boundaries": [[[0, 1, 2, 3], [4, 5, 6]]]}      # outer + one hole
    assert [r for r, _ in iter_faces(geom)] == [[0, 1, 2, 3]]


# --- wireframe ------------------------------------------------------------

def test_cube_wireframe_has_twelve_edges_and_uniform_degree():
    _, faces = unit_cube()
    rings = [f[0] for f in faces]
    assert len(wireframe_edges(rings)) == 12
    assert set(vertex_degrees(rings).values()) == {3}


def test_connected_components_counts_disjoint_pieces():
    _, faces = unit_cube()
    rings = [f[0] for f in faces]
    assert connected_components(rings) == 1
    shifted = [[v + 8 for v in r] for r in rings]
    assert connected_components(rings + shifted) == 2


# --- watertightness and volume -------------------------------------------

def test_closed_cube_is_watertight_with_unit_volume():
    verts, faces = unit_cube()
    assert is_watertight(faces)
    assert signed_volume(faces, verts) == pytest.approx(1.0)


def test_cube_missing_a_face_is_not_watertight():
    verts, faces = unit_cube()
    assert not is_watertight(faces[:-1])


def test_inward_orientation_gives_negative_volume():
    verts, faces = unit_cube()
    flipped = [[r[::-1] for r in f] for f in faces]
    assert signed_volume(flipped, verts) == pytest.approx(-1.0)


# --- coincident vertices --------------------------------------------------

def test_coincident_vertices_are_counted_by_position_not_index():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [2, 0, 0]], dtype=float)
    assert count_coincident(verts, [0, 1, 2, 3]) == 1
    assert count_coincident(verts, [0, 1, 3]) == 0


# --- per-object metrics ---------------------------------------------------

def test_object_metrics_on_the_unit_cube():
    verts, _ = unit_cube()
    m = object_metrics(cube_object(), verts)
    assert m["n_vertices"] == 8
    assert m["n_faces"] == 6
    assert m["n_levi_nodes"] == 14
    np.testing.assert_allclose(m["com"], [0.5, 0.5, 0.5])
    assert m["diameter"] == pytest.approx(np.sqrt(3))
    assert m["ground_level"] == pytest.approx(0.0)
    assert m["volume"] == pytest.approx(1.0)
    assert m["watertight"]


# --- defect predicates ----------------------------------------------------

def test_a_clean_cube_has_no_defects():
    verts, _ = unit_cube()
    assert object_defects(cube_object(), verts) == []


def test_inverted_roof_is_flagged():
    verts, faces = unit_cube()
    obj = cube_object()
    obj["geometry"][0]["boundaries"][0][1] = [faces[1][0][::-1]]   # flip the roof
    assert "roof_normal_down" in object_defects(obj, verts)


def test_ground_not_pointing_down_is_flagged():
    verts, faces = unit_cube()
    obj = cube_object()
    obj["geometry"][0]["boundaries"][0][0] = [faces[0][0][::-1]]   # flip the ground
    assert "ground_normal_up" in object_defects(obj, verts)


def test_non_vertical_wall_is_flagged():
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    verts[4] = [0.5, 0, 1]                                  # lean one wall inwards
    verts[5] = [1.5, 0, 1]
    assert "wall_not_vertical" in object_defects(cube_object(), verts)


def test_outer_ceiling_surface_may_point_down_without_being_flagged():
    """The CityGML class for an overhang underside legitimately faces down."""
    verts, faces = unit_cube()
    obj = cube_object(("GroundSurface", "OuterCeilingSurface", "WallSurface",
                       "WallSurface", "WallSurface", "WallSurface"))
    obj["geometry"][0]["boundaries"][0][1] = [faces[1][0][::-1]]   # downward-facing
    defects = object_defects(obj, verts)
    assert "roof_normal_down" not in defects
    assert "unknown_semantic_label" not in defects
    assert "missing_roof" in defects                        # but it still has no roof


def test_missing_semantic_classes_are_flagged():
    verts, _ = unit_cube()
    obj = cube_object(("GroundSurface",) * 6)
    defects = object_defects(obj, verts)
    assert "missing_roof" in defects and "missing_wall" in defects


def test_unrecognised_semantic_label_is_flagged():
    verts, _ = unit_cube()
    obj = cube_object(("GroundSurface", "RoofSurface", "Nonsense",
                       "WallSurface", "WallSurface", "WallSurface"))
    assert "unknown_semantic_label" in object_defects(obj, verts)


def test_repeated_vertex_in_a_ring_is_flagged():
    verts, _ = unit_cube()
    obj = cube_object()
    obj["geometry"][0]["boundaries"][0][1] = [[4, 5, 5, 7]]
    assert "repeated_ring_vertex" in object_defects(obj, verts)


def test_disconnected_wireframe_is_flagged():
    verts = np.vstack([unit_cube()[0], unit_cube()[0] + 10])
    obj = cube_object()
    solid = obj["geometry"][0]
    solid["boundaries"][0].append([[8, 9, 10]])
    solid["semantics"]["surfaces"].append({"type": "RoofSurface"})
    solid["semantics"]["values"][0].append(len(solid["semantics"]["surfaces"]) - 1)
    assert "disconnected" in object_defects(obj, verts)
