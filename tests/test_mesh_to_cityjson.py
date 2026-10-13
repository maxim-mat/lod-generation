"""Coplanar-merge writeback: triangle mesh -> CityJSON polygons.

A bare triangle writeback is trivially correct and useless -- it emits hundreds
of 3-vertex "surfaces" where the source had a few dozen planar ones. These tests
are about the merge, so the shapes are boxes whose polygon count is known.
"""
import numpy as np

from src.analysis.cityobject_analysis import is_watertight
from src.post_process.post_process import mesh_to_cityjson


def _box(lo=(0.0, 0.0, 0.0), hi=(2.0, 2.0, 2.0)):
    """Closed axis-aligned box, 8 vertices / 12 outward-wound triangles."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    verts = np.array([[x, y, z] for x in (lo[0], hi[0])
                      for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    faces = np.array([
        [0, 1, 3], [0, 3, 2],      # x = lo
        [4, 6, 7], [4, 7, 5],      # x = hi
        [0, 4, 5], [0, 5, 1],      # y = lo
        [2, 3, 7], [2, 7, 6],      # y = hi
        [0, 2, 6], [0, 6, 4],      # z = lo, ground
        [1, 5, 7], [1, 7, 3],      # z = hi, roof
    ], dtype=np.int64)
    return verts, faces


def _solid(cj):
    return next(iter(cj["CityObjects"].values()))["geometry"][0]


def _faces_of(cj):
    return _solid(cj)["boundaries"][0]


def _types_of(cj):
    sem = _solid(cj)["semantics"]
    return [sem["surfaces"][i]["type"] for i in sem["values"][0]]


def test_box_merges_twelve_triangles_into_six_quads():
    """The whole point: 12 triangles in, 6 four-sided polygons out."""
    cj = mesh_to_cityjson(*_box())
    faces = _faces_of(cj)

    assert len(faces) == 6
    assert all(len(face) == 1 for face in faces)          # no holes
    assert all(len(face[0]) == 4 for face in faces)       # each a quad
    assert len(cj["vertices"]) == 8


def test_disjoint_coplanar_patches_stay_separate():
    """Plane equality is not connectivity.

    Two boxes sitting on z=0 but not touching share a ground plane exactly. A
    merge keyed only on the plane equation would fuse them into one surface
    with a nonsense boundary.
    """
    v1, f1 = _box()
    v2, f2 = _box(lo=(10.0, 0.0, 0.0), hi=(12.0, 2.0, 2.0))
    verts = np.vstack([v1, v2])
    faces = np.vstack([f1, f2 + len(v1)])

    cj = mesh_to_cityjson(verts, faces)
    assert len(_faces_of(cj)) == 12                        # 6 + 6, not 10
    assert _types_of(cj).count("GroundSurface") == 2


def test_semantics_come_from_winding_not_eigenvectors():
    """`straighten_face` takes its normal from eigh, whose sign is arbitrary.

    Using it here would assign RoofSurface and GroundSurface at random. The
    winding-derived normal cannot: reverse every triangle and the two swap.
    """
    verts, faces = _box()
    types = _types_of(mesh_to_cityjson(verts, faces))
    assert types.count("RoofSurface") == 1
    assert types.count("GroundSurface") == 1
    assert types.count("WallSurface") == 4

    flipped = _types_of(mesh_to_cityjson(verts, faces[:, ::-1]))
    assert flipped.count("RoofSurface") == 1
    assert flipped.count("GroundSurface") == 1

    # The face that was the roof (highest vertices) is now the ground.
    cj = mesh_to_cityjson(verts, faces[:, ::-1])
    v = np.asarray(cj["vertices"])
    ground = _faces_of(cj)[_types_of(cj).index("GroundSurface")][0]
    assert v[ground][:, 2].min() == 2.0


def test_box_output_is_watertight():
    """Stronger than checking the nesting shape: one assertion covers every
    ring-orientation and vertex-remapping bug at once."""
    cj = mesh_to_cityjson(*_box())
    assert is_watertight(_faces_of(cj))


def test_sloped_roof_planes_do_not_merge_with_each_other():
    """A hip roof must stay two surfaces, not become one folded polygon."""
    # Two triangles meeting at a ridge, tilting opposite ways.
    verts = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 4.0, 0.0],
                      [0.0, 4.0, 0.0], [0.0, 2.0, 2.0], [4.0, 2.0, 2.0]])
    faces = np.array([[0, 1, 5], [0, 5, 4],        # +y-facing slope
                      [2, 3, 4], [2, 4, 5]],       # -y-facing slope
                     dtype=np.int64)
    cj = mesh_to_cityjson(verts, faces)
    assert len(_faces_of(cj)) == 2
    assert _types_of(cj) == ["RoofSurface", "RoofSurface"]


def test_hole_becomes_an_inner_ring():
    """Courtyards are the only non-trivial ring case in real data."""
    outer = [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [6.0, 6.0, 0.0], [0.0, 6.0, 0.0]]
    inner = [[2.0, 2.0, 0.0], [4.0, 2.0, 0.0], [4.0, 4.0, 0.0], [2.0, 4.0, 0.0]]
    verts = np.array(outer + inner)
    # Triangulate the square annulus: each outer edge to its inner counterpart.
    faces = []
    for i in range(4):
        j = (i + 1) % 4
        faces += [[i, 4 + i, j], [j, 4 + i, 4 + j]]
    cj = mesh_to_cityjson(verts, np.array(faces, dtype=np.int64))

    assert len(_faces_of(cj)) == 1
    rings = _faces_of(cj)[0]
    assert len(rings) == 2
    assert len(rings[0]) == 4 and len(rings[1]) == 4
    # The hole must wind against the outer ring, or the face has no net area.
    v = np.asarray(cj["vertices"])

    def newell(ring):
        p = v[ring]
        return np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)

    assert float(newell(rings[0]) @ newell(rings[1])) < 0


def test_unreferenced_vertices_are_dropped():
    verts, faces = _box()
    padded = np.vstack([verts, [[9.0, 9.0, 9.0], [8.0, 8.0, 8.0]]])
    assert len(mesh_to_cityjson(padded, faces)["vertices"]) == 8


def test_collinear_vertices_are_removed():
    """Fan triangulation leaves mid-edge vertices; a quad must stay a quad."""
    verts, faces = _box()
    # Split the roof's ridge edge with a midpoint, then retriangulate that face.
    mid = (verts[1] + verts[7]) / 2.0
    verts = np.vstack([verts, [mid]])
    faces = np.vstack([faces[:-2], [[1, 5, 8], [8, 5, 7], [1, 8, 3], [8, 7, 3]]])

    cj = mesh_to_cityjson(verts, faces)
    roof = _faces_of(cj)[_types_of(cj).index("RoofSurface")][0]
    assert len(roof) == 4


def test_open_mesh_does_not_raise():
    """Untrained output is routinely open; it must degrade, not crash."""
    verts, faces = _box()
    cj = mesh_to_cityjson(verts, faces[:-2])          # roofless
    assert isinstance(cj, dict) and cj
    assert len(_faces_of(cj)) == 5


def test_empty_and_degenerate_meshes_return_empty_dict():
    assert mesh_to_cityjson(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)) == {}
    # All-degenerate: three coincident points span no plane.
    assert mesh_to_cityjson(np.zeros((3, 3)), np.array([[0, 1, 2]], dtype=np.int64)) == {}
    # Fewer than three vertices cannot make a face.
    assert mesh_to_cityjson(np.eye(2, 3), np.array([[0, 1, 1]], dtype=np.int64)) == {}
