"""Roof extraction and the textbook Hausdorff distance for the post-hoc eval.

Roof means a winding-derived unit normal with z > 0.1: the rule
`mesh_to_cityjson` uses to label RoofSurface. Winding decides it, not position.
"""
import numpy as np
import trimesh

from src.eval.mesh_metrics import hausdorff_max, roof_submesh


def _box():
    """2 m cube centred on the origin, outward-wound, 12 triangles."""
    b = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    return np.asarray(b.vertices, dtype=float), np.asarray(b.faces, dtype=np.int64)


def test_roof_of_an_outward_box_is_its_top_face():
    v, f = roof_submesh(*_box())
    assert f.shape == (2, 3)
    assert len(v) == 4                      # unreferenced vertices dropped
    assert np.allclose(v[:, 2], 1.0)
    assert f.max() == len(v) - 1            # indices point into the compacted verts


def test_flipped_winding_reads_the_floor_as_roof():
    v, f = _box()
    rv, rf = roof_submesh(v, f[:, ::-1])
    assert rf.shape == (2, 3)
    assert np.allclose(rv[:, 2], -1.0)


def test_a_walls_only_mesh_has_no_roof():
    v, f = _box()
    normals = trimesh.Trimesh(vertices=v, faces=f, process=False).face_normals
    rv, rf = roof_submesh(v, f[np.abs(normals[:, 2]) < 0.5])
    assert rv.shape == (0, 3) and rf.shape == (0, 3)


def test_an_empty_mesh_has_no_roof():
    rv, rf = roof_submesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    assert rv.shape == (0, 3) and rf.shape == (0, 3)


def test_hausdorff_max_is_the_worse_direction():
    assert hausdorff_max(np.array([0.1, 0.4]), np.array([0.2, 0.3])) == 0.4
    assert np.isnan(hausdorff_max(np.zeros(0), np.array([1.0])))
