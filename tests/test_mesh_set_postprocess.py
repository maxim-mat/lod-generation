"""Face set to welded mesh.

The assertion that matters: without the grid snap, welding cannot fire, so
every shared vertex stays split and the mesh is a pile of loose triangles. That
is the crack-at-every-edge failure this step exists to prevent, and it is
invisible in chamfer distance -- only the vertex count and watertightness show
it.
"""
import numpy as np
import pytest
import torch

from src.eval.mesh_metrics import is_watertight_mesh
from src.models.mesh_set_postprocess import faces_to_mesh

NUM_BINS = 128


def _cube_faces():
    """A unit cube's 12 triangles, as explicit per-face corners [12,3,3]."""
    v = np.array([[x, y, z] for x in (-0.4, 0.4) for y in (-0.4, 0.4)
                  for z in (-0.4, 0.4)], dtype=float)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 7, 5], [4, 6, 7],
                  [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v[f]


def _to_x(tri, n_pad=4):
    """[F,3,3] plus padding -> the [10, F] channel layout a sampler emits."""
    f = len(tri)
    x = np.zeros((10, f + n_pad), dtype=np.float32)
    x[:9, :f] = tri.reshape(f, 9).T
    x[9, :f] = 0.5
    x[9, f:] = -0.5
    return torch.from_numpy(x)


def test_padding_is_dropped_by_presence():
    verts, faces, stats = faces_to_mesh(_to_x(_cube_faces(), n_pad=6), NUM_BINS)
    assert len(faces) == 12
    assert stats["n_dropped_absent"] == 6


def test_snap_welds_the_cube_to_eight_vertices():
    verts, faces, _ = faces_to_mesh(_to_x(_cube_faces()), NUM_BINS, snap=True)
    assert len(verts) == 8
    assert is_watertight_mesh(faces)


def test_without_snap_jitter_prevents_welding():
    rng = np.random.default_rng(0)
    tri = _cube_faces() + rng.normal(0, 1e-5, size=(12, 3, 3))
    _, _, stats = faces_to_mesh(_to_x(tri), NUM_BINS, snap=False)
    assert stats["n_verts"] == 36          # nothing merged: 12 faces x 3 corners
    verts, faces, stats = faces_to_mesh(_to_x(tri), NUM_BINS, snap=True)
    assert stats["n_verts"] == 8           # the snap put them back on the grid


def test_degenerate_faces_are_dropped():
    tri = _cube_faces()
    tri[0] = tri[0, 0]                     # all three corners coincide
    _, faces, stats = faces_to_mesh(_to_x(tri), NUM_BINS)
    assert stats["n_dropped_degenerate"] == 1
    assert len(faces) == 11


def test_accepts_face_major_layout_too():
    x = _to_x(_cube_faces())
    a = faces_to_mesh(x, NUM_BINS)[1]
    b = faces_to_mesh(x.T, NUM_BINS)[1]
    assert np.array_equal(a, b)


def test_empty_sample_returns_empty_arrays():
    x = torch.full((10, 8), -0.5)          # nothing present
    verts, faces, stats = faces_to_mesh(x, NUM_BINS)
    assert len(verts) == 0 and len(faces) == 0
    assert stats["n_faces"] == 0


def test_stats_report_duplicate_removal():
    tri = np.concatenate([_cube_faces(), _cube_faces()[:2]])   # 2 exact dupes
    _, faces, stats = faces_to_mesh(_to_x(tri), NUM_BINS)
    assert stats["n_dropped_duplicate"] == 2
    assert len(faces) == 12
