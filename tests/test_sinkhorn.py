"""Debiased Sinkhorn divergence, reported as sqrt(2 S) so it reads in metres.

With cost |x - y|^2 / 2, translating one cloud by d adds exactly d^2 / 2 to
OT_eps(a, b) and nothing to the two self terms, so the reported value of a
translated copy is d at any blur -- a closed-form target, not a tuned one.
"""
import numpy as np
import trimesh

from src.eval.sinkhorn import mesh_sinkhorn, sinkhorn_divergence


def _cloud(n=200, seed=0):
    return np.random.default_rng(seed).random((n, 3))


def _box():
    b = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    return np.asarray(b.vertices, dtype=float), np.asarray(b.faces, dtype=np.int64)


def test_a_cloud_against_itself_is_zero():
    x = _cloud()
    assert sinkhorn_divergence(x, x, blur=0.05) == 0.0


def test_a_translated_copy_scores_the_translation():
    x = _cloud()
    got = sinkhorn_divergence(x, x + np.array([1.0, 0.0, 0.0]), blur=0.05,
                              max_iters=2000, tol=1e-9)
    assert abs(got - 1.0) < 0.02


def test_world_coordinates_do_not_cost_precision():
    """EPSG:28992 puts y near 450,000; float32 squared distances there lose the
    metre scale. The solver shifts to a shared origin and runs in float64."""
    v, f = _box()
    world = v + np.array([80_000.0, 450_000.0, 0.0])
    lifted = world + np.array([0.0, 0.0, 0.5])
    got = mesh_sinkhorn((world, f), (lifted, f), n_points=256, blur=0.05,
                        max_iters=2000, tol=1e-9)
    assert abs(got - 0.5) < 0.02


def test_identical_meshes_score_zero_because_both_sides_share_a_seed():
    v, f = _box()
    assert mesh_sinkhorn((v, f), (v.copy(), f.copy()), n_points=256) == 0.0


def test_empty_input_is_nan():
    assert np.isnan(sinkhorn_divergence(np.zeros((0, 3)), _cloud()))
    v, f = _box()
    assert np.isnan(mesh_sinkhorn((np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)),
                                  (v, f), n_points=64))
