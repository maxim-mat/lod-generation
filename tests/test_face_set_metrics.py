"""Position-free face, roof-face and LOD1-copy metrics on quantized bins."""
import math

import numpy as np
import pytest

from src.eval.mesh_metrics import FACE_COLUMNS, face_set_metrics

S = np.ones(3)
EMPTY = (np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3), dtype=np.int64))
FACE_KEYS = ("face_acc", "face_precision", "face_recall", "face_f1")


def _box_bins(top=10):
    """Closed, outward-wound 10x10xtop box in bins: 12 faces, top 2 are roof."""
    v = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],
                  [0, 0, top], [10, 0, top], [10, 10, top], [0, 10, top]])
    f = np.array([[0, 2, 1], [0, 3, 2],            # bottom, facing down
                  [4, 5, 6], [4, 6, 7],            # top, facing up
                  [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                  [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return v, f


def _four_faces():
    v = np.array([[0, 0, 0], [5, 0, 0], [0, 5, 0], [5, 5, 0], [0, 0, 5], [9, 9, 9]])
    f = np.array([[0, 1, 2], [1, 3, 2], [0, 1, 4], [2, 3, 5]])
    return v, f


def test_keys_are_the_face_columns_in_order():
    gt = _four_faces()
    assert tuple(face_set_metrics(gt, gt, gt, S)) == FACE_COLUMNS


def test_order_vertex_order_and_winding_are_ignored():
    v, f = _four_faces()
    g = f[[2, 0, 3, 1]]
    g = np.roll(g, 1, axis=1)
    g[0] = g[0][::-1]
    out = face_set_metrics((v, g), (v, f), EMPTY, S)
    assert all(out[k] == 1.0 for k in FACE_KEYS)


def test_partial_match():
    v, f = _four_faces()
    extra = np.array([[0, 4, 5]])
    out = face_set_metrics((v, np.vstack([f[:2], extra])), (v, f), EMPTY, S)
    assert out["face_precision"] == pytest.approx(2 / 3)
    assert out["face_recall"] == 0.5
    assert out["face_f1"] == pytest.approx(4 / 7)
    assert out["face_acc"] == pytest.approx(2 / 5)


def test_faces_are_a_multiset():
    v, f = _four_faces()
    out = face_set_metrics((v, f[[0, 0]]), (v, f[[0]]), EMPTY, S)
    assert out["face_precision"] == 0.5
    assert out["face_recall"] == 1.0


def test_faces_match_on_coordinates_not_indices():
    v, f = _four_faces()
    perm = np.array([3, 5, 0, 2, 4, 1])            # new position of each old vertex
    v2 = np.empty_like(v)
    v2[perm] = v
    out = face_set_metrics((v2, perm[f]), (v, f), EMPTY, S)
    assert out["face_f1"] == 1.0


def test_roof_classification_uses_metre_scale():
    tri = (np.array([[0, 0, 0], [10, 0, 0], [0, 1, 10]]), np.array([[0, 1, 2]]))
    flat = face_set_metrics(tri, tri, EMPTY, np.array([1.0, 1.0, 1.0]))
    assert all(math.isnan(flat[f"roof_{k}"]) for k in FACE_KEYS)
    stretched = face_set_metrics(tri, tri, EMPTY, np.array([1.0, 2.0, 1.0]))
    assert stretched["roof_face_f1"] == 1.0


def test_roof_metrics_see_only_roof_faces():
    v, f = _box_bins()
    lifted_v = np.vstack([v, [[0, 0, 12], [10, 0, 12], [10, 10, 12], [0, 10, 12]]])
    gen_f = np.vstack([f[[0, 1]], [[8, 9, 10], [8, 10, 11]], f[4:]])
    out = face_set_metrics((lifted_v, gen_f), (v, f), EMPTY, S)
    assert out["face_recall"] == pytest.approx(10 / 12)
    assert out["roof_face_recall"] == 0.0


def test_one_empty_side_scores_zero_and_two_empty_sides_are_nan():
    gt = _four_faces()
    one = face_set_metrics(EMPTY, gt, EMPTY, S)
    assert all(one[k] == 0.0 for k in FACE_KEYS)
    both = face_set_metrics(EMPTY, EMPTY, EMPTY, S)
    assert all(math.isnan(both[k]) for k in FACE_KEYS)


def test_lod1_vert_copy_rate_and_norm():
    lod1 = (np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]), np.array([[0, 1, 2]]))
    tri = np.array([[0, 1, 2], [1, 2, 3]])
    gt = (np.array([[0, 0, 0], [1, 0, 0], [7, 7, 7], [8, 8, 8]]), tri)
    gen = (np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [9, 9, 9]]), tri)
    out = face_set_metrics(gen, gt, lod1, S)
    assert out["lod1_vert_copy_rate"] == 0.75
    assert out["lod1_vert_copy_norm"] == 1.5

    disjoint = (np.array([[5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8]]), tri)
    assert math.isnan(face_set_metrics(gen, disjoint, lod1, S)["lod1_vert_copy_norm"])
    none = face_set_metrics(EMPTY, gt, lod1, S)
    assert math.isnan(none["lod1_vert_copy_rate"])
    assert math.isnan(none["lod1_vert_copy_norm"])
