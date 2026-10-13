"""Quantization collapses vertices; the faces that collapse with them must go.

`canonicalize` merges vertices that land on one grid point. Any face that had
two of those vertices becomes degenerate -- a zero-area triangle with a repeated
index -- and used to be tokenized and taught to the model anyway. Measured on
the corpus that was 1.37% of all faces.

Both reference implementations drop them at exactly this point:
MeshAnythingV2's `mesh_sort` calls `nondegenerate_faces()` + `unique_faces()`,
and TreeMeshGPT's `quantize_remove_duplicates` builds an explicit
`collapsed_mask`.
"""
import numpy as np
import pytest

from src.dataset.mesh_dataset import (NUM_BINS, canonicalize, detokenize, quantize,
                                      tokenize)


def test_face_whose_vertices_merge_is_dropped():
    """Two vertices a hair apart land in one bin; their shared face collapses."""
    eps = 0.4 / (NUM_BINS - 1)                 # well under half a bin
    verts = np.array([[-0.5, -0.5, 0.0],
                      [0.5, -0.5, 0.0],
                      [0.0, 0.5, 0.0],
                      [0.0 + eps, 0.5, 0.0]])  # merges with vertex 2
    faces = np.array([[0, 1, 2], [2, 3, 0]])   # second face collapses to a line
    v, f = canonicalize(quantize(verts, NUM_BINS), faces)
    assert len(f) == 1, f"degenerate face survived: {f.tolist()}"
    assert len(set(f[0].tolist())) == 3


def test_exact_duplicate_faces_are_dropped():
    verts = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2], [0, 1, 2]])
    _, f = canonicalize(verts, faces)
    assert len(f) == 1


def test_a_clean_mesh_is_untouched():
    """The guard must not thin a mesh that has nothing wrong with it."""
    verts = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    _, f = canonicalize(verts, faces)
    assert len(f) == 2


def test_still_idempotent():
    eps = 0.4 / (NUM_BINS - 1)
    verts = np.array([[-0.5, -0.5, 0.], [0.5, -0.5, 0.], [0., 0.5, 0.],
                      [eps, 0.5, 0.]])
    faces = np.array([[0, 1, 2], [2, 3, 0]])
    v1, f1 = canonicalize(quantize(verts, NUM_BINS), faces)
    v2, f2 = canonicalize(v1, f1)
    assert np.array_equal(v1, v2) and np.array_equal(f1, f2)


def test_tokenize_emits_no_degenerate_face():
    eps = 0.4 / (NUM_BINS - 1)
    verts = np.array([[-0.5, -0.5, 0.], [0.5, -0.5, 0.], [0., 0.5, 0.],
                      [eps, 0.5, 0.]])
    faces = np.array([[0, 1, 2], [2, 3, 0]])
    toks = tokenize(verts, faces, NUM_BINS)
    assert len(toks) % 9 == 0
    q = np.asarray(toks).reshape(-1, 3, 3)
    for tri in q:
        assert len({tuple(p) for p in tri}) == 3, "tokenized a collapsed triangle"


def test_round_trip_still_exact_on_a_clean_mesh():
    """Dropping faces must not disturb the tokenize/detokenize pair."""
    verts = np.array([[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
                      [0.5, 0.5, -0.5], [-0.5, 0.5, 0.5]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
    v, f = detokenize(tokenize(verts, faces, NUM_BINS), NUM_BINS)
    assert len(f) == len(faces)
    assert np.abs(np.sort(v, axis=0) - np.sort(verts, axis=0)).max() < 1.0 / (NUM_BINS - 1)
