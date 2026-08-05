"""Tokenizer contract for the mesh transformer: canonical order + exact inverse.

Tiny synthetic meshes only, CPU only. The round-trip assertion is the runnable
check that `detokenize(tokenize(m)) == m` up to the quantization step, which is
what the autoregressive model's whole output pipeline rests on.
"""
import json

import numpy as np
import pytest
import torch

from src.dataset.mesh_dataset import (
    BOS,
    EOS,
    PAD,
    canonicalize,
    detokenize,
    mesh_collate_fn,
    normalize_to_unit_box,
    parse_cityjson_file_to_meshes,
    tokenize,
    vocab_size,
    write_obj,
)

NUM_BINS = 128
# Half a bin: the largest error a round trip through `round()` can introduce.
QUANT_TOL = 0.5 / (NUM_BINS - 1)


def _unit_cube():
    """8 vertices / 12 triangles, already inside the [-0.5, 0.5] box."""
    v = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)],
                 dtype=float)
    f = np.array([
        [0, 1, 3], [0, 3, 2],   # x = -0.5
        [4, 7, 5], [4, 6, 7],   # x = +0.5
        [0, 4, 5], [0, 5, 1],   # y = -0.5
        [2, 3, 7], [2, 7, 6],   # y = +0.5
        [0, 2, 6], [0, 6, 4],   # z = -0.5
        [1, 5, 7], [1, 7, 3],   # z = +0.5
    ], dtype=np.int64)
    return v, f


def _random_mesh(n_faces=7, seed=0):
    rng = np.random.default_rng(seed)
    verts = rng.uniform(-0.5, 0.5, size=(3 * n_faces, 3))
    faces = np.arange(3 * n_faces, dtype=np.int64).reshape(n_faces, 3)
    return verts, faces


# ----------------------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------------------

def test_token_layout():
    assert (BOS, EOS, PAD) == (NUM_BINS, NUM_BINS + 1, NUM_BINS + 2)
    assert vocab_size(NUM_BINS) == NUM_BINS + 3


def test_tokenize_length_is_nine_per_face():
    verts, faces = _unit_cube()
    tokens = tokenize(verts, faces, NUM_BINS)
    assert tokens.shape == (9 * len(faces),)
    assert tokens.min() >= 0 and tokens.max() < NUM_BINS  # no specials inside


@pytest.mark.parametrize("mesh", [_unit_cube(), _random_mesh()])
def test_roundtrip_within_quantization_error(mesh):
    verts, faces = mesh
    v_out, f_out = detokenize(tokenize(verts, faces, NUM_BINS), NUM_BINS)

    assert len(f_out) == len(faces)
    tri_in = canonicalize(verts, faces)
    tri_in = tri_in[0][tri_in[1]]          # [F, 3, 3] canonical triangle soup
    tri_out = v_out[f_out]
    assert np.abs(tri_in - tri_out).max() <= QUANT_TOL + 1e-9


@pytest.mark.parametrize("mesh", [_unit_cube(), _random_mesh(seed=3)])
def test_roundtrip_is_idempotent(mesh):
    """A second pass must be a fixed point, else sampling drifts every step."""
    verts, faces = mesh
    tokens = tokenize(verts, faces, NUM_BINS)
    v_out, f_out = detokenize(tokens, NUM_BINS)
    assert np.array_equal(tokenize(v_out, f_out, NUM_BINS), tokens)


def test_tokens_are_invariant_to_input_ordering():
    """Canonical z-y-x sort: the same geometry must give the same sequence."""
    verts, faces = _unit_cube()
    ref = tokenize(verts, faces, NUM_BINS)

    rng = np.random.default_rng(1)
    perm = rng.permutation(len(verts))
    inv = np.argsort(perm)
    shuffled_faces = inv[faces][rng.permutation(len(faces))]
    assert np.array_equal(tokenize(verts[perm], shuffled_faces, NUM_BINS), ref)


def test_canonicalize_preserves_winding():
    """Face rotation, not reversal: normals must survive canonicalization."""
    verts, faces = _unit_cube()
    v_c, f_c = canonicalize(verts, faces)

    def normals(v, f):
        tri = v[f]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        return sorted(tuple(row) for row in np.round(n + 0.0, 6))

    assert normals(verts, faces) == normals(v_c, f_c)


def test_normalize_to_unit_box_is_invertible():
    verts = np.array([[10.0, 20.0, 0.0], [14.0, 20.0, 0.0], [10.0, 26.0, 8.0]])
    v_n, center, scale = normalize_to_unit_box(verts)
    assert v_n.min() >= -0.5 - 1e-9 and v_n.max() <= 0.5 + 1e-9
    assert np.allclose(v_n * scale + center, verts)


def test_normalize_accepts_a_shared_reference_frame():
    """LOD2 must be placed in the LOD1 frame, not rescaled on its own."""
    ref = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    other = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    _, c_ref, s_ref = normalize_to_unit_box(ref)
    v_n, c, s = normalize_to_unit_box(other, ref=ref)
    assert np.allclose(c, c_ref) and np.isclose(s, s_ref)
    assert np.allclose(v_n * s + c, other)


# ----------------------------------------------------------------------
# CityJSON -> mesh
# ----------------------------------------------------------------------

def _tiny_cityjson(tmp_path):
    """One box CityObject as a Solid with a quad footprint (needs triangulating)."""
    cj = {
        "type": "CityJSON", "version": "1.1",
        "transform": {"scale": [1.0, 1.0, 1.0], "translate": [100.0, 200.0, 0.0]},
        "vertices": [[0, 0, 0], [4, 0, 0], [4, 6, 0], [0, 6, 0],
                     [0, 0, 3], [4, 0, 3], [4, 6, 3], [0, 6, 3]],
        "CityObjects": {
            "b1": {"type": "Building", "geometry": [{
                "type": "Solid", "lod": "1",
                "boundaries": [[
                    [[0, 3, 2, 1]], [[4, 5, 6, 7]],
                    [[0, 1, 5, 4]], [[1, 2, 6, 5]], [[2, 3, 7, 6]], [[3, 0, 4, 7]],
                ]],
                "semantics": {
                    "surfaces": [{"type": "GroundSurface"}, {"type": "RoofSurface"},
                                 {"type": "WallSurface"}],
                    "values": [[0, 1, 2, 2, 2, 2]],
                },
            }]},
        },
    }
    path = tmp_path / "tiny.city.json"
    path.write_text(json.dumps(cj), encoding="utf-8")
    return path


def test_parse_cityjson_triangulates_and_applies_transform(tmp_path):
    meshes = parse_cityjson_file_to_meshes(_tiny_cityjson(tmp_path))
    assert set(meshes) == {"b1"}
    verts, faces = meshes["b1"]

    assert faces.shape == (12, 3)          # 6 quads fan-triangulated
    assert faces.max() < len(verts)
    assert np.allclose(verts.min(axis=0), [100.0, 200.0, 0.0])   # transform applied
    assert np.allclose(verts.max(axis=0), [104.0, 206.0, 3.0])


def test_parse_cityjson_missing_file_raises(tmp_path):
    with pytest.raises((FileNotFoundError, OSError)):
        parse_cityjson_file_to_meshes(tmp_path / "nope.city.json")


def test_write_obj_roundtrips_through_disk(tmp_path):
    verts, faces = _unit_cube()
    out = tmp_path / "cube.obj"
    write_obj(out, verts, faces)
    text = out.read_text(encoding="utf-8")
    assert text.count("\nv ") + text.startswith("v ") == len(verts)
    assert text.count("f ") == len(faces)
    assert " 0 " not in text  # .obj face indices are 1-based


# ----------------------------------------------------------------------
# Collate
# ----------------------------------------------------------------------

def test_collate_pads_and_masks():
    items = [
        {"cond": torch.arange(5), "tgt": torch.arange(9), "id": "a"},
        {"cond": torch.arange(2), "tgt": torch.arange(4), "id": "b"},
    ]
    batch = mesh_collate_fn(items)

    assert batch["cond"].shape == (2, 5)
    assert batch["tgt"].shape == (2, 9)
    assert batch["cond_pad_mask"].dtype == torch.bool
    # True marks padding, per nn.Transformer's key_padding_mask convention.
    assert batch["cond_pad_mask"][0].sum() == 0
    assert batch["cond_pad_mask"][1].tolist() == [False, False, True, True, True]
    assert (batch["tgt"][1][4:] == PAD).all()
    assert batch["ids"] == ["a", "b"]
