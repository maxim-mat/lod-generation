"""Face-level matching between generated, ground-truth and condition meshes."""
import math

import numpy as np
import pandas as pd
import pytest
import torch

from src.dataset.mesh_dataset import dequantize, mesh_collate_fn, specials, tokenize
from src.eval.face_match import (COUNT_COLUMNS, RATE_COLUMNS, face_match, face_set,
                                 match_rates, split_face_match, summarize)

NUM_BINS = 32


def _box():
    """Closed unit-box mesh, 12 triangles, outward winding."""
    v = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
    f = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4],
                  [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]])
    return v, f


# ---------------------------------------------------------------- face_set

def test_face_set_matches_the_tokenizer_rows():
    # Same definition as the tokenization notebook: a face IS its canonical 9-token row.
    v, f = _box()
    rows = tokenize(v, f, NUM_BINS).reshape(-1, 9)
    assert face_set(v, f, NUM_BINS) == set(map(tuple, rows.tolist()))


def test_face_set_ignores_face_order_vertex_order_and_rotation():
    v, f = _box()
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(v))                       # relabel vertices
    inv = np.argsort(perm)
    f2 = inv[f][rng.permutation(len(f))]                 # reorder faces
    f2 = np.roll(f2, 1, axis=1)                          # rotate every face's corners
    assert face_set(v[perm], f2, NUM_BINS) == face_set(v, f, NUM_BINS)


def test_reversed_winding_is_a_different_face():
    v, f = _box()
    assert face_set(v, f[:, ::-1], NUM_BINS).isdisjoint(face_set(v, f, NUM_BINS))


def test_face_set_drops_faces_collapsed_by_quantization():
    step = 1.0 / (NUM_BINS - 1)
    v = np.array([[0.0, 0.0, 0.0], [0.1 * step, 0.0, 0.0], [0.0, 0.3, 0.0], [0.3, 0.0, 0.0]])
    f = np.array([[0, 1, 2], [0, 3, 2]])                 # corners 0 and 1 share a bin
    assert len(face_set(v, f, NUM_BINS)) == 1


def test_face_set_accepts_dequantized_tokens_exactly():
    # Decoded model output is grid points as floats; re-quantizing must be lossless.
    v, f = _box()
    q = np.random.default_rng(1).integers(0, NUM_BINS, size=(8, 3))
    verts = dequantize(q, NUM_BINS)
    got = face_set(verts, f, NUM_BINS)
    assert all(0 <= t < NUM_BINS for row in got for t in row)
    assert got == face_set(verts + 1e-9, f, NUM_BINS)


def test_face_set_of_an_empty_mesh_is_empty():
    assert face_set(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), NUM_BINS) == set()


# ---------------------------------------------------------------- face_match / rates

def _faces(*names):
    return {(n,) for n in names}       # any hashable stands in for a 9-tuple here


def test_face_match_counts():
    lod1 = _faces("w1", "w2", "g1", "old_roof")
    gt = _faces("w1", "w2", "g1", "roof1", "roof2")
    gen = _faces("w1", "g1", "old_roof", "roof1", "junk")
    c = face_match(gen, gt, lod1)
    assert c == {
        "n_gen": 5, "n_gt": 5, "n_lod1": 4,
        "gen_in_gt": 3,             # w1 g1 roof1
        "gen_in_lod1": 3,           # w1 g1 old_roof
        "gen_in_lod1_not_gt": 1,    # old_roof: copied, and wrong
        "gen_in_gt_not_lod1": 1,    # roof1: new, and right
        "gt_in_lod1": 3,            # w1 w2 g1
        "gt_not_lod1": 2,           # roof1 roof2
    }
    assert list(c) == list(COUNT_COLUMNS)


def test_match_rates():
    c = face_match(_faces("a", "b", "x"), _faces("a", "b", "c", "n"), _faces("a", "x", "c"))
    r = match_rates(c)
    assert list(r) == list(RATE_COLUMNS)
    assert r["gen_copy"] == pytest.approx(2 / 3)           # a, x
    assert r["gen_precision"] == pytest.approx(2 / 3)      # a, b
    assert r["gen_recall"] == pytest.approx(2 / 4)
    assert r["spurious_copy"] == pytest.approx(1 / 3)      # x
    assert r["novel_recall"] == pytest.approx(1 / 2)       # b of {b, n}
    assert r["copy_recall"] == pytest.approx(1 / 2)        # a of {a, c}
    assert r["gt_copy"] == pytest.approx(2 / 4)            # a, c
    assert r["lod1_precision"] == pytest.approx(2 / 3)     # a, c of {a, x, c}


def test_rates_with_empty_denominators_are_nan_not_errors():
    r = match_rates(face_match(set(), _faces("a"), _faces("a")))
    assert math.isnan(r["gen_copy"]) and math.isnan(r["gen_precision"])
    assert math.isnan(r["novel_recall"])                   # every GT face is in LOD1
    assert r["gen_recall"] == 0.0 and r["gt_copy"] == 1.0


def test_summarize_reports_micro_and_macro():
    rows = [dict(split="val", **face_match(_faces("a"), _faces("a"), _faces("a"))),
            dict(split="val", **face_match(_faces("a", "b", "c"), _faces("a", "b", "c"),
                                           _faces("z")))]
    s = summarize(pd.DataFrame(rows), by="split")
    micro, macro = s.loc[("val", "micro")], s.loc[("val", "macro")]
    assert micro["gen_copy"] == pytest.approx(1 / 4)       # pooled faces
    assert macro["gen_copy"] == pytest.approx((1 + 0) / 2) # mean of buildings
    assert micro["buildings"] == macro["buildings"] == 2


def test_summarize_macro_skips_nan_buildings():
    rows = [face_match(set(), _faces("a"), _faces("a")),
            face_match(_faces("a"), _faces("a"), _faces("b"))]
    s = summarize(pd.DataFrame(rows))
    assert s.loc["macro", "gen_copy"] == pytest.approx(0.0)


# ---------------------------------------------------------------- split loop

class _Dataset:
    """Three (cond, tgt) pairs whose targets are known token rows."""

    def __init__(self):
        bos, eos, _ = specials(NUM_BINS)
        v, f = _box()
        rows = tokenize(v, f, NUM_BINS)
        self.items = []
        for k in range(3):
            cond = rows[: 9 * (4 + k)]                   # LOD1 = a prefix of the faces
            tgt = np.concatenate([[bos], rows, [eos]])
            self.items.append({"cond": torch.from_numpy(cond), "tgt": torch.from_numpy(tgt),
                               "id": f"b{k}", "center": torch.zeros(3), "scale": torch.ones(3)})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class _EchoModel:
    """Generates exactly its condition: the pure copier."""
    num_bins = NUM_BINS
    tokenization = "coord"

    def decode_tokens(self, tokens):
        from src.dataset.mesh_dataset import detokenize
        return detokenize(np.asarray(tokens), NUM_BINS)


def _echo(batch, positions):
    return batch["tgt"], batch["cond"]


def test_split_face_match_scores_every_building_on_the_fly():
    ds = _Dataset()
    seen = []

    def generate(batch, positions):
        seen.append(list(positions))
        return _echo(batch, positions)

    df = split_face_match(_EchoModel(), ds, generate, batch_size=2)
    assert seen == [[0, 1], [2]]
    assert list(df["id"]) == ["b0", "b1", "b2"]
    assert (df["n_gt"] == 12).all()
    assert list(df["n_lod1"]) == [4, 5, 6]
    # The echo model copies everything and invents nothing.
    assert (df["gen_in_lod1"] == df["n_gen"]).all()
    assert (df["gen_in_gt_not_lod1"] == 0).all()
    assert (df["gt_in_lod1"] == df["n_lod1"]).all()


def test_split_face_match_limit():
    df = split_face_match(_EchoModel(), _Dataset(), _echo, batch_size=2, limit=1)
    assert list(df["id"]) == ["b0"]


def test_split_face_match_with_a_real_model():
    from src.models.mesh_transformer import MeshTransformerModule

    torch.manual_seed(1)
    model = MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2, num_layers=1,
                                  dropout=0.0, max_seq_len=256).eval()

    def generate(batch, positions):
        with torch.no_grad():
            cond, tgt, cond_pad, _ = model._prepare(batch)
            out = model.generate(cond, cond_pad, max_new_tokens=9 * 4 + 1, temperature=0.0)
        return tgt, out

    df = split_face_match(model, _Dataset(), generate, batch_size=2)
    assert len(df) == 3 and (df["n_gt"] == 12).all()
    assert (df["gen_in_gt"] <= df["n_gen"]).all() and (df["gen_in_lod1"] <= df["n_gen"]).all()
    assert (df["gt_in_lod1"] == df["n_lod1"]).all()
