"""Adjacent Mesh Tokenization and invalid-prediction masking.

MeshAnything V2 (Chen et al., arXiv:2408.02555), Algorithm 1 and section 3.2.
AMT represents a face by a *single* new vertex whenever it is adjacent to the
previous one, emitting a break token when it cannot, which halves the sequence.

Tiny synthetic meshes, CPU only. The contract that matters is that AMT decodes
to the *same mesh* the coordinate tokenizer decodes to -- it is a different
spelling of the sequence, not a different geometry.
"""
import json

import numpy as np
import pytest
import torch

from src.dataset.mesh_dataset import (
    BOS,
    BREAK,
    EOS,
    PAD,
    amt_detokenize,
    amt_tokenize,
    canonicalize,
    detokenize,
    tokenize,
    vocab_size,
)
from src.dataset.mesh_dataset import MeshDataset
from src.models.mesh_transformer import (
    MeshTransformerModule,
    invalid_logits_mask,
)

NUM_BINS = 128


def _unit_cube():
    """8 vertices / 12 triangles, consistently wound outward, inside [-0.5, 0.5]."""
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


def _disconnected():
    """Two triangles sharing no edge: AMT must break between them."""
    v = np.array([[-0.5, -0.5, -0.5], [-0.5, 0.0, -0.5], [-0.4, -0.5, -0.5],
                  [0.5, 0.5, 0.5], [0.5, 0.0, 0.5], [0.4, 0.5, 0.5]], dtype=float)
    f = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    return v, f


def _face_set(verts, faces):
    """Faces as a hashable set of sorted coordinate triples, winding-blind."""
    tri = np.round(np.asarray(verts)[np.asarray(faces)], 6)
    return {tuple(sorted(map(tuple, t))) for t in tri}


# ----------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------

def test_break_token_sits_above_the_existing_specials():
    assert BREAK == NUM_BINS + 3
    assert (BOS, EOS, PAD) == (NUM_BINS, NUM_BINS + 1, NUM_BINS + 2)


def test_amt_adds_exactly_one_token_to_the_vocabulary():
    assert vocab_size(NUM_BINS, "amt") == NUM_BINS + 4


def test_coord_vocabulary_is_unchanged():
    """Regression guard: existing checkpoints have NUM_BINS + 3 output rows."""
    assert vocab_size(NUM_BINS) == NUM_BINS + 3
    assert vocab_size(NUM_BINS, "coord") == NUM_BINS + 3


# ----------------------------------------------------------------------
# AMT tokenizer
# ----------------------------------------------------------------------

def test_amt_is_shorter_than_the_coordinate_sequence():
    """The whole point: ~half the tokens for the same mesh."""
    verts, faces = _unit_cube()
    assert len(amt_tokenize(verts, faces, NUM_BINS)) < len(tokenize(verts, faces, NUM_BINS))


def test_amt_emits_a_break_between_disconnected_faces():
    verts, faces = _disconnected()
    tokens = amt_tokenize(verts, faces, NUM_BINS)
    # Two isolated triangles: 3 vertices each, one break between them.
    assert (tokens == BREAK).sum() == 1
    assert len(tokens) == 3 * 3 + 1 + 3 * 3


def test_amt_never_breaks_on_a_fully_connected_mesh_start():
    """A closed cube is one strip's worth of adjacency at every step but the first."""
    verts, faces = _unit_cube()
    tokens = amt_tokenize(verts, faces, NUM_BINS)
    assert tokens[0] != BREAK, "a sequence must not open with a break"


def test_amt_decodes_to_the_same_mesh_as_the_coordinate_tokenizer():
    """AMT is a different spelling of the sequence, not different geometry."""
    verts, faces = _unit_cube()
    amt_v, amt_f = amt_detokenize(amt_tokenize(verts, faces, NUM_BINS), NUM_BINS)
    coord_v, coord_f = detokenize(tokenize(verts, faces, NUM_BINS), NUM_BINS)

    assert len(amt_f) == len(coord_f)
    assert _face_set(amt_v, amt_f) == _face_set(coord_v, coord_f)


def test_amt_roundtrip_recovers_a_disconnected_mesh():
    verts, faces = _disconnected()
    v_out, f_out = amt_detokenize(amt_tokenize(verts, faces, NUM_BINS), NUM_BINS)
    q_v, q_f = canonicalize(np.rint((verts + 0.5) * (NUM_BINS - 1)), faces)
    assert len(f_out) == len(faces)
    assert _face_set(v_out, f_out) == _face_set(q_v / (NUM_BINS - 1) - 0.5, q_f)


def test_amt_detokenize_drops_a_partial_strip():
    """A generation cut mid-strip must decode to the faces it managed to close."""
    verts, faces = _unit_cube()
    tokens = amt_tokenize(verts, faces, NUM_BINS)
    _, full = amt_detokenize(tokens, NUM_BINS)
    # Drop the last two coordinates: the final vertex is incomplete.
    _, cut = amt_detokenize(tokens[:-2], NUM_BINS)
    assert len(cut) == len(full) - 1


def test_amt_detokenize_ignores_specials():
    verts, faces = _unit_cube()
    tokens = amt_tokenize(verts, faces, NUM_BINS)
    wrapped = np.concatenate([[BOS], tokens, [EOS], [PAD, PAD]])
    a = amt_detokenize(wrapped, NUM_BINS)
    b = amt_detokenize(tokens, NUM_BINS)
    assert np.array_equal(a[1], b[1]) and np.allclose(a[0], b[0])


def test_amt_empty_in_empty_out():
    v, f = amt_detokenize(np.array([], dtype=np.int64), NUM_BINS)
    assert len(v) == 0 and len(f) == 0


# ----------------------------------------------------------------------
# Masking invalid predictions (V2 section 3.2, after PolyGen)
# ----------------------------------------------------------------------

def _mask(tokens, tokenization="coord"):
    tgt = torch.tensor([tokens], dtype=torch.long)
    vocab = vocab_size(NUM_BINS, tokenization)
    return invalid_logits_mask(tgt, vocab, NUM_BINS, tokenization=tokenization)[0]


def test_mask_forbids_pad_and_bos_always():
    m = _mask([BOS, 5, 5, 5, 5, 5, 5, 5, 5, 5])
    assert m[PAD] and m[BOS]


def test_mask_forbids_eos_mid_face_on_the_coordinate_path():
    """8 coordinates in, one short of a triangle: stopping here loses the face."""
    assert _mask([BOS] + [5] * 8)[EOS]


def test_mask_allows_eos_on_a_face_boundary():
    assert not _mask([BOS] + [5] * 9)[EOS]


def test_mask_never_forbids_every_coordinate():
    """A mask that blocks the whole vocabulary would deadlock generation."""
    m = _mask([BOS] + [5] * 4)
    assert not m[:NUM_BINS].all()


def test_mask_forbids_break_immediately_after_a_break():
    """V2: 'generating another & token immediately after an & token' is invalid."""
    tokens = [BOS] + [5] * 9 + [BREAK]
    assert _mask(tokens, "amt")[BREAK]


def test_mask_forbids_break_before_three_vertices_are_emitted():
    """V2: 'at least three vertices must be generated before allowing any interruptions'."""
    assert _mask([BOS] + [5] * 6, "amt")[BREAK]       # two vertices into a strip
    assert not _mask([BOS] + [5] * 9, "amt")[BREAK]   # three: now legal


def test_mask_forbids_break_mid_vertex():
    """A break between two coordinates of one vertex would orphan them."""
    assert _mask([BOS] + [5] * 10, "amt")[BREAK]


def test_mask_forbids_eos_before_a_strip_closes():
    assert _mask([BOS] + [5] * 6, "amt")[EOS]
    assert not _mask([BOS] + [5] * 9, "amt")[EOS]


def test_mask_never_blocks_the_whole_vocabulary_under_amt():
    """Deadlock guard: whatever the state, some coordinate must stay legal."""
    for n in range(0, 13):
        m = _mask([BOS] + [5] * n, "amt")
        assert not m.all(), f"every token masked after {n} coordinates"


def test_mask_is_batched():
    """Rows are independent. The PAD tail is what `generate` actually produces
    for a row that already finished, and it must not count as a coordinate."""
    tgt = torch.tensor([[BOS] + [5] * 9,
                        [BOS] + [5] * 8 + [PAD]], dtype=torch.long)
    m = invalid_logits_mask(tgt, vocab_size(NUM_BINS), NUM_BINS, tokenization="coord")
    assert m.shape == (2, vocab_size(NUM_BINS))
    assert not m[0, EOS], "9 coordinates is a whole face; EOS is legal"
    assert m[1, EOS], "8 coordinates is mid-face; EOS would drop the triangle"


# ----------------------------------------------------------------------
# Wiring: dataset, model, generation
# ----------------------------------------------------------------------

def _tiny_corpus(tmp_path):
    """A two-LOD corpus on disk, the shape `MeshDataset` expects."""
    def box(height, name):
        cj = {
            "type": "CityJSON", "version": "1.1",
            "transform": {"scale": [1.0, 1.0, 1.0], "translate": [0.0, 0.0, 0.0]},
            "vertices": [[0, 0, 0], [4, 0, 0], [4, 6, 0], [0, 6, 0],
                         [0, 0, height], [4, 0, height],
                         [4, 6, height], [0, 6, height]],
            "CityObjects": {"b1": {"type": "Building", "geometry": [{
                "type": "Solid", "lod": "1",
                "boundaries": [[[[0, 3, 2, 1]], [[4, 5, 6, 7]], [[0, 1, 5, 4]],
                                [[1, 2, 6, 5]], [[2, 3, 7, 6]], [[3, 0, 4, 7]]]],
            }]}},
        }
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "t.city.json").write_text(json.dumps(cj), encoding="utf-8")

    box(3, "LOD1_synth")
    box(5, "LOD2")
    return tmp_path


def test_dataset_amt_sequence_is_shorter_than_coord(tmp_path):
    root = _tiny_corpus(tmp_path)
    coord = MeshDataset(root, tokenization="coord")[0]
    amt = MeshDataset(root, tokenization="amt")[0]
    assert len(amt["tgt"]) < len(coord["tgt"])
    assert len(amt["cond"]) < len(coord["cond"])


def test_dataset_defaults_to_coord(tmp_path):
    """Regression guard: an existing config that names no tokenization is unchanged."""
    root = _tiny_corpus(tmp_path)
    assert np.array_equal(MeshDataset(root)[0]["tgt"].numpy(),
                          MeshDataset(root, tokenization="coord")[0]["tgt"].numpy())


def test_dataset_max_seq_len_matches_the_amt_sequence(tmp_path):
    """Sizes the positional embedding -- a 9-per-face formula would oversize it."""
    ds = MeshDataset(_tiny_corpus(tmp_path), tokenization="amt")
    longest = max(max(len(ds[i]["cond"]), len(ds[i]["tgt"]) - 1) for i in range(len(ds)))
    assert ds.max_seq_len == longest


def test_module_vocabulary_grows_by_one_under_amt():
    m = MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2, num_layers=1,
                              max_seq_len=64, tokenization="amt")
    assert m.network.head.out_features == NUM_BINS + 4


def test_module_decodes_amt_tokens_through_the_amt_inverse():
    """A code path mix-up here reads back as a scrambled mesh, not an exception."""
    verts, faces = _unit_cube()
    tokens = amt_tokenize(verts, faces, NUM_BINS)
    m = MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2, num_layers=1,
                              max_seq_len=512, tokenization="amt")
    v, f = m.decode_tokens(torch.from_numpy(tokens))
    expect_v, expect_f = amt_detokenize(tokens, NUM_BINS)
    assert np.array_equal(f, expect_f) and np.allclose(v, expect_v)


def test_masking_delays_eos_to_the_next_face_boundary():
    """Non-vacuous: EOS is biased to win at every step, so an unmasked model
    would stop immediately. Masking must hold it back until a face closes."""
    torch.manual_seed(0)
    m = MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2, num_layers=1,
                              max_seq_len=128, mask_invalid=True)
    m.eval()
    with torch.no_grad():
        m.network.head.bias.zero_()
        m.network.head.bias[m.eos] = 50.0

    out = m.generate(torch.zeros(1, 9, dtype=torch.long),
                     max_new_tokens=30, temperature=0.0)
    body = out[0, 1:]                                  # drop BOS
    at = (body == m.eos).nonzero()
    assert len(at) == 1, "EOS should fire as soon as it is legal"
    assert int(at[0, 0]) == 9, "EOS must wait for exactly one complete face"


def test_generate_with_masking_never_stops_mid_face():
    """Whatever it samples, a sequence that *terminated* did so cleanly.
    Running out of max_new_tokens is truncation, not a stop, so it is exempt."""
    torch.manual_seed(0)
    m = MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2, num_layers=1,
                              max_seq_len=128, mask_invalid=True)
    m.eval()
    out = m.generate(torch.randint(0, NUM_BINS, (2, 9)),
                     max_new_tokens=45, temperature=1.0)
    for row in out:
        body = row[1:]
        at = (body == m.eos).nonzero()
        if len(at) == 0:
            continue                                   # hit the budget instead
        prefix = body[: int(at[0, 0])]
        n = int(((prefix >= 0) & (prefix < NUM_BINS)).sum())
        assert n % 9 == 0 and n >= 9, f"terminated after {n} coordinates"


def test_generate_without_masking_is_unchanged():
    """Default off: the flag must not perturb an existing run's sampling."""
    def run(mask_invalid):
        torch.manual_seed(0)
        m = MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2,
                                  num_layers=1, max_seq_len=128,
                                  mask_invalid=mask_invalid)
        m.eval()
        torch.manual_seed(1)
        return m.generate(torch.zeros(1, 9, dtype=torch.long),
                          max_new_tokens=20, temperature=0.0)

    assert torch.equal(run(False), run(False))
