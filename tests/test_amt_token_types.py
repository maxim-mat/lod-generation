"""AMT token-role embeddings (MeshAnything V2, arXiv:2408.02555 section 4.2).

An AMT sequence is irregular by construction: a face start spends three
vertices, an adjacent face spends one, and `&` spends none. The paper adds a
distinct embedding per role so the transformer can parse that structure --
"one for the three vertices of a newly started face, another for a single new
vertex, and distinct embeddings for the special tokens". Without it the model
sees a flat run of coordinate ids with no signal about which face boundary it
is inside, and V2's own ablation shows regularity matters as much as
compression (unsorted baseline CD 8.151 vs 2.478 sorted).

The roles are recoverable from the prefix alone, so the same function serves
training and sampling.
"""
import numpy as np
import pytest
import torch

from src.dataset.mesh_dataset import NUM_BINS, break_token, specials, vocab_size
from src.models.mesh_transformer import (ADJACENT, FACE_START, MeshEmbedding, N_ROLES,
                                         ROLE_BOS, ROLE_BREAK, ROLE_EOS, ROLE_PAD)

BOS, EOS, PAD = specials(NUM_BINS)
BRK = break_token(NUM_BINS)


def amt_embedding(d_model=16):
    return MeshEmbedding(vocab_size(NUM_BINS, "amt"), d_model,
                         tokenization="amt", num_bins=NUM_BINS)


def coord_embedding(d_model=16):
    return MeshEmbedding(vocab_size(NUM_BINS, "coord"), d_model,
                         tokenization="coord", num_bins=NUM_BINS)


def test_roles_of_a_hand_built_sequence():
    """BOS, one 3-vertex face start, one adjacent vertex, a break, another start.

    Layout is MeshAnythingV2's `OPTLoopEmbedding`: face-start rows 4/5/6 and
    adjacent rows 7/8/9, each cycling x -> y -> z.
    """
    ids = torch.tensor([[BOS] + [7] * 9 + [8] * 3 + [BRK] + [9] * 9])
    roles = amt_embedding().token_roles(ids)[0].tolist()

    assert roles[0] == ROLE_BOS
    assert roles[1:10] == [FACE_START + a % 3 for a in range(9)]     # 4,5,6,4,5,6,4,5,6
    assert roles[10:13] == [ADJACENT + a % 3 for a in range(3)]      # 7,8,9
    assert roles[13] == ROLE_BREAK                                   # its own row
    assert roles[14:23] == [FACE_START + a % 3 for a in range(9)]    # phase resets


def test_each_special_gets_its_own_row():
    """The reference gives BOS/EOS/PAD rows 0/1/2 and the break row 3."""
    ids = torch.tensor([[BOS, 1, 2, 3, EOS, PAD, BRK]])
    roles = amt_embedding().token_roles(ids)[0].tolist()
    assert roles[0] == ROLE_BOS
    assert roles[4] == ROLE_EOS
    assert roles[5] == ROLE_PAD
    assert roles[6] == ROLE_BREAK
    assert len({ROLE_BOS, ROLE_EOS, ROLE_PAD, ROLE_BREAK}) == 4


def test_axis_phase_survives_a_break():
    """The reason the axis split exists at all.

    Under AMT a break desynchronises position-mod-9, so nothing but the role
    tells the model which coordinate it is emitting. A break must restart the
    x/y/z cycle, not carry the old phase across.
    """
    emb = amt_embedding()
    # 4 coordinate tokens (phase now mid-vertex), then a break, then a face.
    ids = torch.tensor([[BOS] + [7] * 4 + [BRK] + [9] * 6])
    roles = emb.token_roles(ids)[0].tolist()
    assert roles[6:12] == [FACE_START + a % 3 for a in range(6)], roles


def test_every_role_is_inside_the_table():
    emb = amt_embedding()
    ids = torch.tensor([[BOS] + [7] * 30 + [BRK] + [9] * 30 + [EOS, PAD]])
    roles = emb.token_roles(ids)
    assert int(roles.min()) >= 0 and int(roles.max()) < N_ROLES


def test_roles_are_causal():
    """A role must not depend on tokens the model has not emitted yet.

    Otherwise the embedding leaks the future at training time and cannot be
    reproduced during sampling.
    """
    emb = amt_embedding()
    full = torch.tensor([[BOS] + [7] * 9 + [8] * 3 + [BRK] + [9] * 9])
    ref = emb.token_roles(full)[0]
    for cut in range(1, full.shape[1] + 1):
        got = emb.token_roles(full[:, :cut])[0]
        assert torch.equal(got, ref[:cut]), f"role changed with more context at {cut}"


def test_a_long_run_without_breaks_stays_adjacent():
    ids = torch.tensor([[BOS] + [5] * (9 + 3 * 20)])
    roles = amt_embedding().token_roles(ids)[0].tolist()
    assert roles[10:] == [ADJACENT + a % 3 for a in range(3 * 20)]


def test_the_embedding_actually_changes_the_output():
    emb = amt_embedding()
    with torch.no_grad():
        emb.role_embed.weight.normal_(std=1.0)
    ids = torch.tensor([[BOS] + [7] * 9 + [8] * 3])
    out = emb.embed_tokens(ids)
    assert not torch.allclose(out[0, 1], out[0, 10]), \
        "a face-start vertex and an adjacent vertex with the same id must differ"


def test_coord_mode_has_no_role_embedding():
    """On the coordinate path every face is 9 tokens; there is no role to learn."""
    emb = coord_embedding()
    assert not hasattr(emb, "role_embed") or emb.role_embed is None
    ids = torch.tensor([[BOS] + [7] * 9])
    assert torch.allclose(emb.embed_tokens(ids), emb.token_embed(ids))


def test_batch_rows_are_independent():
    emb = amt_embedding()
    a = torch.tensor([[BOS] + [7] * 9 + [8] * 3])
    b = torch.tensor([[BOS] + [7] * 3 + [BRK] + [8] * 9 + [1, 1]])
    both = emb.token_roles(torch.cat([a, b[:, :a.shape[1]]], dim=0))
    assert torch.equal(both[0], emb.token_roles(a)[0])
    assert torch.equal(both[1], emb.token_roles(b[:, :a.shape[1]])[0])


def test_same_id_on_different_axes_embeds_differently():
    """The axis split, checked at the embedding rather than the role id.

    Three identical coordinate values in one vertex are x, y and z. Without
    rows 4/5/6 they would be indistinguishable to the model under AMT, where a
    break has broken the position-mod-9 phase.
    """
    emb = amt_embedding()
    with torch.no_grad():
        emb.role_embed.weight.normal_(std=1.0)
    ids = torch.tensor([[BOS] + [7] * 9])
    out = emb.embed_tokens(ids)[0]
    assert not torch.allclose(out[1], out[2]), "x and y are the same vector"
    assert not torch.allclose(out[2], out[3]), "y and z are the same vector"
    assert torch.allclose(out[1], out[4]), "same axis, same role -> same vector"


# ----------------------------------------------------------------------
# Differential check against the reference implementation
# ----------------------------------------------------------------------

def _to_reference_vocab(ids):
    """Our token ids in MeshAnythingV2's layout.

    They put the three specials at 0/1/2 and shift the coordinate bins up by
    three, so `n_discrete_size + 3` (131) is the '&' break -- the same slot we
    use, reached from the other end.
    """
    out = np.array(ids, dtype=np.int64).copy()
    coord = (out >= 0) & (out < NUM_BINS)
    out[coord] += 3
    for ours, theirs in ((BOS, 0), (EOS, 1), (PAD, 2)):
        out[np.array(ids) == ours] = theirs
    return out


def reference_roles(ids):
    """Literal transcription of `OPTLoopEmbedding.forward`.

    MeshAnythingV2, MeshAnything/models/shape_opt.py. Kept as an explicit
    line-by-line port rather than a paraphrase: it is the thing our vectorised
    version has to agree with, and a paraphrase would drift with it.
    """
    n_discrete_size = NUM_BINS + 3
    state, loop_state = 9, 0                 # what their init_state() sets
    roles = []
    for cur in _to_reference_vocab(ids):
        if cur in (0, 1, 2):                 # idx_in_extra
            state, loop_state = 9, 0
            roles.append(int(cur))           # face_ids stays = input_ids
        elif cur == n_discrete_size:         # the '&' break
            roles.append(3)
            state, loop_state = 9, 0
        else:
            if state == 0:
                roles.append(7 + loop_state % 3)
            else:
                state -= 1
                roles.append(4 + loop_state % 3)
            loop_state += 1
    return roles


@pytest.mark.parametrize("name,seq", [
    ("face start, adjacent, break, restart", [BOS] + [7] * 9 + [8] * 3 + [BRK] + [9] * 9),
    ("break mid-vertex", [BOS] + [7] * 4 + [BRK] + [9] * 6),
    ("two breaks in a row", [BOS] + [7] * 9 + [BRK] + [BRK] + [9] * 3),
    ("eos then padding", [BOS] + [7] * 9 + [EOS] + [PAD] * 4),
    ("long adjacent run", [BOS] + [5] * (9 + 3 * 11)),
    ("condition segment, no BOS", [7] * 9 + [8] * 3),
    # Unreachable in this pipeline, but the reference resets on any special and
    # so must we -- otherwise a mesh packed after an EOS inherits the previous
    # one's axis phase.
    ("EOS mid-sequence then coords", [BOS] + [7] * 4 + [EOS] + [9] * 6),
    ("PAD mid-sequence then coords", [BOS] + [7] * 5 + [PAD] + [9] * 7),
])
def test_matches_the_reference_implementation(name, seq):
    ours = amt_embedding().token_roles(torch.tensor([seq]))[0].tolist()
    assert ours == reference_roles(seq), name
