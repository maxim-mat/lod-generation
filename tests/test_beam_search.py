"""Beam search as an alternative to greedy decoding.

Note this is a *departure* from the reference, not a match: MeshAnything V2
passes `num_beams=1` explicitly (`MeshAnything/models/meshanything_v2.py`) and
offers top-k/top-p sampling as its only alternative to greedy.

The stubs below replace `module.network` so the search is tested against a known
optimum rather than an untrained model's noise. Each stub maps "last token" to a
fixed next-token distribution, which is all the decoder ever asks of it.
"""
import math

import pytest
import torch

from src.dataset.mesh_dataset import specials, vocab_size
from src.models.mesh_transformer import MeshTransformerModule

NUM_BINS = 8
V = vocab_size(NUM_BINS)
BOS, EOS, PAD = specials(NUM_BINS)
CAP = 12


def _module():
    return MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2,
                                 num_layers=1, dropout=0.0, max_seq_len=CAP).eval()


class _Scripted(torch.nn.Module):
    """Network stub: `table[last_token] -> probability vector`."""

    max_seq_len = CAP

    def __init__(self, table, default=None):
        super().__init__()
        self.table = table
        self.default = default if default is not None else {EOS: 1.0}

    def generation_budget(self, n_cond):
        return CAP - 1

    def forward(self, cond, tgt, *a, **k):
        # The decoders hand the network `[tgt..., PAD]` -- one slot to predict
        # into -- so the last *real* token is at -2, not -1.
        out = torch.full((tgt.shape[0], tgt.shape[1] - 1, V), -1e30)
        for row in range(tgt.shape[0]):
            probs = self.table.get(int(tgt[row, -2]), self.default)
            for tok, p in probs.items():
                out[row, -1, tok] = math.log(p)
        return out


def _run(table, default=None, **kw):
    """`use_cache=False` throughout: these stubs implement `forward` only, not the
    incremental protocol. Cache equivalence is `tests/test_kv_cache.py`'s job --
    including for beam search, against real networks."""
    m = _module()
    m.network = _Scripted(table, default)
    cond = torch.zeros(1, 4, dtype=torch.long)
    with torch.no_grad():
        return m.generate(cond, temperature=0.0, use_cache=False, **kw)[0].tolist()


# A trap for greedy. Each row must sum to 1: the stub's values become logits and
# `log_softmax` renormalises them, so a row that sums to less would not encode
# the probabilities it looks like it does.
#
# From BOS token 1 looks better (0.6 > 0.4), but every continuation of 1 is a
# 1-in-8 coin toss, so the branch is worth 0.6 * 0.125. Token 2 leads to a
# certainty and is worth 0.4 * 1.0.
TRAP = {
    BOS: {1: 0.6, 2: 0.4},
    1:   {t: 0.125 for t in range(8)},
    2:   {4: 1.0},
    4:   {EOS: 1.0},
}


def test_greedy_takes_the_locally_best_first_token():
    """Establishes the trap actually traps -- without this the next test proves
    nothing."""
    assert _run(TRAP)[:2] == [BOS, 1]


def test_beam_search_finds_the_sequence_greedy_misses():
    out = _run(TRAP, beam_size=2, length_penalty=0.0)
    assert out[:3] == [BOS, 2, 4], f"beam search stayed on the greedy path: {out}"


def test_beam_size_one_reproduces_greedy_exactly():
    """The option must be free when it is off."""
    assert _run(TRAP, beam_size=1) == _run(TRAP)


def test_output_is_eos_terminated_and_padded_like_greedy():
    """Downstream (`decode_tokens`, `_to_metres`) reads one contract; beam
    search must not hand it a different one."""
    out = _run(TRAP, beam_size=2, length_penalty=0.0)
    assert out[0] == BOS and EOS in out
    tail = out[out.index(EOS) + 1:]
    assert all(t == PAD for t in tail), f"non-pad tokens after EOS: {out}"


def test_length_penalty_decides_between_short_and_long():
    """Unnormalised beam search is biased toward short sequences, and "short"
    here means "fewer triangles" -- the degenerate mesh. The penalty is what
    makes a longer, individually-less-likely continuation competitive.

    From BOS: stop now for log(0.5), or take four 0.8 steps for 4*log(0.8).
    Sum favours stopping (-0.69 > -0.89); the per-token mean favours the run
    (-0.22 > -0.69).
    """
    table = {BOS: {EOS: 0.5, 1: 0.5},
             1: {2: 0.8, EOS: 0.2}, 2: {3: 0.8, EOS: 0.2},
             3: {4: 0.8, EOS: 0.2}, 4: {EOS: 0.8, 5: 0.2}}
    assert _run(table, beam_size=2, length_penalty=0.0)[:2] == [BOS, EOS]
    assert _run(table, beam_size=2, length_penalty=1.0)[:2] == [BOS, 1]


def test_beam_search_respects_the_generation_budget():
    """A model that never emits EOS must still stop, exactly as greedy does."""
    out = _run({}, default={1: 1.0}, beam_size=3)
    assert len(out) == CAP, f"ran past the budget: {len(out)}"


def test_sampling_and_beams_together_are_rejected():
    """Beam search is a deterministic argmax over sequences; combining it with
    temperature sampling is a config mistake, not a mode."""
    with pytest.raises(ValueError, match="beam"):
        m = _module()
        m.network = _Scripted(TRAP)
        m.generate(torch.zeros(1, 4, dtype=torch.long), temperature=0.8,
                   beam_size=2, use_cache=False)


def test_beam_search_honours_masked_invalid_predictions():
    """`mask_invalid` must apply per beam, or beam search reintroduces exactly
    the structurally impossible sequences it exists to forbid."""
    m = MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2,
                              num_layers=1, dropout=0.0, max_seq_len=CAP,
                              mask_invalid=True).eval()
    # EOS is overwhelmingly likely but illegal mid-face, so it must not appear
    # until a whole face (9 coordinate tokens) has been emitted.
    m.network = _Scripted({}, default={EOS: 0.99, 0: 0.01})
    with torch.no_grad():
        out = m.generate(torch.zeros(1, 9, dtype=torch.long), temperature=0.0,
                         beam_size=2, use_cache=False)[0].tolist()
    assert out[1] != EOS, f"EOS emitted mid-face under beam search: {out}"
