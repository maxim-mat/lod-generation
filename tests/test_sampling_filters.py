"""Top-p / min-p filtering and per-row seeded sampling for `generate`.

Nucleus (top-p) sampling: Holtzman et al., "The Curious Case of Neural Text
Degeneration", arXiv:1904.09751. Min-p: Nguyen et al., "Turning Up the Heat:
Min-p Sampling for Creative and Coherent LLM Outputs", arXiv:2407.01082.

The network is replaced by a stub with a known next-token distribution, so the
assertions test the sampler, not an untrained model's noise.
"""
import pytest
import torch

from src.dataset.mesh_dataset import specials, vocab_size
from src.models.mesh_transformer import MeshTransformerModule, filter_logits

NUM_BINS = 8
V = vocab_size(NUM_BINS)
BOS, EOS, PAD = specials(NUM_BINS)
CAP = 12

PROBS = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
LOGITS = PROBS.log()


def _kept(filtered):
    return torch.isfinite(filtered[0]).nonzero().flatten().tolist()


def test_top_p_keeps_the_smallest_set_reaching_the_mass():
    assert _kept(filter_logits(LOGITS, 1.0, top_p=0.75)) == [0, 1]
    assert _kept(filter_logits(LOGITS, 1.0, top_p=0.85)) == [0, 1, 2]


def test_min_p_keeps_tokens_above_a_fraction_of_the_top_probability():
    assert _kept(filter_logits(LOGITS, 1.0, min_p=0.2)) == [0, 1, 2]   # p >= 0.10
    assert _kept(filter_logits(LOGITS, 1.0, min_p=0.5)) == [0, 1]      # p >= 0.25


def test_filters_compose_as_an_intersection():
    assert _kept(filter_logits(LOGITS, 1.0, top_p=0.85, min_p=0.5)) == [0, 1]


def test_temperature_is_applied_before_the_filter():
    # At T=2 the probabilities flatten to ~[0.379, 0.294, 0.208, 0.120], so the
    # min-p threshold 0.5 * 0.379 = 0.19 now admits token 2 as well.
    out = filter_logits(LOGITS, 2.0, min_p=0.5)
    assert _kept(out) == [0, 1, 2]
    assert torch.allclose(out[0, 0], LOGITS[0, 0] / 2.0)


def test_masked_tokens_stay_masked_and_a_row_is_never_emptied():
    logits = torch.tensor([[float("-inf"), 0.0, float("-inf"), -50.0]])
    assert _kept(filter_logits(logits, 1.0, top_p=1e-9, min_p=1.0)) == [1]


@pytest.mark.parametrize("temperature, kw", [
    (0.0, {}), (1.0, {"top_p": 0.0}), (1.0, {"top_p": 1.5}),
    (1.0, {"min_p": -0.1}), (1.0, {"min_p": 1.1}),
])
def test_invalid_arguments_raise(temperature, kw):
    with pytest.raises(ValueError):
        filter_logits(LOGITS, temperature, **kw)


class _Die(torch.nn.Module):
    """Network stub: every step is the same distribution over the 8 coordinate
    tokens. `skew` raises token 0's logit; everything else is unreachable."""

    max_seq_len = CAP

    def __init__(self, skew=0.0):
        super().__init__()
        self.skew = skew

    def generation_budget(self, n_cond):
        return CAP - 1

    def forward(self, cond, tgt, *a, **k):
        out = torch.full((tgt.shape[0], tgt.shape[1] - 1, V), -1e30)
        out[:, :, :NUM_BINS] = 0.0
        out[:, :, 0] = self.skew
        return out


def _module(skew=0.0):
    m = MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2,
                              num_layers=1, dropout=0.0, max_seq_len=CAP).eval()
    m.network = _Die(skew)
    return m


def _gen(m, seeds, **kw):
    """`use_cache=False`: the stub implements `forward` only."""
    kw.setdefault("temperature", 1.0)
    cond = torch.zeros(len(seeds), 4, dtype=torch.long)
    with torch.no_grad():
        return m.generate(cond, use_cache=False, seeds=seeds, **kw)


def test_same_seeds_give_the_same_tokens():
    m = _module()
    assert torch.equal(_gen(m, [3, 4]), _gen(m, [3, 4]))


def test_different_seeds_give_different_tokens():
    out = _gen(_module(), [3, 4])
    assert not torch.equal(out[0], out[1])


def test_a_rows_sample_does_not_depend_on_its_batch_neighbours():
    m = _module()
    together = _gen(m, [3, 4, 5])
    assert torch.equal(together[1], _gen(m, [4])[0])
    assert torch.equal(together[[2, 0, 1]], _gen(m, [5, 3, 4]))


@pytest.mark.parametrize("kw", [{"top_p": 0.3}, {"min_p": 0.5}])
def test_a_tight_filter_leaves_only_the_top_token(kw):
    # skew 2.0: p(token 0) = e^2 / (e^2 + 7) = 0.513, every other token 0.070.
    m = _module(skew=2.0)
    assert (_gen(m, [3, 4], **kw)[:, 1:] == 0).all()
    with torch.no_grad():
        unseeded = m.generate(torch.zeros(2, 4, dtype=torch.long),
                              temperature=1.0, use_cache=False, **kw)
    assert (unseeded[:, 1:] == 0).all()


def test_without_a_filter_the_skewed_die_still_rolls_other_faces():
    assert (_gen(_module(skew=2.0), [3, 4])[:, 1:] != 0).any()


def test_unseeded_sampling_still_draws_from_the_global_rng():
    """`seeds=None` without a filter must stay token-identical to before this
    change: one `torch.multinomial` per step on the global RNG."""
    m = _module()
    cond = torch.zeros(2, 4, dtype=torch.long)
    torch.manual_seed(0)
    with torch.no_grad():
        got = m.generate(cond, temperature=1.0, use_cache=False)
    torch.manual_seed(0)
    probs = torch.zeros(2, V)
    probs[:, :NUM_BINS] = 1.0 / NUM_BINS
    expect = torch.stack([torch.multinomial(probs, 1).squeeze(-1)
                          for _ in range(CAP - 1)], dim=1)
    assert torch.equal(got[:, 1:], expect)


@pytest.mark.parametrize("kw", [{"top_p": 0.9}, {"min_p": 0.1}])
def test_filters_with_beam_search_are_rejected(kw):
    with pytest.raises(ValueError, match="beam"):
        _module().generate(torch.zeros(1, 4, dtype=torch.long), temperature=0.0,
                           beam_size=2, use_cache=False, **kw)


def test_filters_without_temperature_are_rejected():
    with pytest.raises(ValueError, match="temperature"):
        _module().generate(torch.zeros(1, 4, dtype=torch.long), temperature=0.0,
                           top_p=0.9, use_cache=False)


def test_seed_count_must_match_the_batch():
    with pytest.raises(ValueError, match="seeds"):
        _module().generate(torch.zeros(3, 4, dtype=torch.long), temperature=1.0,
                           seeds=[1, 2], use_cache=False)
