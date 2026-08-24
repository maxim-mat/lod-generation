"""Incremental decoding must be a pure speedup: same tokens, byte for byte.

`generate` had no KV cache, so every step re-ran the whole stack over the whole
prefix -- O(L^2) per step, and beam search multiplies that by beam_size. The
cache reuses the *same modules and weights* rather than rebuilding the stack, so
no checkpoint changes; what these tests pin is that it computes the same thing.

Equivalence is the only real defence here: the cached path reimplements the
`norm_first` residual order that `nn.TransformerEncoderLayer.forward` owns, and
a drift between the two would be silent.
"""
import torch

from src.dataset.mesh_dataset import specials, vocab_size
from src.models.mesh_transformer import MeshTransformerModule

NUM_BINS = 16
BOS, EOS, PAD = specials(NUM_BINS)


def _module(tokenization="coord", backbone="scratch", **kw):
    torch.manual_seed(0)
    common = dict(num_bins=NUM_BINS, dropout=0.0, tokenization=tokenization,
                  backbone=backbone, **kw)
    if backbone == "opt":
        from transformers import OPTConfig
        cfg = OPTConfig(vocab_size=vocab_size(NUM_BINS, tokenization),
                        hidden_size=32, word_embed_proj_dim=16,
                        num_hidden_layers=2, num_attention_heads=4, ffn_dim=64,
                        max_position_embeddings=256, dropout=0.0)
        return MeshTransformerModule(opt_config=cfg, **common).eval()
    return MeshTransformerModule(d_model=32, n_head=4, num_layers=2,
                                 max_seq_len=128, **common).eval()


def _cond(n=18, pad_tail=0):
    cond = torch.randint(0, NUM_BINS, (2, n + pad_tail))
    mask = torch.zeros(2, n + pad_tail, dtype=torch.bool)
    if pad_tail:
        cond[1, -pad_tail:], mask[1, -pad_tail:] = PAD, True
    return cond, mask


def _both(m, cond, mask=None, **kw):
    with torch.no_grad():
        off = m.generate(cond, mask, temperature=0.0, use_cache=False, **kw)
        on = m.generate(cond, mask, temperature=0.0, use_cache=True, **kw)
    return off, on


def test_scratch_cached_matches_uncached():
    torch.manual_seed(3)
    m = _module()
    off, on = _both(m, *_cond(), max_new_tokens=24)
    assert torch.equal(off, on), f"cached diverged:\n{off}\n{on}"


def test_scratch_cached_matches_uncached_under_amt():
    """AMT roles are derived from the whole prefix, so an incremental path that
    embeds only the newest token would get them wrong."""
    torch.manual_seed(4)
    m = _module(tokenization="amt")
    off, on = _both(m, *_cond(), max_new_tokens=24)
    assert torch.equal(off, on)


def test_cached_decoding_respects_condition_padding():
    """The cached keys include the padded condition slots; failing to mask them
    lets a short building attend to another building's padding."""
    torch.manual_seed(5)
    m = _module()
    off, on = _both(m, *_cond(n=14, pad_tail=6), max_new_tokens=20)
    assert torch.equal(off, on)


def test_opt_cached_matches_uncached():
    torch.manual_seed(6)
    m = _module(backbone="opt")
    off, on = _both(m, *_cond(), max_new_tokens=20)
    assert torch.equal(off, on)


def test_opt_cached_matches_uncached_with_sinusoidal_positions():
    """`OPTSinusoidalPositions` has to honour `past_key_values_length`, which is
    only exercised once there is a past."""
    torch.manual_seed(7)
    m = _module(backbone="opt", pos_embed="sinusoidal")
    off, on = _both(m, *_cond(), max_new_tokens=20)
    assert torch.equal(off, on)


def test_beam_search_cached_matches_uncached():
    """Beams are reordered every step, so the cache has to be reordered with
    them -- otherwise beam k inherits beam j's history."""
    torch.manual_seed(8)
    m = _module()
    off, on = _both(m, *_cond(), max_new_tokens=20, beam_size=3)
    assert torch.equal(off, on), f"cached beams diverged:\n{off}\n{on}"


def test_opt_beam_search_cached_matches_uncached():
    torch.manual_seed(9)
    m = _module(backbone="opt")
    off, on = _both(m, *_cond(), max_new_tokens=16, beam_size=3)
    assert torch.equal(off, on)


def test_cache_actually_avoids_the_quadratic_rescan():
    """Equivalence alone would also be satisfied by a cache that silently fell
    back to a full forward. This pins that the prefix is encoded once."""
    torch.manual_seed(10)
    m = _module()
    calls = {"n": 0, "tokens": 0}
    real = m.network.forward

    def counting(cond, tgt, *a, **k):
        calls["n"] += 1
        calls["tokens"] += tgt.shape[1]
        return real(cond, tgt, *a, **k)

    m.network.forward = counting
    with torch.no_grad():
        m.generate(_cond()[0], temperature=0.0, max_new_tokens=24, use_cache=True)
    assert calls["n"] == 0, f"cached generation still called full forward {calls['n']}x"
