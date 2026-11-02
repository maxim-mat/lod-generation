"""Aligned image prefix on the OPT network (docs/superpowers/specs/2026-11-02-aligned-prefix-design.md).

Tiny OPT + tiny DINOv3 (native grid 4x4 on 64-px crops); no hub, no dataset.
"""
import pytest
import torch
from transformers import DINOv3ViTConfig, OPTConfig

from src.dataset.mesh_dataset import specials, vocab_size
from src.models.mesh_transformer import MeshOPTTransformer
from src.models.prefix_layout import NO_POSITION, mesh_position_ids, prefix_attention_mask

NUM_BINS, S = 16, 64
V = vocab_size(NUM_BINS)
BOS, EOS, PAD = specials(NUM_BINS)
NEW = dict(separate_positions=True, mask_invalid=True, prefix_attention="bidirectional")


def net(fusion=None, seed=0, pos_embed="sinusoidal", tokenization="coord", **flags):
    torch.manual_seed(seed)
    opt = OPTConfig(vocab_size=V, hidden_size=32, word_embed_proj_dim=32, num_hidden_layers=2,
                    num_attention_heads=4, ffn_dim=64, max_position_embeddings=512, dropout=0.0)
    image = None if fusion is None else {
        "fusion": fusion, "name": None, "image_size": S, "beta": 0.1,
        "config": DINOv3ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                  num_attention_heads=2, patch_size=16, image_size=S),
        **flags}
    return MeshOPTTransformer(vocab_size=V, opt_config=opt, dropout=0.0, pos_embed=pos_embed,
                              max_seq_len=256, num_bins=NUM_BINS, tokenization=tokenization,
                              image=image).eval()


def batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    cond = torch.randint(0, NUM_BINS, (2, 18), generator=g)
    tgt = torch.cat([torch.full((2, 1), BOS), torch.randint(0, NUM_BINS, (2, 18), generator=g),
                     torch.full((2, 1), EOS)], dim=1)
    img = torch.randint(0, 255, (2, 3, S, S), generator=g, dtype=torch.uint8)
    return cond, tgt, img


def aligned(base=None, **flags):
    """A new-mode prefix network carrying `base`'s weights where names match."""
    m = net("prefix", **flags)
    if base is not None:
        m.load_state_dict(base.state_dict(), strict=False)
    return m


def _last_hidden(m, cond, tgt, ctx, **kw):
    seen = {}
    h = m.opt.model.decoder.layers[-1].register_forward_hook(
        lambda mod, args, out: seen.update(h=out[0] if isinstance(out, tuple) else out))
    try:
        with torch.no_grad():
            m(cond, tgt, image_ctx=ctx, **kw)
    finally:
        h.remove()
    return seen["h"]


def test_4d_bool_mask_equals_the_default_causal_mask():
    m = net(None)
    x = torch.randn(2, 7, 32)
    keep = torch.ones(2, 7, dtype=torch.bool)
    none = torch.zeros(2, 0, dtype=torch.bool)
    with torch.no_grad():
        ref = m.opt(inputs_embeds=x, attention_mask=keep.long()).logits
        got = m.opt(inputs_embeds=x, attention_mask=prefix_attention_mask(none, keep, none, False),
                    position_ids=mesh_position_ids(0, keep)).logits
    assert torch.allclose(ref, got, atol=1e-6)


def _ctx(m, img, valid):
    return m.encode_image(img, torch.tensor(valid))


def test_masked_image_reproduces_the_image_free_model():
    base = net(None)
    new = aligned(base, **NEW)
    cond, tgt, img = batch()
    with torch.no_grad():
        assert torch.allclose(base(cond, tgt), new(cond, tgt, image_ctx=_ctx(new, img, [False, False])),
                              atol=1e-5)


def test_mixed_validity_rows():
    base = net(None)
    new = aligned(base, **NEW)
    cond, tgt, img = batch()
    with torch.no_grad():
        ref = base(cond, tgt)
        got = new(cond, tgt, image_ctx=_ctx(new, img, [False, True]))
    assert torch.allclose(ref[0], got[0], atol=1e-5)
    assert not torch.allclose(ref[1], got[1], atol=1e-3)


def test_padded_condition_gives_the_same_target_logits():
    new = aligned(**NEW)
    cond, tgt, img = batch()
    padded = torch.cat([cond, torch.full((2, 3), PAD)], dim=1)
    pad_mask = torch.zeros_like(padded, dtype=torch.bool)
    pad_mask[:, -3:] = True
    tgt_mask = torch.zeros_like(tgt, dtype=torch.bool)
    for valid in ([True, True], [False, False]):
        ctx = _ctx(new, img, valid)
        with torch.no_grad():
            ref = new(cond, tgt, torch.zeros_like(cond, dtype=torch.bool), tgt_mask, image_ctx=ctx)
            got = new(padded, tgt, pad_mask, tgt_mask, image_ctx=ctx)
        assert torch.allclose(ref, got, atol=1e-5), valid


def test_bidirectional_lets_image_tokens_see_later_lod1():
    new = aligned(**NEW)
    cond, tgt, img = batch()
    ctx = _ctx(new, img, [True, True])
    other = cond.clone()
    other[:, -1] = (other[:, -1] + 1) % NUM_BINS
    p = ctx.tokens.shape[1]
    a, b = _last_hidden(new, cond, tgt, ctx), _last_hidden(new, other, tgt, ctx)
    assert not torch.allclose(a[:, :p], b[:, :p], atol=1e-4)


def test_masked_image_keeps_lod1_causal():
    new = aligned(**NEW)
    cond, tgt, img = batch()
    ctx = _ctx(new, img, [False, False])
    other = cond.clone()
    other[:, -1] = (other[:, -1] + 1) % NUM_BINS
    p, c = ctx.tokens.shape[1], cond.shape[1]
    a, b = _last_hidden(new, cond, tgt, ctx), _last_hidden(new, other, tgt, ctx)
    assert torch.allclose(a[:, p:p + c - 1], b[:, p:p + c - 1], atol=1e-6)


@pytest.mark.parametrize("flags", [{}, {"separate_positions": True}, NEW])
def test_target_stays_causal(flags):
    new = aligned(**flags)
    cond, tgt, img = batch()
    ctx = _ctx(new, img, [True, False])
    other = tgt.clone()
    other[:, 10] = (other[:, 10] + 1) % NUM_BINS
    with torch.no_grad():
        a, b = new(cond, tgt, image_ctx=ctx), new(cond, other, image_ctx=ctx)
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-6)


def test_mesh_positions_do_not_move_with_the_prefix():
    new = aligned(**NEW)
    cond, tgt, img = batch()
    ctx = _ctx(new, img, [True, True])
    seen = {}
    h = new.opt.model.decoder.embed_positions.register_forward_hook(
        lambda mod, args, kwargs, out: seen.update(ids=kwargs.get("position_ids")),
        with_kwargs=True)
    with torch.no_grad():
        new(cond, tgt, image_ctx=ctx)
    h.remove()
    p, n = ctx.tokens.shape[1], cond.shape[1] + tgt.shape[1] - 1
    ids = seen["ids"]
    assert (ids[:, :p] == NO_POSITION).all()
    assert torch.equal(ids[:, p:], torch.arange(n).expand(2, n))


def _greedy(m, cond, img, valid, use_cache, steps=12):
    ctx = m.encode_image(img, torch.tensor(valid))
    tgt = torch.full((2, 1), BOS)
    cache = m.new_cache(cond, None, image_ctx=ctx) if use_cache else None
    with torch.no_grad():
        for _ in range(steps):
            if cache is None:
                step = torch.cat([tgt, torch.full_like(tgt[:, :1], PAD)], dim=1)
                logits = m(cond, step, image_ctx=ctx)[:, -1]
            else:
                logits = m.cached_logits(cache, tgt)
            tgt = torch.cat([tgt, logits.argmax(-1, keepdim=True)], dim=1)
    return tgt


CACHE_COMBOS = [{}, {"separate_positions": True}, NEW, {**NEW, "prefix_attention": "causal"},
                {**NEW, "prefix_grid": 2}]


@pytest.mark.parametrize("flags", CACHE_COMBOS)
def test_cached_matches_uncached(flags):
    m = aligned(**flags)
    cond, _, img = batch()
    assert torch.equal(_greedy(m, cond, img, [True, False], False),
                       _greedy(m, cond, img, [True, False], True))


@pytest.mark.parametrize("fusion, kw", [
    ("xattn", {"separate_positions": True}),
    ("prefix", {"separate_positions": True, "pos_embed": "learned"}),
    ("prefix", {"mask_invalid": True}),
    ("prefix", {"separate_positions": True, "prefix_attention": "sideways"}),
    ("prefix", {"prefix_attention": "bidirectional"}),
    ("xattn", {"prefix_grid": 2}),
    ("prefix", {"vertex_features": True, "tokenization": "amt"}),
])
def test_invalid_option_combinations_raise(fusion, kw):
    with pytest.raises(ValueError):
        net(fusion, **kw)


B_ON = {**NEW, "vertex_features": True}


def _randomise_b(m):
    with torch.no_grad():
        m.image_encoder.vertex_proj[-1].weight.normal_()


def test_vertex_features_start_at_zero():
    off = aligned(**NEW)
    on = aligned(off, **B_ON)
    cond, tgt, img = batch()
    for valid in ([True, True], [True, False]):
        with torch.no_grad():
            a = off(cond, tgt, image_ctx=_ctx(off, img, valid))
            b = on(cond, tgt, image_ctx=_ctx(on, img, valid))
        assert torch.equal(a, b), valid


def test_vertex_features_follow_the_image():
    m = aligned(**B_ON)
    _randomise_b(m)
    cond, tgt, img = batch()
    other = torch.randint(0, 255, img.shape, generator=torch.Generator().manual_seed(9),
                          dtype=torch.uint8)
    with torch.no_grad():
        on = [m(cond, tgt, image_ctx=_ctx(m, i, [True, True])) for i in (img, other)]
        off = [m(cond, tgt, image_ctx=_ctx(m, i, [False, False])) for i in (img, other)]
    assert not torch.allclose(on[0], on[1], atol=1e-4)
    assert torch.equal(off[0], off[1])


def test_vertex_features_also_work_with_xattn():
    off = net("xattn")
    on = net("xattn", vertex_features=True)
    on.load_state_dict(off.state_dict(), strict=False)
    cond, tgt, img = batch()
    with torch.no_grad():
        assert torch.equal(off(cond, tgt, image_ctx=_ctx(off, img, [True, True])),
                           on(cond, tgt, image_ctx=_ctx(on, img, [True, True])))


def test_cached_matches_uncached_with_vertex_features():
    m = aligned(**B_ON)
    _randomise_b(m)
    cond, _, img = batch()
    assert torch.equal(_greedy(m, cond, img, [True, False], False),
                       _greedy(m, cond, img, [True, False], True))


def test_cached_step_projects_only_the_new_vertex_rows():
    m = aligned(**B_ON)
    _randomise_b(m)
    cond, _, img = batch()
    rows = []
    h = m.image_encoder.vertex_proj.register_forward_hook(
        lambda mod, args, out: rows.append(args[0].shape[1]))
    try:
        _greedy(m, cond, img, [True, True], True, steps=6)
    finally:
        h.remove()
    # The condition once, then exactly one new target row per step.
    assert sorted(rows) == [1] * 6 + [cond.shape[1]], rows
