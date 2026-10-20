"""Image fusion into the OPT backbone: exact v5 at zero gates, cache equivalence."""
import torch
from transformers import DINOv3ViTConfig, OPTConfig

from src.dataset.mesh_dataset import specials, vocab_size
from src.models.mesh_transformer import MeshOPTTransformer

NUM_BINS, S = 16, 64
V = vocab_size(NUM_BINS)
BOS, EOS, PAD = specials(NUM_BINS)


def net(fusion=None, seed=0):
    torch.manual_seed(seed)
    opt = OPTConfig(vocab_size=V, hidden_size=32, word_embed_proj_dim=32, num_hidden_layers=2,
                    num_attention_heads=4, ffn_dim=64, max_position_embeddings=512, dropout=0.0)
    image = None if fusion is None else {
        "fusion": fusion, "name": None, "image_size": S, "beta": 0.1,
        "config": DINOv3ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                  num_attention_heads=2, patch_size=16, image_size=S)}
    return MeshOPTTransformer(vocab_size=V, opt_config=opt, dropout=0.0,
                              pos_embed="sinusoidal", max_seq_len=256, num_bins=NUM_BINS,
                              image=image).eval()


def batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    cond = torch.randint(0, NUM_BINS, (2, 18), generator=g)
    tgt = torch.cat([torch.full((2, 1), BOS), torch.randint(0, NUM_BINS, (2, 18), generator=g),
                     torch.full((2, 1), EOS)], dim=1)
    img = torch.randint(0, 255, (2, 3, S, S), generator=g, dtype=torch.uint8)
    return cond, tgt, img


def test_zero_gates_reproduce_the_image_free_model_exactly():
    base, fused = net(None), net("xattn")
    fused.load_state_dict(base.state_dict(), strict=False)      # same OPT weights
    cond, tgt, img = batch()
    ctx = fused.encode_image(img, torch.tensor([True, True]))
    with torch.no_grad():
        assert torch.equal(base(cond, tgt), fused(cond, tgt, image_ctx=ctx))


def test_open_gates_make_the_image_matter():
    fused = net("xattn")
    with torch.no_grad():
        fused.xattn.gates.fill_(1.0)
    cond, tgt, img = batch()
    ok = torch.tensor([True, True])
    with torch.no_grad():
        a = fused(cond, tgt, image_ctx=fused.encode_image(img, ok))
        b = fused(cond, tgt, image_ctx=fused.encode_image(img.flip(0), ok))
    assert not torch.allclose(a, b)


def test_old_state_dict_keys_are_untouched():
    base, fused = net(None), net("xattn")
    missing, unexpected = fused.load_state_dict(base.state_dict(), strict=False)
    assert unexpected == []
    assert all(k.startswith(("image_encoder.", "xattn.")) for k in missing)


def _greedy(m, cond, img, use_cache):
    """Greedy decode through the network's own cached / uncached paths."""
    ctx = m.encode_image(img, torch.tensor([True, True]))
    tgt = torch.full((2, 1), BOS)
    cache = m.new_cache(cond, None, image_ctx=ctx) if use_cache else None
    with torch.no_grad():
        for _ in range(12):
            if cache is None:
                step = torch.cat([tgt, torch.full_like(tgt[:, :1], PAD)], dim=1)
                logits = m(cond, step, image_ctx=ctx)[:, -1]
            else:
                logits = m.cached_logits(cache, tgt)
            tgt = torch.cat([tgt, logits.argmax(-1, keepdim=True)], dim=1)
    return tgt


def test_cached_decoding_matches_uncached_with_images():
    for fusion in ("xattn", "prefix"):
        m = net(fusion)
        if fusion == "xattn":
            with torch.no_grad():
                m.xattn.gates.fill_(0.7)
        cond, _, img = batch()
        assert torch.equal(_greedy(m, cond, img, False), _greedy(m, cond, img, True)), fusion


def test_reorder_cache_reorders_the_image_context():
    m = net("xattn")
    with torch.no_grad():
        m.xattn.gates.fill_(0.7)
    cond, _, img = batch()
    ctx = m.encode_image(img, torch.tensor([True, True]))
    cache = m.new_cache(cond, None, image_ctx=ctx)
    tgt = torch.full((2, 1), BOS)
    with torch.no_grad():
        m.cached_logits(cache, tgt)
        swapped = m.reorder_cache(cache, torch.tensor([1, 0]))
        nxt = torch.cat([tgt, torch.zeros(2, 1, dtype=torch.long)], dim=1)
        a = m.cached_logits(swapped, nxt)
        ref = m.new_cache(cond.flip(0), None, image_ctx=ctx.select(torch.tensor([1, 0])))
        m.cached_logits(ref, tgt)
        b = m.cached_logits(ref, nxt)
    torch.testing.assert_close(a, b)
