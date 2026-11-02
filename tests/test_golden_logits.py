"""Backward compatibility pin: today's image paths, frozen as a fixture.

Recorded once from the tree *before* the aligned-prefix change
(docs/superpowers/specs/2026-11-02-aligned-prefix-design.md, test 1), so every
default code path -- image-free, prefix, xattn -- must keep producing the same
logits and the same cached greedy tokens. Regenerate only on purpose:

    python -m tests.test_golden_logits
"""
from pathlib import Path

import torch
from transformers import DINOv3ViTConfig, OPTConfig

from src.dataset.mesh_dataset import specials, vocab_size
from src.models.mesh_transformer import MeshOPTTransformer

NUM_BINS, S = 16, 64
V = vocab_size(NUM_BINS)
BOS, EOS, PAD = specials(NUM_BINS)
FIXTURE = Path(__file__).parent / "fixtures" / "golden_logits.pt"


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


def build_cases():
    """``{case: {"logits", "greedy"}}`` for the image-free, prefix and xattn networks."""
    cases = {}
    for name, fusion in (("image_free", None), ("prefix", "prefix"), ("xattn", "xattn")):
        m = net(fusion)
        if fusion == "xattn":
            with torch.no_grad():
                m.xattn.gates.fill_(0.7)
        cond, tgt, img = batch()
        # One valid, one invalid row, so the null-token path is pinned too.
        ctx = None if fusion is None else m.encode_image(img, torch.tensor([True, False]))
        kw = {} if ctx is None else {"image_ctx": ctx}
        with torch.no_grad():
            logits = m(cond, tgt, **kw)
            cache = m.new_cache(cond, None, **kw)
            seq = torch.full((2, 1), BOS)
            for _ in range(12):
                seq = torch.cat([seq, m.cached_logits(cache, seq).argmax(-1, keepdim=True)], dim=1)
        cases[name] = {"logits": logits, "greedy": seq}
    return cases


def test_defaults_reproduce_golden_logits():
    golden = torch.load(FIXTURE)
    now = build_cases()
    assert golden.keys() == now.keys()
    for name in golden:
        assert torch.equal(golden[name]["greedy"], now[name]["greedy"]), name
        assert torch.allclose(golden[name]["logits"], now[name]["logits"], atol=1e-6, rtol=0), name


if __name__ == "__main__":
    torch.save(build_cases(), FIXTURE)
    print(f"wrote {FIXTURE}")
