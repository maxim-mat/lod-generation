"""Frozen DINOv3 wrapper: orientation, null token, checkpoint hygiene. Tiny config, no hub."""
from types import SimpleNamespace

import pytest
import torch
from transformers import DINOv3ViTConfig

from src.models.ortho_encoder import OrthoImageEncoder

S, P = 64, 16
G = S // P


def tiny(**kw):
    torch.manual_seed(0)
    cfg = DINOv3ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=2, patch_size=P, image_size=S)
    return OrthoImageEncoder(out_dim=24, config=cfg, image_size=S, beta=0.1, **kw)


class FakeBackbone(torch.nn.Module):
    """Patch feature = mean of channel 0 over the patch, in the order it was fed."""
    config = SimpleNamespace(num_register_tokens=0, patch_size=P, hidden_size=1)

    def forward(self, pixel_values):
        self.seen = pixel_values
        pooled = torch.nn.functional.avg_pool2d(pixel_values[:, :1], P)       # [B,1,G,G]
        tokens = pooled.flatten(2).transpose(1, 2)                             # row-major
        cls = torch.zeros(tokens.shape[0], 1, 1)
        return SimpleNamespace(last_hidden_state=torch.cat([cls, tokens], dim=1))


def test_backbone_sees_north_up_and_features_come_back_in_mesh_order():
    enc = tiny()
    enc.backbone = FakeBackbone()
    enc.n_skip = 1
    img = torch.zeros(1, 3, S, S, dtype=torch.uint8)
    img[:, :, -P:, :] = 255                       # mesh order: last rows = +y = north
    feats = enc.features(img)                     # [1, G*G, 1]
    grid = feats.view(G, G)
    assert grid[-1].min() > grid[:-1].max()       # north patches stay at the +y end
    seen = enc.backbone.seen[0, 0]
    assert seen[:P].mean() > seen[P:].mean()      # ...but the backbone saw them on top


def test_invalid_rows_are_the_null_token_and_valid_rows_are_not():
    enc = tiny().eval()
    img = torch.randint(0, 255, (2, 3, S, S), dtype=torch.uint8)
    out = enc(img, valid=torch.tensor([True, False]))
    assert out.shape == (2, G * G, 24)
    assert torch.equal(out[1], enc.null.expand(G * G, -1))
    assert not torch.allclose(out[0], out[1])


def test_coordinate_embedding_distinguishes_patches_of_a_flat_image():
    enc = tiny().eval()
    out = enc(torch.full((1, 3, S, S), 128, dtype=torch.uint8))
    assert torch.unique(out[0], dim=0).shape[0] == G * G


def test_backbone_is_frozen_and_kept_out_of_checkpoints():
    enc = tiny()
    enc.train()
    assert not enc.backbone.training
    assert not any(p.requires_grad for p in enc.backbone.parameters())
    sd = enc.state_dict()
    assert sd and not any(k.startswith("backbone.") for k in sd)
    fresh = tiny()
    fresh.load_state_dict(sd, strict=True)        # backbone restored by the pre-hook
    assert torch.equal(fresh.null, enc.null)


def test_wrong_image_size_is_refused():
    with pytest.raises(ValueError, match="image_size"):
        tiny()(torch.zeros(1, 3, 2 * S, 2 * S, dtype=torch.uint8))


def test_backbone_stays_out_of_a_parent_modules_checkpoint():
    # How it is used: nested inside the Lightning module, with a key prefix.
    parent = torch.nn.Module()
    parent.image_encoder = tiny()
    sd = parent.state_dict()
    assert "image_encoder.null" in sd
    assert not any(".backbone." in k for k in sd)
    fresh = torch.nn.Module()
    fresh.image_encoder = tiny()
    fresh.load_state_dict(sd, strict=True)


# ---------------------------------------------------------------------------
# Aligned prefix (docs/superpowers/specs/2026-11-02-aligned-prefix-design.md)
# ---------------------------------------------------------------------------

def test_sample_map_reads_a_ramp_back_exactly():
    from src.dataset.mesh_dataset import dequantize
    from src.dataset.ortho import pixel_to_uv
    from src.models.ortho_encoder import sample_map
    n, beta = 8, 0.1
    fmap = torch.zeros(1, 2, n, n)
    for i in range(n):
        for j in range(n):
            fmap[0, 0, i, j], fmap[0, 1, i, j] = pixel_to_uv(i, j, n, beta)
    bins = [(20, 100), (64, 64), (100, 30)]
    u = torch.tensor([[float(dequantize(bx, 128)) for bx, _ in bins]])
    v = torch.tensor([[float(dequantize(by, 128)) for _, by in bins]])
    got = sample_map(fmap, u, v, beta)                                   # [1, 3, 2]
    assert torch.allclose(got[0, :, 0], u[0], atol=1e-5)
    assert torch.allclose(got[0, :, 1], v[0], atol=1e-5)


def test_pooled_prefix_has_grid_squared_tokens():
    img = torch.randint(0, 255, (1, 3, S, S), dtype=torch.uint8)
    assert tiny(prefix_grid=2)(img).shape == (1, 4, 24)
    with pytest.raises(ValueError, match="prefix_grid"):
        tiny(prefix_grid=3)


def test_encode_without_null_leaves_invalid_rows_unreplaced():
    enc = tiny()
    img = torch.randint(0, 255, (1, 3, S, S), dtype=torch.uint8)
    ctx = enc.encode(img, torch.tensor([False]), use_null=False)
    assert torch.equal(ctx.tokens, enc.encode(img, torch.tensor([True])).tokens)
    assert ctx.valid.tolist() == [False]
    assert enc.encode(img).valid.tolist() == [True]
    assert enc.encode(img).maps is None


def test_vertex_features_start_at_zero():
    enc = tiny(vertex_dim=32)
    img = torch.randint(0, 255, (2, 3, S, S), dtype=torch.uint8)
    ctx = enc.encode(img)
    fine, coarse = ctx.maps
    assert fine.shape[-2:] == (S // 4, S // 4) and coarse.shape[-2:] == (G, G)
    bx = torch.randint(0, 128, (2, 5))
    out = enc.vertex_features(ctx.maps, bx, bx.flip(-1), 128)
    assert out.shape == (2, 5, 32) and not out.any()
    assert "pixel_cnn.0.weight" in enc.state_dict()


def test_fine_map_is_in_mesh_order():
    enc = tiny(vertex_dim=32)
    enc.pixel_cnn = torch.nn.AvgPool2d(4)
    img = torch.zeros(1, 3, S, S, dtype=torch.uint8)
    img[..., :4, :4] = 255                         # mesh-order rows 0-3, columns 0-3
    fine = enc.encode(img).maps[0][0, 0]
    assert fine.argmax().item() == 0


def test_image_context_select_and_repeat_carry_valid_and_maps():
    enc = tiny(vertex_dim=32)
    img = torch.randint(0, 255, (2, 3, S, S), dtype=torch.uint8)
    ctx = enc.encode(img, torch.tensor([True, False]))
    sw = ctx.select(torch.tensor([1, 0]))
    assert sw.valid.tolist() == [False, True]
    assert torch.equal(sw.maps[0][0], ctx.maps[0][1])
    rep = ctx.repeat(3)
    assert rep.valid.tolist() == [True] * 3 + [False] * 3
    assert rep.maps[1].shape[0] == 6 and rep.tokens.shape[0] == 6
    assert ctx.caching().valid is ctx.valid and ctx.caching().maps is ctx.maps
