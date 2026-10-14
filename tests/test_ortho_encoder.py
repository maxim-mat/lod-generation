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
