"""Frozen DINOv3 encoding of the per-building orthophoto crop.

DINOv3 (Siméoni et al., arXiv:2508.10104) is used as a frozen feature extractor,
as the frozen-vision-encoder VLMs do (Flamingo arXiv:2204.14198, LLaVA
arXiv:2304.08485); only the projector, the coordinate embedding and the null
token train.

Orientation: the dataset's crop is in mesh order (row index along +y). DINOv3
was trained on north-up images, so rows are flipped before the backbone and the
patch grid is flipped back after it -- the tokens leave in mesh order again.

The backbone is excluded from `state_dict` and restored on load, so checkpoints
stay the size of the trainable model; it is rebuilt from the Hugging Face hub
(or the test config) at construction.
"""
import math

import numpy as np
import torch
import torch.nn as nn

from src.dataset.ortho import pixel_to_uv

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _uv_features(grid, beta, n_freq):
    """``[grid*grid, 4*n_freq]`` sin/cos code of each patch centre's (u, v), mesh order."""
    uv = np.array([pixel_to_uv(i, j, grid, beta) for i in range(grid) for j in range(grid)])
    freqs = (2.0 ** np.arange(n_freq)) * math.pi
    ang = uv[:, :, None] * freqs                                     # [P, 2, F]
    feats = np.concatenate([np.sin(ang), np.cos(ang)], axis=-1)      # [P, 2, 2F]
    return torch.tensor(feats.reshape(len(uv), -1), dtype=torch.float32)


class OrthoImageEncoder(nn.Module):
    """``uint8 [B, 3, S, S]`` mesh-order crops -> ``[B, (S/p)^2, out_dim]`` image tokens.

    Args:
        out_dim: token width the fusion expects (OPT hidden size for
            cross-attention, its embedding width for prefix tokens).
        name: Hugging Face id, e.g. ``facebook/dinov3-vitl16-pretrain-lvd1689m``.
        config: a `DINOv3ViTConfig` instead of `name` (tests; no download).
        image_size: crop side S; must be a multiple of the patch size.
        beta: the crop's margin beyond the unit box, for the coordinate code.
        n_freq: frequencies per axis of the coordinate code.
    """

    def __init__(self, out_dim, name=None, config=None, image_size=256, beta=0.1, n_freq=8):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel, DINOv3ViTModel

        if (name is None) == (config is None):
            raise ValueError("Give exactly one of `name` or `config`.")
        if config is None:
            self.backbone = AutoModel.from_pretrained(name)
            proc = AutoImageProcessor.from_pretrained(name)
            mean, std = proc.image_mean, proc.image_std
        else:
            self.backbone = DINOv3ViTModel(config)
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        cfg = self.backbone.config
        if image_size % cfg.patch_size:
            raise ValueError(f"image_size {image_size} is not a multiple of the "
                             f"patch size {cfg.patch_size}.")
        self.image_size = image_size
        self.grid = image_size // cfg.patch_size
        self.n_skip = 1 + int(getattr(cfg, "num_register_tokens", 0) or 0)

        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("uv", _uv_features(self.grid, beta, n_freq), persistent=False)
        self.proj = nn.Sequential(nn.Linear(cfg.hidden_size, out_dim), nn.GELU(),
                                  nn.Linear(out_dim, out_dim))
        self.coord = nn.Linear(4 * n_freq, out_dim)
        self.null = nn.Parameter(torch.randn(out_dim) * 0.02)

        self.register_state_dict_post_hook(_drop_backbone)
        self.register_load_state_dict_pre_hook(_restore_backbone)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()          # frozen means no dropout either
        return self

    def features(self, image):
        """Frozen backbone patch features ``[B, G*G, C]`` in mesh order."""
        if image.shape[-1] != self.image_size or image.shape[-2] != self.image_size:
            raise ValueError(f"Expected {self.image_size}px crops (image_size), got "
                             f"{tuple(image.shape[-2:])}.")
        x = image.float().div(255.0).flip(-2)                  # mesh order -> north-up
        x = (x - self.mean) / self.std
        with torch.no_grad():
            tokens = self.backbone(pixel_values=x).last_hidden_state[:, self.n_skip:]
        b, g = tokens.shape[0], self.grid
        return tokens.reshape(b, g, g, -1).flip(1).reshape(b, g * g, -1)   # back to mesh order

    def forward(self, image, valid=None):
        tokens = self.proj(self.features(image)) + self.coord(self.uv)
        if valid is not None:
            tokens = torch.where(valid.view(-1, 1, 1), tokens,
                                 self.null.to(tokens.dtype).expand_as(tokens))
        return tokens


def _drop_backbone(module, state_dict, prefix, local_metadata):
    for key in [k for k in state_dict if k.startswith(prefix + "backbone.")]:
        del state_dict[key]


def _restore_backbone(module, state_dict, prefix, local_metadata, strict, missing_keys,
                      unexpected_keys, error_msgs):
    for key, value in module.backbone.state_dict().items():
        state_dict.setdefault(prefix + "backbone." + key, value)
