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
import torch.nn.functional as F

from src.dataset.ortho import pixel_to_uv
from src.models.ortho_fusion import ImageContext

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
        prefix_grid: average-pool the patch features to this grid before
            `proj` (aligned-prefix spec §3.3); None keeps the native grid.
        vertex_dim: build the per-vertex feature path (spec §3.4) with this
            output width; None (the default) builds nothing.
        vertex_width: channels of its stride-4 pixel CNN.
    """

    def __init__(self, out_dim, name=None, config=None, image_size=256, beta=0.1, n_freq=8,
                 prefix_grid=None, vertex_dim=None, vertex_width=64):
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
        if prefix_grid is not None and (prefix_grid < 1 or self.grid % prefix_grid):
            raise ValueError(f"image_prefix_grid {prefix_grid} does not divide the native "
                             f"{self.grid}x{self.grid} patch grid.")
        self.prefix_grid = None if prefix_grid in (None, self.grid) else prefix_grid
        self.beta = beta

        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("uv", _uv_features(self.prefix_grid or self.grid, beta, n_freq),
                             persistent=False)
        self.proj = nn.Sequential(nn.Linear(cfg.hidden_size, out_dim), nn.GELU(),
                                  nn.Linear(out_dim, out_dim))
        self.coord = nn.Linear(4 * n_freq, out_dim)
        self.null = nn.Parameter(torch.randn(out_dim) * 0.02)
        self.vertex_dim = vertex_dim
        if vertex_dim is not None:
            # Fine map for the per-vertex lookup: stride 4, i.e. 1.2*L/64 per
            # cell -- ~0.375 m on a 20 m building, against DINOv3's 1.2*L/16.
            w = vertex_width
            self.pixel_cnn = nn.Sequential(
                nn.Conv2d(3, w, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(w, w, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(w, w, 3, stride=1, padding=1))
            width = w + cfg.hidden_size
            self.vertex_proj = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, vertex_dim),
                                             nn.GELU(), nn.Linear(vertex_dim, vertex_dim))
            # Zero at step 0, so turning the path on starts at the model it extends.
            nn.init.zeros_(self.vertex_proj[-1].weight)
            nn.init.zeros_(self.vertex_proj[-1].bias)

        self.register_state_dict_post_hook(_drop_backbone)
        self.register_load_state_dict_pre_hook(_restore_backbone)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()          # frozen means no dropout either
        return self

    def _normalise(self, image):
        """Mesh-order uint8 crops -> normalised float, still in mesh order."""
        if image.shape[-1] != self.image_size or image.shape[-2] != self.image_size:
            raise ValueError(f"Expected {self.image_size}px crops (image_size), got "
                             f"{tuple(image.shape[-2:])}.")
        return (image.float().div(255.0) - self.mean) / self.std

    def features(self, image):
        """Frozen backbone patch features ``[B, G*G, C]`` in mesh order."""
        x = self._normalise(image).flip(-2)                    # mesh order -> north-up
        with torch.no_grad():
            tokens = self.backbone(pixel_values=x).last_hidden_state[:, self.n_skip:]
        b, g = tokens.shape[0], self.grid
        return tokens.reshape(b, g, g, -1).flip(1).reshape(b, g * g, -1)   # back to mesh order

    def encode(self, image, valid=None, use_null=True):
        """Crops -> an `ImageContext` with tokens, `valid` and (if built) vertex maps.

        Args:
            valid: [B] bool; None means every row has an image.
            use_null: replace invalid rows' tokens by the null token. False when
                the network masks those tokens out instead (spec §3.2).
        """
        feats = self.features(image)                                   # [B, G*G, C]
        b, g = feats.shape[0], self.grid
        grid = feats.transpose(1, 2).reshape(b, -1, g, g)              # [B, C, G, G], mesh order
        pooled = feats
        if self.prefix_grid is not None:
            pooled = F.avg_pool2d(grid, g // self.prefix_grid).flatten(2).transpose(1, 2)
        tokens = self.proj(pooled) + self.coord(self.uv)
        if valid is None:
            valid = torch.ones(b, dtype=torch.bool, device=image.device)
        if use_null:
            tokens = torch.where(valid.view(-1, 1, 1), tokens,
                                 self.null.to(tokens.dtype).expand_as(tokens))
        maps = None
        if self.vertex_dim is not None:
            # No flip: the CNN learns from scratch, so it reads mesh order directly.
            maps = (self.pixel_cnn(self._normalise(image)), grid)
        return ImageContext(tokens, valid=valid, maps=maps)

    def forward(self, image, valid=None):
        return self.encode(image, valid).tokens

    def vertex_features(self, maps, bx, by, num_bins):
        """``[B, L, vertex_dim]`` image features at each vertex's ground cell.

        Bins dequantize to (u, v) as `mesh_dataset.dequantize` does, which is
        the unit box only because x/y margins are 0 (`check_vertex_margins`).
        """
        u = bx.float() / (num_bins - 1) - 0.5
        v = by.float() / (num_bins - 1) - 0.5
        feats = torch.cat([sample_map(m, u, v, self.beta) for m in maps], dim=-1)
        return self.vertex_proj(feats)


def sample_map(fmap, u, v, beta):
    """Bilinear ``[B, L, C]`` read of a mesh-order map ``[B, C, H, W]`` at (u, v).

    The map spans [-0.5-beta, 0.5+beta] on both axes like the crop; columns run
    along +x (u) and rows along +y (v), so grid x = u, grid y = v.
    """
    grid = torch.stack([u, v], dim=-1) / (0.5 + beta)                  # [B, L, 2]
    out = F.grid_sample(fmap, grid[:, None].to(fmap.dtype), mode="bilinear",
                        align_corners=False)                           # [B, C, 1, L]
    return out[:, :, 0].transpose(1, 2)


def _drop_backbone(module, state_dict, prefix, local_metadata):
    for key in [k for k in state_dict if k.startswith(prefix + "backbone.")]:
        del state_dict[key]


def _restore_backbone(module, state_dict, prefix, local_metadata, strict, missing_keys,
                      unexpected_keys, error_msgs):
    for key, value in module.backbone.state_dict().items():
        state_dict.setdefault(prefix + "backbone." + key, value)
