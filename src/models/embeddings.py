"""Fixed sinusoidal position embedding shared by both mesh-transformer backbones."""
import math

import torch
import torch.nn as nn


class SinusoidalPositionEmbedding(nn.Module):
    """Fixed sinusoidal absolute positions (Vaswani et al. 2017,
    arXiv:1706.03762 section 3.5).

    Unlike a learned `nn.Embedding(max_seq_len, d)` this has no row count, so it
    is defined at every position and the sequence-length ceiling disappears
    rather than moving. That is the reason it exists here: both mesh backbones
    were capped by a table sized from a default -- OPT's 2048 rides in from a
    hub config whose weights are not even loaded.

    No parameters and a non-persistent buffer, so a checkpoint saved with this
    option carries no position rows at all and can be reloaded at any length.

    Args:
        dim (int): output width; must be even.
        base (float): geometric wavelength range, 10000 in the paper.
    """

    def __init__(self, dim, base=10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"position embedding dim must be even, got {dim}.")
        self.dim = dim
        # Half the width is sin, half cos, which is *not* the paper's
        # interleaving. Nothing downstream
        # reads individual channels, so the permutation is free.
        self.register_buffer(
            "inv_freq",
            torch.exp(torch.arange(0, dim, 2, dtype=torch.float)
                      * (-math.log(base) / dim)),
            persistent=False,
        )

    def forward(self, pos):
        """`pos` of any shape -> that shape with a trailing `dim` axis."""
        args = pos.unsqueeze(-1).float() * self.inv_freq
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
