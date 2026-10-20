"""Gated cross-attention from OPT's decoder layers into image tokens.

Flamingo-style (Alayrac et al., arXiv:2204.14198): a cross-attention block per
decoder layer whose output is scaled by tanh(gate) with the gate initialised
at zero, so a model loaded from a text-only (here: image-free) checkpoint is
reproduced exactly at step 0 and the image's influence grows only as far as it
lowers the loss.

Injected by forward pre-hooks on the existing OPT layers rather than by
subclassing them: OPT's modules -- and therefore every v5 state-dict key --
stay untouched, and no Hugging Face internals are copied.
"""
import contextlib
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ImageContext:
    """Image tokens for one batch, plus per-layer cached K/V during decoding."""
    tokens: torch.Tensor                       # [B, P, D]
    kv: dict = field(default=None)            # layer -> (k, v); None = do not cache

    def select(self, order):
        kv = None if self.kv is None else {i: (k[order], v[order]) for i, (k, v) in self.kv.items()}
        return ImageContext(self.tokens[order], kv)

    def repeat(self, n):
        return ImageContext(self.tokens.repeat_interleave(n, dim=0))

    def caching(self):
        return ImageContext(self.tokens, {})


class CrossAttention(nn.Module):
    """Pre-norm multi-head attention from hidden states into image tokens."""

    def __init__(self, dim, n_head):
        super().__init__()
        if dim % n_head:
            raise ValueError(f"dim {dim} is not divisible by n_head {n_head}.")
        self.n_head = n_head
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, h, ctx, kv=None):
        b, l, d = h.shape
        heads = lambda x: x.view(b, -1, self.n_head, d // self.n_head).transpose(1, 2)
        q = heads(self.q(self.norm(h)))
        if kv is None:
            k, v = self.kv(ctx.to(h.dtype)).chunk(2, dim=-1)
            kv = (heads(k), heads(v))
        o = F.scaled_dot_product_attention(q, *kv)
        return self.out(o.transpose(1, 2).reshape(b, l, d)), kv


class GatedCrossAttention(nn.Module):
    """One gated cross-attention per decoder layer, applied before that layer.

    Args:
        layers: OPT's `decoder.layers` ModuleList (hooked, not modified).
        dim, n_head: OPT hidden size and head count.
    """

    def __init__(self, layers, dim, n_head):
        super().__init__()
        self.blocks = nn.ModuleList(CrossAttention(dim, n_head) for _ in layers)
        self.gates = nn.Parameter(torch.zeros(len(layers)))
        self.active = None
        for i, layer in enumerate(layers):
            layer.register_forward_pre_hook(self._hook(i), with_kwargs=True)

    @contextlib.contextmanager
    def attending(self, ctx):
        self.active = ctx
        try:
            yield
        finally:
            self.active = None

    def _hook(self, i):
        def hook(module, args, kwargs):
            ctx = self.active
            if ctx is None:
                return None
            h = args[0] if args else kwargs["hidden_states"]
            cached = None if ctx.kv is None else ctx.kv.get(i)
            out, kv = self.blocks[i](h, ctx.tokens, cached)
            if ctx.kv is not None:
                ctx.kv[i] = kv
            h = h + torch.tanh(self.gates[i]).to(h.dtype) * out
            if args:
                return (h, *args[1:]), kwargs
            return args, {**kwargs, "hidden_states": h}
        return hook
