"""Gated cross-attention from OPT's decoder layers into image tokens.

Flamingo-style (Alayrac et al., arXiv:2204.14198): a cross-attention block per
decoder layer whose output is scaled by tanh(gate) with the gate initialised
at zero, so a model loaded from a text-only (here: image-free) checkpoint is
reproduced exactly at step 0 and the image's influence grows only as far as it
lowers the loss.

Optionally each cross-attention is followed by a gated FFN, completing
Flamingo's GATED XATTN-DENSE block.

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
    """Image tokens for one batch, plus per-layer cached K/V during decoding.

    `valid` ([B] bool) says which rows really have an image; `maps` is the
    ``(fine, coarse)`` pair of mesh-order feature maps the per-vertex features
    sample (aligned-prefix spec §3.4), None unless that path is on.
    """
    tokens: torch.Tensor                       # [B, P, D]
    kv: dict = field(default=None)            # layer -> (k, v); None = do not cache
    valid: torch.Tensor = None                 # [B] bool
    maps: tuple = None                         # (fine [B,Cf,H,W], coarse [B,C,G,G])

    def _rows(self, pick):
        return dict(valid=None if self.valid is None else pick(self.valid),
                    maps=None if self.maps is None else tuple(pick(m) for m in self.maps))

    def select(self, order):
        kv = None if self.kv is None else {i: (k[order], v[order]) for i, (k, v) in self.kv.items()}
        return ImageContext(self.tokens[order], kv, **self._rows(lambda t: t[order]))

    def repeat(self, n):
        pick = lambda t: t.repeat_interleave(n, dim=0)
        return ImageContext(pick(self.tokens), **self._rows(pick))

    def caching(self):
        return ImageContext(self.tokens, {}, self.valid, self.maps)


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


class FeedForward(nn.Module):
    """Pre-norm two-layer GELU MLP, the dense half of Flamingo's GATED XATTN-DENSE."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))

    def forward(self, h):
        return self.net(h)


class GatedCrossAttention(nn.Module):
    """One gated cross-attention per decoder layer, applied before that layer.

    Args:
        layers: OPT's `decoder.layers` ModuleList (hooked, not modified).
        dim, n_head: OPT hidden size and head count.
        ffn_dim: if given, follow each cross-attention with a gated FFN of this
            hidden width, as Flamingo's GATED XATTN-DENSE block does
            (arXiv:2204.14198, section 2.2). Its own tanh gate starts at zero
            too, so step 0 is still the image-free model. None (the default)
            builds nothing and keeps the old state-dict keys.
    """

    def __init__(self, layers, dim, n_head, ffn_dim=None):
        super().__init__()
        self.blocks = nn.ModuleList(CrossAttention(dim, n_head) for _ in layers)
        self.gates = nn.Parameter(torch.zeros(len(layers)))
        self.ffns = None
        if ffn_dim is not None:
            self.ffns = nn.ModuleList(FeedForward(dim, ffn_dim) for _ in layers)
            self.ffn_gates = nn.Parameter(torch.zeros(len(layers)))
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
            if self.ffns is not None:
                h = h + torch.tanh(self.ffn_gates[i]).to(h.dtype) * self.ffns[i](h)
            if args:
                return (h, *args[1:]), kwargs
            return args, {**kwargs, "hidden_states": h}
        return hook
