"""Regression tests for the fp32 overflow that killed the `debug` run.

Symptom (wandb maxim-lod/lod-generation/gptwstyp): train loss was 1.21 at global
step 12,499 and NaN by 12,549, with no divergence ramp, after which every metric
stayed NaN for the rest of the run.

Root cause: `Xtoy` / `Etoy` / `EtoX` pool their input into four statistics --
mean, min, max and a *variance* misnamed `std` -- and concatenate them into a
single Linear. Mean/min/max are homogeneous of degree 1 in the activations; the
variance is degree 2. That channel feeds `new_y`, which then FiLM-multiplies X
and E in the next layer, so the global-feature branch grows super-polynomially
and overflows fp32 on a batch containing a large building. `LayerNorm(inf)` is
NaN, and AdamW then writes NaN into every parameter.

The upstream MiDi implementation (cvignac/MiDi, midi/models/layers.py) has the
same degree-2 term. It survives there because molecular coordinates are Angstrom
(scale ~1-2, narrow spread across molecules) and padding is masked out of the
pooled statistics.
"""
import pytest
import torch
import torch.nn as nn

from src.models.layers import EtoX, Etoy, Xtoy

B, N, D = 2, 5, 4


def _select_spread_channel(module, d):
    """Rewire `module.lin` so its first output is exactly the pooled spread feature.

    The four pooled statistics are concatenated as (mean, min, max, spread), each
    `d` wide, so the spread block starts at index `3 * d`.
    """
    with torch.no_grad():
        module.lin.weight.zero_()
        module.lin.weight[0, 3 * d] = 1.0
        module.lin.bias.zero_()


def _xtoy_spread(X):
    mod = Xtoy(D, D)
    _select_spread_channel(mod, D)
    return mod(X, torch.ones(B, N, 1))[:, 0]


def _etoy_spread(E):
    mod = Etoy(D, D)
    _select_spread_channel(mod, D)
    return mod(E, torch.ones(B, N, 1, 1), torch.ones(B, 1, N, 1))[:, 0]


def _etox_spread(E):
    mod = EtoX(D, D)
    _select_spread_channel(mod, D)
    return mod(E, torch.ones(B, 1, N, 1))[:, 0, 0]


@pytest.mark.parametrize(
    "spread_fn, shape",
    [
        (_xtoy_spread, (B, N, D)),
        (_etoy_spread, (B, N, N, D)),
        (_etox_spread, (B, N, N, D)),
    ],
    ids=["Xtoy", "Etoy", "EtoX"],
)
def test_pooled_spread_is_homogeneous_degree_one(spread_fn, shape):
    """Scaling the input by c must scale the spread feature by c, not by c^2.

    A variance scales quadratically, which is what let the global-feature branch
    outrun its mean/min/max siblings and overflow fp32.
    """
    torch.manual_seed(0)
    x = torch.randn(*shape)
    c = 10.0

    base = spread_fn(x)
    scaled = spread_fn(c * x)

    assert torch.allclose(scaled, c * base, rtol=1e-4, atol=1e-5), (
        f"spread feature scaled by {(scaled / base).mean():.1f}x for a {c}x input; "
        f"expected {c}x (degree 1). A variance would give {c ** 2}x."
    )


def test_pooled_spread_gradient_is_finite_at_zero_variance():
    """sqrt has an infinite gradient at 0, so a constant input must not produce NaN."""
    x = torch.ones(B, N, D, requires_grad=True)  # variance is exactly zero
    _xtoy_spread(x).sum().backward()
    assert torch.isfinite(x.grad).all(), "non-finite gradient through a zero-variance input"
