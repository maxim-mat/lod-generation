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
import math

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


# --- end-to-end regression -------------------------------------------------
# The conditions that killed gptwstyp: the real model size, a weight state drifted
# away from init, and a batch holding the largest building in the dataset.

N_MAX, BATCH, N_ACTIVE = 100, 2, 20
WEIGHT_DRIFT = 4.0
# Largest trainable building in The Hague LOD2 spans ~154 m, i.e. a coordinate
# std of ~17 m. Raw metres, so this exercises the network, not the coord scaling.
LARGEST_BUILDING_COORD_STD = 17.0


def _drifted_model(weight_scale):
    from src.models.diffusion import CityJSONDiffusionModule

    torch.manual_seed(0)
    model = CityJSONDiffusionModule(
        n_max=N_MAX, hidden_dim=64, edge_dim=32, global_dim=32, n_head=8,
        num_layers=4, T=500, dropout=0.0,
        x_marginals=torch.tensor([0.2001, 0.7999]),
        e_marginals=torch.tensor([0.9936, 0.0064]),
    )
    with torch.no_grad():
        for module in model.network.modules():
            if isinstance(module, nn.Linear):
                module.weight.mul_(weight_scale)
                if module.bias is not None:
                    module.bias.mul_(weight_scale)
    return model


def _large_building_batch(coord_std):
    torch.manual_seed(1)
    x = torch.zeros(BATCH, N_MAX, 3)
    node_mask = torch.zeros(BATCH, N_MAX)
    node_categories = torch.zeros(BATCH, N_MAX, 2)
    adjacency = torch.zeros(BATCH, N_MAX, N_MAX, 1)

    for b in range(BATCH):
        x[b, :N_ACTIVE] = torch.randn(N_ACTIVE, 3) * coord_std
        node_mask[b, :N_ACTIVE] = 1.0
        node_categories[b, :N_ACTIVE, 0] = 1.0
        node_categories[b, N_ACTIVE:, 1] = 1.0
        ring = torch.arange(N_ACTIVE)
        adjacency[b, ring, (ring + 1) % N_ACTIVE, 0] = 1.0
        adjacency[b, (ring + 1) % N_ACTIVE, ring, 0] = 1.0

    return {"x": x, "node_categories": node_categories, "y": adjacency, "node_mask": node_mask}


def test_forward_and_backward_stay_finite_for_a_large_building_under_weight_drift():
    """The exact regression: fp32 overflow in the global-feature branch.

    Before the pooled-std fix this overflowed the forward pass, `LayerNorm(inf)`
    returned NaN, and AdamW propagated NaN into every parameter.
    """
    model = _drifted_model(WEIGHT_DRIFT)
    batch = _large_building_batch(LARGEST_BUILDING_COORD_STD)
    t_int = torch.full((BATCH, 1), 250, dtype=torch.long)

    total, coord_loss, node_loss, edge_loss = model._shared_step(batch, t_int=t_int)[:4]

    assert torch.isfinite(total), (
        f"forward overflowed: total={total.item()} coord={coord_loss.item()} "
        f"node={node_loss.item()} edge={edge_loss.item()}"
    )

    total.backward()
    nonfinite = [
        name for name, p in model.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    assert not nonfinite, f"non-finite gradients in: {nonfinite[:5]}"


def test_global_feature_branch_keeps_headroom_before_fp32_overflows():
    """`y_out` is where the overflow surfaced, so it must stay far from fp32 max.

    Under this same stress the upstream variance reaches 3.8e27 in layer 0 and
    NaN thereafter; the pooled std peaks at 3.7e10. Asserting headroom rather
    than a tuned constant keeps the test meaningful if the model is resized.

    Note the peak is still large: that is raw metres talking, and it is what
    `coord_scale` is for. This test deliberately leaves coordinates unscaled so
    it exercises the network in isolation.
    """
    model = _drifted_model(WEIGHT_DRIFT)
    batch = _large_building_batch(LARGEST_BUILDING_COORD_STD)

    peaks = {}
    for i, layer in enumerate(model.network.tf_layers):
        layer.self_attn.y_out.register_forward_hook(
            lambda _m, _i, out, i=i: peaks.__setitem__(i, out.abs().max().item())
        )

    with torch.no_grad():
        model._shared_step(batch, t_int=torch.full((BATCH, 1), 250, dtype=torch.long))

    assert all(math.isfinite(v) for v in peaks.values()), f"y_out overflowed: {peaks}"

    headroom = torch.finfo(torch.float32).max / max(peaks.values())
    assert headroom > 1e20, (
        f"global-feature branch is amplifying: peaks={peaks}, "
        f"only {headroom:.1e}x headroom before fp32 overflows"
    )
