"""Pairwise-distance featurization: raw (MiDi) vs. sinusoidal / mlp / bessel lift.

Pins that the `dist_embed` flag reaches `lin_dist1`: each lift widens the pair
geometry channel by its feature dim and produces a different, finite denoising
output from the same noisy graph, while 'raw' preserves MiDi's single channel.
"""
import torch

from src.models.embeddings import BesselBasis, SinusoidalDistanceEmbedding
from src.models.regnn import rEGNNTransformer

B, N, K = 2, 5, 16


def _net(dist_embed, equivariance="so2", dist_r_max=None):
    return rEGNNTransformer(num_node_classes=2, num_edge_classes=2, hidden_dim=8,
                            edge_dim=4, global_dim=4, n_head=2, num_layers=1,
                            equivariance=equivariance, dist_embed=dist_embed,
                            dist_embed_dim=K, dist_r_max=dist_r_max)


def _lin_dist1(net):
    return net.tf_layers[0].self_attn.lin_dist1


def _inputs():
    X = torch.zeros(B, N, 2); X[..., 0] = 1.0
    R = torch.randn(B, N, 3)
    Y = torch.zeros(B, N, N, 2); Y[..., 0] = 1.0
    t_norm = torch.rand(B, 1)
    return X, R, Y, t_norm


def test_raw_preserves_midi_pair_width():
    # so2 pair geometry = distance(1) + cosine(1) + dz(1) + horiz_dist(1) = 4.
    assert _lin_dist1(_net("raw", "so2")).in_features == 4
    assert _lin_dist1(_net("raw", "o3")).in_features == 2  # distance + cosine only


def test_lift_widens_pair_channel_by_K():
    for mode in ("sinusoidal", "mlp"):
        assert _lin_dist1(_net(mode, "so2")).in_features == K + 3
        assert _lin_dist1(_net(mode, "o3")).in_features == K + 1
    assert _lin_dist1(_net("bessel", "so2", dist_r_max=3.0)).in_features == K + 3


def test_all_modes_run_finite_and_differ_from_raw():
    torch.manual_seed(0)
    X, R, Y, t = _inputs()
    _, _, X_raw = _net("raw")(X, R, Y, t)
    for mode, kw in (("sinusoidal", {}), ("mlp", {}), ("bessel", {"dist_r_max": 3.0})):
        pos, E, X_out = _net(mode, **kw)(X, R, Y, t)
        for out in (pos, E, X_out):
            assert torch.isfinite(out).all(), mode
        assert not torch.allclose(X_out, X_raw), mode


def test_bessel_requires_r_max():
    try:
        _net("bessel")  # dist_r_max=None
    except ValueError:
        return
    raise AssertionError("bessel without r_max should raise")


def test_bessel_finite_at_zero_distance():
    # Self-distance is 0 -> sin/d hits 0/0; the eps guard must keep it finite.
    out = BesselBasis(dim=K, r_max=3.0)(torch.zeros(B, N, N, 1))
    assert torch.isfinite(out).all()


def test_sinusoidal_odd_dim_rejected():
    try:
        SinusoidalDistanceEmbedding(dim=7)
    except ValueError:
        return
    raise AssertionError("odd embedding dim should raise")


def test_unknown_mode_rejected():
    try:
        _net("gaussian")
    except ValueError:
        return
    raise AssertionError("unknown dist_embed should raise")


if __name__ == "__main__":
    test_raw_preserves_midi_pair_width()
    test_lift_widens_pair_channel_by_K()
    test_all_modes_run_finite_and_differ_from_raw()
    test_bessel_requires_r_max()
    test_bessel_finite_at_zero_distance()
    test_sinusoidal_odd_dim_rejected()
    test_unknown_mode_rejected()
    print("ok")
