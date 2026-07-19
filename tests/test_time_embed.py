"""Timestep conditioning: scalar (MiDi) vs. sinusoidal Fourier lift.

Pins that the `time_embed` flag actually reaches the network -- the sinusoidal
mode must change the global-feature input width and produce a different, finite
denoising output from the same noisy graph.
"""
import torch

from src.models.regnn import SinusoidalTimeEmbedding, rEGNNTransformer

B, N = 2, 5


def _net(time_embed):
    return rEGNNTransformer(num_node_classes=2, num_edge_classes=2, hidden_dim=8,
                            edge_dim=4, global_dim=4, n_head=2, num_layers=1,
                            time_embed=time_embed)


def _inputs():
    X = torch.zeros(B, N, 2); X[..., 0] = 1.0
    R = torch.randn(B, N, 3)
    Y = torch.zeros(B, N, N, 2); Y[..., 0] = 1.0
    t_norm = torch.rand(B, 1)
    return X, R, Y, t_norm


def test_sinusoidal_embedding_shape_and_range():
    emb = SinusoidalTimeEmbedding(dim=8)
    out = emb(torch.rand(B, 1))
    assert out.shape == (B, 8)
    assert out.abs().max() <= 1.0  # sines/cosines only


def test_scalar_mode_matches_midi_input_width():
    assert _net("scalar").mlp_in_y[0].in_features == 1
    assert _net("sinusoidal").mlp_in_y[0].in_features == 4  # == global_dim


def test_both_modes_run_and_differ():
    torch.manual_seed(0)
    X, R, Y, t = _inputs()
    pos_s, E_s, X_s = _net("scalar")(X, R, Y, t)
    pos_f, E_f, X_f = _net("sinusoidal")(X, R, Y, t)
    for out in (pos_s, E_s, X_s, pos_f, E_f, X_f):
        assert torch.isfinite(out).all()
    assert pos_s.shape == pos_f.shape and X_s.shape == X_f.shape
    # Different parameterisations of y -> different conditioning.
    assert not torch.allclose(X_s, X_f)


def test_odd_dim_rejected():
    try:
        SinusoidalTimeEmbedding(dim=7)
    except ValueError:
        return
    raise AssertionError("odd embedding dim should raise")


if __name__ == "__main__":
    test_sinusoidal_embedding_shape_and_range()
    test_scalar_mode_matches_midi_input_width()
    test_both_modes_run_and_differ()
    test_odd_dim_rejected()
    print("ok")
