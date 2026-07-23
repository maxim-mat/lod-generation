"""Symmetry regressions for the so2 / se2 modes.

Yaw rotation about z must rotate the position output and leave X/E logits
unchanged. Translation is quotiented by the *noise model*, not the network:
the network's input contract is an already-projected position tensor
(PositionsMLP consumes raw norms before re-centring), so the CoM tests live
on GraphNoiseModel. Under se2 the projection must not touch z.
"""
import math

import pytest
import torch

from src.models.layers import remove_mean_with_mask
from src.models.noise import GraphNoiseModel
from src.models.regnn import rEGNNTransformer

B, N = 2, 6


def _rot_z(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _inputs():
    torch.manual_seed(0)
    X = torch.nn.functional.one_hot(torch.randint(0, 5, (B, N)), 5).float()
    E = torch.nn.functional.one_hot(torch.randint(0, 3, (B, N, N)), 3).float()
    E = 0.5 * (E + E.transpose(1, 2))
    R = torch.randn(B, N, 3)
    t = torch.rand(B, 1)
    return X, E, R, t


def _net(equivariance):
    torch.manual_seed(1)
    return rEGNNTransformer(num_node_classes=5, num_edge_classes=3, hidden_dim=8,
                            edge_dim=4, global_dim=4, n_head=2, num_layers=2,
                            equivariance=equivariance).eval()


@pytest.mark.parametrize("equivariance", ["so2", "se2"])
def test_yaw_rotation_equivariance(equivariance):
    net = _net(equivariance)
    X, E, R, t = _inputs()
    Q = _rot_z(0.7)

    with torch.no_grad():
        pos1, E1, X1 = net(X, R, E, t)
        pos2, E2, X2 = net(X, R @ Q.T, E, t)

    assert torch.allclose(pos2, pos1 @ Q.T, atol=1e-4)
    assert torch.allclose(X2, X1, atol=1e-4)
    assert torch.allclose(E2, E1, atol=1e-4)


def _noise_model(xy_only_com):
    return GraphNoiseModel(
        T=10,
        x_marginals=torch.ones(5) / 5,
        e_marginals=torch.ones(3) / 3,
        xy_only_com=xy_only_com,
    )


def test_noise_keeps_positions_on_full_com_subspace():
    """Legacy modes: noised positions stay mean-free in all three axes."""
    torch.manual_seed(0)
    model = _noise_model(xy_only_com=False)
    mask = torch.ones(B, N)
    X = torch.nn.functional.one_hot(torch.randint(0, 5, (B, N)), 5).float()
    E = torch.nn.functional.one_hot(torch.randint(0, 3, (B, N, N)), 3).float()
    pos = remove_mean_with_mask(torch.randn(B, N, 3), mask)

    z = model.apply_noise(pos, X, E, mask, t_int=torch.full((B, 1), 5))

    assert torch.allclose(z["pos_t"].mean(dim=1), torch.zeros(B, 3), atol=1e-5)


def test_se2_noise_removes_xy_mean_only():
    """se2: xy stays mean-free; the z mean of the data survives noising."""
    torch.manual_seed(0)
    model = _noise_model(xy_only_com=True)
    mask = torch.ones(B, N)
    X = torch.nn.functional.one_hot(torch.randint(0, 5, (B, N)), 5).float()
    E = torch.nn.functional.one_hot(torch.randint(0, 3, (B, N, N)), 3).float()
    pos = remove_mean_with_mask(torch.randn(B, N, 3), mask, xy_only=True)
    pos[..., 2] += 10.0                                   # a real z offset

    t_int = torch.full((B, 1), 1)                         # low noise: signal dominates
    z = model.apply_noise(pos, X, E, mask, t_int=t_int)

    assert torch.allclose(z["pos_t"][..., :2].mean(dim=1), torch.zeros(B, 2), atol=1e-4)
    assert (z["pos_t"][..., 2].mean(dim=1) > 5.0).all()   # z mean not projected away

    # the reverse-chain seed is also only xy-projected
    seed_pos, _, _ = model.sample_limit_dist(B, N, pos.device)
    assert torch.allclose(seed_pos[..., :2].mean(dim=1), torch.zeros(B, 2), atol=1e-5)


def test_se2_network_uses_absolute_z():
    """z translation is NOT a symmetry of se2: output must change."""
    net = _net("se2")
    X, E, R, t = _inputs()

    with torch.no_grad():
        pos1, _, X1 = net(X, R, E, t)
        pos2, _, X2 = net(X, R + torch.tensor([0.0, 0.0, 4.0]), E, t)

    assert not torch.allclose(X2, X1, atol=1e-4)


def test_remove_mean_xy_only_leaves_z():
    torch.manual_seed(0)
    x = torch.randn(B, N, 3)
    mask = torch.ones(B, N)

    out = remove_mean_with_mask(x, mask, xy_only=True)

    assert torch.allclose(out[..., :2].mean(dim=1), torch.zeros(B, 2), atol=1e-6)
    assert torch.allclose(out[..., 2], x[..., 2], atol=1e-6)
