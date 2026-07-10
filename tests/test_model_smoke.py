"""Smoke checks: the model constructs and a forward pass has the right shapes.

Both assertions guard a bug that used to make every call raise:
  * CityJSONDiffusionModule passed `transition=` to GraphNoiseModel, which takes
    `transition_x` / `transition_e`.
  * NodeEdgeBlock sized its geometry Linears for `equivariance` but forward()
    always built the "o3" feature count.
"""
import pytest
import torch

from src.models.diffusion import NUM_EDGE_CLASSES, CityJSONDiffusionModule
from src.models.regnn import rEGNNTransformer

B, N = 2, 6


@pytest.mark.parametrize("equivariance", ["so2", "o3"])
def test_network_forward_shapes(equivariance):
    net = rEGNNTransformer(hidden_dim=8, edge_dim=4, global_dim=4, n_head=2,
                           num_layers=1, equivariance=equivariance)

    X = torch.nn.functional.one_hot(torch.randint(0, 2, (B, N)), 2).float()
    R = torch.randn(B, N, 3)
    E = torch.nn.functional.one_hot(torch.randint(0, 2, (B, N, N)), NUM_EDGE_CLASSES).float()
    E = 0.5 * (E + E.transpose(1, 2))
    t = torch.rand(B, 1)

    R_pred, E_pred, X_pred = net(X, R, E, t)

    assert R_pred.shape == (B, N, 3)
    assert E_pred.shape == (B, N, N, NUM_EDGE_CLASSES)
    assert X_pred.shape == (B, N, 2)
    assert torch.allclose(E_pred, E_pred.transpose(1, 2), atol=1e-5)


def test_diffusion_module_shared_step():
    model = CityJSONDiffusionModule(hidden_dim=8, edge_dim=4, global_dim=4,
                                    n_head=2, num_layers=1, T=10, n_max=N)
    batch = {
        "x": torch.randn(B, N, 3),
        "node_categories": torch.nn.functional.one_hot(torch.randint(0, 2, (B, N)), 2).float(),
        "y": torch.randint(0, NUM_EDGE_CLASSES, (B, N, N, 1)),
        "node_mask": torch.ones(B, N),
    }
    total = model._shared_step(batch)[0]
    assert total.isfinite()
