"""The coordinate loss is normalised over real slots, with a weaker Off anchor.

Off is ~74% of n_max on the real dataset and `_centre_positions` pins its target
to exactly 0, so an all-slot mean both dilutes the real-node gradient and makes
the loss depend on how much padding a batch happens to carry. These tests pin the
two properties that fix rests on.

The denoiser is replaced by an analytic stub so the losses have closed forms; a
randomly-initialised network would only allow a comparison, not an assertion.
"""
import torch

from src.models.diffusion import CityJSONDiffusionModule
from src.dataset.dataset import VERTEX, GROUND, ROOF, WALL, OFF


class _ConstPosNet(torch.nn.Module):
    """Predicts a constant position everywhere; echoes the noised X and E.

    Makes the coordinate loss exact: with a constant prediction c, the Off term is
    mean((c - 0)^2) = c^2, and with c = 0 the real term is the mean squared
    magnitude of the real targets -- a quantity independent of the padding width.
    """

    def __init__(self, c=0.0):
        super().__init__()
        self.c = c

    def forward(self, X_t, R_t, Y_t, t_norm, node_mask=None):
        return torch.full_like(R_t, self.c), Y_t, X_t


def _model(n_max, off_anchor_weight=0.1, c=0.0):
    m = CityJSONDiffusionModule(num_node_classes=5, num_edge_classes=3, n_max=n_max,
                                hidden_dim=8, edge_dim=4, global_dim=4, n_head=2,
                                num_layers=1, T=10, coord_scale=1.0,
                                off_anchor_weight=off_anchor_weight)
    m.network = _ConstPosNet(c)
    return m


def _batch(n_max, seed=0):
    """One fixed building (8 vertices + 6 faces) padded out to `n_max` with Off."""
    g = torch.Generator().manual_seed(seed)
    n_real = 14
    labels = torch.tensor([VERTEX] * 8 + [GROUND, ROOF] + [WALL] * 4
                          + [OFF] * (n_max - n_real))
    x = torch.zeros(1, n_max, 3)
    x[0, :n_real] = torch.randn(n_real, 3, generator=g) * 5
    return {"x": x,
            "node_categories": torch.nn.functional.one_hot(labels, 5).float().unsqueeze(0),
            "y": torch.zeros(1, n_max, n_max, 1),
            "node_mask": (labels == VERTEX).float().unsqueeze(0)}


def test_coord_loss_is_invariant_to_padding_width():
    """Same building, 3x the Off padding -> same coordinate loss.

    Under the old all-slot mean this ratio was ~n_max_a / n_max_b, so the amount of
    padding silently rescaled the coordinate gradient.
    """
    torch.manual_seed(0)
    losses = []
    for n_max in (20, 60):
        m = _model(n_max)
        _, coord_loss, *_ = m._shared_step(_batch(n_max))
        losses.append(coord_loss.item())

    assert losses[0] > 0, "degenerate: stub produced a zero coordinate loss"
    assert abs(losses[0] - losses[1]) / losses[0] < 1e-5, losses

    # And confirm the old formula really would have drifted, so this test has teeth.
    old = [torch.nn.functional.mse_loss(
        torch.zeros(1, n, 3),
        m_._prepare(_batch(n))[0]).item()
        for n, m_ in ((20, _model(20)), (60, _model(60)))]
    assert old[0] / old[1] > 2.5, old


def test_off_anchor_is_applied_and_weighted():
    """A non-zero prediction on Off slots costs exactly lambda_pos * w * c^2."""
    torch.manual_seed(0)
    c, w, n_max = 0.3, 0.1, 20
    m = _model(n_max, off_anchor_weight=w, c=c)
    total, coord_loss, node_loss, edge_loss, *_ = m._shared_step(_batch(n_max))

    anchor = (total - (m.lambda_pos * coord_loss + m.lambda_x * node_loss
                       + m.lambda_e * edge_loss)).item()
    assert abs(anchor - m.lambda_pos * w * c ** 2) < 1e-5, anchor

    # w=0 removes it entirely; the real-node term is untouched either way.
    m0 = _model(n_max, off_anchor_weight=0.0, c=c)
    total0, coord0, node0, edge0, *_ = m0._shared_step(_batch(n_max))
    anchor0 = (total0 - (m0.lambda_pos * coord0 + m0.lambda_x * node0
                         + m0.lambda_e * edge0)).item()
    assert abs(anchor0) < 1e-6, anchor0
    assert abs(coord0.item() - coord_loss.item()) < 1e-5
