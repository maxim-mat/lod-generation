"""_centre_positions: real-node centering that keeps the all-ones subspace exact.

Targets must have zero mean over ALL N slots (the noise model and network
project positions with an all-ones mask), while OFF slots stay exactly zero.
"""
import torch

from src.models.diffusion import CityJSONDiffusionModule

B, N = 2, 6


def _cats(labels, num_classes):
    one_hot = torch.nn.functional.one_hot(torch.tensor(labels), num_classes).float()
    return one_hot.unsqueeze(0).expand(B, -1, -1).clone()


def test_centre_positions_translates_faces_and_zeroes_off():
    torch.manual_seed(0)
    R0 = torch.randn(B, N, 3)
    # 3 vertices, 2 faces, 1 off (last class)
    cats = _cats([0, 0, 0, 1, 3, 4], 5)

    out = CityJSONDiffusionModule._centre_positions(R0, cats)

    real = out[:, :5]
    assert torch.allclose(real.mean(dim=1), torch.zeros(B, 3), atol=1e-6)
    assert torch.all(out[:, 5] == 0)                       # OFF pinned to zero
    assert torch.allclose(out.mean(dim=1), torch.zeros(B, 3), atol=1e-6)  # all-N mean 0
    # faces are translated, not zeroed: relative geometry preserved
    rel_before = R0[:, 3] - R0[:, 0]
    rel_after = out[:, 3] - out[:, 0]
    assert torch.allclose(rel_before, rel_after, atol=1e-6)


def test_centre_positions_xy_only_keeps_absolute_z_minus_shift():
    torch.manual_seed(1)
    R0 = torch.randn(B, N, 3)
    cats = _cats([0, 0, 0, 1, 3, 4], 5)

    out = CityJSONDiffusionModule._centre_positions(R0, cats, xy_only=True, z_shift=5.0)

    real = out[:, :5]
    assert torch.allclose(real[..., :2].mean(dim=1), torch.zeros(B, 2), atol=1e-6)
    assert torch.allclose(real[..., 2], R0[:, :5, 2] - 5.0, atol=1e-6)  # z untouched by centering
    assert torch.all(out[:, 5] == 0)


def test_centre_positions_matches_legacy_for_two_class_batches():
    """With 2-class categories (Active/Virtual) and zero virtual coords, the
    real mask equals the old node_mask and results are unchanged."""
    torch.manual_seed(2)
    R0 = torch.zeros(B, N, 3)
    R0[:, :4] = torch.randn(B, 4, 3)
    cats = _cats([0, 0, 0, 0, 1, 1], 2)

    out = CityJSONDiffusionModule._centre_positions(R0, cats)

    mean = R0[:, :4].mean(dim=1, keepdim=True)
    assert torch.allclose(out[:, :4], R0[:, :4] - mean, atol=1e-6)
    assert torch.all(out[:, 4:] == 0)
