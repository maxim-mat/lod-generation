import numpy as np
import torch

from src.models.diffusion import CityJSONDiffusionModule


def _tiny_model():
    return CityJSONDiffusionModule(hidden_dim=8, edge_dim=4, global_dim=4,
                                   n_head=2, num_layers=1, T=10, n_max=6)


def test_denormalize_applies_scale_and_zshift():
    m = _tiny_model()
    m.coord_scale = 2.0
    m.z_shift = 5.0
    pos = torch.zeros(3, 3)
    pos[:, 2] = 1.0
    out = m._denormalize_coords(pos)
    assert isinstance(out, np.ndarray)
    # x,y scaled by 2 (still 0); z = 1*2 + 5 = 7
    assert np.allclose(out[:, :2], 0.0)
    assert np.allclose(out[:, 2], 7.0)
