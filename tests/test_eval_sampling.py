import numpy as np
import torch

from src.dataset.dataset import VERTEX, WALL, EDGE_VF, EDGE_VV
from src.eval.sampling import draw_samples
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


def test_draw_samples_counts_drops(monkeypatch):
    m = _tiny_model()

    # one well-formed triangle+face graph, one all-OFF (dropped) graph
    N = 6
    good_pos = torch.zeros(1, N, 3)
    good_pos[0, :3] = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.float)
    good_lab = torch.full((1, N), 4)  # OFF
    good_lab[0, :3] = VERTEX
    good_lab[0, 3] = WALL
    good_edge = torch.zeros(1, N, N, dtype=torch.long)
    for v in range(3):
        good_edge[0, 3, v] = good_edge[0, v, 3] = EDGE_VF
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        good_edge[0, a, b] = good_edge[0, b, a] = EDGE_VV

    bad_pos = torch.zeros(1, N, 3)
    bad_lab = torch.full((1, N), 4)
    bad_edge = torch.zeros(1, N, N, dtype=torch.long)

    calls = iter([(good_pos, good_lab, good_edge), (bad_pos, bad_lab, bad_edge)])
    monkeypatch.setattr(m, "sample", lambda batch_size=1: next(calls))

    records, stats = draw_samples(m, num_batches=2, batch_size=1)
    assert stats == {"attempted": 2, "dropped": 1}
    assert len(records) == 1
    assert records[0]["cityjson"]["type"] == "CityJSON"
    assert records[0]["coords"].shape[1] == 3
    assert records[0]["coords"].shape[0] == 6
