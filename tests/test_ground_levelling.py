"""Generated buildings must come out standing on z = 0.

Under so2 the model trains on CoM-centred coordinates, so a sample's ground
plane lands at an arbitrary negative z. Every downstream geometric check --
validity, ground-level statistics, the LOD1/LOD2 comparison -- assumes a
building sits on its footprint, so the export re-applies the dataset's own
convention (`dataset.get_base_center`): the ground centre goes to the origin.

se2 is exempt: there z_shift deliberately restores absolute height, and
levelling would throw that away.
"""
import numpy as np
import pytest
import torch

from src.dataset.dataset import EDGE_VF, GROUND, VERTEX, WALL
from src.models.diffusion import CityJSONDiffusionModule

N = 10


def _model(equivariance="so2", coord_scale=1.0, z_shift=0.0):
    return CityJSONDiffusionModule(
        num_node_classes=5, num_edge_classes=3, hidden_dim=8, edge_dim=4,
        global_dim=4, n_head=2, num_layers=1, T=10, n_max=N,
        coord_scale=coord_scale, equivariance=equivariance, z_shift=z_shift)


def _graph():
    """4 base + 4 top vertices, one GROUND face node, one WALL face node.

    The base sits at z=5 and is centred on (10, 20), so a correct levelling
    moves it to (0, 0, 0).
    """
    coords = np.zeros((N, 3))
    coords[0] = [9, 19, 5]
    coords[1] = [11, 19, 5]
    coords[2] = [11, 21, 5]
    coords[3] = [9, 21, 5]
    coords[4:8] = coords[0:4] + [0, 0, 4]        # top ring at z=9
    coords[8] = [10, 20, 5]                       # ground face node
    coords[9] = [10, 19, 7]                       # wall face node

    labels = np.array([VERTEX] * 8 + [GROUND, WALL])
    edges = np.zeros((N, N), dtype=int)
    for v in range(4):                            # ground face -> base vertices
        edges[8, v] = edges[v, 8] = EDGE_VF
    for v in (0, 1, 4, 5):                        # wall face -> its own ring
        edges[9, v] = edges[v, 9] = EDGE_VF
    return coords, labels, edges


# --- the ground centre itself ---------------------------------------------

def test_ground_centre_uses_vertices_of_ground_faces():
    coords, labels, edges = _graph()
    centre = CityJSONDiffusionModule._ground_centre(coords, labels, edges)
    np.testing.assert_allclose(centre, [10.0, 20.0, 5.0])


def test_ground_centre_ignores_vertices_that_are_not_on_the_ground():
    """The top ring and the wall face must not drag the centre upward."""
    coords, labels, edges = _graph()
    centre = CityJSONDiffusionModule._ground_centre(coords, labels, edges)
    assert centre[2] == pytest.approx(5.0)        # not the 7.0 mean of all z


def test_ground_centre_falls_back_to_the_lowest_vertices():
    """A sample with no GROUND node still has to be placed somewhere sane.

    Mirrors dataset.get_base_center's own 10 cm fallback.
    """
    coords, labels, edges = _graph()
    labels[8] = WALL                              # no ground label anywhere
    centre = CityJSONDiffusionModule._ground_centre(coords, labels, edges)
    np.testing.assert_allclose(centre, [10.0, 20.0, 5.0])


def test_ground_centre_is_none_without_vertices():
    coords, labels, edges = _graph()
    labels[:] = GROUND
    assert CityJSONDiffusionModule._ground_centre(coords, labels, edges) is None


# --- applied on export ----------------------------------------------------

def _captured_coords(monkeypatch, model):
    coords, labels, edges = _graph()
    monkeypatch.setattr(
        model, "sample",
        lambda batch_size=1: (torch.tensor(coords, dtype=torch.float32).unsqueeze(0),
                              torch.tensor(labels).unsqueeze(0),
                              torch.tensor(edges).unsqueeze(0)))
    seen = {}

    def fake(coords_arg, *a, **k):
        seen["coords"] = np.asarray(coords_arg)
        return {"type": "CityJSON"}

    import src.post_process.post_process as pp
    monkeypatch.setattr(pp, "graph_to_cityjson", fake)
    model.generate_cityjson(batch_size=1)
    return seen["coords"]


def test_generated_building_stands_on_the_origin(monkeypatch):
    out = _captured_coords(monkeypatch, _model("so2"))
    base = out[:4]
    assert np.allclose(base[:, 2], 0.0, atol=1e-5)
    assert np.allclose(base[:, :2].mean(axis=0), 0.0, atol=1e-5)
    assert out[4:8, 2] == pytest.approx(4.0, abs=1e-5)   # height preserved


def test_levelling_scales_before_it_shifts(monkeypatch):
    """coord_scale multiplies metres back; the shift must follow, not precede."""
    out = _captured_coords(monkeypatch, _model("so2", coord_scale=2.0))
    assert np.allclose(out[:4, 2], 0.0, atol=1e-5)
    assert out[4:8, 2] == pytest.approx(8.0, abs=1e-5)   # 4 m * 2


def test_se2_keeps_its_absolute_height(monkeypatch):
    """z_shift restores real elevation there, so z must not be re-levelled."""
    out = _captured_coords(monkeypatch, _model("se2", z_shift=3.0))
    assert out[:4, 2] == pytest.approx(8.0, abs=1e-5)    # 5 + z_shift, unlevelled
    assert np.allclose(out[:4, :2].mean(axis=0), 0.0, atol=1e-5)   # xy still centred
