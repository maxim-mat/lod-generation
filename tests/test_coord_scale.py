"""Coordinate standardisation: training targets are divided, generation multiplies back.

Building coordinates are raw metres (pooled std ~4.3 m on The Hague LOD2, with a
per-building spread of ~18x). The position diffusion adds unit-variance Gaussian
noise and seeds the reverse chain from N(0, I), so the schedule only compresses
into the last ~20% of timesteps: at t = T/2 the SNR is 16.3 instead of 2.3, and
the low-SNR regime where sampling actually starts is barely trained.

The scale is a single dataset-wide scalar, computed on the train split only:
  * per-building would erase building size and is not invertible at sampling time
  * per-axis would break the yaw-equivariance `so2` mode relies on (measured
    z/xy std ratio is 0.77, so it buys nothing anyway)
"""
import torch

from src.dataset.datamodule import CityJSONDataModule
from src.models.diffusion import CityJSONDiffusionModule

B, N = 2, 6


def _batch(scale=1.0):
    x = torch.zeros(B, N, 3)
    node_mask = torch.zeros(B, N)
    node_categories = torch.zeros(B, N, 2)
    for b in range(B):
        x[b, :4] = torch.randn(4, 3) * scale
        node_mask[b, :4] = 1.0
        node_categories[b, :4, 0] = 1.0
        node_categories[b, 4:, 1] = 1.0
    return {
        "x": x,
        "node_categories": node_categories,
        "y": torch.randint(0, 2, (B, N, N, 1)),
        "node_mask": node_mask,
    }


def _model(coord_scale):
    # 2-class fixtures: this file pins coordinate scaling, not the Levi classes.
    return CityJSONDiffusionModule(num_node_classes=2, num_edge_classes=2,
                                   hidden_dim=8, edge_dim=4, global_dim=4, n_head=2,
                                   num_layers=1, T=10, n_max=N, coord_scale=coord_scale)


def test_compute_coord_scale_is_pooled_std_of_centred_active_coords():
    """Mirrors compute_marginals: a train-split-only statistic, no val/test leakage."""
    a = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    b = torch.tensor([[1.0, 1.0, 1.0], [-1.0, 3.0, 5.0]])

    items = []
    for coords in (a, b):
        n = coords.shape[0]
        x = torch.zeros(N, 3)
        x[:n] = coords
        mask = torch.zeros(N)
        mask[:n] = 1.0
        items.append({"x": x, "node_mask": mask})

    dm = CityJSONDataModule(dataset_dir="unused", lods=1)
    dm.train_dataset = items

    pooled = torch.cat([c - c.mean(dim=0, keepdim=True) for c in (a, b)]).flatten()
    expected = float(pooled.std(unbiased=False))

    assert abs(dm.compute_coord_scale() - expected) < 1e-5


def test_compute_coord_scale_ignores_virtual_padding():
    """Virtual nodes sit at the origin; counting them would shrink the scale."""
    coords = torch.tensor([[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0]])
    x = torch.zeros(N, 3)
    x[:2] = coords
    mask = torch.zeros(N)
    mask[:2] = 1.0

    dm = CityJSONDataModule(dataset_dir="unused", lods=1)
    dm.train_dataset = [{"x": x, "node_mask": mask}]

    expected = float((coords - coords.mean(0, keepdim=True)).flatten().std(unbiased=False))
    assert abs(dm.compute_coord_scale() - expected) < 1e-5


def test_prepare_divides_targets_by_coord_scale():
    torch.manual_seed(0)
    batch = _batch()

    R0_raw = _model(coord_scale=1.0)._prepare(batch)[0]
    R0_scaled = _model(coord_scale=4.0)._prepare(batch)[0]

    assert torch.allclose(R0_scaled, R0_raw / 4.0, atol=1e-6)


def test_prepare_yields_unit_variance_when_scale_is_the_data_std():
    torch.manual_seed(0)
    batch = _batch(scale=7.0)
    mask = batch["node_mask"].bool()

    raw = _model(coord_scale=1.0)._prepare(batch)[0]
    scale = float(raw[mask].flatten().std(unbiased=False))

    scaled = _model(coord_scale=scale)._prepare(batch)[0]
    assert abs(float(scaled[mask].flatten().std(unbiased=False)) - 1.0) < 1e-4


def test_generate_cityjson_restores_metres(monkeypatch):
    """The reverse chain runs in scaled units; CityJSON output must be in metres."""
    import src.post_process.post_process as pp

    scale = 5.0
    model = _model(coord_scale=scale)

    pos = torch.arange(4 * 3, dtype=torch.float32).reshape(1, 4, 3)
    node_labels = torch.zeros(1, 4, dtype=torch.long)      # all vertices
    edge_labels = torch.zeros(1, 4, 4, dtype=torch.long)
    monkeypatch.setattr(model, "sample",
                        lambda batch_size=1: (pos, node_labels, edge_labels))

    seen = {}

    def fake_graph_to_cityjson(coords, node_classes, edge_classes, building_id):
        seen["coords"] = coords
        return {"type": "CityJSON"}

    monkeypatch.setattr(pp, "graph_to_cityjson", fake_graph_to_cityjson)

    results = model.generate_cityjson(batch_size=1)

    assert len(results) == 1
    assert torch.allclose(torch.as_tensor(seen["coords"]), pos[0] * scale, atol=1e-6)


def test_coord_scale_survives_a_checkpoint_roundtrip(tmp_path):
    """Inference reads the scale from the checkpoint, so it must be a saved hparam."""
    import lightning as L

    model = _model(coord_scale=3.25)
    ckpt = tmp_path / "m.ckpt"
    trainer = L.Trainer(logger=False, enable_checkpointing=False, accelerator="cpu")
    trainer.strategy.connect(model)
    trainer.save_checkpoint(ckpt)

    restored = CityJSONDiffusionModule.load_from_checkpoint(ckpt, map_location="cpu")
    assert restored.coord_scale == 3.25
