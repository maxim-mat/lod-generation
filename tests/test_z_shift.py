"""se2 z standardization: train-empirical mean out in _prepare, back in decode.

The Gaussian forward process converges to N(0, I) regardless of the data, so
matching the empirical z distribution means standardizing with train-split
moments (the analogue of the discrete marginal priors), not sampling a
histogram. z_shift rides in checkpoint hparams exactly like coord_scale.
"""
import torch

from src.dataset.datamodule import CityJSONDataModule
from src.models.diffusion import CityJSONDiffusionModule

B, N = 2, 6


def _batch():
    torch.manual_seed(0)
    x = torch.zeros(B, N, 3)
    x[:, :4] = torch.randn(B, 4, 3) + torch.tensor([0.0, 0.0, 10.0])
    cats = torch.zeros(B, N, 5)
    cats[:, :4, 0] = 1.0
    cats[:, 4:, 4] = 1.0
    return {
        "x": x,
        "node_categories": cats,
        "y": torch.randint(0, 3, (B, N, N, 1)),
        "node_mask": cats[..., 0].clone(),
    }


def _model(**kw):
    return CityJSONDiffusionModule(hidden_dim=8, edge_dim=4, global_dim=4, n_head=2,
                                   num_layers=1, T=10, n_max=N, **kw)


def test_compute_z_shift_is_pooled_vertex_z_mean():
    items = []
    for zs in ([1.0, 3.0], [5.0]):
        n = len(zs)
        x = torch.zeros(N, 3)
        x[:n, 2] = torch.tensor(zs)
        mask = torch.zeros(N)
        mask[:n] = 1.0
        items.append({"x": x, "node_mask": mask})

    dm = CityJSONDataModule(dataset_dir="unused", lods=1)
    dm.train_dataset = items

    assert abs(dm.compute_z_shift() - 3.0) < 1e-6


def test_prepare_se2_standardizes_z_and_centres_xy_only():
    model = _model(equivariance="se2", z_shift=10.0, coord_scale=2.0)
    batch = _batch()

    R0 = model._prepare(batch)[0]

    real = R0[:, :4]
    assert torch.allclose(real[..., :2].mean(dim=1), torch.zeros(B, 2), atol=1e-5)
    expected_z = (batch["x"][:, :4, 2] - 10.0) / 2.0
    assert torch.allclose(real[..., 2], expected_z, atol=1e-5)
    assert torch.all(R0[:, 4:] == 0)


def test_non_se2_ignores_z_shift():
    model = _model(equivariance="so2", z_shift=99.0, coord_scale=1.0)
    assert model.z_shift == 0.0


def test_generate_cityjson_restores_z_shift(monkeypatch):
    import src.post_process.post_process as pp

    model = _model(equivariance="se2", z_shift=7.0, coord_scale=2.0)

    pos = torch.ones(1, 4, 3)
    node_labels = torch.zeros(1, 4, dtype=torch.long)
    edge_labels = torch.zeros(1, 4, 4, dtype=torch.long)
    monkeypatch.setattr(model, "sample",
                        lambda batch_size=1: (pos, node_labels, edge_labels))

    seen = {}

    def fake(coords, node_classes, edge_classes, building_id):
        seen["coords"] = coords.copy()
        return {"type": "CityJSON"}

    monkeypatch.setattr(pp, "graph_to_cityjson", fake)
    model.generate_cityjson(batch_size=1)

    # Export centres the footprint horizontally in every mode, so xy collapses
    # to the origin here (all four vertices share one xy).
    assert torch.allclose(torch.as_tensor(seen["coords"][:, :2]),
                          torch.zeros((4, 2), dtype=torch.float64), atol=1e-6)
    # z is the point of this test: se2 carries real elevation, so z_shift is
    # restored and must survive un-levelled.
    assert torch.allclose(torch.as_tensor(seen["coords"][:, 2]),
                          torch.full((4,), 9.0, dtype=torch.float64), atol=1e-6)   # 1 * 2 + 7


def test_z_shift_survives_checkpoint(tmp_path):
    import lightning as L

    model = _model(equivariance="se2", z_shift=4.5, coord_scale=1.0)
    ckpt = tmp_path / "m.ckpt"
    trainer = L.Trainer(logger=False, enable_checkpointing=False, accelerator="cpu")
    trainer.strategy.connect(model)
    trainer.save_checkpoint(ckpt)

    restored = CityJSONDiffusionModule.load_from_checkpoint(ckpt, map_location="cpu")
    assert restored.z_shift == 4.5
    assert restored.equivariance == "se2"
