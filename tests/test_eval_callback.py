from pathlib import Path

import numpy as np
import torch

from src.utils.config import Config, GenerativeEvalConfig
from src.utils.setup_utils import create_callbacks
from src.eval.callback import face_centroid_consistency, run_generative_eval
from src.dataset.dataset import VERTEX, WALL, EDGE_VF


def test_callback_absent_when_disabled(tmp_path):
    cfg = Config()
    cfg.generative_eval.enabled = False
    names = [type(c).__name__ for c in create_callbacks(cfg, tmp_path)]
    assert "GenerativeEvalCallback" not in names


def test_callback_present_when_enabled(tmp_path):
    cfg = Config()
    cfg.generative_eval.enabled = True
    names = [type(c).__name__ for c in create_callbacks(cfg, tmp_path)]
    assert "GenerativeEvalCallback" in names


def test_lr_monitor_is_skipped_when_there_is_no_logger(tmp_path):
    """`LearningRateMonitor` has nowhere to write when `logging.loggers` is
    empty, and Lightning refuses the Trainer outright rather than ignoring it:

        MisconfigurationException: Cannot use `LearningRateMonitor` callback
        with `Trainer` that has no logger.

    An empty logger list is a legitimate configuration -- the pipeline's smoke
    mode uses it so a throwaway run creates no wandb run -- so the callback has
    to be conditional on there being something to log to.
    """
    cfg = Config()
    cfg.logging.loggers = []
    assert "LearningRateMonitor" not in [type(c).__name__ for c in create_callbacks(cfg, tmp_path)]

    cfg.logging.loggers = ["tensorboard"]
    assert "LearningRateMonitor" in [type(c).__name__ for c in create_callbacks(cfg, tmp_path)]


def test_a_loggerless_fit_survives_the_callbacks(tmp_path):
    """The check that actually reproduces the failure.

    Lightning validates this in `LearningRateMonitor.setup`, which runs on
    `fit` -- not on `Trainer(...)`. An earlier version of this test only built
    the Trainer and passed while the bug was still present.
    """
    import lightning as L
    from torch.utils.data import DataLoader, TensorDataset

    class Tiny(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(1, 1)

        def training_step(self, batch, _):
            return self.layer(batch[0]).square().mean()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    cfg = Config()
    cfg.logging.loggers = []
    cfg.generative_eval.enabled = False
    # This module logs nothing, and the default config's early stopping watches
    # a diffusion metric. Off, so the test fails only on the logger question.
    cfg.training.early_stopping.enabled = False
    loader = DataLoader(TensorDataset(torch.zeros(2, 1)), batch_size=1)
    # No enable_* overrides: create_callbacks supplies the checkpoint and
    # progress bar itself, and Lightning rejects a Trainer that disables what
    # its callback list contains. This is the real configuration.
    L.Trainer(logger=False, callbacks=create_callbacks(cfg, tmp_path), max_epochs=1,
              accelerator="cpu", devices=1,
              default_root_dir=str(tmp_path)).fit(Tiny(), loader)


def test_face_centroid_consistency_zero_when_centered():
    # 3 vertices + 1 face node sitting exactly at their centroid
    coords = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [1, 1, 0]], dtype=float)
    labels = np.array([VERTEX, VERTEX, VERTEX, WALL])
    edge = np.zeros((4, 4), dtype=np.int64)
    for v in range(3):
        edge[3, v] = edge[v, 3] = EDGE_VF
    assert face_centroid_consistency(coords, labels, edge) < 1e-9
    coords[3] = [9, 9, 0]  # displace the face node
    assert face_centroid_consistency(coords, labels, edge) > 1.0


class _StubSplit:
    """Dataset stand-in that records which indices the reference pass touched."""

    def __init__(self, n):
        self.n = n
        self.touched = []

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        self.touched.append(i)
        return {"x": torch.zeros(3, 3), "node_categories": torch.zeros(3, 5),
                "y": torch.zeros(3, 3, 1)}


def test_reference_features_subsamples_and_is_deterministic(monkeypatch):
    """The splits hold O(1e5) graphs; the cap must bound the conversion work
    and draw the same reference set every run, or the metrics drift."""
    import src.eval.callback as cb
    monkeypatch.setattr(cb, "graph_to_cityjson", lambda *a, **k: None)

    dm = type("DM", (), {})()
    dm.test_dataset = _StubSplit(1000)
    cb.reference_features(dm, "test", "full", max_samples=25)
    first = dm.test_dataset.touched
    assert len(first) == 25 and first == sorted(first)

    dm.test_dataset = _StubSplit(1000)
    cb.reference_features(dm, "test", "full", max_samples=25)
    assert dm.test_dataset.touched == first

    # Below the cap, every graph is used.
    dm.test_dataset = _StubSplit(10)
    cb.reference_features(dm, "test", "full", max_samples=25)
    assert len(dm.test_dataset.touched) == 10


def test_run_generative_eval_smoke(tmp_path, monkeypatch):
    from src.models.diffusion import CityJSONDiffusionModule
    import src.eval.callback as cb

    model = CityJSONDiffusionModule(hidden_dim=8, edge_dim=4, global_dim=4,
                                    n_head=2, num_layers=1, T=10, n_max=6)

    # one well-formed record; no real sampling, no val3dity
    rec = {"coords": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
           "node_labels": np.array([VERTEX, VERTEX, VERTEX]),
           "edge_labels": np.zeros((3, 3), dtype=np.int64),
           "cityjson": {"type": "CityJSON", "version": "1.1",
                        "CityObjects": {"b": {"type": "Building", "geometry": [{
                            "type": "Solid", "lod": "2",
                            "boundaries": [[[[0, 1, 2]]]]}]}},
                        "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]}}
    monkeypatch.setattr(cb, "draw_samples", lambda m, nb, bs: ([rec], {"attempted": 1, "dropped": 0}))
    monkeypatch.setattr(cb, "reference_features", lambda *a, **k: (np.array([[1.0, 2.0]]), ["a", "b"]))
    monkeypatch.setattr(cb, "building_features", lambda cj, feature_set="full": {"a": 1.0, "b": 2.0})
    monkeypatch.setattr(cb, "check_validity", lambda cjs, path=None: None)

    cfg = GenerativeEvalConfig(enabled=True, num_batches=1, batch_size=1, log_n_samples=1)
    metrics = run_generative_eval(model, datamodule=None, cfg=cfg, loggers=[],
                                  save_dir=tmp_path, seed=7)

    assert "gen/rejection_rate" in metrics
    assert "gen/face_centroid_consistency" in metrics
    assert (tmp_path / "gen_0.city.json").exists()
