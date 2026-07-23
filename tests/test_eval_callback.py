from pathlib import Path

import numpy as np

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
    monkeypatch.setattr(cb, "reference_features", lambda dm, split, fs: (np.array([[1.0, 2.0]]), ["a", "b"]))
    monkeypatch.setattr(cb, "building_features", lambda cj, feature_set="full": {"a": 1.0, "b": 2.0})
    monkeypatch.setattr(cb, "check_validity", lambda cjs, path=None: None)

    cfg = GenerativeEvalConfig(enabled=True, num_batches=1, batch_size=1, log_n_samples=1)
    metrics = run_generative_eval(model, datamodule=None, cfg=cfg, loggers=[], save_dir=tmp_path)

    assert "gen/rejection_rate" in metrics
    assert "gen/face_centroid_consistency" in metrics
    assert (tmp_path / "gen_0.city.json").exists()
