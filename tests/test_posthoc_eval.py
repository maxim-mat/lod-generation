"""Post-hoc eval: config, strategy grid, per-object scoring, artifacts, CLI.

Synthetic meshes and an untrained tiny model only. The real-data check lives in
tests/test_metric_sanity_real.py.
"""
from pathlib import Path

import numpy as np
import pytest
from omegaconf.errors import OmegaConfBaseException

from src.eval.posthoc_eval import (expand_strategies, load_posthoc_config,
                                   validate_posthoc_config)
from src.utils.config import PosthocEvalConfig, StrategyConfig

ROOT = Path(__file__).resolve().parents[1]


def _cfg(**kw):
    kw.setdefault("reference_config", "unused.yaml")
    kw.setdefault("checkpoint", "unused.ckpt")
    return PosthocEvalConfig(**kw)


# ---------------------------------------------------------------- Task 4

def test_default_strategies_expand_to_the_seven_named_runs():
    names = [r.name for r in expand_strategies(_cfg().strategies)]
    assert names == ["greedy", "beam_k3_lp1.0", "beam_k5_lp1.0",
                     "top_p_p0.95_t0.6", "top_p_p0.95_t1.2",
                     "min_p_p0.05_t0.6", "min_p_p0.05_t1.2"]


def test_each_run_maps_to_the_generate_arguments_it_names():
    runs = {r.name: r.generate_kwargs() for r in expand_strategies(_cfg().strategies)}
    assert runs["greedy"] == {"temperature": 0.0}
    assert runs["beam_k5_lp1.0"] == {"temperature": 0.0, "beam_size": 5,
                                     "length_penalty": 1.0}
    assert runs["top_p_p0.95_t1.2"] == {"temperature": 1.2, "top_p": 0.95}
    assert runs["min_p_p0.05_t0.6"] == {"temperature": 0.6, "min_p": 0.05}


def test_beam_length_penalty_defaults_to_one():
    runs = expand_strategies([StrategyConfig(kind="beam", beam_size=[3])])
    assert [r.name for r in runs] == ["beam_k3_lp1.0"]


@pytest.mark.parametrize("strategy, match", [
    (StrategyConfig(kind="nucleus"), "Unknown strategy kind"),
    (StrategyConfig(kind="greedy", temperature=[0.6]), "takes no"),
    (StrategyConfig(kind="beam", beam_size=[3], top_p=[0.9]), "takes no"),
    (StrategyConfig(kind="top_p", top_p=[0.9]), "temperature"),
    (StrategyConfig(kind="min_p", min_p=[0.05], temperature=[0.0]), "temperature"),
    (StrategyConfig(kind="beam", beam_size=[]), "beam_size"),
])
def test_bad_strategies_are_rejected(strategy, match):
    with pytest.raises(ValueError, match=match):
        expand_strategies([strategy])


def test_duplicate_run_names_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        expand_strategies([StrategyConfig(kind="greedy"), StrategyConfig(kind="greedy")])


@pytest.mark.parametrize("kw, match", [
    ({"metrics": ["chamfer", "emd"]}, "metric"),
    ({"references": ["lod3"]}, "reference"),
    ({"split": "holdout"}, "split"),
    ({"strategies": []}, "strategy"),
    ({"batch_size": 0}, "batch_size"),
])
def test_validation_fails_fast(kw, match):
    with pytest.raises(ValueError, match=match):
        validate_posthoc_config(_cfg(**kw))


def test_loading_merges_yaml_and_overrides(tmp_path):
    path = tmp_path / "eval.yaml"
    path.write_text("reference_config: configs/mesh-v5-full-coord.yaml\n"
                    "checkpoint: some.ckpt\n"
                    "strategies:\n  - {kind: beam, beam_size: [2]}\n")
    cfg = load_posthoc_config(path, ["max_objects=16", "sinkhorn.blur_m=0.2"])
    assert cfg.max_objects == 16 and cfg.sinkhorn.blur_m == 0.2
    assert [r.name for r in validate_posthoc_config(cfg)] == ["beam_k2_lp1.0"]


def test_loading_rejects_unknown_keys(tmp_path):
    path = tmp_path / "eval.yaml"
    path.write_text("reference_config: a.yaml\ncheckpoint: b.ckpt\n"
                    "strategies:\n  - {kind: beam, beams: [3]}\n")
    with pytest.raises(OmegaConfBaseException):
        load_posthoc_config(path, [])


def test_loading_requires_a_checkpoint(tmp_path):
    path = tmp_path / "eval.yaml"
    path.write_text("reference_config: a.yaml\n")
    with pytest.raises(OmegaConfBaseException):
        load_posthoc_config(path, [])


def test_the_shipped_config_is_valid():
    cfg = load_posthoc_config(ROOT / "configs" / "posthoc-eval.yaml", [])
    assert len(validate_posthoc_config(cfg)) == 7


# ---------------------------------------------------------------- Task 5

import trimesh

from src.eval.posthoc_eval import METRICS, score_object
from src.utils.config import SinkhornConfig

SINK = SinkhornConfig(n_points=256, blur_m=0.05, max_iters=2000, tol=1e-9)


def _box(dz=0.0):
    b = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    return (np.asarray(b.vertices, dtype=float) + np.array([0.0, 0.0, dz]),
            np.asarray(b.faces, dtype=np.int64))


def _score(gen, ref, metrics=METRICS):
    return score_object(gen, {"gt": ref}, metrics, n_points=1024, sinkhorn_cfg=SINK)


def test_a_box_against_itself_scores_zero_everywhere():
    row = _score(_box(), _box())
    for key in ("chamfer_m", "hausdorff_p95_m", "hausdorff_max_m", "sinkhorn_m",
                "roof_chamfer_m"):
        assert row[f"{key}_vs_gt"] < 1e-9, key
    assert row["vol_ok_vs_gt"] is True
    assert row["rel_symdiff_vs_gt"] < 1e-9
    assert row["watertight_gen"] == 1.0 and row["n_faces_gen"] == 12


def test_a_lifted_box_scores_the_known_geometry():
    """2 m cube lifted 0.5 m: overlap 2x2x1.5 = 6 m^3, so the symmetric
    difference is 8 + 8 - 2*6 = 4 m^3 and the union 10 m^3. The roofs are
    parallel 2x2 squares 0.5 m apart, so their chamfer is exactly 0.5 m."""
    row = _score(_box(0.5), _box())
    assert row["symdiff_m3_vs_gt"] == pytest.approx(4.0, abs=1e-9)
    assert row["rel_symdiff_vs_gt"] == pytest.approx(0.4, abs=1e-9)
    assert row["roof_chamfer_m_vs_gt"] == pytest.approx(0.5, abs=1e-6)
    assert row["sinkhorn_m_vs_gt"] == pytest.approx(0.5, abs=0.02)
    assert row["chamfer_m_vs_gt"] > 0
    assert row["hausdorff_max_m_vs_gt"] >= row["hausdorff_p95_m_vs_gt"]


def test_an_open_mesh_has_no_volume_but_still_has_distances():
    v, f = _box()
    row = _score((v, f[:-2]), _box())
    assert row["vol_ok_vs_gt"] is False and np.isnan(row["symdiff_m3_vs_gt"])
    assert row["vol_note_vs_gt"]
    assert row["watertight_gen"] == 0.0
    assert np.isfinite(row["chamfer_m_vs_gt"])


def test_an_empty_generation_is_nan_not_an_exception():
    empty = (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    row = _score(empty, _box())
    for key in ("chamfer_m", "hausdorff_max_m", "sinkhorn_m", "roof_chamfer_m",
                "symdiff_m3"):
        assert np.isnan(row[f"{key}_vs_gt"]), key
    assert row["n_faces_gen"] == 0 and row["watertight_gen"] == 0.0


def test_only_requested_metrics_and_references_produce_columns():
    row = score_object(_box(), {"lod1": _box(), "lod2": _box(0.5)}, ["chamfer"],
                       n_points=256)
    metric_cols = sorted(k for k in row if "_vs_" in k)
    assert metric_cols == ["chamfer_m_vs_lod1", "chamfer_m_vs_lod2"]
