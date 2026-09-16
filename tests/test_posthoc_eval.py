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
