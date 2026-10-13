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


# ---------------------------------------------------------------- Task 7

import json

import pandas as pd
import torch

from src.dataset.mesh_dataset import specials
from src.eval.posthoc_eval import checkpoint_info, object_seed, run_posthoc_eval
from src.models.mesh_transformer import MeshTransformerModule
from src.utils.config import Config

NUM_BINS = 32
BOS, EOS, PAD = specials(NUM_BINS)


class _Dataset:
    """Two random-token (cond, tgt) pairs. Same stub as tests/test_mesh_eval.py:
    the meshes are noise, which is the point -- scoring must survive noise."""

    def __init__(self):
        self.ids = ["a", "b"]
        rng = np.random.default_rng(0)
        self.items = []
        for k in range(2):
            n_faces = 4 + k
            self.items.append({
                "cond": torch.from_numpy(rng.integers(0, NUM_BINS, 9 * n_faces)),
                "tgt": torch.cat([torch.tensor([BOS]),
                                  torch.from_numpy(rng.integers(0, NUM_BINS, 9 * n_faces)),
                                  torch.tensor([EOS])]),
                "id": self.ids[k],
                "center": torch.zeros(3),
                "scale": torch.ones(3) * 10.0,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _model():
    torch.manual_seed(1)
    return MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2,
                                 num_layers=1, dropout=0.0, max_seq_len=128)


def _small_cfg(**kw):
    kw.setdefault("batch_size", 2)
    kw.setdefault("n_points", 256)
    kw.setdefault("sinkhorn", SinkhornConfig(n_points=128, max_iters=50))
    return _cfg(**kw)


OBJECT_COLUMNS = [
    "object_id", "seed", "n_faces_lod1", "n_faces_lod2", "eos_emitted", "decode_ok",
    "gen_seconds", "n_faces_gen", "watertight_gen",
] + [f"{m}_vs_{ref}" for ref in ("lod1", "lod2")
     for m in ("chamfer_m", "hausdorff_p95_m", "hausdorff_max_m", "symdiff_m3",
               "rel_symdiff", "vol_ok", "vol_note", "sinkhorn_m", "roof_chamfer_m")]


def test_object_seed_is_stable_and_distinct():
    assert object_seed(1234, "a") == object_seed(1234, "a")
    assert object_seed(1234, "a") != object_seed(1234, "b")
    assert object_seed(1234, "a") != object_seed(7, "a")
    assert object_seed(1234, "a") == 2078608377   # zlib.crc32(b"1234:a"), unsalted


def test_checkpoint_info_hashes_and_reads_metadata(tmp_path):
    path = tmp_path / "x.ckpt"
    torch.save({"epoch": 3, "global_step": 99,
                "hyper_parameters": {"tokenization": "coord"}}, path)
    info = checkpoint_info(path)
    assert info["epoch"] == 3 and info["global_step"] == 99
    assert info["hyper_parameters"] == {"tokenization": "coord"}
    assert len(info["sha256"]) == 64


def test_end_to_end_writes_one_row_per_object_per_run(tmp_path):
    ckpt = tmp_path / "x.ckpt"
    torch.save({"epoch": 0, "global_step": 0, "hyper_parameters": {}}, ckpt)
    cfg = _small_cfg()
    out = tmp_path / "eval"
    summary = run_posthoc_eval(cfg, Config(), _model(), _Dataset(), out,
                               checkpoint=checkpoint_info(ckpt),
                               split_sizes={"train": 0, "val": 2, "test": 0})

    names = [r.name for r in expand_strategies(cfg.strategies)]
    assert [row["run"] for row in summary] == names
    for name in names:
        df = pd.read_csv(out / name / "objects.csv")
        assert df["object_id"].tolist() == ["a", "b"]
        assert list(df.columns) == OBJECT_COLUMNS
        run = json.loads((out / name / "run.json").read_text())
        assert run["name"] == name and run["batch_size"] == 2

    for artifact in ("eval_config.yaml", "reference_config.yaml", "environment.json",
                     "checkpoint.json", "split.json", "summary.csv"):
        assert (out / artifact).is_file(), artifact
    table = pd.read_csv(out / "summary.csv")
    assert len(table) == 7
    assert {"watertight_pct", "decode_rate", "eos_rate", "chamfer_m_vs_lod2_mean",
            "chamfer_m_vs_lod2_median", "vol_defined_pct_vs_lod1",
            "roof_defined_pct_vs_lod2"} <= set(table.columns)
    split = json.loads((out / "split.json").read_text())
    assert split["n_evaluated"] == 2 and split["split_sizes"]["val"] == 2


def test_a_rerun_reproduces_every_row(tmp_path):
    cfg = _small_cfg(strategies=[
        StrategyConfig(kind="top_p", top_p=[0.95], temperature=[1.2]),
        StrategyConfig(kind="beam", beam_size=[3])])
    for sub in ("a", "b"):
        run_posthoc_eval(cfg, Config(), _model(), _Dataset(), tmp_path / sub)
    for name in ("top_p_p0.95_t1.2", "beam_k3_lp1.0"):
        a = pd.read_csv(tmp_path / "a" / name / "objects.csv").drop(columns="gen_seconds")
        b = pd.read_csv(tmp_path / "b" / name / "objects.csv").drop(columns="gen_seconds")
        pd.testing.assert_frame_equal(a, b)


def test_max_objects_takes_a_seeded_subsample(tmp_path):
    cfg = _small_cfg(max_objects=1, strategies=[StrategyConfig(kind="greedy")],
                     metrics=["chamfer"])
    run_posthoc_eval(cfg, Config(), _model(), _Dataset(), tmp_path / "e")
    assert len(pd.read_csv(tmp_path / "e" / "greedy" / "objects.csv")) == 1


def test_an_empty_selection_is_rejected(tmp_path):
    class _Empty(_Dataset):
        def __len__(self):
            return 0
    with pytest.raises(ValueError, match="no objects"):
        run_posthoc_eval(_small_cfg(), Config(), _model(), _Empty(), tmp_path / "e")


# ---------------------------------------------------------------- Task 8

import src.eval.posthoc_eval as posthoc
from src.eval.posthoc_eval import check_tokenization, main, resolve_device


def test_a_tokenization_mismatch_is_refused():
    ref = Config()
    ref.mesh_data.tokenization = "coord"
    check_tokenization({"tokenization": "coord"}, ref)
    # Checkpoints from before AMT existed carry no key and were coordinate runs.
    check_tokenization({}, ref)
    with pytest.raises(ValueError, match="tokenization"):
        check_tokenization({"tokenization": "amt"}, ref)


def test_resolve_device():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cpu", "cuda")


def _write(tmp_path, text):
    path = tmp_path / "eval.yaml"
    path.write_text(text)
    return path


def test_main_rejects_a_bad_metric_before_reading_anything(tmp_path):
    path = _write(tmp_path, "reference_config: does/not/exist.yaml\n"
                            "checkpoint: nope.ckpt\nmetrics: [emd]\n")
    with pytest.raises(ValueError, match="metric"):
        main(["--config", str(path)])


def test_main_rejects_a_non_mesh_reference(tmp_path):
    ref = tmp_path / "ref.yaml"
    ref.write_text("config_set: diffusion\ndata:\n  dataset_dir: x\n")
    path = _write(tmp_path, f"reference_config: {ref.as_posix()}\ncheckpoint: nope.ckpt\n")
    with pytest.raises(ValueError, match="config_set"):
        main(["--config", str(path)])


def test_main_checks_the_checkpoint_before_loading_data(tmp_path, monkeypatch):
    ref = (ROOT / "configs" / "mesh-v5-full-coord.yaml").as_posix()
    missing = (tmp_path / "nope.ckpt").as_posix()
    path = _write(tmp_path, f"reference_config: {ref}\ncheckpoint: {missing}\n")
    monkeypatch.setattr(posthoc, "build_split", lambda *a, **k: pytest.fail(
        "the dataset was loaded before the checkpoint was checked"))
    with pytest.raises(FileNotFoundError, match="nope.ckpt"):
        main(["--config", str(path)])


def test_main_checks_tokenization_before_loading_data(tmp_path, monkeypatch):
    ckpt = tmp_path / "amt.ckpt"
    torch.save({"epoch": 0, "global_step": 0,
                "hyper_parameters": {"tokenization": "amt"}}, ckpt)
    ref = (ROOT / "configs" / "mesh-v5-full-coord.yaml").as_posix()   # coord run
    path = _write(tmp_path, f"reference_config: {ref}\ncheckpoint: {ckpt.as_posix()}\n")
    monkeypatch.setattr(posthoc, "build_split", lambda *a, **k: pytest.fail(
        "the dataset was loaded before tokenization was checked"))
    with pytest.raises(ValueError, match="tokenization"):
        main(["--config", str(path)])
