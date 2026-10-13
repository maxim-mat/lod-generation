"""`training.accumulate_grad_batches` reaches the Trainer from config.

The field is additive with a no-op default, so the thing worth pinning is that
it survives the schema/YAML/CLI merge and that omitting it keeps the old
behaviour (1 = every batch is an optimizer step).
"""
import inspect

import lightning as L
import pytest
from omegaconf import OmegaConf

from src.utils.config import Config, TrainingConfig
from src.utils.initialization import load_config


def write_cfg(tmp_path, body):
    """A config file that resolves.

    `DataConfig.dataset_dir` is MISSING and OmegaConf resolves the whole schema
    on `to_object`, so every config needs one -- the same reason the mesh
    configs carry a dummy `data:` block they never read.
    """
    path = tmp_path / "c.yaml"
    path.write_text("data:\n  dataset_dir: /nonexistent\n" + body, encoding="utf-8")
    return path


def test_default_is_a_no_op():
    assert TrainingConfig().accumulate_grad_batches == 1


def test_absent_from_yaml_falls_back_to_the_default(tmp_path):
    cfg = load_config(write_cfg(tmp_path, "training:\n  lr: 0.001\n"), [])
    assert cfg.training.accumulate_grad_batches == 1


def test_yaml_value_is_picked_up(tmp_path):
    cfg = load_config(
        write_cfg(tmp_path, "training:\n  accumulate_grad_batches: 8\n"), [])
    assert cfg.training.accumulate_grad_batches == 8


def test_cli_override_wins_over_yaml(tmp_path):
    cfg = load_config(write_cfg(tmp_path, "training:\n  accumulate_grad_batches: 8\n"),
                      ["training.accumulate_grad_batches=4"])
    assert cfg.training.accumulate_grad_batches == 4


def test_non_integer_is_rejected_by_the_schema(tmp_path):
    with pytest.raises(Exception):
        load_config(
            write_cfg(tmp_path, "training:\n  accumulate_grad_batches: half\n"), [])


def test_shipped_default_config_carries_the_key():
    cfg = OmegaConf.load("configs/default.yaml")
    assert cfg.training.accumulate_grad_batches == 1


def test_lightning_trainer_accepts_the_argument():
    """Guards the wiring against a Lightning rename."""
    assert "accumulate_grad_batches" in inspect.signature(L.Trainer).parameters


@pytest.mark.parametrize("module", ["train", "train_mesh", "train_mesh_vqvae"])
def test_every_trainer_passes_it_through(module):
    """All three entry points, not just the mesh one."""
    src = (__import__("pathlib").Path("src") / f"{module}.py").read_text(encoding="utf-8")
    assert "accumulate_grad_batches=cfg.training.accumulate_grad_batches," in src


def test_config_object_round_trips_through_omegaconf():
    """train_*.py log hparams via OmegaConf.structured(cfg); it must not choke."""
    hp = OmegaConf.to_container(OmegaConf.structured(Config()), resolve=True)
    assert hp["training"]["accumulate_grad_batches"] == 1
