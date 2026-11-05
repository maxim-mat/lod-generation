from pathlib import Path

import pytest
import torch

from src.utils.config import Config
from tests.test_ortho_module import S, batch, module


def test_config_defaults_keep_v5_image_free():
    m = Config().mesh_model
    assert m.image_encoder is None and m.init_from is None
    assert (m.image_fusion, m.train_scope, m.backbone_lr_scale) == ("xattn", "all", 1.0)


def test_train_mesh_refuses_an_image_model_without_ortho_dir():
    from src.train_mesh import train_mesh
    cfg = Config()
    cfg.mesh_data.dataset_dir = "unused"
    cfg.mesh_model.image_encoder = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    with pytest.raises(ValueError, match="ortho_dir"):
        train_mesh(cfg)


class _Seen(Exception):
    """Raised by the generate spy: the test is about the call, not the metrics after it."""


def test_run_mesh_eval_passes_the_image_to_generate(monkeypatch):
    from src.eval import mesh_eval
    m = module().eval()
    seen = {}

    def spy(self, cond, cond_pad_mask=None, **kw):
        seen.update(kw)
        raise _Seen
    monkeypatch.setattr(type(m), "generate", spy)
    b = batch()
    items = [{"cond": b["cond"][i], "tgt": b["tgt"][i], "id": b["ids"][i],
              "center": b["center"][i], "scale": b["scale"][i],
              "image": b["image"][i]} for i in range(2)]
    cfg = type("Cfg", (), dict(batch_size=2, beam_size=1, length_penalty=1.0, taus=[0.25],
                               n_points=64, voxel_m=0.5, save_samples=0))()
    with pytest.raises(_Seen):
        mesh_eval.run_mesh_eval(m, items, [0, 1], cfg, max_new_tokens=4)
    assert seen["image"].shape == (2, 3, S, S) and seen.get("image_valid") is None


def test_inference_refuses_image_models():
    from src.inference import _check_config
    cfg = Config()
    cfg.inference.checkpoint_path, cfg.inference.input_path = "x", "y"
    cfg.mesh_model.image_encoder = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    with pytest.raises(ValueError, match="orthophoto"):
        _check_config(cfg)


# Every v6 arm runs on v5's random split for now (prefix 2026-10-24, xattn
# 2026-11-05; time), from the random-split v5 checkpoint. The planned split is
# tile-buffer50m (see the v5-tile baseline test below).
@pytest.mark.parametrize("name, split", [("xattn-a", "random"),
                                         ("xattn-b", "random"),
                                         ("smoke", "random"),
                                         ("prefix", "random"), ("prefix-smoke", "random"),
                                         ("prefix-aligned", "random"),
                                         ("prefix-aligned-smoke", "random")])
def test_shipped_v6_configs_load(name, split):
    from src.dataset.splits import split_name
    from src.utils.initialization import load_config
    cfg = load_config(Path(f"configs/mesh-v6-ortho-{name}.yaml"), [])
    assert cfg.mesh_model.image_encoder and cfg.mesh_data.ortho_dir == "ortho"
    t = cfg.training
    assert split_name(t.split, t.split_block_m, t.split_buffer_m) == split


@pytest.mark.parametrize("name", ["prefix", "xattn-a", "smoke"])
def test_v6_inits_from_the_v5_run_on_its_split(name):
    from src.dataset.splits import split_name
    from src.utils.initialization import load_config
    run = load_config(Path(f"configs/mesh-v6-ortho-{name}.yaml"), [])
    v5 = load_config(Path("configs/mesh-v5-full-opt-coord.yaml"), [])
    assert "/mesh-v5-full-opt-coord/checkpoints/" in run.mesh_model.init_from
    for t in (run.training, v5.training):
        assert split_name(t.split, t.split_block_m, t.split_buffer_m) == "random"
    assert run.seed == v5.seed
    assert list(run.training.train_val_test_split) == list(v5.training.train_val_test_split)


def test_v5_tile_baseline_uses_the_v6_split():
    # load_weights_from refuses an init whose data_split differs, so the
    # baseline that initialises every v6 run must name the same split.
    from src.dataset.splits import split_name
    from src.utils.initialization import load_config
    t = load_config(Path("configs/mesh-v5-full-opt-coord-tile.yaml"), []).training
    assert split_name(t.split, t.split_block_m, t.split_buffer_m) == "tile-buffer50m"


def test_prefix_smoke_matches_the_prefix_run_but_for_size():
    from src.utils.initialization import load_config
    run = load_config(Path("configs/mesh-v6-ortho-prefix.yaml"), [])
    smoke = load_config(Path("configs/mesh-v6-ortho-prefix-smoke.yaml"), [])
    for key in ("image_fusion", "train_scope", "backbone_lr_scale", "init_from"):
        assert getattr(smoke.mesh_model, key) == getattr(run.mesh_model, key)
    assert smoke.mesh_data.dataset_dir.endswith("mini_cleaner")
    assert smoke.training.max_epochs == 1 and not smoke.logging.loggers


def test_check_vertex_margins():
    from src.train_mesh import check_vertex_margins
    check_vertex_margins(True, [0, 0, 0], [0, 0, 0.1])
    check_vertex_margins(False, [0.1, 0, 0], [0, 0, 0.1])
    with pytest.raises(ValueError, match="margin"):
        check_vertex_margins(True, [0.1, 0, 0], [0, 0, 0.1])
    with pytest.raises(ValueError, match="margin"):
        check_vertex_margins(True, [0, 0, 0], [0, 0.05, 0.1])


def test_config_defaults_keep_the_old_prefix():
    m = Config().mesh_model
    assert (m.image_separate_positions, m.image_mask_invalid, m.image_prefix_grid,
            m.image_vertex_features, m.image_prefix_attention, m.image_vertex_width,
            m.image_xattn_ffn) == (False, False, None, False, "causal", 64, False)


@pytest.mark.parametrize("name, ffn", [("xattn-a", True), ("xattn-b", True), ("smoke", True),
                                       ("prefix", False), ("prefix-aligned", False)])
def test_gated_ffn_is_on_in_the_xattn_configs_only(name, ffn):
    from src.utils.initialization import load_config
    cfg = load_config(Path(f"configs/mesh-v6-ortho-{name}.yaml"), [])
    assert cfg.mesh_model.image_xattn_ffn is ffn


ALIGNED_FLAGS = {"image_separate_positions": True, "image_mask_invalid": True,
                 "image_prefix_grid": 8, "image_vertex_features": True,
                 "image_prefix_attention": "bidirectional"}


def _flat(cfg):
    from dataclasses import asdict
    return {f"{sec}.{k}": v for sec, d in asdict(cfg).items() if isinstance(d, dict)
            for k, v in d.items()}


def test_aligned_differs_from_prefix_only_in_the_new_flags():
    from src.utils.initialization import load_config
    run = _flat(load_config(Path("configs/mesh-v6-ortho-prefix.yaml"), []))
    aligned = _flat(load_config(Path("configs/mesh-v6-ortho-prefix-aligned.yaml"), []))
    diff = {k for k in run if run[k] != aligned[k]}
    assert diff == {f"mesh_model.{k}" for k in ALIGNED_FLAGS} | {"logging.run_name"}
    assert all(aligned[f"mesh_model.{k}"] == v for k, v in ALIGNED_FLAGS.items())
    assert aligned["logging.run_name"] == "mesh-v6-ortho-prefix-aligned-random"


def test_aligned_smoke_matches_aligned_but_for_size():
    from src.utils.initialization import load_config
    run = load_config(Path("configs/mesh-v6-ortho-prefix-aligned.yaml"), [])
    smoke = load_config(Path("configs/mesh-v6-ortho-prefix-aligned-smoke.yaml"), [])
    for key in (*ALIGNED_FLAGS, "image_fusion", "init_from"):
        assert getattr(smoke.mesh_model, key) == getattr(run.mesh_model, key), key
    assert smoke.mesh_data.dataset_dir.endswith("mini_cleaner")
    assert smoke.training.max_epochs == 1 and not smoke.logging.loggers
