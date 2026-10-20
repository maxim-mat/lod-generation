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
