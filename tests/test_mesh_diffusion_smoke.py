"""End-to-end smoke: the Phase A arm must train a step and sample a mesh.

Synthetic on-disk data is out of scope here -- these tests build a
`MeshDiffusionModule` directly and feed it a hand-made batch, which is the
smallest thing that proves the four subsystems are wired to each other.
"""
import pytest
import torch
from omegaconf import OmegaConf

from src.models.mesh_diffusion_module import MeshDiffusionModule
from src.utils.config import Config


def _cfg(**kw):
    cfg = OmegaConf.structured(Config)
    cfg.config_set = "mesh_diffusion"
    cfg.mesh_data.num_bins = 128
    cfg.mesh_diffusion.d_model = 32
    cfg.mesh_diffusion.time_dim = 32
    cfg.mesh_diffusion.n_head = 2
    cfg.mesh_diffusion.num_layers = 2
    cfg.mesh_diffusion.noise_steps = 100
    cfg.mesh_diffusion.eval_steps = 5
    for k, v in kw.items():
        OmegaConf.update(cfg, f"mesh_diffusion.{k}", v)
    return cfg


def _batch(b=2, f=16, fc=8):
    x = torch.zeros(b, 10, f)
    x[:, :9] = torch.rand(b, 9, f) - 0.5
    mask = torch.zeros(b, f, dtype=torch.bool)
    mask[:, : f - 4] = True
    x[:, 9] = torch.where(mask, 0.5, -0.5)
    cond = torch.zeros(b, 10, fc)
    cond[:, :9] = torch.rand(b, 9, fc) - 0.5
    cond[:, 9] = 0.5
    return {"x": x, "x_mask": mask, "cond": cond,
            "cond_mask": torch.ones(b, fc, dtype=torch.bool),
            "center": torch.zeros(b, 3), "scale": torch.ones(b, 3),
            "ids": ["a", "b"][:b]}


def test_training_step_produces_a_finite_scalar_with_gradients():
    m = MeshDiffusionModule(_cfg())
    loss = m.training_step(_batch(), 0)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_generate_returns_the_batch_shape():
    m = MeshDiffusionModule(_cfg()).eval()
    out = m.generate(_batch(), n_steps=3)
    assert out.shape == (2, 10, 16)
    assert torch.isfinite(out).all()


def test_condition_dropout_still_produces_a_loss():
    m = MeshDiffusionModule(_cfg(cond_dropout=1.0))   # always dropped
    assert torch.isfinite(m.training_step(_batch(), 0))


def test_guidance_changes_the_sample():
    torch.manual_seed(0)
    m = MeshDiffusionModule(_cfg(cond_dropout=0.1, guidance=1.0)).eval()
    # The output head is zero-initialised on purpose, so an untrained denoiser
    # predicts exactly 0 whether or not it is given the condition -- and
    # guidance, which is a difference of two predictions, is then provably
    # zero too. Break that degeneracy before asking whether guidance bites.
    torch.nn.init.normal_(m.denoiser.outc.weight, std=0.05)
    torch.nn.init.normal_(m.denoiser.outc.bias, std=0.05)
    torch.manual_seed(0)
    a = m.generate(_batch(), n_steps=3)
    m.guidance = 3.0
    torch.manual_seed(0)
    b = m.generate(_batch(), n_steps=3)
    assert not torch.allclose(a, b)


def test_x0_target_arm_also_trains():
    m = MeshDiffusionModule(_cfg(target="original"))
    assert torch.isfinite(m.training_step(_batch(), 0))


def test_illegal_arm_is_rejected_at_construction():
    with pytest.raises(ValueError, match="hungarian"):
        MeshDiffusionModule(_cfg(order="none", pos_embed="none", loss="mse"))


def test_run_mesh_set_eval_returns_paired_metrics(tmp_path):
    """The eval must produce metrics from a model that has learned nothing.

    A random model's chamfer is meaningless; that it is a finite float, keyed
    the way the AR branch keys its metrics, is not.
    """
    import numpy as np
    from src.eval.mesh_set_eval import run_mesh_set_eval

    class _FakeDataset:
        """Two identical cubes, in the shape MeshSetDataset returns."""

        def __init__(self):
            v = np.array([[x, y, z] for x in (-0.4, 0.4) for y in (-0.4, 0.4)
                          for z in (-0.4, 0.4)], dtype=float)
            f = np.array([[0, 1, 3], [0, 3, 2], [4, 7, 5], [4, 6, 7],
                          [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                          [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
            self.tri = v[f]
            self.ids = ["cube0", "cube1"]
            self.pairs = [((v, f), (v, f))] * 2

        def __len__(self):
            return 2

        def mesh_pair(self, i):
            return self.pairs[i]

        def __getitem__(self, i):
            x = torch.zeros(12, 10)
            x[:, :9] = torch.from_numpy(self.tri.reshape(12, 9)).float()
            x[:, 9] = 0.5
            return {"x": x, "cond": x.clone(), "id": self.ids[i],
                    "center": torch.zeros(3), "scale": torch.ones(3)}

    m = MeshDiffusionModule(_cfg()).eval()
    out = run_mesh_set_eval(m, _FakeDataset(), [0, 1], _cfg(),
                            save_dir=tmp_path, seed=0)
    assert "chamfer_m" in out and np.isfinite(out["chamfer_m"])
    assert "n_faces" in out and "gt_chamfer_m" in out
    assert list(tmp_path.glob("*.obj"))


def test_eval_callback_is_gated_by_epoch():
    from src.eval.mesh_set_eval import MeshSetEvalCallback

    cfg = _cfg()
    cfg.mesh_diffusion.every_n_epochs = 5
    cb = MeshSetEvalCallback(cfg, save_dir=None)
    assert not cb._due(epoch=0) and not cb._due(epoch=3)
    assert cb._due(epoch=4) and cb._due(epoch=9)
