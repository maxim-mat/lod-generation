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


def test_unordered_hungarian_arm_trains():
    m = MeshDiffusionModule(_cfg(order="none", pos_embed="none",
                                 loss="hungarian", denoiser="transformer"))
    loss = m.training_step(_batch(), 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.parameters())


def test_flow_arm_trains_and_samples():
    m = MeshDiffusionModule(_cfg(process="flow", target="velocity"))
    assert torch.isfinite(m.training_step(_batch(), 0))
    m.eval()
    assert m.generate(_batch(), n_steps=4).shape == (2, 10, 16)


def _batch_with_bins(b=2, f=16, num_bins=128):
    batch = _batch(b=b, f=f)
    batch["x_bins"] = torch.randint(0, num_bins, (b, 9, f))
    return batch


def test_quantized_ce_arm_trains_and_lands_on_the_grid():
    """state: quantized, Gaussian noise, categorical readout -- the reference's
    scheme on coordinate channels."""
    from src.dataset.mesh_dataset import quantize

    m = MeshDiffusionModule(_cfg(state="quantized", loss="ce",
                                 process="ddpm", target="original"))
    assert torch.isfinite(m.training_step(_batch_with_bins(), 0))
    m.eval()
    out = m.generate(_batch_with_bins(), n_steps=4)
    assert out.shape == (2, 10, 16)
    coords = out[:, :9].detach().numpy()
    # Hard clamping must leave every coordinate on a grid point.
    import numpy as np
    from src.dataset.mesh_dataset import dequantize
    assert np.allclose(coords, dequantize(quantize(coords, 128), 128), atol=1e-6)


def test_onehot_ce_arm_trains():
    """state: onehot -- 9 x 128 indicator channels, exactly what the reference
    diffuses."""
    m = MeshDiffusionModule(_cfg(state="onehot", loss="ce",
                                 process="ddpm", target="original"))
    assert torch.isfinite(m.training_step(_batch_with_bins(), 0))


def test_onehot_denoiser_input_width_is_nine_times_bins_plus_one():
    m = MeshDiffusionModule(_cfg(state="onehot", loss="ce",
                                 process="ddpm", target="original"))
    x = m._to_state(_batch_with_bins())
    assert x.shape == (2, 9 * 128 + 1, 16)


def test_soft_clamp_leaves_the_grid():
    import numpy as np
    from src.dataset.mesh_dataset import dequantize, quantize

    m = MeshDiffusionModule(_cfg(state="quantized", loss="ce", process="ddpm",
                                 target="original", x0_clamp="soft")).eval()
    coords = m.generate(_batch_with_bins(), n_steps=4)[:, :9].detach().numpy()
    assert not np.allclose(coords, dequantize(quantize(coords, 128), 128), atol=1e-6)


def test_ce_with_noise_target_is_rejected():
    with pytest.raises(ValueError, match="signal-free"):
        MeshDiffusionModule(_cfg(state="quantized", loss="ce",
                                 process="ddpm", target="noise"))


def test_module_construction_rejects_flow_with_ce():
    """Same rule as the compat suite's `test_flow_with_ce_is_rejected`, checked
    through the constructor -- the path a training run actually takes."""
    with pytest.raises(ValueError, match="incompatible"):
        MeshDiffusionModule(_cfg(state="quantized", loss="ce",
                                 process="flow", target="velocity"))


def test_discrete_arm_trains_and_samples():
    cfg = _cfg(process="d3pm", target="original", loss="ce", state="bins")
    m = MeshDiffusionModule(cfg)
    batch = _batch()
    batch["x_bins"] = torch.randint(0, 128, (2, 9, 16))
    assert torch.isfinite(m.training_step(batch, 0))
    m.eval()
    out = m.generate(batch, n_steps=4)
    assert out.shape == (2, 10, 16)
    assert set(out[:, 9].unique().tolist()) <= {0.5, -0.5}


def test_scaffold_enabled_eval_path_runs():
    """The scaffold hook is built inside run_mesh_set_eval, per batch, and is
    the one piece of Task 13 no unit test reaches: a shape error there would
    only surface hours into the c1 arm."""
    from src.eval.mesh_set_eval import run_mesh_set_eval

    cfg = _cfg()
    cfg.mesh_diffusion.scaffold.enabled = True
    cfg.mesh_diffusion.scaffold.apply_below_t = 0.9   # fire on most steps

    batch = _batch()

    class _Ds:
        ids = ["a", "b"]

        def __len__(self):
            return 2

        def mesh_pair(self, i):
            import numpy as np
            v = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1.0]])
            f = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            return ((v, f), (v, f))

        def __getitem__(self, i):
            return {"x": batch["x"][i].T.clone(), "cond": batch["cond"][i].T.clone(),
                    "id": self.ids[i], "center": torch.zeros(3),
                    "scale": torch.ones(3)}

    m = MeshDiffusionModule(cfg).eval()
    out = run_mesh_set_eval(m, _Ds(), [0, 1], cfg, seed=0)
    assert out and "chamfer_m" in out


def test_unet_hungarian_arm_trains():
    """a4: conv U-Net over file order with a set loss. Legal since D5 was
    relaxed -- file order carries most of the locality the convolution needs
    (72.4% of consecutive faces share a corner on this corpus)."""
    m = MeshDiffusionModule(_cfg(order="none", pos_embed="none",
                                 loss="hungarian", denoiser="unet"))
    loss = m.training_step(_batch(), 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.parameters())
    m.eval()
    assert m.generate(_batch(), n_steps=3).shape == (2, 10, 16)


def test_face_count_is_predicted_not_read_from_the_input():
    """The denoiser must never see which slots are real.

    It used to: `generate` passed `batch["x_mask"]`, which the U-Net's masking
    turned into an exact signal -- the presence output at unused slots was
    literally the head bias, identical for any input. Face count then came from
    the label, not the model, and plan D2's "presence decides face count at
    generation time" was false.

    The property that replaces it: unused slots are DETR no-object slots, so
    their output must depend on their content like any other slot.
    """
    m = MeshDiffusionModule(_cfg()).eval()
    torch.nn.init.normal_(m.denoiser.outc.weight, std=0.05)
    torch.nn.init.normal_(m.denoiser.outc.bias, std=0.05)

    batch = _batch()                       # 12 real slots of 16
    outs = []
    for seed in range(3):
        torch.manual_seed(seed)
        b = {**batch, "x": torch.randn_like(batch["x"])}
        with torch.no_grad():
            outs.append(m._denoise(b["x"], torch.rand(2), b["cond"], None,
                                   b["cond_mask"]))
    unused = torch.stack([o[:, 9, 12:] for o in outs])
    assert unused.std(dim=0).max().item() > 1e-3, (
        "output at unused slots is input-independent -- the model is being "
        "told where the mesh ends instead of predicting it")


def test_slot_budget_is_fixed_at_eval_and_jittered_at_train():
    from src.dataset.mesh_set_dataset import mesh_set_collate_fn

    items = [{"x": torch.zeros(n, 10), "cond": torch.zeros(3, 10), "id": "a",
              "center": torch.zeros(3), "scale": torch.ones(3)} for n in (5, 11)]
    fixed = mesh_set_collate_fn(items, width=64)
    assert fixed["x"].shape[-1] == 64
    assert fixed["x_mask"].sum(1).tolist() == [5, 11]   # supervision survives

    torch.manual_seed(0)
    widths = {mesh_set_collate_fn(items, jitter_to=64)["x"].shape[-1]
              for _ in range(30)}
    assert len(widths) > 1 and min(widths) >= 16 and max(widths) <= 64

    with pytest.raises(ValueError, match="below this batch"):
        mesh_set_collate_fn(items, width=8)


def _gen_ds():
    """Two buildings in the shape MeshSetDataset returns."""
    import numpy as np

    class _Ds:
        ids = ["a", "b"]

        def __len__(self):
            return 2

        def mesh_pair(self, i):
            v = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1.0]])
            f = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            return ((v, f), (v, f))

        def __getitem__(self, i):
            g = torch.Generator().manual_seed(i)
            x = torch.zeros(12, 10)
            x[:, :9] = torch.rand(12, 9, generator=g) - 0.5
            x[:, 9] = 0.5
            return {"x": x, "cond": x.clone(), "id": self.ids[i],
                    "center": torch.zeros(3), "scale": torch.ones(3)}
    return _Ds()


def test_generative_monitor_keys_and_direction():
    """Tier 2 must produce one lower-is-better key for every arm, plus the
    diagnostics that make a face-count collapse visible."""
    from src.eval.mesh_set_eval import run_generative_monitor

    cfg = _cfg()
    cfg.mesh_data.max_faces = 32          # the gate requires budget >= filter
    cfg.mesh_diffusion.slot_budget = 32
    cfg.mesh_diffusion.gen_eval_steps = 3
    m = MeshDiffusionModule(cfg).eval()
    out = run_generative_monitor(m, _gen_ds(), [0, 1], cfg, seed=0)
    for key in ("gen_coord_mse", "gen_presence_acc", "gen_n_faces",
                "gen_n_faces_target"):
        assert key in out, key
    assert out["gen_coord_mse"] >= 0.0
    assert 0.0 <= out["gen_presence_acc"] <= 1.0
    assert "gen_bin_acc" not in out          # continuous arm has no alphabet


def test_generative_monitor_adds_bin_accuracy_on_a_categorical_arm():
    from src.eval.mesh_set_eval import run_generative_monitor

    cfg = _cfg(state="bins", process="d3pm", loss="ce", target="original")
    cfg.mesh_data.max_faces = 32          # the gate requires budget >= filter
    cfg.mesh_diffusion.slot_budget = 32
    cfg.mesh_diffusion.gen_eval_steps = 3
    m = MeshDiffusionModule(cfg).eval()
    out = run_generative_monitor(m, _gen_ds(), [0, 1], cfg, seed=0)
    assert 0.0 <= out["gen_bin_acc"] <= 1.0


def test_monitor_is_reproducible_and_leaves_the_training_rng_alone():
    """Two properties at once.

    Reproducible: the reverse trajectory starts from a random prior, so without
    a fixed seed the monitor moves epoch to epoch on sampling noise and early
    stopping fires on that.

    Non-invasive: seeding globally inside a training loop would also reset the
    TRAINING rng. With the monitor running every epoch that would correlate
    dropout and shuffling across the whole run -- a real contaminant, not a
    curiosity.
    """
    from src.eval.mesh_set_eval import run_generative_monitor

    cfg = _cfg()
    cfg.mesh_data.max_faces = 32          # the gate requires budget >= filter
    cfg.mesh_diffusion.slot_budget = 32
    cfg.mesh_diffusion.gen_eval_steps = 3
    m = MeshDiffusionModule(cfg).eval()
    ds = _gen_ds()

    a = run_generative_monitor(m, ds, [0, 1], cfg, seed=7)
    b = run_generative_monitor(m, ds, [0, 1], cfg, seed=7)
    assert a["gen_coord_mse"] == pytest.approx(b["gen_coord_mse"], rel=1e-9)

    torch.manual_seed(1234)
    before = torch.randn(4)
    torch.manual_seed(1234)
    run_generative_monitor(m, ds, [0, 1], cfg, seed=7)
    after = torch.randn(4)
    assert torch.allclose(before, after), "the monitor perturbed the training RNG"


def test_test_step_computes_no_single_step_loss():
    """Test is the full reverse trajectory only."""
    m = MeshDiffusionModule(_cfg())
    assert m.test_step(_batch(), 0) is None
