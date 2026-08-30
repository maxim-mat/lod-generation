"""End-to-end smoke: the Phase A arm must train a step and sample a mesh.

Synthetic on-disk data is out of scope here -- these tests build a
`MeshDiffusionModule` directly and feed it a hand-made batch, which is the
smallest thing that proves the four subsystems are wired to each other.
"""
import math

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


def test_warmup_ramps_on_the_step_interval():
    """Lightning defaults `interval` to "epoch". A step-scaled warmup left on
    that default would ramp over 500 EPOCHS -- ~63k steps instead of 500. It
    trains, badly and silently, so the interval is asserted here."""
    cfg = _cfg()
    cfg.training.warmup_steps = 50
    cfg.training.lr = 1e-3
    cfg.training.lr_scheduler = "none"
    m = MeshDiffusionModule(cfg)
    out = m.configure_optimizers()
    assert out["lr_scheduler"]["interval"] == "step"

    opt, sched = out["optimizer"], out["lr_scheduler"]["scheduler"]
    seen = []
    for _ in range(120):
        seen.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert seen[0] == pytest.approx(1e-3 / 50, rel=1e-6)   # ramping from ~0
    assert seen[49] == pytest.approx(1e-3, rel=1e-6)       # at peak by step 50
    assert seen[119] == pytest.approx(1e-3, rel=1e-6)      # then flat


def test_no_warmup_and_no_schedule_returns_a_bare_optimizer():
    m = MeshDiffusionModule(_cfg())
    assert isinstance(m.configure_optimizers(), torch.optim.Optimizer)


def test_step_scheduler_is_supported_like_the_other_branches():
    """`TrainingConfig` advertises "step" and mesh_transformer/mesh_vqvae both
    implement it; this module used to raise on it."""
    cfg = _cfg()
    cfg.training.lr_scheduler = "step"
    out = MeshDiffusionModule(cfg).configure_optimizers()
    assert out["lr_scheduler"]["interval"] == "epoch"


def test_ema_is_checkpointed_and_lags_the_live_weights():
    cfg = _cfg()
    cfg.mesh_diffusion.ema_decay = 0.999
    cfg.mesh_diffusion.ema_start_step = 0
    m = MeshDiffusionModule(cfg)
    assert any(k.startswith("ema.") for k in m.state_dict()), \
        "EMA must ride in state_dict -- checkpoints are selected on a metric computed with it"

    # Untouched EMA is still the initialisation, so sampling must NOT use it.
    assert int(m.ema.n_averaged) == 0
    assert m._eval_net() is m.denoiser

    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    for i in range(5):
        opt.zero_grad(); m.training_step(_batch(), i).backward(); opt.step()
        m.ema.update_parameters(m.denoiser)
    assert int(m.ema.n_averaged) == 5
    m.eval()
    assert m._eval_net() is m.ema.module          # now it is worth using
    live = next(m.denoiser.parameters())
    shadow = next(m.ema.module.parameters())
    assert not torch.allclose(live, shadow), "EMA never diverged from the live weights"


def test_ema_can_be_disabled():
    cfg = _cfg()
    cfg.mesh_diffusion.ema_decay = None
    m = MeshDiffusionModule(cfg).eval()
    assert m.ema is None and m._eval_net() is m.denoiser


def test_ema_decay_ramps_in():
    """A constant 0.999 leaves a fresh EMA 90% initialisation after 100 updates.
    The ramp tracks the live weights early and tightens as it earns length."""
    from src.models.mesh_diffusion_module import _ramped_ema

    fn = _ramped_ema(0.999)
    ema = [torch.zeros(1)]
    for n in range(200):
        fn(ema, [torch.ones(1)], n)
    assert ema[0].item() > 0.95        # a constant-0.999 EMA would be ~0.18 here


def test_monitor_scores_ema_and_live_weights_differently():
    """Dual logging: the same trajectory scored from both weight sets.

    EMA leaves the gradients untouched, so the live-weight reading of an EMA
    run IS the no-EMA result -- which is why EMA on/off costs no arm and only
    the decay LENGTH needs separate runs.
    """
    from src.eval.mesh_set_eval import run_generative_monitor

    cfg = _cfg()
    cfg.mesh_data.max_faces = 32
    cfg.mesh_diffusion.slot_budget = 32
    cfg.mesh_diffusion.gen_eval_steps = 3
    cfg.mesh_diffusion.ema_decay = 0.9
    cfg.mesh_diffusion.ema_start_step = 0
    m = MeshDiffusionModule(cfg)
    torch.nn.init.normal_(m.denoiser.outc.weight, std=0.05)

    opt = torch.optim.AdamW(m.denoiser.parameters(), lr=1e-2)
    for i in range(6):
        opt.zero_grad(); m.training_step(_batch(), i).backward(); opt.step()
        m.ema.update_parameters(m.denoiser)
    m.eval()

    ds = _gen_ds()
    ema = run_generative_monitor(m, ds, [0, 1], cfg, seed=3, weights="ema")
    live = run_generative_monitor(m, ds, [0, 1], cfg, seed=3, weights="live")
    assert ema["gen_coord_mse"] != live["gen_coord_mse"], \
        "the two weight sets produced identical scores -- the switch is inert"
    assert m.eval_weights == "ema", "using_weights must restore the previous setting"


def test_optimizer_excludes_the_ema_copy():
    """The EMA is exactly as large as the denoiser; handing it to the optimizer
    is harmless today only because its grads stay None."""
    m = MeshDiffusionModule(_cfg())
    opt = m.configure_optimizers()
    opt = opt if isinstance(opt, torch.optim.Optimizer) else opt["optimizer"]
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n_opt == sum(p.numel() for p in m.denoiser.parameters())
    assert n_opt < sum(p.numel() for p in m.parameters())


# --- the x0 diagnostic, and min-SNR weighting --------------------------------

class _Const(torch.nn.Module):
    """A denoiser that returns a fixed tensor, so x0 is exactly known."""

    def __init__(self, out):
        super().__init__()
        self.out = out

    def forward(self, x, t, cond, mask, cond_mask):
        return self.out


def _logged(module, batch, stage="train"):
    """Run one _shared_step, capturing what it logs instead of a real logger."""
    seen = {}
    module.log = lambda name, value, **kw: seen.__setitem__(name, float(value))
    module._shared_step(batch, stage)
    return seen


def test_coord_err_bins_is_a_per_channel_mean_on_the_bin_grid():
    """A perfect x0 estimate reads 0 bins; a saturated one cannot exceed the grid.

    The regression: the divisor omitted the nine coordinate channels, so the
    number came out 9x too large -- 378 bins on a 128-bin grid, which is what
    made it look like a broken metric rather than a diagnosis.
    """
    m = MeshDiffusionModule(_cfg(target="original"))
    batch = _batch()
    # target="original" means the denoiser's output IS the x0 estimate, so a
    # stubbed identity denoiser gives an exactly-known answer.
    m.denoiser = _Const(batch["x"].clone())
    seen = _logged(m, batch)
    assert seen["train_coord_err_bins"] == pytest.approx(0.0, abs=1e-4)

    m.denoiser = _Const(-batch["x"].clone())
    seen = _logged(m, batch)
    # Worst case is the full clamped box (width 1.0) x (num_bins - 1).
    assert 0.0 < seen["train_coord_err_bins"] <= 127.0


def test_coord_err_bins_is_bucketed_by_per_sample_t():
    """Four t buckets, and a batch must land in the bucket of its OWN t.

    The regression: the bucket came from `t.mean()` over the batch and the x0
    conversion from `float(t[0])`, so both stood in one row's noise level for
    every row.
    """
    m = MeshDiffusionModule(_cfg())
    torch.manual_seed(0)
    seen = _logged(m, _batch(b=8))
    buckets = [k for k in seen if k.startswith("train_coord_err_bins_t")]
    assert buckets, "no per-t buckets logged"
    assert all(k[-1] in "0123" for k in buckets)
    assert len(buckets) > 1, "8 uniform draws collapsed into a single bucket"


def test_min_snr_gamma_changes_the_loss_it_does_not_rescale_it():
    torch.manual_seed(0)
    batch = _batch()
    plain = MeshDiffusionModule(_cfg())
    torch.manual_seed(1)
    a = float(plain._shared_step(batch, "val"))
    weighted = MeshDiffusionModule(_cfg(min_snr_gamma=5.0))
    weighted.load_state_dict(plain.state_dict())
    torch.manual_seed(1)
    b = float(weighted._shared_step(batch, "val"))
    assert a != pytest.approx(b), "min_snr_gamma had no effect"
    assert 0.1 * a < b < 10 * a, "reweighting turned into a rescaling"


def test_min_snr_gamma_is_rejected_where_it_is_undefined():
    """It is an alpha_bar construction: no flow matching, no categorical arms."""
    from src.utils.config import validate_combination
    with pytest.raises(ValueError, match="min_snr_gamma"):
        validate_combination(_cfg(process="flow", target="velocity",
                                  min_snr_gamma=5.0))


def test_adaln_arm_trains_and_samples():
    """c7's axes end to end: the module builds, steps and generates."""
    m = MeshDiffusionModule(_cfg(denoiser="transformer", time_cond="adaln"))
    loss = m.training_step(_batch(), 0)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    # Step 0 reaches the head only: `outc` is zero-initialised, so nothing
    # upstream of it has a gradient path yet. True of the additive arm too.
    assert m.denoiser.outc.weight.grad.abs().sum() > 0
    assert m.denoiser.blocks[0].emb[1].weight.grad.abs().sum() == 0

    # Once the head opens, the zero-init modulation must start receiving
    # gradient -- otherwise the gates never open and the network stays the
    # identity forever, which is the one way adaLN-Zero can fail silently.
    opt = torch.optim.AdamW(m.denoiser.parameters(), lr=1e-3)
    opt.step()
    opt.zero_grad()
    m.training_step(_batch(), 0).backward()
    assert m.denoiser.blocks[0].emb[1].weight.grad.abs().sum() > 0

    out = m.eval().generate(_batch(), n_steps=3)
    assert out.shape == (2, 10, 16) and torch.isfinite(out).all()


# --- the monitor has to be able to see an empty mesh --------------------------

def _monitor_cfg():
    cfg = _cfg()
    cfg.mesh_data.max_faces = 32
    cfg.mesh_diffusion.slot_budget = 32
    cfg.mesh_diffusion.gen_eval_steps = 3
    return cfg


class _Fixed(torch.nn.Module):
    """A model stub whose `generate` returns a chosen sample verbatim."""

    def __init__(self, module, presence):
        super().__init__()
        self.m = module
        self.presence = presence

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.m, name)

    def generate(self, batch, n_steps=None, scaffold=None):
        out = batch["x"].clone()          # exactly the target coordinates
        out[:, 9] = self.presence
        return out


def test_monitor_scores_an_empty_mesh_as_the_worst_case_not_the_best():
    """An all-absent sample generates NO MESH. It must not score better than a
    correct one -- under `gen_coord_mse` alone it scored a perfect 0.0, because
    that metric masks by the target and never reads the presence channel.
    """
    from src.eval.mesh_set_eval import run_generative_monitor

    cfg = _monitor_cfg()
    m = MeshDiffusionModule(cfg).eval()
    ds = _gen_ds()

    good = run_generative_monitor(_Fixed(m, 0.5), ds, [0, 1], cfg, seed=0)
    empty = run_generative_monitor(_Fixed(m, -0.5), ds, [0, 1], cfg, seed=0)

    assert math.isfinite(empty["gen_chamfer_m"]), "empty sample must not be nan"
    assert empty["gen_chamfer_m"] > good["gen_chamfer_m"], (
        f"empty mesh scored {empty['gen_chamfer_m']} against a correct "
        f"sample's {good['gen_chamfer_m']}")
    # And the old metric is exactly as blind as it was -- kept as a diagnostic,
    # which is why it must not be what selects the checkpoint.
    assert empty["gen_coord_mse"] == pytest.approx(good["gen_coord_mse"])


def test_monitor_counts_buildings_not_batches():
    """`n_eval` came from the number of CHUNKS, so it read 1 for 8 buildings
    and clobbered tier 3's own `val_n_eval` of 16 in the same namespace."""
    from src.eval.mesh_set_eval import run_generative_monitor

    cfg = _monitor_cfg()
    cfg.mesh_diffusion.eval_batch_size = 8     # both buildings in ONE chunk
    m = MeshDiffusionModule(cfg).eval()
    out = run_generative_monitor(m, _gen_ds(), [0, 1], cfg, seed=0)
    assert out["n_eval"] == 2.0


def test_the_shipped_monitor_is_the_geometric_one():
    """Early stopping and checkpointing must select on the metric that can see
    a face-count collapse, not on the one that cannot."""
    from omegaconf import OmegaConf

    base = OmegaConf.load("configs/mesh-diff-base.yaml")
    assert base.training.early_stopping.monitor == "val_gen_chamfer_m"
    assert base.training.checkpoint.monitor == "val_gen_chamfer_m"


def test_a_real_fit_binds_early_stopping_and_checkpointing_to_the_monitor(tmp_path):
    """End to end: the shipped config's monitor must exist in a real fit.

    `EarlyStopping(strict=True)` raises when its metric is missing, and
    ModelCheckpoint's filename template interpolates it -- so a monitor that is
    named in the config but never logged kills the run on epoch 1 rather than
    degrading quietly. Unit-testing `run_generative_monitor` does not cover the
    wiring; this does.
    """
    import lightning as L
    from omegaconf import OmegaConf
    from src.eval.mesh_set_eval import MeshSetEvalCallback
    from src.utils.setup_utils import create_callbacks

    base = OmegaConf.load("configs/mesh-diff-base.yaml")
    monitor = base.training.early_stopping.monitor

    cfg = _cfg()
    cfg.mesh_data.max_faces = 32
    cfg.mesh_diffusion.slot_budget = 32
    cfg.mesh_diffusion.gen_eval_steps = 3
    cfg.mesh_diffusion.n_val_gen = 2
    cfg.mesh_diffusion.every_n_epochs = 0          # tier 3 off; tier 2 must survive
    cfg.training.early_stopping.enabled = True
    cfg.training.early_stopping.monitor = monitor
    cfg.training.early_stopping.mode = "min"
    cfg.training.checkpoint.enabled = True
    cfg.training.checkpoint.monitor = monitor
    cfg.training.checkpoint.mode = "min"
    cfg.training.checkpoint.filename = "{epoch:02d}-{" + monitor + ":.5f}"

    ds = _gen_ds()

    class _DM(L.LightningDataModule):
        train_dataset = val_dataset = test_dataset = ds

        def _dl(self):
            from src.dataset.mesh_set_dataset import mesh_set_collate_fn
            from functools import partial
            return torch.utils.data.DataLoader(
                ds, batch_size=2, collate_fn=partial(
                    mesh_set_collate_fn, multiple_of=8,
                    width=cfg.mesh_diffusion.slot_budget,
                    num_bins=cfg.mesh_data.num_bins))

        train_dataloader = val_dataloader = test_dataloader = _dl

    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    # Only the two under test: the progress bar and the lr monitor need a
    # logger/progress-bar-enabled Trainer and are irrelevant here.
    callbacks = [c for c in create_callbacks(cfg, tmp_path)
                 if isinstance(c, (EarlyStopping, ModelCheckpoint))]
    assert len(callbacks) == 2, f"config did not build both callbacks: {callbacks}"
    callbacks.append(MeshSetEvalCallback(cfg, tmp_path, seed=0))
    trainer = L.Trainer(max_epochs=2, accelerator="cpu", devices=1,
                        callbacks=callbacks, logger=False,
                        enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(MeshDiffusionModule(cfg), datamodule=_DM())

    assert monitor in trainer.callback_metrics, (
        f"{monitor} never reached callback_metrics; early stopping would raise. "
        f"Logged: {sorted(trainer.callback_metrics)}")
    assert math.isfinite(float(trainer.callback_metrics[monitor]))
    written = list(tmp_path.rglob("*.ckpt"))
    assert written, "no checkpoint was written"
    assert monitor.split("_", 1)[1][:3] in written[0].name or "epoch" in written[0].name
