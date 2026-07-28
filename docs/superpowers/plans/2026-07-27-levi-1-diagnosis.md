# levi-1 diagnosis — why good metrics gave bad samples

**Date:** 2026-07-27
**Run:** `levi-1`, wandb `1zvfdzhm` (entity `maxim-lod`, project `lod-generation`),
100 epochs / ~100 h wall clock, config `configs/levi1-train.yaml`.
**Diagnostics:** `notebooks/test_generated.ipynb`, cells 14+.

Status at the end of this session: **cause 1 fixed, causes 2 and 3 open.**

---

## 1. Ruled out — do not re-investigate

### The sampler is correct

Feeding an oracle x0 prediction through `sample_zs_from_zt_and_pred` reconstructs
perfectly at every `t0` from 10 to T:

| t0 | vertex RMSE | node acc | vv recall | vf recall |
|---|---|---|---|---|
| 10 … 500 | 0.0000 m | 1.000 | 1.000 | 1.000 |

No posterior bug, no prefactor bug, no discrete-transition bug. The chain also
*attenuates* independent per-step error — injecting a calibrated 0.10 m x0 error
per step yields 0.058 m at the output (~0.6x) at every `t0`. **Error growth is
never the sampler's doing.**

> Gotcha when reproducing: `sample_zs_from_zt_and_pred` applies `F.softmax` to
> `pred_X`/`pred_E`. An oracle must pass large logits — a 5-class one-hot
> softmaxes to 0.47/0.13/0.13/0.13/0.13 and silently becomes a weak predictor.
> This produced a false "the discrete posterior is broken" result mid-session.

### There is no coordinate collapse

`footprint_diag_m` gen/real = **1.03**. Generated footprints are the right size.
An earlier claim of collapse came from eyeballing a single sample and was wrong.

### The t0=500 reconstruction cliff is definitional

At `alpha_bar ~ 1e-4` the input carries zero information about the original
graph, so "reconstruction" is undefined — a *perfect* generative model would also
score `vf_recall ~= the marginal`. Not evidence of a schedule problem.

Schedule reality (`cosine_beta_schedule_discrete`, T=500):

| t | ᾱ_pos (ν=2.5) | ᾱ_x (ν=1.0) | ᾱ_e (ν=1.5) |
|---|---|---|---|
| 250 | **0.920** | 0.495 | 0.716 |
| 300 | 0.815 | 0.343 | 0.551 |
| 350 | 0.636 | 0.205 | 0.365 |
| 400 | 0.386 | 0.096 | 0.187 |
| 450 | 0.129 | 0.025 | 0.053 |
| 500 | 0.0001 | 0.0000 | 0.0000 |

Positions are still 92% intact at t=250. The original sweep grid
`{10, 50, 100, 250, 500}` skipped the entire transition (t=262–456).

### The denoiser is healthy per-step

Single-step x0 error beats copy-the-input past t≈60 and converges to exactly the
CoM ceiling at high t — which *is* the optimal prediction as t→T. Single-step
vertex error at t=1 is **0.106 m** on ~13 m buildings.

---

## 2. Confirmed causes

### Cause 1 — CoM leak through Off slots  ✅ FIXED

Under se2 the zero-CoM projection spans **all** `n_max` slots in xy, so
`sum(real) == -sum(off)`. Off positions are pinned to 0 in the *target* only, not
in the chain, so drift there forces the real centroid to `-sum(off)/n_real`.

Measured at t0=10:

| quantity | value |
|---|---|
| off-slot drift | 1.125 m |
| xy shift observed | 1.033 m |
| xy shift predicted by the identity | 1.180 m |
| direction cosine | **+0.82** |

Magnitude within 12% *and* direction, stable across all t0. Confirmed.

### Cause 2 — Exposure bias  ⬜ OPEN

`shape_rmse / single_step_error` at the same t:

| t0 | shape error | single-step | compounding |
|---|---|---|---|
| 10 | 0.828 m | 0.106 m | **7.8x** |
| 50 | 1.568 m | 0.121 m | **12.9x** |
| 100 | 1.510 m | 0.198 m | 7.6x |
| 250 | 2.258 m | 1.004 m | 2.2x |

~1.23x growth per step, and **worst in the low-t regime** where per-step error is
smallest and the chain runs the most steps. Since the sampler attenuates
independent error to 0.6x, this is a feedback loop: the network is accurate on
properly-noised inputs and inaccurate on its own trajectory.

This is a train/inference mismatch. It will not yield to loss reweighting, and it
will not yield to more epochs.

### Cause 3 — z-bias  ⬜ OPEN, unexplained

`translation_m` is 3D (1.230 m), `xy_shift_observed_m` is horizontal (1.033 m),
so the vertical shift is `sqrt(1.230^2 - 1.033^2) = 0.67 m`.

**z is not CoM-constrained under se2** — it is absolute, standardised by
`z_shift`. So cause 1 cannot explain it. Corroborated by `height_m` gen/real =
**1.58** in the distribution stats. Untouched this session.

---

## 3. Metric traps

- **`val_coord_mse` is in scaled units.** Multiply by `coord_scale` (4.792) for
  metres: the reported 0.0455 was really **~1.02 m** RMSE per axis.
- **`_eval_step` scores a fixed `t = T//2`** with the real graph as noising
  anchor — teacher-forced one-shot denoising, never the reverse chain. Nothing in
  the levi-1 dashboard measured generation, yet it is what `early_stopping` and
  `checkpoint` monitor. `val_real_precision = 0.995` means "tell vertex from off
  in a half-noised real building", which is trivial.
- **Aggregate `edge_ce` is meaningless** at 98.5% `EDGE_OFF`. Use per-class
  precision/recall. `EDGE_VF` is 1.09% of entries and collapsed unnoticed.
- Run marginals for reference:
  `x = [0.146 vertex, 0.012 ground, 0.023 roof, 0.081 wall, 0.738 off]`,
  `e = [0.9847 off, 0.0045 vv, 0.0109 vf]`. Mean ~26 real nodes vs `n_max=100`.

---

## 4. Shipped this session

Nothing committed. `91 passed, 1 skipped`.

### Coordinate loss — real-node normalisation + Off anchor (`diffusion.py`)

```python
real = (1.0 - X0[..., -1]).unsqueeze(-1)
sq_err = (R_pred - R0) ** 2
coord_loss = (sq_err * real).sum() / (real.sum() * 3).clamp(min=1.0)
off_anchor = (sq_err * (1 - real)).sum() / ((1 - real).sum() * 3).clamp(min=1.0)
total = self.lambda_pos * (coord_loss + self.off_anchor_weight * off_anchor) + ...
```

The anchor is **not optional**: dropping Off from the loss removes the only thing
holding those positions at the origin, which is exactly cause 1.
`off_anchor_weight` defaults to 0.1, wired through `ModelConfig` and
`setup_utils`, saved in hparams; `0.0` is the ablation.

`train_coord_mse` now logs the real-node MSE alone (anchor rides in the total),
so it is comparable to `val_coord_mse` — but **not** to levi-1's series.

### Generative eval — reference cap + stage logging (`eval/callback.py`)

The slowness was not only converting every graph in test *and* train (O(1e5)
each). `kernel_mmd` is **O(n^2) in the reference count**:
`_median_bandwidth(vstack([gen, ref]))` and `k(ref, ref)` each build an n×n
float64 gram matrix — ~80 GB at n=100k. `GenerativeEvalConfig.ref_max_samples =
2000` caps both, with a seeded sorted subsample so the reference distribution is
identical across runs. Added a `_stage()` contextmanager logging
`[gen-eval] <name> ... / done in Xs` around each arm.

### Notebook diagnostics (`notebooks/test_generated.ipynb`, cells 14+)

`graph_stats` (shared by generated and real, bypasses the converter so collapsed
samples can't vanish) · `reconstruct_from` · reconstruction sweep with per-class
precision/recall and vertex/face RMSE split · `single_step_error` with
copy-input and CoM baselines · `error_decomposition` (translation vs shape, with
the exact CoM-leak prediction and `compounding`) · distribution comparison vs the
test split · class-marginal check · overfit-100 test.

### Tests

`tests/test_coord_loss_masking.py` — analytic stub denoiser pins padding-width
invariance (the old formula drifted >2.5x between `n_max` 20 and 60) and exact
anchor weighting.

### Incidental

Restored `import lightning as L` in `datamodule.py:7` — the `L` had been clipped,
breaking every `src.dataset` import and the whole test suite.

---

## 5. Left to do

### Before launching a retrain

- **`lambda_pos=3.0` is now an effective ~4x stronger position weight** (the
  denominator went from 100 slots to ~26). Intended, but it is the first knob to
  turn if training destabilises — see the y-branch overflow history.
- **`gradient_clip_val` is still `null`** in `levi1-train.yaml`, while the
  earlier debug runs used `1`. Put it back.
- **Nothing monitors sample quality.** Checkpoint selection still uses a metric
  orthogonal to it.

### Open work, in priority order

1. **Self-conditioning** — the fix for cause 2, which is the larger term. Feed
   the previous step's `x0_hat` into the network alongside `z_t`, with 50%
   dropout during training so it still works unconditioned. ~15 lines across
   `regnn.py` and `_shared_step`. Standard remedy for this mismatch in the
   DiGress/MiDi lineage. **Awaiting go-ahead.**
2. **Chase the z-bias** (cause 3). Clean signature to work from: 0.67 m
   systematic vertical shift, `height_m` gen/real = 1.58.
3. **Periodic generative validation** (16–32 samples every N epochs: drop rate,
   `valid_fraction`, face coherence) and monitor *that* for checkpointing.
   Replace the fixed `t = T//2` eval with a sweep over `t`.
4. **Re-run the reconstruction sweep on t ∈ {250, 300, 350, 400, 450}** — the
   transition region levi-1's grid skipped entirely.
5. **Untested levers already built:** `dist_embed` was `raw`/`Identity` in
   levi-1; the sinusoidal/Bessel lift exists and has never been run. `n_max=100`
   against a 26-node mean is 74% waste, quadratic in the edge tensor — MiDi-style
   `n ~ p(n)` sampling is the bigger version of this.

### On "just train 10x longer"

Hold. A model that is accurate single-step and diverges over its own chain has a
train/inference mismatch. More epochs sharpen the on-manifold predictor and leave
the off-manifold behaviour untouched. Fix cause 2 first, then decide.
