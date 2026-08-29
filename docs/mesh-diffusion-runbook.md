# Mesh-diffusion experiments: running guide

How to run the `config_set: mesh_diffusion` grid end to end. For *why* the
branch exists and what it gives up, read
`docs/research/2026-08-25-mesh-set-diffusion-notes.md`; for the design
decisions, `docs/superpowers/plans/2026-08-25-mesh-set-diffusion.md`.

**18 arms, four stages, ~4 days of GPU time.** Stages are sequential: each
one's configs must be edited to the previous one's winner before launching.
That edit is the part to get right — everything else is one command.

---

## 1. Preflight

```bash
python -m pytest tests/ -q                       # expect 609 passed, 1 skipped
SMOKE=1 ./run_experiments.sh --diff-stage 1      # 4 arms, 1 epoch each, ~2 min
```

`--smoke` runs 1 epoch on 20 files with no logger, into a scratch dir deleted on
exit. It creates no wandb run and writes no checkpoint. Run it after *any* config
edit — it catches a bad key in minutes instead of at hour four.

Also check:

- **Data.** `data/The Hague/mini_cleaner/{LOD1,LOD2}` → 5076 pairs after the
  `max_faces: 200` filter (train 4060 / val 507 / test 509). The scan takes ~50 s
  per process, so `mesh_data.num_workers: 10` pays for itself on the real runs.
- **wandb.** The base config logs to `wandb_entity: maxim-lod`,
  `project: lod-generation`, `experiment_name: initial-runs`. Confirm with
  `python -c "import wandb; print(wandb.Api().viewer.username)"`.
- **GPU.** Peak VRAM is 508 MiB at batch 32 (235 MiB at batch 4). Any CUDA card
  will do; the model is launch-bound, not compute-bound.

---

## 2. Cost, measured

Batch 32, slot budget 200, 127 steps/epoch, `max_epochs: 1000`. Times are the
ceiling — early stopping (`patience: 100` on `val_gen_chamfer_m`) will usually
cut them short.

| stage | arms | ms/step | hours |
|---|---|---|---|
| **1** | a1 4.4 h, a2 3.4, **a3 10.3**, **a4 11.9** | 78 / 49 / 245 / 289 | **30.0** |
| **2** | b1 4.3, b2 4.4, b3 4.4, b4 4.3, b5 4.6, b6 4.8, b7 4.9 | 73–91 | **31.7** |
| **3** | c1 4.4, c2 4.3, c3 4.4, c4 4.4, c5 4.4 | 74–79 | **21.9** |
| **revisit** | r1 3.3, **r2 10.2** | 47 / 241 | **13.5** |
| | | | **≈97 h (4.0 days)** |

**The Hungarian arms (a3, a4, r2) cost 3–4× the rest**, and that is a batch-size
effect, not an architectural one: `hungarian_match` runs one scipy
`linear_sum_assignment` per *sample*, so it scales with batch while the network
stays launch-bound. At batch 4 they were comparable to everything else. If you
need them cheaper, lower `training.batch_size` for those arms specifically — the
network cost will not rise.

---

## 3. Running the stages

```bash
nohup ./run_experiments.sh --diff-stage 1 > stage1.out 2>&1 &
```

Then read the results, **edit the next stage's configs** (section 4), and:

```bash
nohup ./run_experiments.sh --diff-stage 2       > stage2.out 2>&1 &
nohup ./run_experiments.sh --diff-stage 3       > stage3.out 2>&1 &
nohup ./run_experiments.sh --diff-stage revisit > revisit.out 2>&1 &
```

The runner merges each arm config over `configs/mesh-diff-base.yaml` into
`logs/<run>/<arm>.merged.yaml` and launches that, so the merged file is the exact
config a run saw. `--keep-going` is on by default: one arm failing does not cost
the others their slot. Per-arm logs land in `logs/experiments-<timestamp>/`.

Useful overrides, appended to any launch:

```bash
./run_experiments.sh --diff-stage 1 mesh_data.max_files=200   # subset, for a fast read
./run_experiments.sh --diff-stage 1 training.max_epochs=200   # shorter ceiling
./run_experiments.sh --diff-stage 3 mesh_diffusion.x0_clip=null   # ablate the clamp
```

---

## 4. The edit between stages — the part to get right

Every stage-2 and stage-3 arm ships at **a1's** stage-1 axes
(`order: morton`, `pos_embed: sinusoidal`, `denoiser: unet`). They are valid
runs as shipped; they just answer the question at a1's settings rather than at
the measured winner.

**If a1 wins stage 1:** launch stage 2 unchanged.

**If a2 wins:** set `denoiser: transformer` in all seven b configs.

**If a3 or a4 wins (both `order: none`): stop and read this.** `loss: ce` is
scored slot-to-slot and requires `order: morton` (plan D4, enforced by
`validate_combination`), so **b4–b7 have no unordered form and cannot be
ported.** You will get a startup `ValueError`, not a bad run. Choose one:

- keep stage 2 ordered anyway — it then answers "which objective wins *given* a
  canonical order", still a real question, just not on the stage-1 winner; or
- run only b1–b3 unordered with `loss: hungarian`, and record that the four
  on-grid arms are unreachable. That is a result about the grid, not a gap to
  route around.

The same conditional governs `r2-loss` in the revisit stage.

**Stage 3** ships at a1+b1 axes; edit to stage 2's winner. `c3-clamp-soft` is
only meaningful if the winner is b4 or b5 (it ablates the clamping trick, which
only exists for a continuous process reading out categorically). Skip it for
b6/b7 (no clamp in a categorical chain) and for b1/b2/b3 (no alphabet).

After any edit: `SMOKE=1 ./run_experiments.sh --diff-stage N`.

---

## 5. What to watch

**`val_gen_chamfer_m`** — the monitor. Free-running reverse trajectory on 8
fixed buildings at 20 steps, taken through `faces_to_mesh` and scored in
metres, every validation epoch. Early stopping and checkpointing both read it,
`mode: min`, and it is comparable across every arm. A sample that generates no
mesh scores the LOD1 box diagonal — tens of metres — rather than nan.

Read it against `val_chamfer_m` (tier 3) only within an arm: this runs at 20
reverse steps and tier 3 at 50, so the monitor is the pessimistic one.

**`val_gen_coord_mse`** — a diagnostic, and *not* a selection metric. It masks
by the target's face mask and never reads the predicted presence channel, so it
cannot see face count: a model that marks every slot absent scores a perfect
0.0. It was the monitor until that was measured. Useful for watching
coordinate accuracy in isolation, useless for ranking checkpoints.

**`val_gen_coord_mse_live`** — the same trajectory from the raw (non-EMA)
weights. EMA never touches the gradients, so this *is* the no-EMA result at no
extra training cost. If the two rank a1–a4 identically, the EMA confound is
closed; if not, that is a result the paired-diffusion literature does not report.

**`val_gen_n_faces` vs `val_gen_n_faces_target`** — face count is genuinely
predicted now (nothing tells the model where the mesh ends), so this is where a
presence channel that collapses to all-absent, or fires across the whole slot
budget, becomes visible. An untrained model marks ~107 of 200 slots absent and
lands ~37 faces against a median target of 20.

**`val_bin_acc_t3`** on b6/b7 only — the high-noise bucket, where plan D11's
ordinal-alphabet claim lives. Watch it from epoch 1; if b7 does not beat b6 the
ordinality argument is wrong for this data, which is a result.

**Every 10 epochs**, the geometric tier adds `val_chamfer_m`, `val_vol_iou`,
`val_watertight_gen`, and the **`val_gt_*` ceiling** — the ground truth pushed
through the same snap/weld pipeline. `gt_chamfer_m` is ~0.016 m and is **not
zero**: read every generated number against that row, never against 0.

---

## 6. Gotchas

**A discontinuity in the monitor around epoch 8 is expected.**
`ema_start_step: 1000` is ~8 epochs at batch 32; before it, sampling uses live
weights. The jump when the EMA takes over is not a bug.

**`EarlyStopping(strict=True)` raises on a missing metric.** The monitor is
produced by `MeshSetEvalCallback`, which is now always attached —
`every_n_epochs` gates only the *geometric* tier. Do not set
`mesh_diffusion.n_val_gen: 0` to "disable" the monitor; it is rejected at the
gate precisely because it would kill the run rather than disable anything.

**`EarlyStopping(check_finite=True)` is also on.** A NaN monitor stops the run
immediately. That is usually what you want, but it means a diverged arm ends
early rather than burning its slot.

**Do not disable `x0_clip` casually.** Without it an untrained ε-model turns the
DDIM step into a pure rescaling that telescopes to ×157: samples leave the box,
`quantize` clips every axis to bin 0 or 127, and the mesh welds to the 8 corners
of the normalized box. The monitor reads 23216 instead of 0.36. It is exposed as
`x0_clip: null` purely so the ablation is runnable.

**The AR comparison is not controlled.** Corpus, split and seed match the
mesh-v3/v4 arms; batch (32 vs 16 effective) and lr (1e-3 vs 5e-5) do not.
Report stage 4 as best-effort against best-effort.

**Smoke mode overrides more than it looks.** It forces `max_epochs=1`,
`max_files=20`, no logger, a scratch save dir, `num_workers=0`, and a shrunken
eval (`n_test=4`, `eval_steps=5`). It answers "does this config run", never
"how well does it score".

---

## 7. The arms

**Stage 1 — denoiser and loss regime.** Fixed: `state: continuous`,
`process: ddpm`, `target: original`, `presence_weight: 0.383`.

The objective here was `target: noise` until run `mesh-diff-a1-unet-mse`
showed the epsilon loss decoupled from geometry — it fell 5–6x below the
echo-the-input floor while chamfer sat at 0.93 m against a 0.020 m ceiling,
because a uniform epsilon MSE is an x0 objective weighted by an SNR spanning
~1e4 to 4e-5. `target: original` is in x0 units at every t. Epsilon is now
ablated by `b1-eps` rather than assumed.

| arm | order | pos_embed | loss | denoiser | question |
|---|---|---|---|---|---|
| a1-unet-mse | morton | sinusoidal | mse | unet | the reference every other arm is read against |
| a2-tf-mse | morton | sinusoidal | mse | transformer | convolution against attention at matched order — *not* a test of the sort, which is a1 vs a4 |
| a3-tf-hungarian | none | none | hungarian | transformer | is the set formulation worth a Hungarian solve per sample? |
| a4-unet-hungarian | none | none | hungarian | unet | what does the canonical sort buy, given file order already has 72.4% adjacency? |

**Stage 2 — state, corruption, readout.** Fixed at stage 1's winner.

| arm | state | process | loss | target | on-grid | question |
|---|---|---|---|---|---|---|
| b1-eps | continuous | ddpm | mse | **noise** | no | is ε-prediction worse conditioned than x0? (`presence_weight` 1.0, its regime's parity) |
| b2-flow | continuous | flow | mse | velocity | no | do straight paths beat a strided DDPM at 50 NFE? |
| b3-quant-mse | quantized | ddpm | mse | original | no | **control** — snapping the target alone, readout unchanged |
| b4-quant-ce | quantized | ddpm | ce | original | **yes** | categorical head on 9 coordinate channels |
| b5-onehot-ce | onehot | ddpm | ce | original | **yes** | the reference's scheme exactly, 9×128 indicator channels |
| b6-d3pm-uniform | bins | d3pm | ce | original | **yes** | categorical corruption, nominal transition |
| b7-d3pm-gauss | bins | d3pm | ce | original | **yes** | categorical corruption, **ordinal** transition (D11) |

Read **b3 against b4** first: identical target, differing only in readout, so
their gap *is* the value of the categorical head. Then b4 vs b5 (does one-hot
earn its width — it is free on both compute and memory, so only quality is at
stake), and b6 vs b7 (D11's ordinality claim).

**Stage 3 — sampling-time levers**, on stage 2's winner.

| arm | change | question |
|---|---|---|
| c1-scaffold | `scaffold.enabled: true` | does the LOD1 legal region help? |
| c2-guidance | `guidance: 3.0` | plausible meshes, but plausible answers to *this* LOD1? |
| c3-clamp-soft | `x0_clamp: soft` | **only if the winner is b4/b5** — is the clamping trick load-bearing? |
| c4-ema-short | `ema_decay: 0.99` | EMA length sweep, short end |
| c5-ema-long | `ema_decay: 0.9999` | EMA length sweep, long end (Palette's value) |

c4 / base 0.999 / c5 is a three-point sweep; each run's own `*_live` curve is the
zero-length end of the same axis, so EMA on/off needs no arm.

**Revisit — the check on a greedy grid.**

| arm | change | question |
|---|---|---|
| r1-denoiser | flip the denoiser under stage 2's winner | did stage 1's backbone choice survive the objective change? |
| r2-loss | the set regime under stage 2's winner | did stage 1's loss choice survive? **Illegal if the winner's loss is `ce`.** |

**Stage 4 — no training.** Score the stage-3 winner and the five AR checkpoints
(`mesh-v3-coord-sin`, `-amt-sin`, `-opt-coord-sin`, `-opt-amt-sin`,
`mesh-v4-scratch-sin`) plus the LOD1 passthrough floor
(`src/eval/lod1_baseline.py`) on the same test indices with the same
`mesh_metrics` call, and fill in the table in section 5 of the research note.
Include the **NFE** column: an AR sample is ~1800 *sequential* forward passes, a
50-step diffusion sample is 50 (100 with guidance). A speed claim without it is
not a claim.

---

## 8. Knobs you may actually want to turn

| knob | shipped | note |
|---|---|---|
| `training.batch_size` | 32 | the U-Net is launch-bound to batch 256; 32 was chosen with lr, not against VRAM |
| `training.lr` | 1e-3 | from a 250-step probe: 5e-5 dropped the loss 51%, 1e-3 dropped it 91%. That probe ranks *early* progress and does not settle 1e-3 vs 3e-4 at convergence |
| `training.warmup_steps` | 500 | step interval, not epoch. Only `mesh_diffusion` honours this field |
| `training.max_epochs` | 1000 | ~127k updates. The dominant cost, and a guess |
| `mesh_diffusion.slot_budget` | 200 | must be ≥ `mesh_data.max_faces`, multiple of 8. Costs nothing measurable — the model is launch-bound |
| `mesh_diffusion.n_val_gen` / `gen_eval_steps` | 8 / 20 | the monitor's size. ~10% of an epoch |
| `mesh_diffusion.every_n_epochs` | 10 | geometric tier only |
| `mesh_diffusion.ema_decay` | 0.999 | `null` disables; the `*_live` metric already gives you that comparison |
| `mesh_diffusion.presence_weight` | per-regime | **do not flatten this.** Calibrated so presence and coordinates start at parity in each arm's loss regime; a flat value makes b3-vs-b4 differ 18× in effective presence weight. `test_shipped_configs_equalise_presence_weight` pins it |
