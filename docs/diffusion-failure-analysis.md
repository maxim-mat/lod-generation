# Why the Levi-graph diffusion model fails at unconditional generation

Diagnostic report, 2026-08-05. Branch `main`, no code changed.
Figures and raw numbers: `outputs/diagnostics/`.

---

## Verdict up front

The model is not broken by a bug and it is not undertrained. It fails because
**the reverse chain has to commit to the building's topology during the phase
where both discrete heads provably carry zero information.** Everything the
converter needs — which slots are real, which vertices are adjacent, which
vertices belong to which face — is decided by an unguided random walk over the
first 20–30 % of the sampling steps. The rest of the chain then politely refines
whatever noise handed it.

The continuous channel is *fine*. Coordinates denoise at or better than the
optimal linear-shrinkage baseline at every timestep. Chasing the coordinate
pipeline further is chasing the healthy half of the model.

Ranked by how much each one explains the observed output:

| # | Finding | Severity |
|---|---------|----------|
| F1 | Discrete heads carry ≤0 bits for the first ~120 reverse steps; topology is sampled from the prior | **fatal** |
| F2 | MiDi's schedule rationale ("bonds follow from conformation") is false for a Levi graph | **fatal, design-level** |
| F3 | Representation cannot express planarity or ring order; 96.4 % of generated faces have no valid cycle | **fatal, representation-level** |
| F4 | `val_coord_mse` at `t=T/2` drives early stopping and checkpointing, and measures the easiest regime | severe (masks all of the above) |
| F5 | `Off` as a diffused node class → 61 % padding majority, bimodal coordinate target | severe |
| F6 | 2/3 of the schedule's steps are spent on already-easy coordinates | moderate |
| F7 | Face-node coordinates are redundant, unread, and drift ~0.9 m | moderate |
| F8 | Coordinate scaling: shape-preserving (hypothesis rejected), but prior is anisotropically mismatched and ignores the size mixture | moderate |
| F9 | Inverted `nu` comment in config/docstring, likely misled tuning | documentation |

---

## Method and constraints

* Data: **mini dataset only** (`data/The Hague/mini`, LOD2, `n_max=50`,
  `upper_limit_nodes=50`, `normalize_coords=true`) — 7 238 buildings, matching
  `configs/mini2-train.yaml`.
* Checkpoint probed: `outputs/initial-runs/mini-2/checkpoints/last.ckpt`, batch
  size 10 throughout, 50 val buildings per sweep point.
* W&B: entity `maxim-lod`, project `lod-generation`, runs `mini-1` (7knakv9o),
  `mini-2` (f5prlyl3), `mini-3` (z0irwnge), `levi-1` (1zvfdzhm).
* Reference: Vignac et al., *MiDi: Mixed Graph and 3D Denoising Diffusion for
  Molecule Generation*, arXiv:2302.09048.

Caveat: the checkpoint stores `coord_scale=3.1866`; recomputing it today on the
mini set gives `3.3566`. The mini dataset changed after those runs
(`b2a7591 Add sample_mini_dataset`). Absolute MSE values are therefore ~5 %
off between the two, which does not affect any conclusion below.

---

## F1 — The discrete heads are uninformative exactly when topology is decided

**Claim.** For the first ~120 of the 500 reverse steps, the node-class and
edge-class heads return *less* information than simply emitting the training
marginals.

**Evidence** (`outputs/diagnostics/09_discrete_information.png`,
`discrete_probe.json`). Measured on the *debiased* posterior the sampler
actually consumes (`_debias`, `src/models/diffusion.py:185`), reported as
information gain over the marginal prior in bits — a class-weight-invariant
measure, unlike raw accuracy.

Prior entropy: nodes 1.5302 bits, edges 0.2939 bits.

| t | node info (bits) | edge info (bits) | F1 vertex-vertex edge | F1 vertex-face edge | F1 GroundSurface |
|---|---|---|---|---|---|
| 500 | **−0.039** | **−0.010** | 0.000 | 0.000 | 0.000 |
| 450 | **−0.019** | **−0.007** | 0.000 | 0.000 | 0.000 |
| 400 | 0.133 (8.7 %) | 0.017 (5.9 %) | 0.059 | 0.043 | 0.000 |
| 350 | 0.654 (43 %) | 0.094 (32 %) | 0.413 | 0.387 | 0.242 |
| 300 | 1.216 (79 %) | 0.190 (65 %) | 0.708 | 0.677 | 0.674 |
| 250 | 1.445 (94 %) | 0.243 (83 %) | 0.852 | 0.845 | 0.926 |

The reverse chain starts at `t=T` and walks down (`src/models/diffusion.py:435`).
So steps 500→400 — where the graph's global structure is laid down and can no
longer be undone by a marginal-transition posterior — run on a head that is
worse than a coin weighted by the class frequencies. `F1(vertex-vertex) = 0.000`
means the model does not place a single correct adjacency edge there.

**Why this is structural, not a training failure.** The network is fully
permutation-equivariant over the `n_max` slots and carries no positional
encoding (`src/models/regnn.py:370-415`). At high `t` every slot's features are
i.i.d. draws from the same marginal and the positions are near-zero, so the
network's output *must* be near-identical for every slot; the only admissible
answer is the marginal. Nothing in more training fixes this.

**Consequence in the runs.** `gen/uniqueness = 0` on both `mini-2` and `mini-3`
(W&B summary) — every sampled building collapses to the same feature vector.
`mini-3` additionally reports `gen/valid_fraction = 0`, `gen/rejection_rate = 1`.

---

## F2 — MiDi's schedule rationale does not transfer to a Levi graph

**Claim.** The adaptive schedule (`nu_pos=2.5 > nu_e=1.5 > nu_x=1.0`) is copied
from MiDi, and it is justified by a premise that is false here.

**Paper.** MiDi §4.2: *"while the 2D connectivity structure can be predicted
relatively well from the 3D conformation, the converse is not true"* — hence the
schedule resolves coordinates first, then bonds, then atom types. The same
section gives QM9's exponents as `ν_r = 2.5, ν_y = 1.5, ν_x = ν_c = 1`, which
are exactly this repo's defaults (`src/utils/config.py:51-53`).

**Why the premise fails.** For molecules, connectivity genuinely follows from
geometry — EDM+OpenBabel recovers bonds from interatomic distances alone, and
MiDi's whole comparison is against that baseline. For a Levi building graph it
does not:

* `EDGE_VV` is *ring adjacency along a face boundary*. Eight corner points of a
  box admit many distinct edge sets; the point cloud does not select one.
* `EDGE_VF` is bipartite membership between a face node and its ring. Its only
  geometric signature is "the face node sits near the ring centroid"
  (`src/dataset/dataset.py:230`), which is a weak, non-identifying cue —
  the face nodes of adjacent walls sit close together.

So the schedule spends its expensive early steps resolving the one channel that
does *not* determine the others, and leaves the determining channel until it is
too late to matter.

**Also worth noting**: MiDi itself lowered `ν_r` from 2.5 to 2 when moving from
QM9 (≤9 heavy atoms) to GEOM-DRUGS (avg 44 atoms, up to 181) — the harder
dataset got a *less* position-dominated schedule. This repo kept the QM9 value.

---

## F3 — The representation cannot express planarity or ring order

**Claim.** A CityJSON face is a planar, cyclically-ordered ring. The Levi graph
encodes neither constraint, so the model cannot satisfy them and the converter
invents the missing structure.

**Evidence** (`outputs/diagnostics/face_report.json`), 10 real val buildings vs
10 generated samples:

| | real | generated |
|---|---|---|
| faces measured | 73 | 122 |
| mean ring size | 4.30 | 5.00 |
| rings with >6 vertices | 2.7 % | 21.3 % |
| rings with <3 vertices (dropped by converter) | 0.0 % | 9.8 % |
| planarity RMS, median | **0.0000 m** | **0.1774 m** |
| planarity RMS, p90 | 0.0003 m | 0.5080 m |
| **faces whose members form a clean VV cycle** | **100 %** | **3.6 %** |
| face-centroid drift | 0.0000 m | 0.9117 m |

**96.4 % of generated faces have no valid boundary cycle.** `_order_ring`
(`src/post_process/post_process.py:101`) therefore falls through to its
angle-sort fallback (`post_process.py:122`) for essentially every face, so the
polygon winding in the exported CityJSON is manufactured by the post-processor,
not generated by the model.

99.2 % of real faces are non-triangular (mean ring 4.39, mostly quads —
`outputs/diagnostics/data_stats.json`). Four freely-diffused points in R³ are
coplanar with probability zero, so non-planarity is guaranteed by construction,
not by underfitting. This is exactly what val3dity reports: `mini-3` logged
`gen/err/203 = 84` (non-planar polygon) against `gen/err/301 = 7` and
`gen/err/302 = 3`.

---

## F4 — The monitored metric measures the easiest point on the chain

**Claim.** `val_coord_mse` is evaluated at a single fixed `t = T//2`
(`src/models/diffusion.py:357`), and every config uses it for both early
stopping and checkpoint selection (`configs/*.yaml`, `src/utils/config.py:78,86`).

**Evidence.** At `t=T/2` with `nu_pos=2.5`: `alpha_bar_pos = 0.9187`, i.e.
**SNR = 5.41**. That is the high-signal end of the chain. By that point the node
head has already recovered 94 % of the prior entropy (F1). The metric is
structurally blind to the regime where the model actually fails.

The gap is visible in the logs: for `mini-1`, `train_coord_mse_epoch` sits at
0.22–0.34 (averaged over `t ~ U[1,T]`) while `val_coord_mse` sits at 0.071–0.082
— a 3–4× difference that is entirely an artefact of the fixed evaluation `t`,
not train/val generalisation.

`mini-1` then early-stopped at epoch 61 of 250, on a metric that had been flat
and noisy since epoch ~19. Model selection is being driven by noise on a
near-trivial task.

---

## F5 — `Off` as a diffused node class is a departure from MiDi

**Claim.** Making padding a generated class is this repo's own design decision,
documented at `src/models/diffusion.py:10-15` and implemented at
`src/models/diffusion.py:251` (`net_mask = torch.ones_like(...)`, every slot
participates). MiDi does not do this.

**Paper.** MiDi §4: *"we consider the absence of a bond as a particular bond
type and generate dense adjacency tensors"* — that is **bonds only**. Atom types
have no virtual/absent class; a molecule's node count is its actual size, and
GEOM-DRUGS averages 44 atoms. There is no padding majority in MiDi's node
channel.

**Cost here.** Measured on the mini set (`data_stats.json`): mean real nodes
19.65 of 50 slots → **60.7 % of slots are Off**, `x_marginals =
[0.232, 0.020, 0.028, 0.112, 0.609]`. Consequences:

1. The coordinate target is bimodal — a point mass at exactly 0 with weight 0.61
   (`_centre_positions` pins Off to zero, `src/models/diffusion.py:234`) plus a
   continuous part — and *which* mode a slot belongs to is settled by the X
   channel, which resolves **last** under `nu_x=1.0`. For most of the chain the
   position head is regressing a mixture it cannot disambiguate.
2. `GroundSurface`, the class that defines the base plane, is 2.0 % of slots.
   The `class_balance=0.5` weighting and `_debias` were added to fight this, and
   they work as designed, but they cannot create signal that isn't there (F1
   measures the debiased posterior).

---

## F6 — Two thirds of the schedule is spent on already-solved coordinates

**Evidence** (`outputs/diagnostics/01_noise_schedule.png`, `08_shrinkage_bound.png`).

* Position SNR > 1 for **378 of 501 steps** (75 %); SNR < 0.1 for only 17 %.
* Coordinate MSE is already down to 0.0389 by `t=200` and 0.0080 by `t=100`.
* The whole informative range (SNR between 0.1 and 10) is `t ∈ [~250, ~420]` —
  about a third of the chain.

**Positions are not the problem.** Against the optimal linear-shrinkage
predictor `Var(x₀)/(1+SNR)` on the same 50 buildings (`Var(x₀) = 1.0087`):

| t | SNR | model MSE | shrinkage bound | ratio |
|---|---|---|---|---|
| 250 | 5.41 | 0.0818 | 0.1573 | **0.52** |
| 300 | 1.94 | 0.1990 | 0.3434 | 0.58 |
| 350 | 0.66 | 0.4235 | 0.6067 | 0.70 |
| 400 | 0.17 | 0.7436 | 0.8623 | 0.86 |
| 450 | 0.016 | 1.0001 | 0.9931 | 1.01 |
| 500 | 0.000 | 1.0086 | 1.0087 | 1.00 |

The model beats the second-order baseline everywhere it can, and at `t=T`
predicts exactly the prior mean — which is the correct Bayes answer, not a
failure. I initially read the `t≥450` numbers as underfitting; that was an
artefact of estimating `Var(x₀)` on a different (10-building) sample than the
MSE curve. Corrected, the position denoiser is healthy.

Oracle ablations agree: feeding the model the **clean** `X₀` and `E₀` at `t=400`
moves coordinate MSE only from 0.744 to 0.626. Perfect knowledge of the graph
barely helps the coordinates — the dependency runs the other way.

---

## F7 — Face-node coordinates are redundant, never read back, and drift

**Claim.** Face nodes are positioned at their ring centroid
(`src/dataset/dataset.py:230`) — a deterministic function of the vertices they
connect to. They are then diffused as free 3-DOF variables, and
`graph_to_cityjson` reads **only** vertex rows
(`src/post_process/post_process.py:161,210`). The face row is written, diffused,
scored by the loss, and discarded.

**Evidence.**

* Face nodes are **41.4 % of all real nodes** (`data_stats.json`).
* They sit at 0.51× the mean squared radius of vertices, so they pull
  `compute_coord_scale` (`src/dataset/datamodule.py:163`) downward and bias the
  centroid `_centre_positions` subtracts — both by design
  (`datamodule.py:195-198`), but both measured on a quantity that never reaches
  the output.
* Generated face-centroid drift: **0.9117 m** (mine, 10 samples) — consistent
  with `gen/face_centroid_consistency` of 0.90 / 1.11 / 1.60 m logged by
  `mini-2` / `mini-1` / `mini-3`. The metric exists precisely because this drift
  was anticipated (`src/eval/callback.py:38-52`).

So ~41 % of the coordinate degrees of freedom are spent enforcing a hard
algebraic constraint (centroid = mean of members) that a Gaussian diffusion
cannot represent, and the result is thrown away. The user's hypothesis here is
**upheld**, though it is a capacity/consistency tax rather than the fatal defect.

---

## F8 — Coordinate scaling: shape is preserved, the prior is not

This was the specific hypothesis to test. Two separable questions, opposite answers.

**Does scaling ruin the shape? No.** `_centre_positions` followed by division by
a single scalar `coord_scale` is an exact isotropic similarity transform.
Measured directly: reconstructing pairwise distances after the round trip on 200
buildings gives a **maximum error of 0.0156 m**, which is float32 `cdist`
numerics, not the transform. Visually confirmed for four buildings in
`outputs/diagnostics/03_scaling_before_after.png` — the scaled graphs are
pixel-for-pixel the same shapes.

**Does it match the prior? No, in two ways.**

1. **Anisotropy.** One global scalar is applied to all three axes, but buildings
   are not isotropic. Per-axis std of the centred real-node coordinates:

   | | x | y | z |
   |---|---|---|---|
   | metres | 4.62 | 4.91 | 2.34 |
   | after ÷ `coord_scale` (3.357) | **1.38** | **1.46** | **0.70** |

   The N(0,1) position prior therefore *overshoots* the vertical variance by ~2×
   and *undershoots* the horizontal by ~2×. Heights get proportionally less
   resolution than footprints throughout the chain. Choosing one scalar is
   deliberate — per-axis scaling would break the `so2` yaw-equivariance
   (`src/dataset/datamodule.py:174-176`) — so this is a real, acknowledged
   trade-off, not an oversight. It is worth measuring against `se2` before
   assuming it's harmless.

2. **Size mixture.** `coord_scale` is a *pooled* std, dominated by large
   buildings. Per-building maximum extent in model units on the val split:
   p5 = 0.88, median = 1.48, p95 = 5.35, max = 39.24 — a **6.1× p95/p5 spread**.
   The median building is smaller than the prior's own 1σ ball while the largest
   is 39 units across. A single unit-variance prior has to cover both. This is
   again deliberate (per-building normalisation would erase building size and is
   not invertible at sampling time) but it materially widens `Var(x₀)` and
   deepens the shrinkage the model must undo.

See `outputs/diagnostics/04_prior_vs_building.png` and `08_shrinkage_bound.png`.

---

## F9 — The `nu` comment is inverted in two places

`src/utils/config.py:50` and `src/models/diffusion.py:52` both state:

> `nu_pos > nu_e destroys coordinates faster than graph structure.`

This is backwards. `src/models/noise.py:26-30` states it correctly (*"A larger
nu retains more signal"*), and the numbers confirm noise.py:

```
t=250:  alpha_bar_pos=0.9187   alpha_bar_e=0.7134   alpha_bar_x=0.4923
steps with alpha_bar > 0.5:  pos 378/501,  edge 313/501,  node 248/501
```

Larger `nu` ⇒ noised **slower** ⇒ resolved **earlier** in reverse. Given that
`mini-3` was configured with `nu_pos: 1.0, nu_x: 1.5` — an inversion of the
default ordering — this comment plausibly misdirected that ablation.

---

## Verdicts on the three starting hypotheses

**"The noise schedule doesn't fit the data."** — **Partly upheld, but not the
root cause.** The schedule is misallocated (F6: 75 % of steps at SNR > 1) and
its ordering rationale doesn't transfer (F2). But retuning `nu` alone will not
help, because the failure at high `t` is an information-theoretic property of a
permutation-equivariant one-shot denoiser on a 61 %-padded graph (F1), not a
property of the schedule. `mini-3` tested exactly this (`nu_pos` 2.5→1.0,
T 500→100) and got *worse* results — though that run is confounded, it also
dropped `lr` by 100× to 1e-5, so treat it as inconclusive rather than as
evidence against schedule changes.

**"Scaling ruins the structure."** — **Rejected.** The transform is an exact
similarity; measured distortion is 1.6 cm of float32 noise. The real scaling
issues are prior mismatch (2× anisotropy) and the unnormalised size mixture
(6.1× spread), which cost resolution but do not deform anything (F8).

**"Face nodes having coordinates hurts the coordinate distribution."** —
**Upheld.** 41 % of real nodes carry forced, redundant coordinates that are
never read back, deflate `coord_scale` by sitting at 0.51× vertex radius, and
drift 0.91 m in generated samples (F7). Genuine cost, secondary to F1–F3.

---

## What these constraints prevented

* **Mini dataset only.** All model probing used the `mini-2` checkpoint
  (`n_max=50`). The full-dataset `levi-1` run (`n_max=100`, `se2`, full The Hague)
  was analysed from W&B metrics only, not probed. F1 should be *worse* at
  `n_max=100` (more padding, weaker symmetry breaking), but that is a prediction,
  not a measurement.
* **Batch size 10 / 50 buildings per sweep point.** The information curves in F1
  are stable enough for the qualitative claim (the sign of the information gain),
  but the per-timestep numbers carry a few percent of sampling noise. The
  face-structure comparison in F3 is 10 vs 10 buildings — the 3.6 % cycle-validity
  figure is stark enough to survive that, but should be re-run at ~200 samples
  before being quoted as a headline number.
* **`levi-1` has no generative-eval metrics** (`gen/*` keys absent from its
  summary — the eval callback postdates that run), so there is no full-dataset
  uniqueness/validity number to compare against `mini-2`'s zero.
* I did not run val3dity myself; the error-code histogram in F3 comes from
  `mini-3`'s logged `gen/err/*`.

---

## Figure index — `outputs/diagnostics/` (≈0.8 MB total)

| file | shows |
|---|---|
| `01_noise_schedule.png` | signal-retention curves, position SNR, true vs spurious edge counts |
| `02_data_distributions.png` | scaled coords vs N(0,1), real-node counts, ring sizes |
| `03_scaling_before_after.png` | four buildings, raw metres vs model units — shape preserved |
| `04_prior_vs_building.png` | one building against the prior's 1σ ball, three projections |
| `05_model_probe.png` | denoising quality vs t, oracle ablations, prediction shrinkage |
| `06_forward_corruption.png` | one real building corrupted at t = 0…500 |
| `07_generated_samples.png` | 10 unconditional samples from `mini-2` |
| `08_shrinkage_bound.png` | model vs linear-shrinkage bound; building-size mixture |
| `09_discrete_information.png` | information recovered by the discrete heads; per-class F1 |

Raw numbers: `data_stats.json`, `model_probe.json`, `discrete_probe.json`,
`face_report.json`, `sample_stats.json`.
