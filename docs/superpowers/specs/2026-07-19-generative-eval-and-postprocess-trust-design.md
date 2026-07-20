# Design: Generative Evaluation in the Lightning Pipeline

**Date:** 2026-07-19 · **Revised:** 2026-07-20 (Levi representation landed; scope re-cut)
**Status:** Phase 0 done · Phase 1 mostly done · Phase 2 (this spec's remaining work) not started

**Sequencing note.** Phase 0 (the Levi-graph representation) has **landed** on branch
`levi-representation` — conversion is now an exact-inverse pair and generation emits correct
semantics + LoD directly. This document is re-cut around what remains: finishing the
post_process trust tests (small) and building the **end-of-pipeline generative evaluation**
(the substantive work).

## 1. Problem

`CityJSONDiffusionModule` generates buildings via a `T`-step reverse chain
(`sample` → `generate_cityjson`, `src/models/diffusion.py`). Evaluation splits into two tiers
with very different cost, and only the cheap tier exists:

- **Tier 1 — single-step denoising proxies (exist, `_eval_step`).** At a fixed mid-chain
  `t = T/2`: `coord_mse`, `real_precision`/`real_recall` (vertex class), `edge_ce`, `loss`.
  Logged every epoch on train + val, and once on the **test** split after training. Fast, so
  they run per-epoch as convergence signals — but they measure single-step fidelity, **not** a
  building sampled end-to-end. Errors compound across the chain, so these can look healthy while
  full samples are malformed.

- **Tier 2 — full-generation quality (missing).** Running the whole reverse chain is slow, so it
  is **not** affordable per-epoch. But **once, at the end of the pipeline**, we can afford to
  sample a configurable number of batches all the way from noise and actually score the
  buildings: validity, distribution realism, novelty, and internal coherence. This is what the
  spec adds.

The Levi detour already removed the old blocker for Tier 2 — evaluation used to be contaminated
by `post_process` guesswork (`find_cycles_dfs`, normal-snapping). That is gone; see §3.

## 2. Goals / Non-goals

**Goals**
- Finish the **post_process trust tests** (§4) — the exact-inverse converter is the foundation
  every Tier-2 number stands on; a handful of gaps remain, chiefly on the *generated* (noisy)
  path.
- Build **`GenerativeEvalCallback`** (§5): fired **once at the end of the pipeline**, it samples
  `num_batches` batches through the full reverse chain, scores each building, and reports to
  WandB — scalars/histograms plus a **per-instance table** — and **saves sample graphs +
  geometries locally and to WandB**.

**Non-goals (v1)**
- Replacing Tier 1. The single-step `_eval_step` stays; it measures a different, cheaper thing.
- **Conditional / guided generation.** We are unconditional now — there is no per-sample ground
  truth, so Tier-2 metrics are reference-distribution- and self-consistency-based. The code is
  structured (§5.8) so guided generation — full sampling **on the test set with ground truth** —
  slots in later by swapping the sample source and adding paired-GT columns/arms, **without**
  restructuring. Design for it; do not build it.
- Inner rings / holes; LoD1/LoD3 population (parametrized, only LoD2 populated); multi-GPU
  distributed sample gathering.

## 3. Phase 0 — Levi representation (DONE, for reference)

Landed on `levi-representation`. What Tier-2 relies on:

- **5 node classes** `0=VERTEX, 1=GROUND, 2=ROOF, 3=WALL, 4=OFF`; **3 edge classes**
  `0=off, 1=vertex-vertex, 2=vertex-face` (`src/dataset/dataset.py`).
- **Face nodes carry a diffused centroid position**; zero-CoM is over vertex nodes only.
- **`graph_to_cityjson` is an exact inverse** of `parse_cityjson_file_to_graphs`
  (`src/post_process/post_process.py`): reads vertex-node rows for coordinates, recovers ring
  order by walking the vv cycle within each face's incident set (angle-sort fallback), orients
  CCW-outward via Newell normal + building-centroid test. Emits correct semantics and `lod:"2"`.
  **No regularization** on this path (would break invertibility). `straighten_face` /
  `regularize_building_geometry` remain in the module as **optional** post-hoc regularizers, not
  on the inverse path.
- `sample()` returns **argmax** class labels — there is no `edge_threshold` any more. **Any
  `threshold` parameter from the earlier design is obsolete and must not reappear** in config or
  in `generate_cityjson`.

## 4. Phase 1 — post_process trust tests (mostly done; finish the gaps)

`tests/test_levi_roundtrip.py` already covers, on synthetic cube + gable-house fixtures:
lossless round-trip (`parse∘convert∘parse == parse`), file-level determinism (`cj3 == cj2`),
semantics pass-through, CCW-outward orientation (exact signed volume), and the `<3`-incident-
vertex skip. **Extend that file** (do not create a parallel `test_post_process.py`) with the
gaps — weighted toward the *generated* path, where the noisy heuristics actually get exercised:

1. **Real LoD2 fixture round-trip.** A small committed real building through
   parse → convert → assert geometric + topological + semantic equality. The synthetic fixtures
   don't exercise real coordinate scales / ring shapes.
2. **Non-convex face order recovery.** An L-shaped / concave face where convex angular sort would
   fail — this pins the `_order_ring` cycle-walk and exposes its documented `ponytail:` ceiling
   (the angle-sort fallback). Untested today; on the generated critical path.
3. **Adversarial generated topology** — the cases sampling actually produces: vv edges among a
   face's members that don't form a single cycle (→ fallback path), duplicate vertices,
   disconnected graph. Assert graceful, documented behavior (skip + warn, no crash, no silent
   garbage).
4. **val3dity validity** on converted output, `pytest.mark.skipif(shutil.which("val3dity") is None)`.

Rationale for the reweighting: the exact inverse is proven on *well-formed* graphs, but
model-generated graphs are noisy and hit `_order_ring`'s fallback and `_orient_outward`'s
centroid heuristic. On the generated path, those heuristics — not the clean inverse — are what
runs, so their failure modes are what the trust tests must characterize.

## 5. Phase 2 — end-of-pipeline generative evaluation

### 5.1 Two-tier architecture (where each metric runs)

| Tier | What | When | Cost | Home |
|------|------|------|------|------|
| 1 | single-step denoise fidelity (`coord_mse`, vertex P/R, `edge_ce`) | every epoch (train+val); once on test | cheap | `_eval_step` (exists) |
| 2 | full-generation quality (validity, realism, novelty, coherence) | **once, end of pipeline** | slow | `GenerativeEvalCallback` (new) |

Tier 1 is the convergence proxy watched during training. Tier 2 is the acceptance measure run
after test-set Tier-1 metrics are in.

### 5.2 Home and trigger (`src/eval/callback.py`)
- Lightning `Callback`, fired on **`on_test_end`** so it runs after the test split's Tier-1
  metrics — i.e. truly at the end of the pipeline. Gated by a new `generative_eval` config block
  (disabled → no-op).
- The scoring body is a **standalone callable** against a checkpoint (thin entrypoint mirroring
  `src/inference.py`); the callback is a thin Lightning adapter over it. This is also the seam
  for §5.8.
- Pulls the run's WandB logger from `trainer.loggers`. Reference/train features are built once
  from `trainer.datamodule` and cached on the instance.

**Config block (`generative_eval`, in `src/utils/config.py`):**
```
enabled: bool = false
num_batches: int              # how many batches to sample end-to-end (the affordability knob)
batch_size: int
seed: int                     # fixed → comparable buildings across runs/epochs
log_n_samples: int            # how many graphs+geometries to persist (local + WandB)
save_dir: str | null          # local output root; null → <run_dir>/generative_eval
feature_set: "full" | "welldefined" = "full"
val3dity_path: str | null     # null → resolve via shutil.which("val3dity")
```
(No `threshold` — §3.)

### 5.3 Flow
1. `L.seed_everything(seed)`.
2. `records = _draw(num_batches, batch_size)` — full reverse sampling. Each **record** is one
   generated instance: `{graph:(coords,node_labels,edge_labels), cityjson, features, valid,
   error_codes, conditioning:None, ground_truth:None}`. The two `None`s are the §5.8 seam.
   Generation drops (empty/`<3`-vertex graphs) are counted, not silently dropped (§5.3.1).
3. Validity (§5.4) → features (§5.5) → distances (§5.6) → novelty (§5.7) → log + save (§5.8).

**5.3.1 The callback samples directly, not through `generate_cityjson`.** `generate_cityjson`
returns only a list of CityJSON dicts and is depended on by `src/inference.py`; changing its
contract would break that, and it hides the raw graph the callback needs for `.npz` saving and
the face-centroid metric (§5.8). So the callback calls `model.sample(batch_size)` itself, then
converts each instance with `graph_to_cityjson` in its own loop — where drop counting
(`{}`-result or `<3` vertices) falls out naturally. The only shared logic is the metres
un-scaling (`* coord_scale`, `+ z_shift`): extract it as a small `_denormalize_coords(pos_i)`
helper on the model so `generate_cityjson` and the callback share one definition. `generate_cityjson`'s
public return type is unchanged; only its internals are refactored to call the helper.

### 5.4 Validity + rejection rate (`src/eval/validity.py`)
- Gated by `shutil.which("val3dity")` / `val3dity_path`. Absent → warn, skip the validity arm;
  the generation-drop rejection rate still logs.
- Serialize the batch to **CityJSONSeq**, one `val3dity stdin --report` subprocess (stdin
  streaming, val3dity ≥ 2.5.0); parse the flat per-feature error list.
- Logs: `gen_valid_fraction`, **per-error-code histogram**, and **rejection rate** =
  `(generation drops) + (val3dity-invalid)` / `attempted`, with the breakdown.
- val3dity version-checked and smoke-validated on a known-good building at eval start (fail
  loud, not silently all-invalid).

### 5.5 Feature extraction (`src/eval/building_features.py`)
- 3dSAGER property vector (Genossar et al., arXiv:2511.06300, Table 1) per building.
- `full` (default, LoD2) = all 17: `area, volume, height_diff, num_vertices, perimeter,
  circumference, perimeter_index, convex_hull_area, ave_centroid_distance, shape_index,
  fractality, elongation, hemisphericality, cubeness, axes_symmetry, density, num_floors`.
  `welldefined` drops the LoD1-ambiguous ones (`num_floors, fractality, circumference`).
- **numpy + scipy only** (`scipy.spatial.ConvexHull`, OBB via covariance eigendecomposition,
  volume via divergence theorem). Non-obvious descriptors (`shape_index`, `hemisphericality`,
  `cubeness`, `elongation`) **ported** from `tudelft3d/3d-building-metrics` with per-formula
  attribution — **not** a dependency (its pins conflict with the torch/lightning env).

### 5.6 Distribution distance (`src/eval/distribution.py`)
- `log(1 + x)` normalize (3dSAGER §3.1).
- **Reference = the test-split graphs pushed through the same converter + featurizer** as the
  generated samples. The datamodule holds parsed **graphs** (padded tensors, metres), not raw
  files, so this is what's directly available — and because `graph_to_cityjson` is a proven exact
  inverse, featurizing reference graphs through the converter equals featurizing the raw
  CityJSON. The earlier design's raw-vs-pipelined split therefore **collapses**: there is one
  reference, and running both sides through the identical pipeline removes any converter bias from
  the comparison. (Reaching for the raw files instead would be extra plumbing for a provably
  ~zero difference.)
- **Per-feature 1-Wasserstein** (`scipy.stats.wasserstein_distance`) of generated vs reference,
  per feature + mean aggregate.
- **Kernel MMD** (RBF, median-heuristic bandwidth) on the stacked normalized vectors vs the same
  reference.

### 5.7 Novelty / uniqueness (`src/eval/novelty.py`)
- Cache **train-split** feature vectors. `uniqueness` = fraction of samples that aren't mutual
  near-duplicates in feature space; `novelty` = fraction whose nearest train-set neighbour
  exceeds a logged, config-default tolerance.

### 5.8 Logging, saving, and the conditional seam (`src/eval/callback.py`)

**Scalars / histograms** under a `gen/` namespace: `gen_valid_fraction`, rejection rate + its
breakdown, per-feature Wasserstein + mean, MMD, uniqueness, novelty, and the per-error-code
histogram. Plus one Tier-2-native coherence metric the Levi representation uniquely enables:

- **`gen_face_centroid_consistency`** — mean `‖generated face-node position − centroid(its member
  vertices)‖`. Face position and its vertices are diffused **independently** and the converter
  *ignores* face positions (reads vertex rows only), so this drift is invisible to conversion but
  is a pure internal-coherence signal that needs no val3dity, no features, and no reference set.
  Nearly free; add it.

**Per-instance WandB Table.** One row per generated building (no ground truth — we are
unconditional), columns = `{sample_id, num_vertices, num_faces, valid, error_codes,
face_centroid_consistency, <the 17 features>}`, so quality is inspectable **per instance**, not
just in aggregate. A `wandb.Object3D` (minimal OBJ from the CityJSON faces) is embedded per row
for the first `log_n_samples`. Fixed `seed` → the same buildings recur across runs for
comparison.

**Local + WandB artifact saving.** For the first `log_n_samples` instances, write to
`save_dir` (default `<run_dir>/generative_eval/`): the **graph** (`.npz`: `pos`, `node_labels`,
`edge_labels`) and the **geometry** (`.city.json` via `save_to_file`, and the OBJ). Log the same
files as a WandB artifact so they're retrievable off the run. Local copy is the source of truth;
WandB is the shareable mirror.

**Conditional-generation seam (design-for, don't build).** Everything above is one shape away
from guided generation:
- `_draw` is the only conditional-aware unit. Today it samples unconditionally and sets
  `conditioning=None`, `ground_truth=None`. Later it iterates the **test dataloader**, conditions
  each sample on the input, and fills `ground_truth` with the paired real building.
- When `ground_truth` is present, a new **paired-metrics arm** (`src/eval/paired.py`, not written
  now) adds per-instance reconstruction columns (e.g. Chamfer / vertex error / face-set IoU to
  GT) — the Table already has a row per instance to hang them on, and the distribution arm gains
  a paired mode alongside the reference mode.
- No restructuring required: swap the sample source, add columns to the existing record + Table.
  This is why the scoring body is a standalone callable over records rather than baked into the
  callback.

## 6. Wiring, dependencies, config
- **Wiring:** new `generative_eval` block (`src/utils/config.py`); `create_callbacks`
  (`src/utils/setup_utils.py`) appends the callback when `enabled`.
- **New dependency:** `scipy` (`ConvexHull`, `wasserstein_distance`) — the only new Python dep;
  add to `requirements.txt`. Existing: `wandb>=0.15.0` (`Object3D`, `Table`, `Artifact`), `numpy`.
- **External tool (ops, gated):** `val3dity` binary on PATH. Dev (Windows): prebuilt `.exe`.
  Server (Linux): compiled once into the env / image. Decoupled by the `shutil.which` gate.

## 7. Module layout
```
src/models/diffusion.py    # small change: extract _denormalize_coords helper (§5.3.1)
src/post_process/post_process.py  # unchanged (exact-inverse converter already lands here)
src/eval/
  __init__.py
  callback.py              # GenerativeEvalCallback (on_test_end) + standalone body + saving
  validity.py              # val3dity subprocess, CityJSONSeq, rejection rate
  building_features.py     # 3dSAGER vector (numpy + scipy, ported formulas)
  distribution.py          # log1p, per-feature Wasserstein, MMD (+ optional converter-fidelity)
  novelty.py               # uniqueness + novelty vs train features
  # paired.py              # (future) conditional GT metrics — seam only, not built now
tests/
  test_levi_roundtrip.py   # Phase 1: extend with real-fixture, non-convex, adversarial, val3dity
  test_eval_*.py           # Phase 2 unit tests
```

## 8. Testing strategy (eval code)
- `building_features`: analytic solids (unit cube → known area/volume/cubeness; tall box → known
  elongation) assert exact values.
- `distribution`: identical distributions → distance ≈ 0; shifted → monotonically larger.
- `validity`: parse a captured val3dity report fixture → correct valid/invalid + histogram
  without the binary in CI.
- `novelty`: sample identical to a train building → novelty 0; all-distinct → uniqueness 1.
- `callback`: smoke test with a tiny model + monkeypatched `generate_cityjson` and a fake logger
  (reuse the `model.log = lambda ...` pattern from `test_model_smoke.py`); assert the `gen/`
  scalar keys logged, the Table has one row per sample, files written to `save_dir`, and graceful
  degradation when val3dity is absent.
- `face_centroid_consistency`: a graph whose face node sits exactly at its members' centroid → 0;
  a displaced face node → the exact displacement.

## 9. Open items resolved
- **Representation:** Levi graph, landed (§3). Conversion is the exact inverse; heuristics moved
  off the real-data path and onto the generated path (§4).
- **Two tiers:** single-step proxies per-epoch (exist); full generative eval once at end of
  pipeline (`on_test_end`), `num_batches` configurable.
- **No threshold:** argmax sampling; the old `threshold` knob is deleted everywhere.
- **Reference:** one reference — test-split graphs through the same converter+featurizer as the
  samples (the datamodule exposes graphs, not raw files; the exact inverse makes pipelined == raw,
  and same-pipeline comparison removes converter bias). The old raw-vs-pipelined split is dropped.
- **Unconditional now, guided later:** per-instance WandB Table + standalone record-based body +
  `_draw`-only conditioning seam make the switch additive (§5.8). Paired-GT metrics deferred.
- **Persistence:** sample graphs (`.npz`) + geometries (`.city.json` + OBJ) saved locally and as
  a WandB artifact, plus per-instance Table with embedded `Object3D`.

## 10. References
- Vignac et al., *MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation*,
  arXiv:2302.09048 — the base model this pipeline adapts.
- Genossar, Dalyot, Shraga, Gal, *3dSAGER: Geospatial Entity Resolution over 3D Objects*,
  arXiv:2511.06300 — the geometric property vector (Table 1) for distribution realism.
- Ledoux, *val3dity: validation of 3D GIS primitives according to the international standards*,
  Open Geospatial Data, Software and Standards, 2018 — 3D validity.
- `tudelft3d/3d-building-metrics` — source of the ported shape-descriptor formulas (attributed at
  each implementation; not a dependency).
