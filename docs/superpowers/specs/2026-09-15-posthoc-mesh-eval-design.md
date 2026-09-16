# Post-hoc mesh-transformer evaluation across inference strategies

Date: 2026-09-15 · Branch: `mesh-transformer` · Status: approved design, pending spec review

## Goal

Score one trained `MeshTransformerModule` checkpoint on a data split under
several decoding strategies, with per-object results and enough recorded
state to reproduce any row later.

Not in scope: resume-after-crash logic, new decoding strategies beyond the four
kinds below, wiring into `main.py` or the training callback, raw (non-round-trip)
references.

## 1. Inference change — the only edit to existing code

File: `src/models/mesh_transformer.py`.

Blast radius of `MeshTransformerModule.generate` (upstream): gitnexus reports
LOW, flagged as a lower bound (one call site dropped at index time). Text search
finds `src/eval/mesh_eval.py::run_mesh_eval` (training-time `MeshEvalCallback`),
5 test files (`test_beam_search`, `test_kv_cache`, `test_mesh_amt`,
`test_mesh_transformer`, `test_mesh_code_transformer`) and the `test-mesh*`
notebooks. The change is additive: new keyword arguments default to off.

### `filter_logits(logits, temperature, top_p=None, min_p=None)` — new, module level

- Input logits already carry the invalid-prediction mask (`-inf`).
- Order: divide by `temperature` first, then apply the filter (HuggingFace
  logits-warper order).
- top-p / nucleus (Holtzman et al., arXiv:1904.09751): keep the smallest
  highest-probability set whose cumulative mass is >= `top_p`.
- min-p (Nguyen et al., arXiv:2407.01082): keep tokens with
  `p >= min_p * max(p)`.
- The argmax token is always kept, so a row can never be emptied.
- `temperature <= 0` is rejected here; greedy never calls this function.

### `generate(..., top_p=None, min_p=None, seeds=None)`

- `top_p` / `min_p` require `temperature > 0` and `beam_size == 1`; combining
  either with beams raises `ValueError`, matching today's beam + temperature rule.
- `seeds`: optional sequence of one int per row. When given, each row owns a
  `torch.Generator` (on the model device) and samples by Gumbel-max over the
  filtered logits, so a row's draw depends only on its own seed — not on its
  batch neighbours or batch order.
- `seeds=None`: the existing `torch.multinomial` path, token-for-token unchanged.
- Known limit, recorded rather than fixed: batched float kernels can differ at
  ~1e-6 across batch sizes, so `batch_size` is part of the reproduction recipe.

## 2. Configuration

Dataclasses `StrategyConfig` and `PosthocEvalConfig` in `src/utils/config.py`
(standalone; not added to the root `Config`). The eval YAML holds only
eval-specific parameters and names a training config that is loaded with the
existing `src.utils.initialization.load_config`.

Example, which is also the shipped default `configs/posthoc-eval.yaml`:

```yaml
reference_config: configs/mesh-v5-full-coord.yaml
checkpoint: outputs/initial-runs/mesh-v5-full-coord/checkpoints/last.ckpt
reference_overrides: []   # dot-list applied to the reference config
split: val                # train | val | test, reproduced via the reference seed
max_objects: null         # null = whole split; else eval_indices(n, max_objects, seed)
batch_size: 8
seed: 1234
device: auto
output_dir: null          # null -> <save_dir>/<experiment>/<run>/posthoc_eval
strategies:
  - {kind: greedy}
  - {kind: beam,  beam_size: [3, 5], length_penalty: [1.0]}
  - {kind: top_p, top_p: [0.95], temperature: [0.6, 1.2]}
  - {kind: min_p, min_p: [0.05], temperature: [0.6, 1.2]}
metrics: [chamfer, hausdorff, vol_symdiff, sinkhorn, roof_chamfer]
references: [lod1, lod2]
n_points: 4096
sinkhorn: {n_points: 2048, blur_m: 0.1, max_iters: 500, tol: 1.0e-6}
```

Rules:

- List-valued strategy params expand as a cartesian product within that
  strategy. Defaults expand to 7 runs: `greedy`, `beam_k3_lp1.0`,
  `beam_k5_lp1.0`, `top_p_p0.95_t0.6`, `top_p_p0.95_t1.2`,
  `min_p_p0.05_t0.6`, `min_p_p0.05_t1.2`.
- Allowed params per kind: greedy — none; beam — `beam_size`,
  `length_penalty`; top_p — `top_p`, `temperature`; min_p — `min_p`,
  `temperature`.
- Fail fast, before any model or data is loaded: unknown strategy kind, unknown
  param for a kind, unknown metric, unknown reference, empty strategy list.
- After loading the checkpoint, before generating: raise if the checkpoint's
  saved `tokenization` hyperparameter differs from the reference config's
  `mesh_data.tokenization` (the silent mis-decode trap documented in
  `notebooks/test-mesh-normal.ipynb`).
- VQ-VAE runs load their tokenizer with `src.train_mesh._load_vqvae`, exactly as
  training does. A missing stage-1 checkpoint raises with the path named.
- Generation budget: `9 * (mesh_model.n_max_triangles or mesh_data.max_faces or 200) + 1`,
  the `MeshEvalCallback` rule; `generate` clamps it to the position budget.
- Per-object seed: `zlib.crc32(f"{seed}:{object_id}".encode())`, identical
  across strategies (common random numbers). Never Python's salted `hash()`.

Entry point: `python -m src.eval.posthoc_eval --config configs/posthoc-eval.yaml [dot.overrides]`.

## 3. Metrics

References, both through the tokenizer round trip as `MeshEvalCallback` does:

- `lod2`: `model.decode_tokens(tgt)` in metres.
- `lod1`: the condition decoded by the dataset inverse (`detokenize` or
  `amt_detokenize`) in metres.

Every selected metric is computed against every selected reference; columns
are suffixed `_vs_lod2` / `_vs_lod1`.

| metric key | columns | implementation |
|---|---|---|
| `chamfer` | `chamfer_m` | existing `surface_distances` + `chamfer_distance` |
| `hausdorff` | `hausdorff_p95_m`, `hausdorff_max_m` | same distance arrays; p95 is the repo convention, max the textbook definition |
| `vol_symdiff` | `symdiff_m3`, `rel_symdiff`, `vol_ok`, `vol_note` | existing `src.analysis.exact_volume.exact_pair_volumes` (manifold3d `Mesh64`); `symdiff_m3`/`rel_symdiff` are NaN when `vol_ok` is False |
| `sinkhorn` | `sinkhorn_m` | new `src/eval/sinkhorn.py` |
| `roof_chamfer` | `roof_chamfer_m` | new roof sub-mesh extraction + existing chamfer |

### Sinkhorn (`src/eval/sinkhorn.py`, torch)

- Debiased Sinkhorn divergence
  `S = OT_eps(a,b) - 0.5 OT_eps(a,a) - 0.5 OT_eps(b,b)` with squared Euclidean
  cost, uniform weights on area-weighted surface samples (`sample_surface`),
  log-domain iterations, `eps = blur_m ** 2`. Reported as `sqrt(max(S, 0))` in
  metres so a mesh against itself scores ~0, like chamfer.
- Cite Cuturi 2013 (arXiv:1306.0895) and Feydy et al. 2019 (arXiv:1810.08278)
  at the implementation.
- Both meshes are sampled with the **same** seed. Identical meshes then give
  identical clouds and S ~ 0; with independent seeds the self-score would be
  sampling noise on the order of the sample spacing, not zero.
- Both clouds are shifted to one shared origin (min of the pair) and solved in
  float64: at EPSG:28992 magnitudes (y ~ 450,000) float32 squared distances lose
  the metre scale entirely.
- Torch, not numpy: `mesh_metrics.py` is numpy/scipy-only by its own rule, and
  three 2048x2048 log-domain solves per pair are too slow on CPU numpy.
- No new dependency (POT and geomloss are not installed).
- NaN when either mesh has no positive surface area.

### Roof extraction

- A triangle is roof when its winding-derived unit normal has `z > 0.1` — the
  rule `mesh_to_cityjson` uses to label `RoofSurface` (`post_process.py:445`)
  and `mesh_metrics._UP`. One shared helper, applied identically to generated,
  LOD1 and LOD2 meshes.
- `roof_chamfer_m` is NaN when either side has no roof triangles.
- Caveat, recorded in the docstring: winding-based, so a generated triangle with
  flipped winding reads as ground.

### Per-row context columns

`object_id`, `seed`, `n_faces_gen`, `n_faces_lod1`, `n_faces_lod2`,
`eos_emitted`, `decode_ok`, `watertight_gen`, `gen_seconds` (batch wall time
divided by batch size).

### Summary (one row per strategy run in `summary.csv`)

Run name and strategy params; `n_objects`; `watertight_pct` (over all objects);
`decode_rate`; `eos_rate`; for each metric column the nanmean and nanmedian;
`vol_defined_pct` and `roof_defined_pct` per reference.

## 4. Outputs and reproducibility artifacts

```
<output_dir>/<ckpt_stem>/<YYYYmmdd-HHMMSS>/
  eval_config.yaml         resolved eval config
  reference_config.yaml    resolved reference Config, overrides applied
  environment.json         git commit, dirty flag, `git status --porcelain`,
                           python/torch/cuda/numpy/trimesh/manifold3d versions,
                           device, hostname, argv
  git_diff.patch           `git diff HEAD`, written only when dirty
  checkpoint.json          path, sha256, epoch, global_step, hyper_parameters
  split.json               dataset_dir, split sizes, max_objects, sha256 of the
                           ordered evaluated id list
  summary.csv
  <run_name>/objects.csv   one row per object, appended and flushed per batch
  <run_name>/run.json      exact generate() kwargs, max_new_tokens, seed rule,
                           batch_size
```

Untracked files are listed in `git status` but not captured in the patch; that
limit is stated in `environment.json` under a `note` key.

## 5. Units and files

| file | change |
|---|---|
| `src/models/mesh_transformer.py` | `filter_logits`; `generate` gains `top_p`, `min_p`, `seeds` |
| `src/utils/config.py` | `StrategyConfig`, `PosthocEvalConfig` |
| `src/eval/sinkhorn.py` | new: debiased Sinkhorn divergence |
| `src/eval/mesh_metrics.py` | new `roof_submesh`, `hausdorff_max` helpers (numpy) |
| `src/eval/posthoc_eval.py` | new: strategy expansion, loading, per-object scoring, writers, CLI |
| `configs/posthoc-eval.yaml` | new default eval config |
| `tests/test_sampling_filters.py` | new |
| `tests/test_sinkhorn.py` | new |
| `tests/test_roof_and_hausdorff.py` | new |
| `tests/test_posthoc_eval.py` | new |
| `tests/test_metric_sanity_real.py` | new |

## 6. Testing (written first, failing before implementation)

Synthetic, tiny, no real data:

- `filter_logits`: top-p keeps exactly the nucleus on a known distribution;
  min-p keeps exactly the tokens above `min_p * p_max`; a row with masked
  entries is never emptied; `temperature <= 0` raises.
- Seeded `generate` with the scripted network stub from `test_beam_search`:
  same seeds give same tokens; permuting batch rows permutes the outputs;
  `seeds=None` keeps existing behaviour; `beam_size > 1` with `top_p` or
  `min_p` raises.
- Sinkhorn: identical point sets give ~0; a copy translated by `d` gives ~`d`
  (blur small relative to `d`); empty input gives NaN.
- Roof extraction: a closed outward-wound box yields only its two top
  triangles; the same box with flipped winding yields only its two bottom
  triangles (winding, not position, decides roofness); a walls-only mesh
  yields none.
- Strategy expansion: defaults give exactly the 7 named runs; unknown kind,
  param, metric or reference raises.
- End to end with the `_Dataset` stub from `test_mesh_eval`, an untrained tiny
  model and `tmp_path`: one `objects.csv` per run with one row per id and every
  expected column; `summary.csv` has one row per run; all artifact files exist;
  a second run with the same seed produces identical `objects.csv` apart from
  `gen_seconds`.

Real-data sanity and baseline check (`tests/test_metric_sanity_real.py`):

- Reads a small sample only: `MeshDataset("data/The Hague/cleaner",
  max_files=1)` — one tile per LOD, never `MeshDataModule.setup()` — and takes
  the first 8 objects. Skipped when the directory is absent.
- Meshes go through the same tokenizer round trip and the same per-object
  scoring function the eval uses, with every metric enabled.
- LOD1 vs itself and LOD2 vs itself: `chamfer_m`, `hausdorff_p95_m`,
  `hausdorff_max_m`, `roof_chamfer_m` are below 1e-6 m (not exactly 0: the
  proximity query runs on world coordinates at y ~ 450,000); `sinkhorn_m` is
  below `0.1 * blur_m`; where `vol_ok`, `rel_symdiff < 1e-9`.
- Also asserted: at least one sampled object has `vol_ok` on LOD2 vs itself, so
  the volume assertions cannot pass vacuously.
- LOD1 vs its LOD2: every defined value is finite and non-negative and at least
  its self-score; `hausdorff_max_m >= hausdorff_p95_m`;
  `chamfer_m <= hausdorff_max_m`.
- Swapping the arguments: `symdiff_m3` agrees to 1e-3 relative -- exact CSG,
  but measured: manifold3d is bit-commutative on 7 of 8 pairs in the first
  `cleaner` tile and 5.2e-5 off on one, cancellation-amplified from 7e-6 of
  the volume;
  `sinkhorn_m` to 2% (same sampling seed both sides); `chamfer_m` to 5% or 1 cm
  and `hausdorff_p95_m` to 25% or 2 cm, because `surface_distances` draws the
  two sides with seeds `seed` and `seed + 1`, so a swap resamples both clouds.
  The p95 bound is measured, not chosen: re-running one direction under four
  seeds moves p95 by up to 13.9% on the first `cleaner` tile, against a largest
  observed swap gap of 13.3%; chamfer moves at most 10% the same way.
  `hausdorff_max_m` is not compared: a max over samples is not stable under
  resampling.
- Zero is allowed for LOD1 vs LOD2: 31.8% of `cleaner` pairs are effectively
  identical (identity census), so strict positivity is not asserted.
- Prints the per-object LOD1-vs-LOD2 table and its means (visible with
  `pytest -s`) as the do-nothing baseline for every metric.
