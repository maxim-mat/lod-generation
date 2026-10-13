# Post-hoc mesh eval: running guide

How to score a finished mesh-transformer checkpoint under several decoding
strategies. For the design decisions read
`docs/superpowers/specs/2026-09-15-posthoc-mesh-eval-design.md`; for the
implementation steps, `docs/superpowers/plans/2026-09-15-posthoc-mesh-eval.md`.

**What it is.** `src/eval/mesh_eval.py` scores greedy or beam decoding on a
small subsample *during* training, to drive early stopping. This scores a
finished checkpoint on a whole split under a grid of strategies, writes one CSV
row per building, and records everything needed to reproduce any row.

One command:

```bash
python -m src.eval.posthoc_eval --config configs/posthoc-eval-mesh-v5-opt-coord.yaml
```

---

## 1. How the config layers

The eval YAML holds **only** eval settings and names a training config:

```yaml
reference_config: configs/mesh-v5-full-opt-coord.yaml
checkpoint: outputs/initial-runs/mesh-v5-full-opt-coord/checkpoints/last.ckpt
```

Corpus, split ratios, split seed, tokenizer, bins, margins and model shape all
come from that reference, loaded with the same `load_config` training uses. So
the split is rebuilt exactly as the run saw it, and the eval cannot silently
describe a different dataset than the weights were fit on.

When the corpus has moved, override the reference rather than editing it:

```yaml
reference_overrides: ["mesh_data.dataset_dir=data/The Hague/mini_remote"]
```

Any field can also be overridden on the command line:

```bash
python -m src.eval.posthoc_eval --config configs/posthoc-eval-mesh-v5-opt-coord.yaml \
    max_objects=8 device=cpu sinkhorn.n_points=1024
```

Shipped configs: `configs/posthoc-eval.yaml` (defaults, points at the
from-scratch V5 arm) and `configs/posthoc-eval-mesh-v5-opt-coord.yaml` (the
OPT arm).

---

## 2. Preflight

```bash
python -m pytest tests/ -q     # expect 773 passed, 1 skipped (~90 s)
```

Then check three things the eval cannot invent for you:

- **The checkpoint exists.** `save_last: True` is set in `setup_utils`, so every
  run has `checkpoints/last.ckpt`. For the *selected* model use the
  best-scoring file, `{epoch:02d}-{val_gen_chamfer_m:.4f}.ckpt` in the same
  directory — these runs select on `val_gen_chamfer_m`, so the last epoch is
  not necessarily the best one. **As of 2026-09-16 no `mesh-v5-*` run directory
  exists locally**; the V5 configs point at where the run will write.
- **The corpus is where the reference config says.** `data/The Hague/cleaner`
  for the V5 arms. Older mesh runs name `mini`, which is now `mini_remote` —
  that is what `reference_overrides` is for.
- **The tokenizer matches.** The eval refuses a checkpoint whose saved
  `tokenization` differs from the reference config's, because nothing
  downstream reconciles them: a coordinate sequence decoded as AMT keeps its
  vertices and silently re-triangulates into ~1.8x the faces.

Every config-level mistake (unknown metric, unknown strategy kind, a param that
does not belong to its kind, missing checkpoint, non-`mesh` reference) fails
before the dataset is loaded. A bad config costs seconds, not an hour.

---

## 3. What gets run

List-valued strategy params expand as a cartesian product. The default grid is
7 runs:

| run | decoding |
|---|---|
| `greedy` | argmax |
| `beam_k3_lp1.0`, `beam_k5_lp1.0` | beam search, length penalty 1.0 |
| `top_p_p0.95_t0.6`, `top_p_p0.95_t1.2` | nucleus 0.95 at two temperatures |
| `min_p_p0.05_t0.6`, `min_p_p0.05_t1.2` | min-p 0.05 at two temperatures |

Sampled runs draw **one** generation per object, seeded
`crc32("<seed>:<object_id>")` — the same seed in every strategy, so strategies
are compared on common random numbers, and a row does not depend on which
buildings shared its batch.

Allowed params: `beam` takes `beam_size` and `length_penalty` (default `[1.0]`);
`top_p` takes `top_p` and `temperature`; `min_p` takes `min_p` and
`temperature`; `greedy` takes none.

---

## 4. Cost, and how to calibrate it

**Not measured yet — no real checkpoint has been scored.** What is known: each
object costs up to `9 * max_faces + 1` = 1801 sequential forward passes at
`max_faces: 200`, beam-5 carries five candidates through that loop, and the
KV cache is on by default. The val split of `cleaner` is ~25,892 objects, so
the full grid over the whole split is a multi-day job.

Calibrate instead of guessing:

```bash
python -m src.eval.posthoc_eval --config configs/posthoc-eval-mesh-v5-opt-coord.yaml \
    max_objects=8
```

Then read `gen_seconds` (per object, per strategy) out of the CSVs and scale.
The metric side costs real time too: Sinkhorn runs three n×n log-domain solves
per reference per object, so on `device: cpu` set `sinkhorn.n_points=1024`.

There is **no resume**. `objects.csv` is appended and flushed after every
batch, so a crash keeps the finished rows, but re-running starts from the top.

---

## 5. Output layout

```
outputs/initial-runs/<run>/posthoc_eval/<ckpt-stem>/<timestamp>/
  eval_config.yaml         resolved eval config
  reference_config.yaml    resolved training config, overrides applied
  environment.json         git commit + dirty flag + status, library versions, device, argv
  git_diff.patch           written only when the tree was dirty
  checkpoint.json          path, sha256, epoch, global_step, hyper_parameters
  split.json               dataset_dir, split sizes, sha256 of the evaluated id list
  summary.csv              one row per strategy
  <run_name>/objects.csv   one row per building
  <run_name>/run.json      exact generate() kwargs, budget, seeds, batch_size
```

`objects.csv` columns: `object_id`, `seed`, face counts for generated/LOD1/LOD2,
`eos_emitted`, `decode_ok`, `gen_seconds`, `watertight_gen`, then each metric
suffixed `_vs_lod1` and `_vs_lod2`:

| column | meaning |
|---|---|
| `chamfer_m` | mean bidirectional surface error, metres, unsquared |
| `hausdorff_p95_m` / `hausdorff_max_m` | 95th percentile and true worst case |
| `symdiff_m3`, `rel_symdiff`, `vol_ok`, `vol_note` | exact CSG symmetric difference; NaN unless both solids are closed |
| `sinkhorn_m` | debiased Sinkhorn divergence, metres |
| `roof_chamfer_m` | chamfer between the upward-facing parts only |

`summary.csv` adds mean and median per metric, `watertight_pct`, `decode_rate`,
`eos_rate`, and `vol_defined_pct` / `roof_defined_pct` — read those last two
before trusting a volume or roof average, since they say what fraction of the
split the average covers.

Both references are the **tokenizer round trip**, as `MeshEvalCallback` uses,
so these numbers are comparable with the `val_gen_chamfer_m` curves in W&B.

---

## 6. Reading the numbers

Scored against LOD2 the metrics say how close the generation is to the truth.
Scored against LOD1 they say how far it moved from its own input — a run that
matches LOD1 more closely than LOD2 has learned to copy its condition.

The do-nothing floor, measured on the first `cleaner` tile (8 buildings,
`tests/test_metric_sanity_real.py`, means of LOD1 scored against its own LOD2):

| metric | baseline |
|---|---|
| chamfer | 0.171 m |
| p95 Hausdorff | 1.032 m |
| max Hausdorff | 1.764 m |
| Sinkhorn | 1.005 m |
| roof chamfer | 0.587 m |
| relative symmetric difference | 0.110 |
| watertight | 8/8 |

A generation that does not beat these has added nothing its input did not
already say. That tile is 8 buildings, not a corpus estimate — re-run that test
with more objects before quoting it as one.

Two properties of the metrics worth knowing before reading small differences:

- **p95 Hausdorff carries ~14% resampling noise** at `n_points: 4096`. Measured:
  re-running the same comparison under four different sampling seeds moves it by
  up to 13.9%, while chamfer moves at most 10%. Do not read a 5% p95 gap between
  two strategies as a result.
- **Volume symmetric difference is exact** (manifold3d, float64), but only where
  both meshes are closed. Generated meshes frequently are not, which is why
  `watertight_pct` and `vol_defined_pct` sit next to it.

---

## 7. Reproducing a row later

Everything needed is in the run directory: `eval_config.yaml` and
`reference_config.yaml` are the resolved inputs, `checkpoint.json` pins the
weights by sha256, `split.json` pins the evaluated id list by sha256, and
`run.json` holds the exact `generate()` kwargs and the seed rule.

Re-running with the same eval config, checkpoint and **batch size** reproduces
`objects.csv` exactly, apart from `gen_seconds` — pinned by
`tests/test_posthoc_eval.py::test_a_rerun_reproduces_every_row`. Batch size is
part of the recipe because batched float kernels can differ at ~1e-6 across
batch sizes; the per-object seeding removes the RNG dependence, not that one.
