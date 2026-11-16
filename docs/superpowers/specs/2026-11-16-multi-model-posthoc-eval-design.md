# Multi-model post-hoc eval: design

## Intent

Score several trained mesh-transformer models (v5-tile, v6 ortho-prefix-aligned,
v6 xattn-a, v6 xattn-b) and a do-nothing LOD1-copy baseline on their test splits
with one command. The output is one table of means and one table of rows per
building for each model, plus a few interactive examples of good, average and
bad results. This extends `src/eval/posthoc_eval.py` instead of adding a new tool.

## Goals

- `posthoc_eval` takes a list of training configs and a matching list of
  checkpoints, and scores each pair on that config's own split.
- New metrics per building: position-free face accuracy, precision, recall and
  F1, the same four on roof faces only, and two LOD1-copy measures normalized by
  the truth.
- A `lod1_copy` run per model, scored exactly like a generated mesh.
- 5 good, 5 average and 5 bad buildings per model x strategy, saved as
  interactive plotly HTML showing LOD1 | LOD2 | generated.

## Non-goals

- Splits. Each model is scored on whatever split its own config defines. Making
  the models comparable across splits is out of scope.
- Teacher-forced token accuracy. The position-free face metrics replace it.
- Roof-area IoU. The xy shadow of a roof is essentially the footprint, so it
  would sit near 1 for every model, as `footprint_iou` already does. Roof chamfer
  and roof face F1 cover roof shape.

## Anti-goals

- Breaking `src/analysis/ortho_strata.py`. It reads `<model dir>/<strategy>/objects.csv`,
  so that layout stays.
- Dropping any reproducibility artifact the script writes today. That covers the
  configs, git state, checkpoint sha, split ids and seeds, and they are now
  written once per model.

## Approaches

1. **Loop over `(config, checkpoint)` pairs inside `main`, keeping today's
   per-model eval, and add a combined summary on top.** Chosen. It is the
   smallest change, and each model's directory keeps the existing layout.
2. A wrapper script that runs the current CLI once per model. Rejected, because
   the user asked for the script itself to take the lists.
3. A separate eval notebook. Rejected by the user in favour of changing the script.

## Design

**Config (`PosthocEvalConfig`).**
- `reference_config: str` becomes `reference_configs: List[str]`.
- `checkpoint: str` becomes `checkpoints: List[Optional[str]]`. It must have the
  same length, checked before anything loads.
- A `null` checkpoint resolves to the lowest `val_gen_chamfer_m` file in that
  run's `checkpoints/`, falling back to `last.ckpt`.
- `reference_overrides` applies to every config.
- The default `split` becomes `test`.
- The shipped `posthoc-eval*.yaml` configs move to the list form.
- A new `configs/posthoc-eval-all.yaml` lists the four models.

**Per-building metrics (new columns in `objects.csv`).** Discrete metrics compare
vertices in quantized-bin space. The decoded unit-box vertices are re-quantized,
which is exact. Generated, truth and LOD1 meshes share each building's frame.

| column | definition |
|---|---|
| `face_acc` | \|G ∩ T\| / \|G ∪ T\|, with faces as multisets (G = generated, T = truth). There are no true negatives, so accuracy is the Jaccard index. |
| `face_precision`, `face_recall`, `face_f1` | \|G ∩ T\| / \|G\|, then / \|T\|, then their harmonic mean |
| `roof_face_{acc,precision,recall,f1}` | the same, restricted to faces whose winding normal has z > 0.1, each mesh by its own normals (the `roof_submesh` rule) |
| `lod1_vert_copy_rate` | \|V_gen ∩ V_lod1\| / \|V_gen\|, over unique vertex bins |
| `lod1_vert_copy_norm` | `lod1_vert_copy_rate(gen) / lod1_vert_copy_rate(truth)`. Above 1 means it copies LOD1 more than the truth does. |
| `lod1_chamfer_norm` | `chamfer_m_vs_lod1(gen) / chamfer(LOD1, LOD2)`, where the denominator is `chamfer_floor`. Below 1 means it is closer to LOD1 than the truth is (over-copying). |

A face is the unordered set of its three vertex bin-triplets, so position in the
sequence, vertex order within the face and winding are all ignored. Values are
never an exception.
- The face metrics are NaN only when both sides are empty, for example no roof
  faces on either side. When exactly one side is empty, as with a failed
  generation, they score 0, so NaN-skipping means do not inflate the runs that
  fail most. This was decided at plan time.
- The copy measures are NaN when their denominator is empty or zero.

These columns sit beside the existing ones, which are unchanged: face count,
chamfer, Hausdorff, volume symmetric difference, Sinkhorn and roof chamfer,
scored against both LOD1 and LOD2.

**Baseline.** For each model, the run `lod1_copy` uses the decoded LOD1
condition as the "generation" and goes through the same scoring. It has no
generation step, and its seeds and `gen_seconds` are recorded as such.

**Tables.**
- Per model: `summary.csv` holds the mean and median of every metric for each
  strategy and for `lod1_copy`, as today plus the new columns.
- `<strategy>/objects.csv` holds one row per building.
- At the top level, `summary.csv` concatenates every model's summary with a
  `model` column. That is the main table.

**Samples.**
- For each model x strategy, buildings are ranked by `roof_chamfer_m_vs_lod2`.
  Buildings where it is NaN are excluded from the ranking.
- good = the 5 lowest, avg = the 5 nearest the median, bad = the 5 highest.
- Each sample is written to `<model dir>/<strategy>/samples/{good,avg,bad}_<rank>_<id>.html`.
  It is a three-panel `side_by_side` (`src/visualize_mesh.py`) of LOD1, LOD2 and
  the generation, with metrics in the titles and plotly loaded from the CDN.
- Generated meshes are held in memory until the run ends, so there is no
  regeneration. A crash keeps the CSV rows but loses the samples.
- Raw roof chamfer is in metres, so small buildings lean towards "good". The
  user chose this.

**Output layout.**

```
<output_dir or outputs/posthoc_eval>/<timestamp>/
  summary.csv                       all models, one row per model x strategy
  <run_name>/                       per model, today's layout and artifacts
    eval_config.yaml  reference_config.yaml  environment.json  checkpoint.json  split.json
    summary.csv
    <strategy>/objects.csv  run.json  samples/*.html
    lod1_copy/objects.csv   samples/*.html
```

## Constraints

- Generation over a full test split takes days per model on GPU, and this
  machine has no GPU. The script must work unchanged on the remote box, and a
  `max_objects=8` smoke run must work on CPU.
- Impact: `PosthocEvalConfig`, `run_posthoc_eval` and `main` are MEDIUM risk.
  Their callers are `main`, `tests/test_posthoc_eval.py`, the two shipped eval
  configs and `src/inference.py` (helper imports only).

## Left to the builder

- Function boundaries. Expected: one pure `face_set_metrics(gen_bins, gt_bins, lod1_bins)`
  in `mesh_metrics.py`, called from `_score_batch`.
- Exact column order and summary naming, following the existing `_summarize`
  pattern.
- How `lod1_copy` reuses the batch and scoring code path.

## Revision 2026-11-16: aligned with `src/eval/face_match.py`

`main` gained `src/eval/face_match.py` (2026-11-13), the project's own definition
of face matching and face-level copying, while this branch was in flight. To keep
a single definition of "face precision" and "copy rate" in the project, the
discrete columns above were replaced by `face_match.face_scores`:

- A face is `face_set`'s canonical 9-token row. Reversed winding is a different
  face, faces collapsed by quantization are dropped, and matching uses sets, not
  multisets.
- `face_match`'s counts and rates are written as they are (`gen_precision`,
  `gen_recall`, `gen_copy`, `gt_copy`, ...). Precision and copy rates are NaN on
  an empty denominator, which is that module's convention.
- Added on top: `face_acc` (Jaccard) and `face_f1` from the counts. Both are 0
  when one side is empty, so failed generations still count. Also added:
  `gen_copy_norm = gen_copy / gt_copy` and the four `roof_face_*` columns.
- The vertex-level `lod1_vert_copy_rate` / `_norm` are dropped. The module's
  rationale: "a shared vertex says little, a shared face is something the model
  could have echoed".
- `summary.csv` adds face_match's micro averages (`<rate>_micro`).
