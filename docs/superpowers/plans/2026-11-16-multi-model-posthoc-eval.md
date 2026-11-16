# Multi-model post-hoc eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `python -m src.eval.posthoc_eval` scores a list of (training config, checkpoint) pairs plus a LOD1-copy baseline on each config's own split. It writes per-building and mean tables with new face, roof-face and LOD1-copy metrics, and saves good, average and bad samples as plotly HTML.

**Architecture:** One pure numpy function, `face_set_metrics`, in `src/eval/mesh_metrics.py` computes the discrete metrics in quantized-bin space. `posthoc_eval._score_batch` takes decoded unit-box meshes, so a generated sequence and the LOD1 condition go through the same scoring path. `run_posthoc_eval` adds a `lod1_copy` run and writes samples. `main` checks every model first, then loops over them and writes a combined `summary.csv`.

**Tech Stack:** numpy, pandas, torch, plotly (already used by `src/visualize_mesh.py`), OmegaConf structured configs, pytest.

**Spec:** `docs/superpowers/specs/2026-11-16-multi-model-posthoc-eval-design.md`

## Global Constraints

- Python: the base conda env, CPU-only on this machine (memory `env-base-conda-cpu`).
- Run test files **one at a time**. The full suite aborts on a pre-existing MKL fault (memory `pytest-mkl-abort-run-per-file`). Run tests sparingly.
- `src/eval/mesh_metrics.py` stays numpy/scipy/trimesh only: no torch, no `src.dataset` import.
- Before editing any existing symbol, run `impact({target, direction: "upstream"})`. Stop and ask if it reports HIGH or CRITICAL.
- Before each commit, run `detect_changes({scope: "all"})`. A `partial` or `truncated` result does not count as clean.
- Work on branch `feat/multi-model-posthoc-eval`, created from `main` before Task 1. Do not commit `configs/mesh6-train.yaml`, which is untracked and belongs to the user.
- Keep every reproducibility artifact written today (`eval_config.yaml`, `reference_config.yaml`, `environment.json`, `git_diff.patch`, `checkpoint.json`, `split.json`, `run.json`, seeds), now written once per model.
- Roof means a winding normal with `z > 0.1` (`_UP`), measured in metres.
- Samples are ranked by `roof_chamfer_m_vs_lod2`, 5 per bucket, and buildings where it is NaN are excluded.

## Review Focus

1. **A typo in the 4th model's checkpoint.** It must fail before any data loads or any model generates, because the first three models take days. Task 4 adds `test_main_preflights_every_model_before_loading_data`.
2. **A failed generation (empty mesh) scored for face metrics.** It must score 0, not NaN. Otherwise NaN-skipping means inflate exactly the runs that fail most. NaN is only correct when both sides are empty. Task 1 adds `test_one_empty_side_scores_zero_and_two_empty_sides_are_nan`.
3. **Object ids with characters that are illegal in Windows file names** (`:`, `/`). Sample file names must be sanitized. Task 3 adds `test_sample_file_names_are_sanitized`.
4. **Two entries resolving to the same `logging.run_name`** would write into one directory and silently mix CSVs. Reject that in preflight. Task 4 adds `test_duplicate_run_names_are_rejected_in_preflight`.
5. **A split with fewer than 15 rankable buildings** (smoke runs, `max_objects=8`). Sampling must return smaller buckets, not crash. Task 3 adds `test_select_samples_with_few_rows`.

---

### Task 1: `face_set_metrics` in `mesh_metrics.py`

**Files:**
- Modify: `src/eval/mesh_metrics.py` (`roof_submesh` at lines 377-406; new functions after it)
- Test: `tests/test_face_set_metrics.py` (new)

**Interfaces:**
- Produces: `face_set_metrics(gen, gt, lod1, scale) -> dict[str, float]`. `gen`, `gt` and `lod1` are each `(vert_bins [V,3] int, faces [F,3] int)`. `scale` is `[3]` metres per unit-box axis. It returns exactly these keys, in this order: `face_acc, face_precision, face_recall, face_f1, roof_face_acc, roof_face_precision, roof_face_recall, roof_face_f1, lod1_vert_copy_rate, lod1_vert_copy_norm`.
- Produces: `FACE_COLUMNS` (module-level tuple of those 10 keys, in that order).
- Produces: `roof_mask(verts, faces, up=_UP) -> np.ndarray[bool] [F]`. `roof_submesh` now uses it, and its behavior is unchanged.

- [ ] **Step 0: Impact.** Run `impact({target: "roof_submesh", direction: "upstream"})` and report the result. Expected callers are `score_object` and `tests/test_roof_and_hausdorff.py`.

- [ ] **Step 1: Write the failing tests** in `tests/test_face_set_metrics.py`. Build meshes from integer bins directly. Use `S = np.ones(3)` unless stated.
  - `test_order_vertex_order_and_winding_are_ignored`: `gen` is `gt` with the face rows permuted, each face rolled by one and one face reversed (`f[::-1]`). Assert that every `face_*` value is `1.0`.
  - `test_partial_match`: `gt` has 4 distinct faces, `gen` has 2 of them plus 1 that is not in `gt`. Assert `face_precision == 2/3`, `face_recall == 0.5`, `face_f1 == pytest.approx(4/7)` and `face_acc == pytest.approx(2/5)`.
  - `test_faces_are_a_multiset`: `gen` has face X twice and `gt` has it once. Assert `face_precision == 0.5` and `face_recall == 1.0`.
  - `test_faces_match_on_coordinates_not_indices`: `gen` repeats `gt`'s geometry with a shuffled vertex array and remapped indices. Assert `face_f1 == 1.0`.
  - `test_roof_classification_uses_metre_scale`: one triangle `(0,0,0),(10,0,0),(0,1,10)` in both `gen` and `gt`. Its bin-space normal z is about 0.0995. With `scale=[1,1,1]`, all `roof_face_*` are NaN, because there are no roof faces on either side. With `scale=[1,2,1]`, the normal z is about 0.196, so `roof_face_f1 == 1.0`.
  - `test_roof_metrics_see_only_roof_faces`: `gt` is a closed box (12 faces, top 2 are roof). `gen` is the same box with the 2 top faces replaced by a lifted pair. Assert `face_recall == 10/12` and `roof_face_recall == 0.0`.
  - `test_one_empty_side_scores_zero_and_two_empty_sides_are_nan`: if `gen` is empty and `gt` is not, all four `face_*` are `0.0`. If both are empty, all four are NaN.
  - `test_lod1_vert_copy_rate_and_norm`: `lod1` has vertices L. `gt` has 4 unique vertices, 2 of them in L. `gen` has 4 unique vertices, 3 of them in L. Assert `lod1_vert_copy_rate == 0.75` and `lod1_vert_copy_norm == 1.5`. If `gt` shares no vertex with L, the norm is NaN. If `gen` is empty, the rate and the norm are NaN.

- [ ] **Step 2: Run them and confirm they fail.** Run `python -m pytest tests/test_face_set_metrics.py -q`. Expected: an ImportError on `face_set_metrics`.

- [ ] **Step 3: Implement.**
  - `roof_mask`: lift the normal computation out of `roof_submesh` (lines 395-401) and have `roof_submesh` call it.
  - `face_set_metrics` defines a face key as `tuple(sorted(map(tuple, vert_bins[f])))`, so it matches on coordinates and ignores order and winding.
  - Match multisets with `collections.Counter`: intersection `sum((G & T).values())` and union `sum((G | T).values())`.
  - Shared rule for `(acc, p, r, f1)` over G and T:
    - If `|G| + |T| == 0`, all four are NaN.
    - Otherwise `p = |G∩T|/|G|`, or `0` if `|G| == 0`. `r` is the same over `|T|`.
    - `f1 = 2pr/(p+r)`, or `0` when `p + r == 0`. `acc = |G∩T| / |G∪T|`.
  - The roof subsets select each mesh's faces with `roof_mask(vert_bins * scale, faces)`.
  - Copy rate uses the sets of unique vertex rows. Use NaN when the denominator set is empty or the truth's rate is 0.

- [ ] **Step 4: Run the new tests and the roof tests, and confirm they pass.** Run `python -m pytest tests/test_face_set_metrics.py -q`, then `python -m pytest tests/test_roof_and_hausdorff.py -q`. Expected: all pass.

- [ ] **Step 5: Commit.** First run `detect_changes`.
  ```bash
  git add src/eval/mesh_metrics.py tests/test_face_set_metrics.py
  git commit -m "feat(eval): position-free face, roof-face and LOD1-copy metrics"
  ```

---

### Task 2: Score face metrics and add the `lod1_copy` run

**Files:**
- Modify: `src/eval/posthoc_eval.py` (`DecodeRun.generate_kwargs` at 79-87, `_summarize` at 338-357, `_score_batch` at 360-384, `run_posthoc_eval` at 387-472)
- Test: `tests/test_posthoc_eval.py`

**Interfaces:**
- Consumes: `face_set_metrics` and `FACE_COLUMNS` (Task 1); `chamfer_floor` and `chamfer_ratio` (`mesh_metrics.py`); `quantize` (`src.dataset.mesh_dataset`).
- Produces: `LOD1_COPY = "lod1_copy"`, the run name and kind of the baseline.
- Produces: `_score_batch(model, batch, gen_unit, eos_emitted, cfg, seeds, seconds, device) -> (rows: list[dict], meshes: dict[object_id, {"lod1", "lod2", "gen"}])`.
  - `gen_unit` is a list of unit-box `(verts, faces)`, one per item.
  - `eos_emitted` is a `list[bool]`, or `None` for `lod1_copy`, which writes NaN.
  - `meshes` holds metre meshes for Task 3.
  - It calls `model._prepare(batch)` and `model._item_cond(...)` itself. The current `cond/tgt/cond_pad` parameters go away.
- New `objects.csv` columns, appended after today's columns in this order: the 10 `FACE_COLUMNS`, then `lod1_chamfer_norm`.
  - `lod1_chamfer_norm` is `chamfer_ratio(row["chamfer_m_vs_lod1"], chamfer_floor(lod1, lod2, cfg.n_points, SURFACE_SEED))`.
  - It is only written when `chamfer_m_vs_lod1` is in the row.

- [ ] **Step 0: Impact.** Run impact upstream on `_score_batch`, `_summarize`, `run_posthoc_eval` and `DecodeRun` and report the results. The 2026-11-16 check found `run_posthoc_eval` at MEDIUM.

- [ ] **Step 1: Write the failing tests and update the existing ones** in `tests/test_posthoc_eval.py`.
  - `OBJECT_COLUMNS` gets `list(FACE_COLUMNS) + ["lod1_chamfer_norm"]` appended.
  - `test_end_to_end_writes_one_row_per_object_per_run`:
    - The expected run list becomes `names + ["lod1_copy"]`, with `len(table) == 8`.
    - `lod1_copy/objects.csv` exists with the same columns.
    - `{"face_f1_mean", "roof_face_f1_mean", "lod1_vert_copy_norm_mean", "lod1_chamfer_norm_mean"} <= set(table.columns)`.
  - `test_lod1_copy_scores_the_condition_as_its_generation`: run greedy only with `metrics=["chamfer"]`. In `lod1_copy/objects.csv`:
    - `n_faces_gen == n_faces_lod1`
    - `chamfer_m_vs_lod1 == pytest.approx(0, abs=1e-9)`
    - `lod1_vert_copy_rate == 1.0`
    - `eos_emitted` is NaN
  - `test_a_perfect_generation_scores_perfectly`:
    - Wrap `model._prepare` so it stashes `tgt`, and monkeypatch `type(model).generate` to return the stash.
    - In `greedy/objects.csv`: `face_f1 == 1.0`, `face_acc == 1.0` and `chamfer_m_vs_lod2 == pytest.approx(0, abs=1e-9)`.
  - `test_a_rerun_reproduces_every_row` keeps passing unchanged. It now also covers the new columns.

- [ ] **Step 2: Run them and confirm they fail.** Run `python -m pytest tests/test_posthoc_eval.py -q`. Expected: the 3 tests above fail, on a column mismatch or a missing `lod1_copy`.

- [ ] **Step 3: Implement.**
  - `generate_kwargs` returns `{}` for kind `lod1_copy`.
  - `run_posthoc_eval` appends `DecodeRun(LOD1_COPY, LOD1_COPY, {})` after the expanded runs.
    - For that run it skips `generate` and passes the decoded condition as `gen_unit`. Decode it with `amt_detokenize`/`detokenize` on `batch["cond"][k]`, as `_score_batch` does today.
    - `run.json` records `generate_kwargs: {}`.
  - `_score_batch`:
    - The vertex bins are `quantize(unit_verts, model.num_bins)` for gen, lod2 and lod1.
    - Call `face_set_metrics(..., scale=batch["scale"][k])`.
    - Keep the metre conversion as today.
  - `_summarize` adds `_mean`/`_median` for each `FACE_COLUMNS` entry and for `lod1_chamfer_norm`, when present.
  - `decode_rate`/`eos_rate` stay as they are. Pandas `mean` skips the NaN `eos_emitted` of `lod1_copy`.

- [ ] **Step 4: Run the tests and confirm they pass.** Run `python -m pytest tests/test_posthoc_eval.py -q`. Expected: all pass.

- [ ] **Step 5: Commit** after `detect_changes`: `git commit -m "feat(eval): face metrics, LOD1 chamfer norm and lod1_copy baseline in posthoc_eval"`.

---

### Task 3: Good, average and bad samples as HTML

**Files:**
- Modify: `src/eval/posthoc_eval.py` (new functions; called at the end of each run in `run_posthoc_eval`)
- Test: `tests/test_posthoc_eval.py`

**Interfaces:**
- Consumes: the `meshes` dict from `_score_batch` (Task 2); `mesh_figure` and `side_by_side` from `src/visualize_mesh.py`, imported inside the function as `load_model` does.
- Produces: `select_samples(df, column="roof_chamfer_m_vs_lod2", k=5) -> dict[str, list[str]]` with keys `good`, `avg` and `bad`.
  - It returns object ids and drops NaN rows.
  - `good` is the k lowest, best first. `bad` is the k highest, worst first. `avg` is the k with the smallest `|value - median|`, closest first.
  - Buckets may overlap when there are fewer than 3k rows.
- Produces: `write_samples(run_dir, picks, meshes, rows) -> list[Path]`.
  - It writes `run_dir/samples/{bucket}_{rank}_{safe_id}.html`, with rank starting at 1 and `safe_id = re.sub(r"[^\w.-]", "_", id)`.
  - Each file is `side_by_side([mesh_figure(*lod1), mesh_figure(*lod2), mesh_figure(*gen)], titles)`, written with `write_html(..., include_plotlyjs="cdn")`.
  - The titles show the face counts and the panel names "LOD1 input", "LOD2 truth" and "generated (<run>)". The figure title shows `roof_chamfer_m_vs_lod2`, `chamfer_m_vs_lod2` and `face_f1`.

- [ ] **Step 1: Write the failing tests.**
  - `test_select_samples_buckets`: a frame with ids `o0..o20` valued 0..20 plus 2 NaN rows. Assert `good == ["o0".."o4"]`, `bad == ["o20","o19","o18","o17","o16"]`, `set(avg) == {"o8","o9","o10","o11","o12"}` and `avg[0] == "o10"`.
  - `test_select_samples_with_few_rows`: 3 finite rows. Each bucket has 3 ids and nothing raises.
  - `test_sample_file_names_are_sanitized`: `write_samples` on one synthetic box triple with id `"a/b:c"` writes `samples/good_1_a_b_c.html`, and its text contains `"plotly"`.
  - `test_samples_are_written_per_run`: use the perfect-generation spy from Task 2 with `metrics=["chamfer", "roof_chamfer"]`. Assert that `greedy/samples/*.html` is non-empty.

- [ ] **Step 2: Run them and confirm they fail.** Run `python -m pytest tests/test_posthoc_eval.py -q`. Expected: an ImportError on `select_samples`.

- [ ] **Step 3: Implement both functions.**
  - Call them once per run, `lod1_copy` included, after the run's CSV is complete.
  - Accumulate `meshes` across batches in a dict local to the run, dropped after the samples are written.
  - If the ranking column is absent (no `roof_chamfer` metric or no `lod2` reference), log one warning and skip samples.

- [ ] **Step 4: Run the tests and confirm they pass.** Run `python -m pytest tests/test_posthoc_eval.py -q`.

- [ ] **Step 5: Commit** after `detect_changes`: `git commit -m "feat(eval): good/avg/bad samples ranked by roof chamfer, as plotly HTML"`.

---

### Task 4: List of configs and checkpoints, combined table, configs and docs

**Files:**
- Modify: `src/utils/config.py` (`PosthocEvalConfig` at 421-448)
- Modify: `src/eval/posthoc_eval.py` (`validate_posthoc_config` at 147-170, `main` at 541-585; new `resolve_checkpoint`)
- Modify: `configs/posthoc-eval.yaml`, `configs/posthoc-eval-mesh-v5-opt-coord.yaml`
- Create: `configs/posthoc-eval-all.yaml`
- Modify: `docs/posthoc-eval-runbook.md` (§1, §5, §6), `docs/ortho-conditioning-runbook.md` (the post-hoc paragraph near line 130)
- Test: `tests/test_posthoc_eval.py`

**Interfaces:**
- `PosthocEvalConfig`:
  - `reference_configs: List[str] = MISSING` and `checkpoints: List[Optional[str]] = MISSING` replace `reference_config` and `checkpoint`.
  - `split` defaults to `"test"`.
  - The `output_dir` comment changes to say that `None` means `outputs/posthoc_eval`.
- `validate_posthoc_config` also raises `ValueError` mentioning `reference_configs` when the list is empty, and `checkpoints` when the lengths differ.
- Produces `resolve_checkpoint(ref_cfg, checkpoint: Optional[str]) -> Path`.
  - Given a path, it returns that path if the file exists, else raises `FileNotFoundError`.
  - Given `None`, it looks in `<save_dir>/<experiment_name>/<run_name>/checkpoints/` for the file with the lowest `val_gen_chamfer_m=<x>.ckpt` (regex `val_gen_chamfer_m=([0-9.]+)\.ckpt$`). Then it falls back to `last.ckpt`, and otherwise raises `FileNotFoundError` naming the directory.
- `main(argv) -> Path` returns the root `<output_dir or outputs/posthoc_eval>/<stamp or mode-stamp>`.
  - Phase 1 is preflight, for every pair and before anything loads:
    - `load_config` and the `config_set == "mesh"` check
    - `resolve_checkpoint`, `checkpoint_info`, `check_tokenization`, `check_vertex_margins`
    - unique `logging.run_name`, else `ValueError` mentioning `run_name`
  - Phase 2, for each pair: `build_split`, `load_model`, then `run_posthoc_eval(..., out_dir=root / run_name)`. Write `root/summary.csv` after each model, so a crash keeps the finished models. The rows get a `model` column (the run_name) as the first column.
  - Free the model between pairs (`del model`, plus `torch.cuda.empty_cache()` when on CUDA).

- [ ] **Step 0: Impact.** Run impact upstream on `PosthocEvalConfig` (MEDIUM on 2026-11-16), `validate_posthoc_config` and `main`, and report the results.

- [ ] **Step 1: Write the failing tests and migrate the existing ones.**
  - `_cfg` defaults to `reference_configs=["unused.yaml"]` and `checkpoints=["unused.ckpt"]`.
  - Every YAML string in the tests moves to the list form.
  - `test_loading_requires_a_checkpoint` checks that `checkpoints` is missing.
  - `test_validation_fails_fast` gains `({"checkpoints": ["a", "b"]}, "checkpoints")` and `({"reference_configs": [], "checkpoints": []}, "reference_configs")`.
  - `test_the_shipped_config_is_valid` becomes `test_every_shipped_posthoc_config_is_valid`. It globs `configs/posthoc-eval*.yaml`, loads and validates each one, and asserts that the list lengths match.
  - `test_resolve_checkpoint`, in `tmp_path`:
    - files `epoch=03-val_gen_chamfer_m=0.2100.ckpt`, `epoch=05-val_gen_chamfer_m=0.1800.ckpt` and `last.ckpt`: picks `0.1800`;
    - only `last.ckpt`: picks it;
    - empty directory: `FileNotFoundError`;
    - an explicit missing path: `FileNotFoundError`.
  - `test_main_scores_every_model_into_one_table`:
    - Two reference YAMLs, `"config_set: mesh\nlogging: {run_name: m1}\n"` and `m2`, with two `torch.save`d checkpoints whose `hyper_parameters` are `{}`.
    - Strategies `[{kind: greedy}]`, `metrics: [chamfer]`, `output_dir: <tmp>`.
    - Monkeypatch `posthoc.build_split` to return `(_Dataset(), {"train": 0, "val": 0, "test": 2})` and `posthoc.load_model` to return `_model()`.
    - Assert that `root/summary.csv` has `model == ["m1","m1","m2","m2"]` and `run == ["greedy","lod1_copy"]*2`, and that `root/m2/greedy/objects.csv` and `root/m1/checkpoint.json` exist.
  - `test_main_preflights_every_model_before_loading_data`:
    - The first pair is valid and the second checkpoint is missing.
    - `build_split` is monkeypatched to `pytest.fail`.
    - Expect `FileNotFoundError`.
    - This replaces `test_main_checks_the_checkpoint_before_loading_data`.
  - `test_duplicate_run_names_are_rejected_in_preflight`: two pairs with the same run_name raise `ValueError` matching `run_name`, again with `build_split` set to `pytest.fail`.

- [ ] **Step 2: Run them and confirm they fail.** Run `python -m pytest tests/test_posthoc_eval.py -q`.

- [ ] **Step 3: Implement** the config, validation, `resolve_checkpoint` and `main` as specified in Interfaces.

- [ ] **Step 4: Migrate the configs.**
  - Change the two shipped YAMLs to `reference_configs: [<old>]` and `checkpoints: [<old>]`, keeping their comments.
  - Change `split: val` to `split: test`. Leave `max_objects` as it is.
  - Create `configs/posthoc-eval-all.yaml` with `reference_configs` set to `mesh-v5-full-opt-coord-tile.yaml`, `mesh-v6-ortho-prefix-aligned.yaml`, `mesh-v6-ortho-xattn-a.yaml` and `mesh-v6-ortho-xattn-b.yaml`, and `checkpoints: [null, null, null, null]`.
  - Copy the strategy, metric and sinkhorn blocks from `posthoc-eval-mesh-v5-opt-coord.yaml`.
  - Add a header comment saying that `null` means the lowest `val_gen_chamfer_m`, that each model uses its own config's split, and that a full test split is a multi-day GPU job: calibrate with `max_objects=8`.

- [ ] **Step 5: Run the tests and confirm they pass.** Run `python -m pytest tests/test_posthoc_eval.py -q`, then `python -m pytest tests/test_face_set_metrics.py -q`. Expected: all pass.

- [ ] **Step 6: Update the docs.**
  - `posthoc-eval-runbook.md`:
    - §1: show the list form.
    - §5: the new layout, the per-model directories, the top-level `summary.csv` with `model`, `lod1_copy/`, `samples/`, and the new columns with one line each, copied from the spec's table.
    - §6: read `lod1_copy` as the floor, `lod1_vert_copy_norm > 1` / `lod1_chamfer_norm < 1` as over-copying, and note that the sample buckets lean small because they are in metres.
  - `ortho-conditioning-runbook.md`: update the output path sentence to `<root>/<run_name>/...`.

- [ ] **Step 7: CPU smoke run, only if a checkpoint is available locally.** Run `python -m src.eval.posthoc_eval --config configs/posthoc-eval-all.yaml max_objects=2 device=cpu "strategies=[{kind: greedy}]" sinkhorn.n_points=256`. Expected: a root directory with `summary.csv` and one directory per model. If no checkpoint exists locally (true on 2026-11-16), skip this step and say so in the handoff.

- [ ] **Step 8: Commit** after `detect_changes`:
  ```bash
  git add src/utils/config.py src/eval/posthoc_eval.py tests/test_posthoc_eval.py \
      configs/posthoc-eval.yaml configs/posthoc-eval-mesh-v5-opt-coord.yaml \
      configs/posthoc-eval-all.yaml docs/posthoc-eval-runbook.md docs/ortho-conditioning-runbook.md
  git commit -m "feat(eval): score a list of configs/checkpoints into one combined table"
  ```
