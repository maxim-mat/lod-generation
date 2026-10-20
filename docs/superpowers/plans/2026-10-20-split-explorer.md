# Split Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Streamlit app that shows each sampled test building beside its nearest training building
(by distance, volume symmetric difference, or chamfer) under a choice of split. The pairs come from a
precompute run over `cleaner`.

**Architecture:**
- **Split logic:** `split_pairs.split_parts` composes the same `splits.py` functions
  `MeshDataModule.setup` calls; a test pins it equal to the datamodule for every scheme.
  (Amended 2026-10-20: refactoring `setup` itself was impact-rated HIGH.)
- **Precompute (`src/analysis/split_pairs.py`):**
  - parses `cleaner` once into a `MeshDataset` (raw LOD2 meshes in metres live in `dataset.pairs`);
  - samples test buildings for each split;
  - shortlists the K nearest training buildings;
  - scores every unique pair in a process pool;
  - writes CSVs.
- **App (`src/split_explorer.py`):** reads the CSVs, and loads a CityJSON tile only when a pair from
  it is shown.

**Tech Stack:** Python 3.12, numpy 1.26, scipy `cKDTree`, manifold3d (via `exact_volume`), trimesh
(via `mesh_metrics`), pandas, Streamlit, plotly. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-20-split-explorer-design.md`

## Global Constraints

- **Splits:** `random`, `tile`, `tile-buffer50m`, `grid500m-buffer50m`, `grid1000m-buffer50m`.
  Seed 42, ratios (0.8, 0.1, 0.1). Names come from `splits.split_name`.
- **Defaults:** about 1000 test buildings per split (`--n-sample 1000`), K = 50, chamfer points 1024
  (recorded in `run.json`). The sample seed is the split seed, 42.
- **Data arguments** come from `configs/mesh-v5-full-opt-coord.yaml` (dataset_dir, lod_in/out,
  num_bins, margins, max_faces, tokenization), so building order and filtering match training.
- **Shape metrics:**
  - computed on raw LOD2 in metres;
  - each mesh is translated so its footprint centroid (`splits.footprint`) sits at x = y = 0 and its
    lowest vertex at z = 0;
  - no rotation.
- **Volume:** `exact_pair_volumes(test, cand)`. `ok = False` gives NaN `symdiff_m3`/`rel_symdiff`,
  and such a candidate never wins the volume criterion.
- **Chamfer:** `surface_distances(test, cand, n=n_points, seed=0)` then `chamfer_distance`.
- **Never touch the real dataset in tests.** Tiny synthetic boxes only.
- **Run gitnexus `impact` before editing `MeshDataModule.setup`; run `detect_changes` before each
  commit.**
- **`run.json` records:** git commit, seeds, K, n_points, n_sample, splits with their parameters,
  per-split counts, data arguments and wall time.

## Review Focus

1. **A split whose test set is smaller than `n_sample`** (e.g. a tiny subset). It must use every
   test building, not raise. Tested in Task 2 (`test_sample_takes_all_when_fewer`).
2. **Fewer than K training buildings.** The shortlist is clipped to the training count. Tested in
   Task 2 (`test_shortlist_clips_k`).
3. **A candidate where CSG or the BVH throws** (C++ code is not total). The pair must get
   NaN/`ok = False` and the run must continue. Tested in Task 2 (`test_pair_metrics_survives_a_failure`).
4. **All candidates have `ok = False`.** There is then no volume-criterion row for that building,
   rather than a NaN winner. Tested in Task 2 (`test_best_matches_skips_volume_when_none_ok`).
5. **A tile missing on disk when the app opens a pair.** The app warns instead of crashing (same
   pattern as `identity_explorer.draw_object`); this is UI, so it is checked by reading the code.

---

### Task 1: `split_parts`, shared by training and the precompute

**Files:**
- Modify: `src/dataset/splits.py` (add `split_parts`)
- Modify: `src/dataset/mesh_datamodule.py` (`MeshDataModule.setup`, lines 111-126, call it)
- Test: `tests/test_splits.py`

**Interfaces:**
- Produces: `split_parts(dataset, split, ratios, seed, block_m=None, buffer_m=0.0) -> (parts, assign, dropped)`
  - `dataset` needs `len()`, `.files` (tile per item) and `.xy` (`[N, 2]`). `.xy` is read only for
    `grid` or a buffer.
  - `parts` is `[train, val, test]` index lists. `assign` is `{group: 0|1|2}` (`{}` for random).
    `dropped` is the buffer's count.

- [ ] **Step 1:** Run `impact` on `MeshDataModule.setup` (upstream) and ledger the result.
- [ ] **Step 2: Failing tests** in `tests/test_splits.py`:
  - `test_split_parts_random_is_v5`: on a stub with 20 items, `split_parts(stub, "random", (0.8, 0.1, 0.1), 42)[0] == random_split_indices(20, (0.8, 0.1, 0.1), 42)`.
  - `test_split_parts_grid_buffer_matches_its_parts`: on a stub with 4 tiles and a 2×2-km lattice of
    points, `split_parts(stub, "grid", r, 42, block_m=1000, buffer_m=150)` equals
    `tile_split_indices(grid_labels(xy, 1000), r, 42)` with `buffer_train` applied to train.
  - `test_split_parts_random_never_reads_xy`: a stub whose `.xy` raises still splits under `"random"`.
- [ ] **Step 3:** Run `python -m pytest tests/test_splits.py -q -p no:warnings`. Expected: ImportError on `split_parts`.
- [ ] **Step 4:** Implement `split_parts` by moving the block in `setup`. `setup` becomes
  `parts, assign, self.buffer_dropped = split_parts(self.full_dataset, self.split, ratios, self.seed, self.split_block_m, self.split_buffer_m)`.
  `self.split_groups` and the buffer log line stay as they are.
- [ ] **Step 5:** Run `python -m pytest tests/test_splits.py tests/test_mesh_dataset_ortho.py -q -p no:warnings`. Expected: all pass.
- [ ] **Step 6:** `detect_changes`, then commit `refactor(data): split_parts shared by MeshDataModule and analysis`.

### Task 2: pair metrics and match selection

**Files:**
- Create: `src/analysis/split_pairs.py`
- Test: `tests/test_split_pairs.py`

**Interfaces:**
- Consumes: `split_parts` (Task 1), `footprint` (splits), `exact_pair_volumes`, `surface_distances`, `chamfer_distance`.
- Produces:
  - `SPLITS: dict[str, tuple[str, float|None, float]]`: name → `(scheme, block_m, buffer_m)` for the five splits; names equal `split_name(...)`.
  - `sample_test(test_idx, n, seed) -> list[int]`: seeded (`np.random.default_rng(seed)`), sorted, all of them when fewer than n.
  - `shortlist(xy, train_idx, query_idx, k) -> (idx [Q, k'], dist [Q, k'])`, with k' = min(k, len(train)); `idx` holds dataset indices.
  - `centred(mesh) -> (verts, faces)`.
  - `pair_metrics(a, b, n_points) -> dict(symdiff_m3, rel_symdiff, vol_ok, chamfer_m)`.
  - `CRITERIA = {"distance": "distance_m", "volume": "symdiff_m3", "chamfer": "chamfer_m"}`.
  - `best_matches(candidates: DataFrame) -> DataFrame`: one row per (split, test_id, criterion) with the candidate minimising that column. Volume uses only `vol_ok` rows.

- [ ] **Step 1: Failing tests** (synthetic box meshes, made by an 8-vertex/12-triangle helper):
  - `test_split_names_match_split_name`: `all(n == split_name(*p) for n, p in SPLITS.items())` and the set equals the five names.
  - `test_sample_is_seeded_and_sorted`, `test_sample_takes_all_when_fewer`.
  - `test_shortlist_is_nearest_first` (points on a line) and `test_shortlist_clips_k`.
  - `test_centred_puts_footprint_at_origin_and_ground_at_zero`: a box at (81000, 455000, 3) comes back with its centroid at (0, 0) and min z = 0.
  - `test_identical_boxes_score_zero`: `pair_metrics(box, box_shifted_far_away, 256)` gives symdiff ≈ 0 and chamfer ≈ 0 (centring removes the shift).
  - `test_taller_box_scores_its_extra_volume`: a 10×10×6 box against a 10×10×8 box gives symdiff_m3 ≈ 200, vol_ok True, chamfer > 0.
  - `test_pair_metrics_survives_a_failure`: with `exact_pair_volumes` monkeypatched to raise, the result is vol_ok False, symdiff NaN, and a finite chamfer.
  - `test_best_matches_picks_each_criterion` and `test_best_matches_skips_volume_when_none_ok`.
- [ ] **Step 2:** Run `python -m pytest tests/test_split_pairs.py -q -p no:warnings`. Expected: ModuleNotFoundError.
- [ ] **Step 3:** Implement. `pair_metrics` catches `Exception` around both C++ calls, as `identity_pairs.scan_tile` does, and logs a warning.
- [ ] **Step 4:** Rerun the tests. Expected: all pass.
- [ ] **Step 5:** Commit `feat(analysis): split_pairs metrics and match selection`.

### Task 3: precompute run and CLI

**Files:**
- Modify: `src/analysis/split_pairs.py` (add `candidate_table`, `main`)
- Test: `tests/test_split_pairs.py`

**Interfaces:**
- Produces:
  - `candidate_table(dataset, splits, n_sample, k, n_points, seed, workers=1) -> DataFrame` with columns
    `split, test_id, test_tile, cand_id, cand_tile, rank, distance_m, symdiff_m3, rel_symdiff, vol_ok, chamfer_m, test_faces, cand_faces`.
    - `dataset` needs `.ids`, `.files`, `.xy`, and `.pairs[i][1]` (the LOD2 `(verts, faces)`).
    - Each unique (test, cand) pair is scored once across all splits. A `multiprocessing.Pool` with
      ordered `imap` runs when `workers > 1`.
  - `main(argv)`:
    - arguments `--config` (default `configs/mesh-v5-full-opt-coord.yaml`), `--out` (default
      `outputs/split_pairs`), `--n-sample 1000`, `--k 50`, `--n-points 1024`, `--workers 8`;
    - builds `MeshDataset` from the config's `mesh_data` (no ortho);
    - writes `candidates.csv`, `pairs.csv` (`best_matches` plus a `criterion` column) and `run.json`.

- [ ] **Step 1: Failing tests:**
  - `test_candidate_table_scores_each_unique_pair_once`: on a 30-box stub with monkeypatched
    `pair_metrics` counting calls, two splits that share test buildings give one call per unique pair.
  - `test_candidate_table_columns_and_ranks`: the columns are exactly the list above; ranks run 0..k'-1 by distance.
  - `test_main_writes_outputs` (stub dataset monkeypatched into `main`): it writes the three files, and
    `run.json` has `git_commit`, `k`, `n_points`, `splits`.
- [ ] **Step 2:** Run them. Expected: ImportError on `candidate_table`.
- [ ] **Step 3:** Implement. Workers receive `(i, j, mesh_i, mesh_j, n_points)` tuples; meshes are small, so pickling is cheap.
- [ ] **Step 4:** Rerun `tests/test_split_pairs.py`. Expected: all pass.
- [ ] **Step 5:** Commit `feat(analysis): split_pairs precompute CLI`.

### Task 4: the explorer

**Files:**
- Create: `src/split_explorer.py`
- Test: `tests/test_split_explorer.py` (pure helpers only, like `test_ortho_explorer.py`)

**Interfaces:**
- Consumes: `outputs/split_pairs/{pairs,candidates}.csv`, `run.json`; `cityjson_figure`, `load_cityjson`.
- Produces: `matches(pairs, split, criterion) -> DataFrame` (sorted by the criterion's column,
  ascending); `METRIC_LABELS` for the metric row.

- [ ] **Step 1: Failing test:** `test_matches_filters_and_sorts` on a 6-row frame.
- [ ] **Step 2:** Run it. Expected: ModuleNotFoundError.
- [ ] **Step 3:** Implement the page:
  - **Sidebar:** split, criterion ("nearest by"), sort column, shade toggle.
  - **Distribution:** a histogram of the criterion's metric over the split's matches. A second trace
    shows the same criterion under `random` for comparison.
  - **Picker:** a selectbox over the sorted matches.
  - **Pair view:** two LOD2 3D views (test, match), with `draw_object` loading tiles lazily and a
    `max_entries=6` cache.
  - **Metrics:** distance, symdiff (m³ and relative), chamfer, faces.
  - **Expander:** the K-candidate table for the selected building.
  - **Theme:** palette and Streamlit setup as in `identity_explorer.py`. Usage goes in the module docstring.
- [ ] **Step 4:** Rerun the test, then the full suite (`python -m pytest -q -p no:warnings`). Expected: all pass.
- [ ] **Step 5:** `detect_changes`, then commit `feat(analysis): split explorer app`.
