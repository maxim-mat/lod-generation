# Split explorer: each test building beside its nearest training building

## Intent

`docs/superpowers/plans/2026-10-14-orthophoto-conditioning.md` (Tasks 2a–2c) proposes replacing v5's
random per-building split with a geographic one. The sweep in `outputs/split_sweep` makes the
argument with aggregate numbers. Under the random split:

- a test building's nearest training building is a median 6 m away;
- that neighbour correlates 0.45 with the test building on `faces_resid`.

This app makes the same argument pair by pair. You pick a split and a test building, and see that
building next to the training building most like it, with the numbers between them. The question
it answers: **under this split, does training contain a near-copy of this test building close by?**

## Goals

- Splits to choose from: `random` (v5's exact split) plus a few geographic splits: `tile`,
  `tile-buffer50m`, `grid500m-buffer50m`, `grid1000m-buffer50m`. Every split is the one
  `MeshDataModule` builds with seed 42 and ratios 0.8/0.1/0.1.
- Three meanings of "nearest", selectable in the app:
  1. geographic distance between footprint centroids;
  2. LOD2 volume symmetric difference (exact);
  3. LOD2 chamfer distance.
- Each pair is shown as two LOD2 3D views side by side, plus a metric row: geographic distance,
  volume symmetric difference (m³ and relative), chamfer (m), and face counts.
- Each of the three matches shows all three metrics. For example, the chamfer-nearest building
  also shows how far away it is.
- One distribution plot of the chosen metric to the chosen match, over the split's sampled test
  buildings, so splits can be compared at a glance.
- Everything is precomputed. The app does no geometry, and reads a tile from disk only when a pair
  using it is shown.

## Non-goals

- No search for near-twins anywhere in the city. Shape similarity is searched only among the test
  building's geographic neighbourhood in training (user decision).
- No model in the loop: no predictions, no per-building model error. The "does v5's error grow
  with distance" check stays a separate piece of work.
- No new split schemes. Only those `src/dataset/splits.py` already implements.

## Decided (by the user)

- **Shortlist:** shape-nearest is searched among the training buildings **around** the test
  building, not the whole training set.
- **All metrics precomputed.**
- **Corpus:** `data/The Hague/cleaner`, so real neighbours exist. Splits and the test sample
  depend only on building order and centroids; meshes are read lazily for the sampled pairs only.
  The compute target is about an hour.
- **Sample:** about 1000 test buildings per split.

## Design

### Precompute: `src/analysis/split_pairs.py`

1. Build the split the way training builds it, by calling `MeshDataModule(...).setup()` with the
   v5 config's data arguments for each split scheme. That way building order, the `max_faces`
   filter and the split are exactly the training ones, with no reimplementation.
   - Cost: one parse of `cleaner`, about 2 GB of RAM and a few minutes.
   - `MeshDataset.xy` supplies the footprint centroids.
   - `files` and `ids` supply each building's tile and object id.
   *My call:* parse `cleaner` **once** and derive all five splits from that one dataset, using the
   same `splits.py` functions `MeshDataModule.setup` calls. A test asserts the result equals
   `MeshDataModule`'s split on a tiny fixture, so the orchestration cannot drift. This avoids five
   full parses.
2. Draw a seeded sample of 1000 of the split's **test** buildings. The seed is recorded.
3. For each sampled test building, take its **K = 50 nearest training buildings** by centroid
   (`cKDTree` over the split's training buildings, after the buffer). This is the shortlist.
4. For every (test, candidate) pair, compute:
   - geographic distance;
   - exact LOD2 volume symmetric difference, absolute and relative (`exact_volume.exact_pair_volumes`);
   - LOD2 chamfer (`mesh_metrics.surface_distances` + `chamfer_distance`, fixed seed).

   Both shape metrics run in **metres on the raw CityJSON LOD2**, not on the quantised tokens. The
   *My call:* CityJSON surfaces are turned into triangles with the same ring-aware triangulation
   the identity scan (`src/analysis/identity_pairs.py`) uses for its exact volumes:
   `mesh_dataset.parse_cityjson_file_to_meshes`, the same parser training uses. The face[0]
   truncation and fan-triangulation bugs recorded on 2026-10 must not come back through a second
   converter. The
   two meshes are first moved to a common origin: footprint centroid at x = y = 0, lowest vertex at
   z = 0. They are not rotated.
5. A pair whose LOD2 is not a closed, correctly wound solid gets NaN volume metrics (`ok = False`)
   and is ignored when picking the volume-nearest match. Chamfer is still defined.
6. Unique pairs are computed once and shared across splits. Work is grouped by tile, so each tile
   is parsed once per worker, and runs in a multiprocessing pool.
7. Outputs in `outputs/split_pairs/`:
   - `pairs.csv`: one row per split × test building × criterion. Columns: test id and tile, match
     id and tile, distance, symdiff_m3, rel_symdiff, chamfer_m, face counts, ok.
   - `candidates.csv`: all K rows per test building, so a later change of criterion needs no
     recompute.
   - `run.json`: git commit, seeds, K, chamfer points, split parameters, and counts per split.

### App: `src/split_explorer.py`

Same structure as `identity_explorer.py` and `duplicate_explorer.py`:
- the `REPO` path fix-up;
- `@st.cache_data` on the CSV, and a small `max_entries` cache on tiles;
- `cityjson_figure` for the 3D views;
- usage in the module docstring.

Layout follows the sketch: a sidebar with split, criterion and sort order; a distribution plot; a
test-building picker; two 3D views; a metric row.

## Constraints

- **Reproducibility:** seed, K, chamfer points and git commit all go in `run.json`. Reused
  functions keep their docstrings.
- **No new dependencies.** Streamlit, plotly, manifold3d and scipy are already in use.
- **Tests:** the precompute's pure logic (sampling, shortlist, centring, best-match selection, NaN
  handling) is TDD'd on tiny synthetic meshes and centroids. The real dataset is never touched in
  tests. The Streamlit page is exploratory UI and is not unit-tested, consistent with the other
  explorers; `test_ortho_explorer.py` is the exception, so a smoke import test is optional.
- Runs only when the user starts it.

## Left to the builder

- Chamfer sample count. 1024 points keeps 5 splits × 1000 × 50 within about an hour on 8 workers;
  4096 is the posthoc default. Record whichever is used.
- Plot styling, following the dataviz palette the other explorers already use.
- Whether the app shows the full K-candidate table for the selected building. It is cheap, since
  `candidates.csv` exists.
