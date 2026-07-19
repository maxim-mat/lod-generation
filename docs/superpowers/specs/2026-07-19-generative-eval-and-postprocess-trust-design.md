# Design: Levi-Graph Representation + Generative Evaluation + `post_process` Trust

**Date:** 2026-07-19
**Status:** Approved design, pending implementation plan
**Sequencing note:** the current branch (`fix/y-branch-numerical-stability`) will be
merged; this work resumes on a **new branch**. Phase 0 (representation) lands first and
alone, then the evaluation work builds on top.

## 1. Problem

The model (`CityJSONDiffusionModule`) unconditionally generates building geometry via a
`T`-step reverse diffusion chain (`sample` → `generate_cityjson`). Two things are missing
and one is structurally wrong.

**Missing:** nothing measures the quality of generated buildings. The metrics in
`_eval_step` (`src/models/diffusion.py`) — `coord_mse`, `real_precision`/`real_recall`,
`edge_ce` — are all **single-step denoising fidelity at a fixed `t = T/2`**, not a
measure of a building sampled end-to-end from noise. Errors compound across the chain, so
those metrics can look healthy while sampled buildings are malformed.

**Structurally wrong (address first):** the representation views a CityJSON building as a
**wireframe graph** — nodes = vertices (Active/Virtual), edges = binary vertex-vertex
adjacency built from surface *outer rings*. Face membership is **discarded at parse
time** (`dataset.py:123-134` keeps only ring edges), which forces `post_process` to
reverse-engineer faces with a cycle-finding DFS (`find_cycles_dfs`) and to *guess*
semantic surface type by snapping face normals (`straighten_face`). Both are lossy
heuristics that fail on non-convex faces and ambiguous normals, and they sit directly on
the critical path of every quality number. Building evaluation on top of this abstraction
before fixing it would measure post_process's guesswork as much as the model.

## 2. Goals / Non-goals

**Goals**
- **Phase 0 — Levi-graph representation.** Augment the wireframe graph to a Levi
  (incidence) graph: explicit face nodes, vertex-face incidence edges, and semantic
  surface type carried as the face node's class. Make generation predict topology and
  semantics directly, and collapse graph → CityJSON to a trivial per-face operation.
- **Phase 1 — `post_process` trust suite.** Prove the new (trivial) conversion is
  lossless and correct on real LoD2 data. This is the gate on Phase 0.
- **Phase 2 — `GenerativeEvalCallback`.** Sample the full reverse process and score the
  buildings: val3dity validity, rejection rate, 3dSAGER distribution realism, MMD,
  novelty/uniqueness, per-error-code histogram, and WandB 3D logging.

**Non-goals (v1)**
- Inner rings / holes in faces (LoD2 building surfaces rarely have them) — Phase 0 handles
  outer rings only; note the limitation.
- LoD1/LoD3 coverage — the representation and trust suite are parametrized by LoD, but
  only **LoD2** is populated now.
- Semantic surface types beyond Ground/Wall/Roof — mapped to the nearest of the three (or
  a `Face-Other` class), decided at implementation.
- Multi-GPU distributed sample gathering; conditional generation.
- Replacing the denoising `test_step` (it stays; it measures a different thing).

## 3. Phase 0 — Levi-graph representation (foundational, CRITICAL risk)

### 3.1 The representation
- **Node classes (was 2 → 5):** `0 Vertex`, `1 Face-Ground`, `2 Face-Wall`,
  `3 Face-Roof`, `4 Virtual` (padding).
- **Node positions:** vertex node = its 3D coordinate; **face node = centroid of its
  incident vertices** (diffused like any position); virtual = 0. The zero-CoM constraint
  (`_centre_positions`) is computed over **vertex nodes only** — face centroids are
  derived, so including them would double-count and distort the subspace.
- **Edge classes (was 2 → 3):** `0 no-edge`, `1 vertex-vertex wireframe`,
  `2 vertex-face incidence`. Face-face pairs and any virtual-involving pair are `no-edge`.
  The dense adjacency `y` now holds class indices `{0,1,2}`, not a binary flag.

The wireframe edges are **retained** (not replaced) — they are what recovers per-face
vertex *order* at conversion time (see §3.3). Incidence edges give the unordered vertex
*set* of each face; the wireframe cycle among those vertices gives their order.

### 3.2 Parsing (`parse_cityjson_file_to_graphs`)
For each surface in a building's geometry:
- Create a **face node** with its semantic class read from the geometry's
  `semantics.surfaces`/`values` (the parser already reads these in `get_base_center`;
  reuse that traversal).
- Add **incidence edges** between the face node and each outer-ring vertex.
- **Retain** the vertex-vertex ring (wireframe) edges as today.
- Face node position = centroid of the ring's vertices.
Unknown/other semantic surface types map to the nearest of Ground/Wall/Roof (or a
`Face-Other` class) — decided at implementation. Inner rings are ignored in v1.

### 3.3 Conversion graph → CityJSON (the payoff — now trivial)
`graph_to_cityjson` is rewritten to:
1. For each face node, gather its incident vertices (via incidence edges).
2. Order them by walking the **wireframe cycle** restricted to those vertices — a single
   cycle over ~4-8 nodes, unambiguous, no global search, robust to non-convex faces.
3. Emit the ordered ring; the surface type reads directly off the face node's class.
4. Assemble the `Solid` boundaries + semantics; set the LoD tag correctly (this fixes the
   `lod:"1"` hardcode at `post_process.py:193,204`).

`find_cycles_dfs`, `straighten_face` (normal-snapping), and
`regularize_building_geometry` are **removed**. A light optional planarity projection may
be retained as post-hoc regularization, gated by config — decided at implementation.

### 3.4 Model / noise changes
- `num_node_classes = 5`, `NUM_EDGE_CLASSES = 3`.
- `x_marginals` shape `[5]`, `e_marginals` shape `[3]`, recomputed from the train split
  (`compute_marginals`); the `marginal` limit distribution uses them.
- `rEGNNTransformer` node/edge output heads resized to 5 / 3.
- `GraphNoiseModel` limit-distribution sampling and transitions over the new class counts.
- `_prepare` one-hots `E` over 3 classes; edge cross-entropy is over 3 classes
  (off-diagonal only, as today).
- `n_max` grows to `max(#vertices + #faces)` per building; recompute; revisit
  `upper_limit_nodes`.
- Denoising eval metrics (`real_precision`/`real_recall`, currently over the Active-vertex
  class) generalize to per-class (vertex vs face-types vs virtual) — adjust so they stay
  meaningful under class imbalance.

### 3.5 Blast radius (CRITICAL)
This rewrites the full pipeline: **dataset** (`parse_cityjson_file_to_graphs`,
`_pad_graph`, `NUM_NODE_CLASSES`, `n_max`, `compute_marginals`, `compute_coord_scale`) →
**noise** (`GraphNoiseModel`) → **model** (`CityJSONDiffusionModule`,
`rEGNNTransformer`) → **post_process** (rewritten) → **inference** → **tests**. Per
project rules, `impact({target, direction:"upstream"})` MUST be run and reported before
editing each shared symbol, and HIGH/CRITICAL findings surfaced before proceeding. This
lands **first, on its own branch**, and is gated by the §4 round-trip test before any
evaluation work builds on it.

### 3.6 Verification gate (TDD anchor for Phase 0)
The headline test: real LoD2 CityJSON → parse to Levi graph → convert back to CityJSON →
assert **lossless** round-trip: vertices within tolerance, identical face set, identical
surface-type assignment, and val3dity-valid output. If parse↔convert is not lossless on
real buildings, the representation is wrong and nothing downstream can be trusted. This is
written before the implementation (TDD) and is the Phase 0 completion gate.

## 4. Phase 1 — `post_process` trust suite (`tests/test_post_process.py`)

With conversion now trivial, the suite centers on proving the round-trip is faithful.
Focus **LoD2**; parametrize by LoD so lod1/lod3 slot in later.

1. **Lossless round-trip on real data (headline).** As §3.6, over a committed fixture of
   real LoD2 buildings: parse → convert → assert geometric + topological + **semantic**
   equality and val3dity validity.
2. **Determinism.** Same graph → byte-identical CityJSON.
3. **Order recovery.** Hand-built faces (incl. a non-convex L-shaped face) → the wireframe
   walk recovers the correct ring order; convex-only angular sort would fail here, proving
   the wireframe retention earns its keep.
4. **Semantics pass-through.** Face node class → correct CityJSON surface type, no normal
   guessing.
5. **Degenerate / adversarial inputs.** Face node with <3 incident vertices; incidence set
   whose wireframe edges don't form a single cycle (inconsistent generation); duplicate
   vertices; disconnected graph → graceful, documented behavior (no crash, no silent
   garbage).

Plain `pytest` over pure functions; val3dity-dependent assertions
`pytest.mark.skipif(shutil.which("val3dity") is None)`. Real-data tests skip if the
dataset fixture is absent.

## 5. Phase 2 — `GenerativeEvalCallback`

Unchanged by Phase 0 except that `generate_cityjson` now uses the trivial converter and
emits correct semantics/LoD. All scoring operates on the emitted CityJSON, so the feature,
validity, distribution, and novelty arms are representation-agnostic.

### 5.1 Home and trigger (`src/eval/callback.py`)
- Lightning `Callback`, fired on `on_fit_end`, gated by a new `generative_eval` config
  block (disabled → no-op). The evaluation body is also callable standalone against a
  checkpoint (thin entrypoint mirroring `src/inference.py`); the callback is a thin
  Lightning adapter over it.
- Gets the run's WandB logger from `trainer.loggers`. Reference + train features are built
  once from `trainer.datamodule` and cached on the instance.

**Config block (`generative_eval`):**
```
enabled: bool = false
num_batches: int
batch_size: int
seed: int                     # fixed → comparable buildings across runs/epochs
log_n_meshes: int
feature_set: "full" | "welldefined" = "full"
val3dity_path: str | null     # null → resolve via shutil.which("val3dity")
threshold: float = 0.5
```

### 5.2 Flow
1. `L.seed_everything(seed)`.
2. Draw `num_batches` via `model.generate_cityjson(batch_size, threshold)`, tracking drops
   (samples that fail to form a valid building).
3. Validity (§5.3) → features (§5.4) → distances (§5.5) → novelty (§5.6) → log (§5.7).

### 5.3 Validity + rejection rate (`src/eval/validity.py`)
- Gated by `shutil.which("val3dity")` / `val3dity_path`. Absent → warn, skip validity
  arm; rejection rate from generation drops still logs.
- Serialize the batch to **CityJSONSeq**, one `val3dity stdin --report` subprocess
  (stdin streaming, val3dity ≥ 2.5.0), parse the flat per-feature error list.
- Logs: `gen_valid_fraction`, **per-error-code histogram**, and **rejection rate** =
  `(dropped in generation) + (val3dity-invalid)` / `attempted`, with the breakdown.
- val3dity version-checked and smoke-validated on a known-good building at eval start
  (fail loud, not silently all-invalid).

### 5.4 Feature extraction (`src/eval/building_features.py`)
- 3dSAGER property vector (Genossar et al., arXiv:2511.06300, Table 1) per building.
- `feature_set: "full"` (default, LoD2) — all 17 features: `area, volume, height_diff,
  num_vertices, perimeter, circumference, perimeter_index, convex_hull_area,
  ave_centroid_distance, shape_index, fractality, elongation, hemisphericality, cubeness,
  axes_symmetry, density, num_floors`. `feature_set: "welldefined"` drops the
  LoD1-ambiguous ones (`num_floors`, `fractality`, `circumference`).
- **numpy + scipy only** (`scipy.spatial.ConvexHull`, OBB via covariance
  eigendecomposition, mesh volume via divergence theorem). Non-obvious descriptors
  (`shape_index`, `hemisphericality`, `cubeness`, `elongation`) **ported** from
  `tudelft3d/3d-building-metrics` (`cityStats.py`, `shape_index.py`) with per-formula
  attribution — **not** a dependency (its pins `pyvista==0.36.1`, `shapely==1.8.5`,
  `sklearn==1.3.2`, `geopandas`, `pymeshfix`, `miniball` would conflict with the
  torch/lightning env).

### 5.5 Distribution distance (`src/eval/distribution.py`)
- `log(1 + x)` normalize (3dSAGER §3.1).
- **Per-feature 1-Wasserstein** (`scipy.stats.wasserstein_distance`) vs both references
  (`_vs_pipelined` = real test graphs through the same converter; `_vs_raw` = raw test
  CityJSON featurized directly), per feature + mean aggregate.
- **Kernel MMD** (RBF, median-heuristic bandwidth) on the stacked normalized vectors vs
  each reference.

### 5.6 Novelty / uniqueness (`src/eval/novelty.py`)
- Cache **train-split** feature vectors. `uniqueness` = fraction of samples not mutual
  near-duplicates in feature space; `novelty` = fraction whose nearest train-set neighbour
  exceeds a (logged, config-default) tolerance.

### 5.7 WandB 3D + scalar logging
- Log `log_n_meshes` sampled buildings as `wandb.Object3D` (minimal OBJ from the CityJSON
  faces). Fixed `seed` → same buildings recur across epochs. Scalars/histograms under a
  `gen/` namespace.

## 6. Wiring, dependencies, config

- **Wiring:** new `generative_eval` config block (`src/utils/config.py`);
  `create_callbacks` (`src/utils/setup_utils.py`) appends the callback when `enabled`.
- **New dependency:** `scipy` (`ConvexHull`, `wasserstein_distance`) — the only new Python
  dep; added to `requirements.txt`. Existing: `wandb>=0.15.0` (`Object3D`), `numpy`.
- **External tool (ops, gated):** `val3dity` binary on PATH. Dev (Windows): prebuilt
  `.exe`. Server (Linux): compiled once into the env / Docker image. Decoupled from code by
  the `shutil.which` gate. `val3ditypy` bindings and the web API were rejected (§9).

## 7. Module layout

```
src/dataset/dataset.py     # Phase 0: parse to Levi graph, 5 node / 3 edge classes
src/models/diffusion.py    # Phase 0: class counts, marginals, edge loss, metrics
src/models/noise.py        # Phase 0: limit dist / transitions over new class counts
src/models/regnn.py        # Phase 0: resized node/edge heads
src/post_process/post_process.py  # Phase 0: trivial converter; DFS/straighten removed
src/eval/
  __init__.py
  callback.py              # Phase 2: GenerativeEvalCallback + standalone body
  validity.py              # val3dity subprocess, CityJSONSeq, rejection rate
  building_features.py     # 3dSAGER vector (numpy + scipy, ported formulas)
  distribution.py          # log1p, per-feature Wasserstein, MMD
  novelty.py               # uniqueness + novelty vs train features
tests/
  test_post_process.py     # Phase 1 trust suite (Levi round-trip)
  test_levi_representation.py  # Phase 0: parse/_pad_graph shapes, class counts
  test_eval_*.py           # Phase 2 unit tests
```

## 8. Testing strategy (eval code)

- `building_features`: known solids with analytic properties (unit cube → known
  area/volume/cubeness; tall box → known elongation) assert exact values.
- `distribution`: identical distributions → distance ≈ 0; shifted → monotonically larger.
- `validity`: parse a captured val3dity report fixture → correct valid/invalid + histogram
  without the binary in CI.
- `novelty`: sample identical to a train building → novelty 0; all-distinct → uniqueness 1.
- `callback`: smoke test with a tiny model + monkeypatched `generate_cityjson` and a fake
  logger (reuse the `model.log = lambda ...` pattern from `test_model_smoke.py`); assert
  scalar keys logged and graceful degradation when val3dity is absent.

## 9. Open items resolved during design

- **Representation:** Levi graph — keep wireframe edges (for per-face order), face nodes
  carry diffused centroid positions, semantic surface type is the face node's class. §3.
- **val3dity integration:** subprocess to a PATH binary via CityJSONSeq stdin, one process
  per batch, availability-gated. Rejected `val3ditypy` (from-source CGAL, no wheels) and
  the web API (network + size cap).
- **Reference distribution:** log both vs-pipelined and vs-raw. §5.5.
- **Feature-set fidelity:** full 17 default (LoD2); `welldefined` subset for LoD1.
- **Extra metrics:** MMD, novelty/uniqueness, per-error-code histogram all in v1.
- **LoD:** generating LoD2; representation/eval assume LoD2; the `lod:"1"` hardcode is
  fixed in Phase 0 (face nodes carry real semantics, LoD tag set correctly). post_process
  must eventually support lod1/2/3; only lod2 is in scope now.

## 10. References

- Vignac et al., *MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation*,
  arXiv:2302.09048 — the base model this pipeline adapts.
- Genossar, Dalyot, Shraga, Gal, *3dSAGER: Geospatial Entity Resolution over 3D Objects*,
  arXiv:2511.06300 — the geometric property vector (Table 1) for distribution realism.
- Ledoux, *val3dity: validation of 3D GIS primitives according to the international
  standards*, Open Geospatial Data, Software and Standards, 2018 — 3D validity.
- `tudelft3d/3d-building-metrics` — source of the ported shape-descriptor formulas
  (attributed at each implementation; not a dependency).
```
