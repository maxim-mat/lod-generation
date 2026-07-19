# Levi Graph Representation — Design

Date: 2026-07-19. Branch: `levi-representation`. Status: approved.

## Goal

Shift the dataset representation from vertex-only graphs (Active/Virtual nodes,
binary adjacency) to Levi graphs: faces become nodes, face membership becomes
edges. This makes faces explicit (no post-hoc cycle search) and prepares for an
SE(2)-symmetric denoiser (out of scope here).

## Representation

Per building, the Levi graph has vertex nodes (the CityJSON vertices used by the
building) and one face node per outer ring of each surface.

- `node_categories` `[n_max, 5]` one-hot:
  `0=vertex, 1=ground-face, 2=roof-face, 3=wall-face, 4=off` (off = padding;
  replaces "Virtual"). `vertex=0` preserves model-side `argmax == 0` masks.
- `x` `[n_max, 3]`: vertex coordinates in raw metres. Face and off nodes are
  zeros — vertex coordinates are the only continuous features.
  Graphs are **not centered**: `normalize_coords` stays as a flag but defaults
  to `False`. The dataset-wide `coord_scale` contract is unchanged.
- `y` `[n_max, n_max, 1]` integer edge labels:
  `0=off, 1=vertex-vertex (ring adjacency), 2=vertex-face (membership)`.
  Symmetric, zero diagonal. No face-face edges.
- `node_mask` `[n_max]`: 1 for vertex and face nodes, 0 for off nodes.
- Face semantics from the file's `semantics` (GroundSurface/RoofSurface/
  WallSurface, present in both LOD1 and LOD2 sources); when absent, inferred
  from the face normal with the same snap thresholds `straighten_face` uses
  (|nz| > 0.9 horizontal, < 0.1 wall, else sloped → roof if up).
- `n_max` / `upper_limit_nodes` count vertex + face nodes.

## Inverse conversion (`graph_to_cityjson`)

New signature: `(coords, node_classes, edge_classes, building_id)` — integer
class labels, not probabilities. For each face node:

1. Collect vertex neighbors via vertex-face edges.
2. Recover ring order by walking the vertex-vertex cycle restricted to that
   vertex set. If not a clean cycle (malformed generated graph or a chord from
   another face), fall back to angle-sort around the centroid on the best-fit
   plane (ceiling: non-convex faces).
3. Orient CCW viewed from outside: Newell normal, flip ring if the normal
   points toward the building centroid (ground faces end up pointing down).
4. Surface type comes from the face node's class.

`find_cycles_dfs` and the post-hoc classification path are deleted.
`regularize_building_geometry` / `straighten_face` remain as an optional
post-processing step for generated outputs, outside the conversion, so the
conversion stays exactly invertible.

## Inverse verification

`tests/test_levi_roundtrip.py` (TDD, written first): synthetic CityJSON
buildings (cube with semantics, gable-roof house, one without semantics) →
parse → `graph_to_cityjson` → parse again → assert identical graphs (coords,
node classes, canonical edge sets), ring sets equal up to rotation, and outward
orientation via positive total signed volume.

## Model wiring (minimum to keep training runnable)

- `CityJSONDiffusionModule`: `num_node_classes` **and** `num_edge_classes` are
  both constructor arguments, defaults `5` and `3` (module-level
  `NUM_EDGE_CLASSES` constant removed in favor of the argument).
- Config: `num_node_classes: 5`, new `num_edge_classes: 3`; wired through
  `create_model`.
- `compute_marginals` generalized to 3 edge classes.
- `sample` / `generate_cityjson` pass argmaxed class labels to the new
  conversion (no more edge-probability threshold).
- SE(2) denoiser symmetry: explicitly out of scope.

## Visualization

`src/visualize_levi.py` — exploratory/throwaway script (exempt from TDD and
reproducibility requirements per project instructions): sample ~3 dataset
graphs with 30–50 vertex nodes, render interactive Plotly 3D HTML; vertex nodes
at their coordinates, face nodes drawn at their face centroid (display position
only), node/edge colors + hover labels for all discrete and continuous
features.
