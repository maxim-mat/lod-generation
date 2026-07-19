# SE(2) Mode + Levi Face Geometry — Design

Date: 2026-07-19. Branch: `levi-representation`. Status: approved (conversation).
Builds on `2026-07-19-levi-representation-design.md`.

## Goal

Adapt the model architecture to the Levi representation: give face nodes real
geometry (ring centroids, diffused), and add `se2` as an additional
equivariance mode (yaw rotation + xy translation quotiented; z absolute).
Existing `so2`/`o3` behavior is preserved bit-for-bit under default config.

## Decisions (with rationale from the design conversation)

1. **Face node position = ring centroid, diffused and in the loss.**
   `parse_cityjson_file_to_graphs` writes `x[face] = mean of ring vertex
   coords`. No information is added (centroid is a deterministic function of
   vertices + v-f edges); the justification is (a) minimal surgery — the whole
   existing position pipeline applies unchanged, (b) meaningful pairwise
   distances for v-f and f-f pairs (fixes the garbage distance-to-origin
   currently fed to `lin_dist1`), (c) auxiliary supervision / early-resolving
   face anchors in the reverse chain. Decode ignores face coords, so the
   conversion inverse is untouched. Ablation lever documented: masking face
   slots out of the coordinate loss gives the unsupervised variant.
   No normals (deferred until sample quality demands them; prefer Newell area
   vector then). No engineered v-v / v-f edge features (dense `cdist` covers
   lengths already).

2. **`_centre_positions` centers over real (vertex + face) nodes and zeroes
   only OFF slots.** The real mask is `1 − node_categories[..., -1]` (last
   class is Virtual/Off in both the legacy 2-class and 5-class conventions).
   Centering over real nodes keeps the target on the all-ones zero-CoM
   subspace that `noise.py` and the network project onto (OFF slots are 0, so
   the all-N mean of the centered target is exactly 0) — avoiding an
   irreducible loss floor.

3. **`equivariance="se2"`**: same yaw-invariant feature set as `so2` in
   `NodeEdgeBlock`; the difference is translation handling — all zero-CoM
   projections remove the **xy** mean only, z passes through. Implemented as
   `remove_mean_with_mask(x, node_mask, xy_only=False)`; the flag threads:
   `ModelConfig.equivariance` ("so2" default) → `create_model` →
   `CityJSONDiffusionModule` → `rEGNNTransformer` (mask_graph, velocity
   projection, PositionsMLP) and → `GraphNoiseModel(xy_only_com=...)`
   (apply_noise, sample_limit_dist, sample_zs_from_zt_and_pred).

4. **z prior matches the train empirical distribution by moment-matching, not
   by sampling a histogram.** Gaussian diffusion converges to N(0, I)
   regardless of data, so the empirical-prior analogue of the discrete
   marginal transitions is standardization: `z_shift` = pooled train-split
   mean of vertex z, computed by `CityJSONDataModule.compute_z_shift()`
   (mirror of `compute_coord_scale`), stored in checkpoint hparams, applied in
   `_prepare` as `(z − z_shift) / coord_scale` (se2 only; 0.0 otherwise) and
   inverted in `generate_cityjson`. `coord_scale` semantics unchanged (its
   per-building-centered pooled std slightly understates absolute-z spread on
   flat terrain; accepted).

## Non-goals

- Face normals / area vectors (deferred).
- Non-Gaussian empirical z prior (would require changing the forward kernel).
- Changing `so2`/`o3` numerics: with default config every existing test and
  checkpoint contract must behave identically except for the centroid data
  change and the `_centre_positions` real-mask fix.

## Verification

- Parse test asserts face coords equal ring centroids (cube fixture).
- `_centre_positions`: face slots translated (not zeroed), OFF slots zero,
  all-N mean exactly 0; se2 variant leaves z unshifted by centering.
- `remove_mean_with_mask(xy_only=True)` leaves z untouched, removes xy mean.
- Equivariance regression: random yaw rotation of network inputs → position
  output rotates accordingly, X/E logits invariant (parametrized over so2 and
  se2).
- z pipeline: `compute_z_shift` value on synthetic items; `_prepare`
  standardizes z; `generate_cityjson` restores metres + z offset.
- Full existing suite stays green.
