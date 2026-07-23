# Distance Featurization & Scale — Exploration Spec

Date: 2026-07-19. Branch: `levi-representation`. Status: exploration (no decision).
Related: `2026-07-19-se2-levi-model-design.md` (decision #1 already flags the
`lin_dist1` distance channel as the geometry bottleneck).

## Problem

The network's only view of geometry is a set of rotation-invariant scalars built
in `NodeEdgeBlock.forward` (`src/models/regnn.py:138-160`): per-node norm
`‖posᵢ‖`, pairwise `cdist(pos, pos)`, cosines, and in se2 also `z`, `dz`,
`horiz_dist`. The structurally important one — **pairwise distance — is fed raw
through a single `Linear`** (`lin_dist1`). `PositionsMLP` runs a small MLP on the
node norm, but pairwise distance gets only a linear map.

A linear map of a raw distance can express only a monotonic near→far ramp. It
cannot give the network distinct responses at different length scales. This is
the same low-frequency / spectral-bias limitation addressed for the timestep by
`time_embed`, but for geometry.

### Why this bites us and did not bite MiDi

The absolute unit (metre vs MiDi's ångström) is **not** the mechanism — `coord_scale`
standardizes coordinates to ~unit pooled variance before the network, so
distances are O(1) either way. What the standardization does **not** remove is the
**dynamic range** of the distance distribution:

- Molecular distances are tightly peaked (bonded atoms ~1–1.5 Å; molecule spans a
  few Å). A raw-distance → linear featurization is adequate on a narrow support.
- Building distances are heavy-tailed and multi-scale: **~18× per-building size
  spread** (see `tests/test_coord_scale.py`), and a Levi graph mixes very short
  vertex–vertex edges with long footprint diagonals and vertex–face spans.

So the raw-distance featurization is weaker for this data than it was for MiDi,
and the gap is a direct consequence of the domain/scale change — not a defect
inherited from MiDi. **Open question to settle first:** is the issue the
*featurization* (linear map too weak) or the *scale/normalization* (a single
`coord_scale` is the wrong normalizer for a multi-scale distance distribution),
or both? Cheapest avenues target the latter; richer ones target the former.

## Prerequisite: measure before choosing

Before picking an avenue, plot the **train-split pairwise-distance histogram** on
the normalized coordinate scale, split by pair type (v–v edge, v–v non-edge, v–f,
f–f). This decides (a) whether the distribution is genuinely multi-scale or just
wide, (b) a sensible cutoff `r_max` / basis range for any expansion, (c) whether a
log transform alone flattens it. Every avenue below needs this histogram to set or
justify its parameters; guessing a molecular default is the failure mode.

## Avenues for exploration

Ordered cheapest → richest. Not mutually exclusive.

1. **Rescale / renormalize distance only (cheapest).** Keep the linear map but
   feed a transformed distance: `log1p(d)`, or divide by a per-dataset `r_max`
   from the histogram. Tests whether the problem is purely scale. One-line change,
   no new parameters, fully reversible. If this closes most of the gap, stop here.
   - Con: still a single monotonic channel; no multi-scale resolution.

2. **Radial basis expansion (SchNet Gaussian smearing / DimeNet Bessel).**
   Replace the raw distance channel with `K` basis functions centred across
   `[0, r_max]` (Gaussian: Schütt et al. 2017; Bessel + envelope: Gasteiger et al.
   2020, arXiv:2003.03123). Standard in molecular ML precisely for multi-scale
   distance. Equivariance-safe (distance is already invariant).
   - Pro: localized, multi-scale basis; strong prior; well-understood.
   - Con: `K`, `r_max`, (and cutoff envelope) are calibration knobs tied to the
     histogram; adds width to `pos_info` before `lin_dist1`.

3. **Fourier / random-features on distance.** Sinusoidal or Gaussian-random
   Fourier features of `d` (Tancik et al. 2020, arXiv:2006.10739) — the direct
   analogue of the `time_embed` sinusoidal option, reused for geometry.
   - Pro: mirrors machinery already in the repo; breaks the low-frequency bias.
   - Con: non-local (a single frequency responds everywhere); less physically
     motivated for distances than RBFs; frequency range still scale-dependent.

4. **Cutoff / envelope on interactions.** Independent of the basis: down-weight or
   drop long-range pairs via a smooth cutoff so the dense `cdist` attention is not
   dominated by large-building diagonals. Addresses "far pairs swamp near pairs"
   rather than "near/far indistinguishable."
   - Pro: cheap; composes with 1–3; can sharpen local structure.
   - Con: introduces a range hyperparameter; risks dropping real long v–f spans.

5. **Learned small MLP on distance (match `PositionsMLP` treatment).** Give
   pairwise distance the same nonlinear-MLP-of-a-scalar treatment the node norm
   already gets, instead of a single Linear.
   - Pro: minimal, symmetric with existing code; no basis-range calibration.
   - Con: still one input scalar → MLPs are low-frequency-biased on raw scalars
     (the very reason 2/3 exist); likely a weaker version of an RBF/Fourier lift.

## Evaluation (how we would know it helped)

- **Coordinate MSE vs t**: current suspicion is it is flat across timesteps
  (denoiser can't localize) — improvement should show t-dependence and a lower
  low-SNR-regime error.
- **Sampled distance distribution** vs train histogram (per pair type): does the
  multi-scale structure reproduce?
- **Edge accuracy** on v–v / v–f, since distance drives `lin_dist1` → edge logits.
- Hold `equivariance`/`time_embed`/schedule fixed; ablate the distance avenue
  alone. Reuse the `time_embed` flag-threading pattern
  (`config → setup_utils → CityJSONDiffusionModule → rEGNNTransformer`).

## Non-goals

- Absolute-position (NeRF-style xyz) encoding: **excluded** — it is a function of
  pose and would destroy the yaw/rotation invariance the architecture is built on.
  Only invariant scalars (distances, norms, cosines, z/dz) may be re-featurized.
- Re-deriving `coord_scale`: out of scope here, but a per-pair-type or per-axis
  normalization is a candidate the histogram may motivate as a follow-up.
