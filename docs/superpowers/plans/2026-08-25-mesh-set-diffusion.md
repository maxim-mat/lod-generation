# Mesh-Set Diffusion (LOD1 to LOD2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a non-autoregressive branch that generates a whole LOD2 triangle mesh in one denoising trajectory, conditioned on the paired LOD1 mesh, with swappable ordering/loss, denoiser, and diffusion formulation.

**Architecture:** A building is a padded *face set* `[F, 10]` -- 9 coordinate channels (3 vertices x xyz, in the existing LOD1-normalized unit box) plus 1 presence channel. A denoiser (attention-augmented 1D U-Net, or a pure transformer) maps `(x_t, t, LOD1)` to a prediction; a process object (DDPM/DDIM, flow matching, or D3PM over the 128 coordinate bins) owns the noising and the reverse loop. Conditioning is cross-attention to encoded LOD1 faces at every level, never elementwise addition -- LOD1 and LOD2 have different face counts.

**Tech Stack:** PyTorch, Lightning, OmegaConf, numpy, scipy (`linear_sum_assignment` -- already a dependency), trimesh (metrics only). No new dependencies.

**Spec:** `docs/research/2026-08-24-preliminary-future-plan.md` section 3.1 (diffusion prior over building geometry), section 3.2 (cross-attention conditioning, scaffold masking), plus the *Design Decisions* section below, which is the spec of record for everything not in that doc.

## Cold Start

You have no context from the session that wrote this plan. Everything you need is in this file plus the repo. Before Task 1:

1. **Where you are.** Repo root `D:\Projects\lod-generation`, branch `mesh-transformer` (branched from `main`). Windows; the shell is PowerShell, with a Git Bash tool also available — they take different syntax. Verify: `git rev-parse --abbrev-ref HEAD` prints `mesh-transformer`.
2. **Read these three first**, in order, and do not start writing code until you have:
   - `CLAUDE.md` at the repo root — the GitNexus rules are binding, in particular "run `impact({target, direction: 'upstream'})` before modifying any existing symbol". This plan creates new files almost everywhere precisely to keep that surface small; the four places it modifies an existing file are listed under *File Structure*.
   - `docs/research/2026-08-24-preliminary-future-plan.md` — the spec's other half. Sections 3.1 and 3.2 are what justify this branch existing.
   - `src/dataset/mesh_dataset.py` and `src/eval/mesh_metrics.py` — the two modules this plan reuses most heavily. You need `normalize_to_unit_box`, `quantize`/`dequantize` and `mesh_metrics` in your head, not paraphrased.
3. **Environment.** `pip install -r requirements.txt`. Everything this plan needs is already listed — scipy (for `linear_sum_assignment`), trimesh + rtree (for metrics), lightning, omegaconf. **Adding a dependency is out of scope**; if you think you need one, that is a signal you have mis-read a task.
4. **Running tests.** `python -m pytest tests/ -q` from the repo root. There is no `pytest.ini`/`pyproject.toml` — pytest is invoked with defaults and finds `tests/` by rootdir. Every test this plan adds is CPU-only and uses synthetic meshes.
5. **Running training.** `python main.py --train --config <path>` plus optional dotted overrides (`training.max_epochs=1`). The project rule is **do not run training scripts unless the task explicitly says to** — three tasks do (6, 8, 14), and they say so.
6. **Data.** Tasks 8 and 14 touch `data/The Hague/mini_cleaner`, which is not in git. If it is absent, the unit tests still all pass; only the two "run the real config" steps are blocked. Report that rather than synthesising a dataset.
7. **Reference code.** The architecture is ported from `maxim-mat/trace-denoise-refactor` on GitHub (`src/denoisers/`, `src/diffusion/`, `src/modules/`). You do not need it — every block is reproduced here — but plan D10 quotes it, and if you want to check that quote, `src/diffusion/base_diffusion.py::noise_data` and `src/diffusion/diffusion_module.py::training_step` are the two functions in question.
8. **Order.** Tasks 1-8 are Phase A and are strictly sequential. Tasks 9-13 are Phase B, depend only on Phase A, and are independent of each other — they can be done in any order or in parallel. Task 14 depends on all of Phase B; Task 15 depends on runs finishing.

## Global Constraints

- Python >=3.10, torch >=2.0, lightning >=2.0, numpy >=1.24, scipy >=1.10. **No new dependency may be added.**
- New config lives in `MeshDiffusionConfig`, selected by `config_set: mesh_diffusion`. Existing `mesh`, `mesh_vqvae`, `diffusion`, `mini` config sets must keep working unchanged.
- Reuse, do not reimplement: `normalize_to_unit_box`, `quantize`, `dequantize`, `canonicalize`, `fix_winding` (`src/dataset/mesh_dataset.py`); `weld`, `drop_duplicate_faces`, `n_degenerate` (`src/eval/mesh_postprocess.py`); `mesh_metrics`, `_raster` (`src/eval/mesh_metrics.py`); `mesh_to_cityjson`, `save_to_file` (`src/post_process/post_process.py`); `create_loggers`, `create_callbacks` (`src/utils/setup_utils.py`); `cosine_beta_schedule_discrete` (`src/models/noise.py`).
- **Do not modify `run_mesh_eval` or `MeshTransformer`.** The diffusion branch gets its own eval module. If a task appears to need an edit to either, stop and run `impact({target: "<symbol>", direction: "upstream"})` first and report before proceeding (project CLAUDE.md rule).
- Reproducibility artifacts are mandatory and never trimmed: `seed` threaded to every sampler, full config logged via `save_hyperparameters`, docstrings on anything used in more than one module.
- Tests: CPU-only, synthetic meshes, never load the full dataset. Run with `python -m pytest`.
- Every task ends with a commit on the current branch (`mesh-transformer`). Do not push.

---

## Design Decisions

These are settled. Do not re-litigate them during implementation.

**D1 -- Representation is a face set, not vertices + connectivity.** `x` has shape `[F, 10]`: channels 0-8 are `(v0.x, v0.y, v0.z, v1.x, ..., v2.z)` in the LOD1-normalized box (range `[-0.5, 0.5]`), channel 9 is presence with target `+0.5` for a real face and `-0.5` for a pad slot. Vertices are *repeated per face*; shared vertices are recovered post-hoc by welding (Task 13). Generating connectivity is the problem the AR tokenizers exist to solve; this branch dodges it deliberately.

**D2 -- Presence is diffused, not a side head.** It is channel 9, noised and denoised like any other. At sampling, faces with `presence < 0` are dropped. This is how face count is decided at generation time, and it doubles as the DETR "no-object" logit for the Hungarian matcher. Coordinate loss is masked to real slots; presence loss is computed on *all* slots.

**D3 -- Faces are cyclically rotated to a canonical corner before anything else.** `(v0,v1,v2)`, `(v1,v2,v0)` and `(v2,v0,v1)` are the same triangle with the same winding, so an unrotated target makes MSE penalize an identical face. Rotation puts the `(z,y,x)`-smallest corner first, decided on the quantized grid so the comparison is exact.

**D4 -- Two ordering/loss regimes, and they are not independent.**
- `order: morton` + `pos_embed: sinusoidal` + `loss: mse` -- faces sorted by Morton code of their quantized centroid, slot-to-slot MSE.
- `order: none` + `pos_embed: none` + `loss: hungarian` -- permutation-equivariant model, `scipy.optimize.linear_sum_assignment` over a coord+presence cost.
- **`order: none` with `loss: mse` is unlearnable** and must raise: a model that cannot see position is being asked which slot to fill.

**D5 -- The U-Net requires a sorted order.** Stride-2 1D convolution assumes locality along the face axis. With `order: none` the axis is arbitrary and the convolution is noise. `denoiser: unet` + `order: none` raises.

**D6 -- Padding is within-batch, rounded up to a multiple of 8.** Three stride-2 downsamples in the U-Net. `F_pad = ceil(max_F_in_batch / 8) * 8`. Pad coords are filled with `0.0` (box centre) and masked from the coordinate loss.

**D7 -- Timestep is always handed to the denoiser as a float in `[0, 1]`.** DDPM/D3PM divide their integer index by `noise_steps`; flow matching passes its continuous `t` straight through. This is the one interface that lets all three processes share both denoisers.

**D8 -- Welding needs a snap first.** `weld` merges by exact equality; two continuous floats never coincide. The post-hoc pipeline is `snap to the 128-bin grid -> weld -> drop duplicate faces -> fix winding -> drop degenerate`. The snap is `dequantize(quantize(v))`, reusing the existing pair.

**D9 -- Scaffold masking is a sampler-side operation, applied only below a threshold timestep.** Applied at every step from the start it fights the early denoising, when `x_t` is nearly pure noise and every coordinate is out of bounds. Continuous processes project out-of-scaffold coordinates onto the nearest legal voxel centre; the discrete process masks illegal bin logits.

**D10 -- State, noise and readout are three independent axes, not one.** Conflating them is the easiest mistake to make here, and the reference implementation is what makes it easy: maxim-mat/trace-denoise-refactor runs a **Gaussian** forward process (`base_diffusion.noise_data` is `sqrt(abar) x + sqrt(1-abar) eps`) and then reads it out **categorically** (`diffusion_module.training_step` takes `argmax(target, dim=1)` and applies `CrossEntropyLoss`). That is neither "continuous diffusion" nor D3PM; it is a third thing, and it is a shipped, working configuration. The three axes are:

- `state` -- what `x` *is*: `continuous` (raw coordinates), `quantized` (coordinates snapped to the `num_bins` grid, still carried as floats), `onehot` (9 x `num_bins` indicator channels), or `bins` (integer indices).
- `process` -- how it is corrupted: `ddpm` / `flow` (Gaussian) or `d3pm` (categorical).
- `loss` -- how it is read out: `mse`, `hungarian` (regression) or `ce` (categorical).

Gaussian noise with a categorical readout (`state: quantized|onehot`, `process: ddpm`, `loss: ce`) is a legal and independently interesting arm. So is a categorical process with a categorical readout. Task 11 covers the first, Task 12 the second.

**D11 -- The 128 bins are an *ordinal* alphabet, and that decides the transition matrix.** The Levi-graph branch's discrete variables were node and edge *class labels* -- nominal, unordered, where bin 3 and bin 4 have no relation. Coordinate bins are the opposite: adjacent bins are nearly the same coordinate, and the alphabet carries a metric. A uniform transition throws that structure away and corrupts a coordinate to a uniformly random position; a **discretized-Gaussian transition** (Austin et al. arXiv:2107.03006 section 3.2, their `D3PM-gauss`) corrupts it to a *nearby* bin, which is both the better-matched process and the one whose reverse the model has a chance of learning from local evidence. Both ship as `transition: uniform | gaussian`, and the pair is a controlled A/B rather than a default plus a fallback.

That difference in modality is also why the Levi branch's outcome does not transfer as a prediction. It is a reason to instrument, not a reason to expect failure: log per-noise-bucket bin accuracy from epoch 1 so the answer arrives in an epoch instead of a weekend.

## Compatibility Matrix

`validate_combination()` (Task 2) enforces this. **R** = raise, **W** = warn and continue.

**Order / positional encoding / loss** -- unchanged, and independent of everything below:

| order / pos_embed | `loss: mse` | `loss: hungarian` | `loss: ce` |
|---|---|---|---|
| `morton` / `sinusoidal` | allowed | W (legal, wasteful) | allowed |
| `morton` / `none` | R (D4) | allowed | R (D4) |
| `none` / `sinusoidal` | R (D4) | R (PE over an arbitrary order) | R (D4) |
| `none` / `none` | R (D4) | allowed | R (D4) |

`loss: ce` follows `loss: mse` here: a categorical readout is still slot-to-slot, so it needs the same order and positional encoding. The set-matched categorical arm is `loss: hungarian` with `state: onehot`, whose matching cost is built from the expected coordinate.

**State / process / loss / target** -- the axis this revision adds:

| `state` | `process` | `loss` | `target` | in-grid output | notes |
|---|---|---|---|---|---|
| `continuous` | `ddpm` | `mse` / `hungarian` | `noise` / `original` | no | Phase A reference |
| `continuous` | `flow` | `mse` / `hungarian` | `velocity` | no | |
| `quantized` | `ddpm` | `mse` / `hungarian` | `noise` / `original` | no | control: isolates the snapped *target* from the categorical *readout* |
| `quantized` | `ddpm` | `ce` | `original` | yes | the reference's scheme, coordinate-channel variant |
| `quantized` | `flow` | `ce` | R | -- | see below |
| `onehot` | `ddpm` | `ce` / `hungarian` | `original` | yes | the reference's scheme, exactly |
| `onehot` | `flow` | `ce` | R | -- | see below |
| `bins` | `d3pm` | `ce` | `original` | yes | D3PM, `transition: uniform` or `gaussian` |
| `bins` | `ddpm` / `flow` | any | R | -- | integer state has no Gaussian path |
| `continuous` | `d3pm` | any | R | -- | categorical process needs an alphabet |

Two rules worth spelling out because they are easy to get wrong:

- **`loss: ce` forces `target: original`.** The reference allows `denoiser_output: noise` with `cross_entropy`, which takes `argmax` of a Gaussian noise tensor -- an arbitrary index carrying no signal. It runs and it trains to nothing. Raise on it here rather than inherit it.
- **`process: flow` is incompatible with `loss: ce`.** Flow matching regresses a velocity; there is no categorical quantity in the parameterisation to apply cross-entropy to. Reaching one would mean predicting `x0` from the velocity and adding a second head, which is a different model, not a config.

Plus, unchanged: `denoiser: unet` requires `order: morton` (D5); `scaffold.enabled` is legal with every process, in its projection form for Gaussian states and its bin form for `bins`.

## File Structure

**Create:**
- `src/dataset/mesh_set_dataset.py` -- `MeshSetDataset` (face-set tensors, canonical rotation, Morton sort, optional quantization) + `mesh_set_collate_fn`.
- `src/models/mesh_set_modules.py` -- time embedding, LOD1 face encoder, and the conv/attention blocks both denoisers share.
- `src/models/mesh_set_unet.py` -- `ConditionalMeshUNet`.
- `src/models/mesh_set_transformer.py` -- `MeshSetTransformer`.
- `src/models/mesh_set_losses.py` -- `masked_mse_loss`, `hungarian_loss`, `discrete_ce_loss`.
- `src/models/mesh_processes.py` -- `BaseProcess`, `GaussianProcess` (DDPM + DDIM), `FlowMatchingProcess`, `DiscreteProcess`.
- `src/models/mesh_scaffold.py` -- `lod1_scaffold`, `project_to_scaffold`, `scaffold_bin_mask`.
- `src/models/mesh_set_postprocess.py` -- `faces_to_mesh`.
- `src/models/mesh_diffusion_module.py` -- `MeshDiffusionModule` (LightningModule).
- `src/train_mesh_diffusion.py` -- entry point for `config_set: mesh_diffusion`.
- `src/eval/mesh_set_eval.py` -- `run_mesh_set_eval` + `MeshSetEvalCallback`.
- `configs/mesh-diff-*.yaml` (Task 15).
- `tests/test_mesh_set_dataset.py`, `tests/test_mesh_set_losses.py`, `tests/test_mesh_processes.py`, `tests/test_mesh_set_denoisers.py`, `tests/test_mesh_scaffold.py`, `tests/test_mesh_set_postprocess.py`, `tests/test_mesh_diffusion_compat.py`, `tests/test_mesh_diffusion_smoke.py`.

**Modify:**
- `src/utils/config.py` -- add `MeshDiffusionConfig`, register on `Config`.
- `src/train.py:29-34` -- add the `mesh_diffusion` branch to the `config_set` switch.
- `run_experiments.sh` -- add the diffusion arm queue (Task 15).

---

# Phase A — one working arm end to end

Target arm: `order: morton`, `pos_embed: sinusoidal`, `loss: mse`, `denoiser: unet`, `process: ddpm`, `target: noise`, scaffold off. Phase A alone is shippable, trainable software.

### Task 1: Face-set dataset

**Files:**
- Create: `src/dataset/mesh_set_dataset.py`
- Test: `tests/test_mesh_set_dataset.py`

**Interfaces:**
- Consumes: `normalize_to_unit_box`, `quantize`, `dequantize`, `_scan_lod_dir`, `MeshDataset` (for its scan/filter logic only) from `src/dataset/mesh_dataset.py`.
- Produces:
  - `rotate_faces_canonical(tri, num_bins=128) -> np.ndarray` — `[F,3,3] -> [F,3,3]`
  - `morton_order(tri, num_bins=128) -> np.ndarray` — `[F,3,3] -> [F]` int64 permutation
  - `faces_to_array(verts, faces) -> np.ndarray` — `[V,3],[F,3] -> [F,3,3]`
  - `MeshSetDataset(dataset_dir, lod_in, lod_out, num_bins, margin_lo, margin_hi, max_faces, max_files, order="morton", state="continuous")`; `__getitem__` returns a dict with keys `x [F,10] float32`, `cond [Fc,10] float32`, `id str`, `center [3]`, `scale [3]`, and for every `state` except `"continuous"` also `x_bins [F,9] int64`.
  - `mesh_set_collate_fn(batch, multiple_of=8) -> dict` with keys `x [B,10,Fpad]`, `x_mask [B,Fpad] bool (True=real)`, `cond [B,10,Fcpad]`, `cond_mask [B,Fcpad] bool`, `ids list[str]`, `center [B,3]`, `scale [B,3]`, and `x_bins [B,9,Fpad]` when present.

**Note on layout:** `__getitem__` returns `[F, C]` (face-major, matching `MeshDataset`'s numpy style); `mesh_set_collate_fn` transposes to `[B, C, F]` so both denoisers and the trace-repo module conventions get channels-first without a per-forward permute.

- [ ] **Step 1: Write the failing test**

```python
"""Face-set representation: canonical rotation, Morton order, batch padding.

Tiny synthetic meshes, CPU only. These assertions are what every downstream
loss and denoiser rests on: a face must survive rotation unchanged as a set,
the order must be a permutation, and padding must not leak into the mask.
"""
import numpy as np
import pytest
import torch

from src.dataset.mesh_set_dataset import (
    faces_to_array,
    mesh_set_collate_fn,
    morton_order,
    rotate_faces_canonical,
)

NUM_BINS = 128


def _tri(a, b, c):
    return np.array([[a, b, c]], dtype=float)


def test_rotate_is_cyclic_and_winding_preserving():
    v0, v1, v2 = [0.4, 0.4, 0.4], [-0.1, 0.0, 0.1], [0.2, -0.3, -0.4]
    tri = _tri(v0, v1, v2)
    out = rotate_faces_canonical(tri, NUM_BINS)
    # Same three corners, as a set.
    assert {tuple(x) for x in out[0]} == {tuple(v0), tuple(v1), tuple(v2)}
    # Smallest by (z, y, x) leads: v2 has z=-0.4, the lowest.
    assert np.allclose(out[0, 0], v2)
    # Winding preserved: the cyclic successor of v2 is still v0.
    assert np.allclose(out[0, 1], v0)


def test_rotate_is_idempotent():
    tri = _tri([0.4, 0.4, 0.4], [-0.1, 0.0, 0.1], [0.2, -0.3, -0.4])
    once = rotate_faces_canonical(tri, NUM_BINS)
    assert np.allclose(rotate_faces_canonical(once, NUM_BINS), once)


def test_rotate_equalises_the_three_spellings():
    v = [[0.4, 0.4, 0.4], [-0.1, 0.0, 0.1], [0.2, -0.3, -0.4]]
    spellings = [_tri(v[0], v[1], v[2]), _tri(v[1], v[2], v[0]), _tri(v[2], v[0], v[1])]
    outs = [rotate_faces_canonical(s, NUM_BINS) for s in spellings]
    assert np.allclose(outs[0], outs[1])
    assert np.allclose(outs[0], outs[2])


def test_morton_order_is_a_permutation_and_deterministic():
    rng = np.random.default_rng(0)
    tri = rng.uniform(-0.5, 0.5, size=(17, 3, 3))
    perm = morton_order(tri, NUM_BINS)
    assert sorted(perm.tolist()) == list(range(17))
    assert np.array_equal(perm, morton_order(tri, NUM_BINS))


def test_morton_order_groups_nearby_faces():
    # Two tight clusters far apart; sorting must not interleave them.
    lo = np.full((5, 3, 3), -0.45) + np.linspace(0, 0.01, 5)[:, None, None]
    hi = np.full((5, 3, 3), 0.45) + np.linspace(0, 0.01, 5)[:, None, None]
    tri = np.concatenate([lo, hi])[[0, 5, 1, 6, 2, 7, 3, 8, 4, 9]]  # interleaved
    perm = morton_order(tri, NUM_BINS)
    is_hi = (tri[perm][:, 0, 0] > 0)
    # Sorted output must be all-low then all-high, never alternating.
    assert is_hi.tolist() == sorted(is_hi.tolist())


def test_faces_to_array_expands_indices():
    verts = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]])
    faces = np.array([[0, 1, 2]])
    out = faces_to_array(verts, faces)
    assert out.shape == (1, 3, 3)
    assert np.allclose(out[0, 1], [1.0, 0, 0])


def test_state_quantized_snaps_the_continuous_channels():
    """Under state: quantized the float channels must already sit on grid
    points, so a regression readout and a categorical one are scored against an
    identical target -- which is what makes the b3-vs-b4 control readable."""
    from src.dataset.mesh_dataset import dequantize
    from src.dataset.mesh_set_dataset import _pack

    rng = np.random.default_rng(0)
    tri = rng.uniform(-0.5, 0.5, size=(6, 3, 3))
    x, bins = _pack(tri, NUM_BINS, "morton", state="quantized")
    assert bins is not None and bins.shape == (6, 9)
    assert np.allclose(x[:, :9], dequantize(bins, NUM_BINS), atol=1e-6)


def test_state_continuous_emits_no_bins():
    from src.dataset.mesh_set_dataset import _pack

    x, bins = _pack(np.zeros((4, 3, 3)), NUM_BINS, "morton", state="continuous")
    assert bins is None and x.shape == (4, 10)


def test_unknown_state_raises():
    from src.dataset.mesh_set_dataset import _pack

    with pytest.raises(ValueError, match="Unknown state"):
        _pack(np.zeros((2, 3, 3)), NUM_BINS, "morton", state="onehotish")


def test_collate_pads_to_multiple_of_eight_and_sets_presence():
    items = [
        {"x": torch.zeros(5, 10), "cond": torch.zeros(3, 10),
         "id": "a", "center": torch.zeros(3), "scale": torch.ones(3)},
        {"x": torch.zeros(11, 10), "cond": torch.zeros(4, 10),
         "id": "b", "center": torch.zeros(3), "scale": torch.ones(3)},
    ]
    for it in items:                       # real faces carry presence +0.5
        it["x"][:, 9] = 0.5
        it["cond"][:, 9] = 0.5
    out = mesh_set_collate_fn(items, multiple_of=8)
    assert out["x"].shape == (2, 10, 16)   # max 11 -> 16
    assert out["x_mask"][0].sum() == 5 and out["x_mask"][1].sum() == 11
    # Pad slots: presence -0.5, coords 0.
    assert torch.allclose(out["x"][0, 9, 5:], torch.full((11,), -0.5))
    assert torch.allclose(out["x"][0, :9, 5:], torch.zeros(9, 11))
    # Condition padded independently, also to a multiple of 8.
    assert out["cond"].shape == (2, 10, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mesh_set_dataset.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'src.dataset.mesh_set_dataset'`

- [ ] **Step 3: Write the implementation**

```python
"""LOD2 as a padded face set, for the non-autoregressive diffusion branch.

`MeshDataset` spells a mesh out as a token sequence for an autoregressive
model. This spells the same pair out as a fixed-width array of faces, which is
what a diffusion denoiser consumes: `[F, 10]`, nine coordinate channels plus a
presence channel, in the same LOD1-normalized frame `MeshDataset` already uses.

Vertices are repeated per face and connectivity is not represented. Shared
vertices come back post-hoc, in `src/models/mesh_set_postprocess.py`.
"""
import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset.mesh_dataset import (
    NUM_BINS,
    _scan_lod_dir,
    dequantize,
    normalize_to_unit_box,
    quantize,
)

logger = logging.getLogger(__name__)

# Presence channel values. Symmetric around 0 so the sign is the decision rule
# and the channel has the same scale as a coordinate -- one Gaussian noise
# level then fits all ten channels without a per-channel weight.
PRESENT, ABSENT = 0.5, -0.5
N_CHANNELS = 10


def faces_to_array(verts, faces):
    """``[V,3]`` vertices + ``[F,3]`` indices to ``[F,3,3]`` explicit corners."""
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        return np.zeros((0, 3, 3), dtype=float)
    return verts[faces]


def _grid_key(tri, num_bins):
    """``[F,3]`` lexicographic (z,y,x) rank of each corner, on the bin grid.

    Decided on the quantized grid rather than on the floats so the comparison
    is exact and total: two corners in the same bin are the same corner as far
    as the model can ever tell, and float lexicographic order on near-equal
    coordinates is not stable across platforms.
    """
    q = quantize(np.asarray(tri, dtype=float).reshape(-1, 3), num_bins)
    q = q.reshape(-1, 3, 3)
    return (q[:, :, 2] * num_bins + q[:, :, 1]) * num_bins + q[:, :, 0]


def rotate_faces_canonical(tri, num_bins=NUM_BINS):
    """Cyclically rotate each face so its ``(z,y,x)``-smallest corner leads.

    A triangle has three spellings with identical geometry and identical
    winding. Without this the coordinate loss punishes a face for being spelled
    from a different corner, which is not an error the model can fix -- it is
    an error in the target. Rotation, not sorting: a sort would reverse winding
    on half the faces, which `fix_winding` would then have to undo.

    Args:
        tri: ``[F,3,3]`` face corners.
        num_bins: grid used to break ties exactly. Must match the dataset's.

    Returns:
        np.ndarray: ``[F,3,3]``, same dtype semantics as the input.
    """
    tri = np.asarray(tri, dtype=float)
    if len(tri) == 0:
        return tri
    start = _grid_key(tri, num_bins).argmin(axis=1)
    idx = (start[:, None] + np.arange(3)[None, :]) % 3
    return np.take_along_axis(tri, idx[:, :, None], axis=1)


def morton_order(tri, num_bins=NUM_BINS):
    """Permutation sorting faces by the Morton code of their centroid.

    Z-order interleaving gives a 1-D sequence in which adjacency implies
    spatial proximity, which is the property a stride-2 convolution along the
    face axis needs and a file-order face list does not have. Ties are broken
    by the canonical corner key so the order is total and reproducible.

    Args:
        tri: ``[F,3,3]`` face corners in ``[-0.5, 0.5]``.
        num_bins: quantization grid; the code uses ``log2(num_bins)`` bits/axis.

    Returns:
        np.ndarray: ``[F]`` int64 permutation.
    """
    tri = np.asarray(tri, dtype=float)
    if len(tri) == 0:
        return np.zeros(0, dtype=np.int64)
    bits = int(np.log2(num_bins))
    if 2 ** bits != num_bins:
        raise ValueError(f"num_bins must be a power of two for Morton codes, got {num_bins}.")

    q = quantize(tri.mean(axis=1), num_bins).astype(np.int64)   # [F,3] centroids
    code = np.zeros(len(q), dtype=np.int64)
    for b in range(bits):
        for axis in range(3):
            code |= ((q[:, axis] >> b) & 1) << (3 * b + axis)
    # Secondary key: the face's own smallest corner, so co-centroid faces
    # (a fold, two triangles of one quad) get a deterministic order.
    tie = _grid_key(tri, num_bins).min(axis=1)
    return np.lexsort((tie, code)).astype(np.int64)


def _pack(tri, num_bins, order, state="continuous"):
    """``[F,3,3]`` corners to the ``[F,10]`` channel layout (+ optional bins).

    `state` decides whether the float channels are snapped to the grid and
    whether integer bins come along (plan D10). The one-hot expansion is NOT
    done here: it is 9 x num_bins channels per face, which would multiply the
    collate buffer and the worker-to-main-process transfer by ~128 for a tensor
    `MeshDiffusionModule` can rebuild from `x_bins` on the accelerator with one
    `scatter_`.
    """
    tri = rotate_faces_canonical(tri, num_bins)
    if order == "morton":
        tri = tri[morton_order(tri, num_bins)]
    elif order != "none":
        raise ValueError(f"Unknown order: {order!r}. Expected 'morton' or 'none'.")

    x = np.zeros((len(tri), N_CHANNELS), dtype=np.float32)
    x[:, :9] = tri.reshape(len(tri), 9)
    x[:, 9] = PRESENT
    if state == "continuous":
        return x, None
    if state not in ("quantized", "onehot", "bins"):
        raise ValueError(
            f"Unknown state: {state!r}. Expected 'continuous', 'quantized', "
            "'onehot' or 'bins'.")
    bins = quantize(tri.reshape(-1, 3), num_bins).reshape(len(tri), 9).astype(np.int64)
    # Snap the float channels onto the same grid so a regression readout and a
    # categorical one are scored against an identical target -- which is what
    # makes the `quantized`/`mse` control arm (b3) interpretable at all.
    x[:, :9] = dequantize(bins, num_bins).astype(np.float32)
    return x, bins


class MeshSetDataset(Dataset):
    """Paired (LOD1, LOD2) buildings as padded face-set arrays.

    Shares `MeshDataset`'s scan, `max_faces` filter and normalization frame
    exactly, so a diffusion run and an autoregressive run on the same
    `dataset_dir` see the same corpus and the same coordinate frame.

    Args:
        dataset_dir: root holding the two LOD directories.
        lod_in, lod_out: directory names, not LOD numbers.
        num_bins: quantization grid, for rotation ties, Morton codes and the
            quantized states. Must match `mesh_data.num_bins` for comparability.
        margin_lo, margin_hi: per-axis headroom, passed to
            `normalize_to_unit_box`. Same values as the AR branch.
        max_faces: drop pairs where either side exceeds this triangle count.
        max_files: read only the first N files per LOD. Smoke tests only.
        order: "morton" (spatially sorted) or "none" (file order).
        state: "continuous" | "quantized" | "onehot" | "bins" (plan D10).
            Anything but "continuous" snaps the float channels to the grid and
            emits `x_bins`. "onehot" is expanded in the model, not here.
    """

    def __init__(self, dataset_dir, lod_in="LOD1", lod_out="LOD2",
                 num_bins=NUM_BINS, margin_lo=(0.0, 0.0, 0.0),
                 margin_hi=(0.0, 0.0, 0.1), max_faces=200, max_files=None,
                 order="morton", state="continuous"):
        self.num_bins = num_bins
        self.margin_lo = np.asarray(margin_lo, dtype=float)
        self.margin_hi = np.asarray(margin_hi, dtype=float)
        self.order = order
        self.state = state

        meshes_in = _scan_lod_dir(dataset_dir, lod_in, max_files)
        meshes_out = _scan_lod_dir(dataset_dir, lod_out, max_files)
        ids = sorted(set(meshes_in) & set(meshes_out))
        if max_faces is not None:
            kept = [i for i in ids
                    if len(meshes_out[i][1]) <= max_faces
                    and len(meshes_in[i][1]) <= max_faces]
            logger.info("Dropped %d/%d buildings over max_faces=%d",
                        len(ids) - len(kept), len(ids), max_faces)
            ids = kept
        self.ids = ids
        self.pairs = [(meshes_in[i], meshes_out[i]) for i in ids]
        if not self.ids:
            logger.warning("No buildings shared between %s and %s under %s",
                           lod_in, lod_out, dataset_dir)
        self.max_faces_seen = max(
            (max(len(f_out), len(f_in))
             for (_, f_in), (_, f_out) in self.pairs), default=0)
        logger.info("MeshSetDataset: %d pairs, longest face set %d",
                    len(self.ids), self.max_faces_seen)

    def __len__(self):
        return len(self.ids)

    def mesh_pair(self, index):
        """Raw ``((verts, faces), (verts, faces))`` in metres, for .obj dumps."""
        return self.pairs[index]

    def __getitem__(self, index):
        (v_in, f_in), (v_out, f_out) = self.pairs[index]
        v_in_n, center, scale = normalize_to_unit_box(
            v_in, margin_lo=self.margin_lo, margin_hi=self.margin_hi)
        v_out_n, _, _ = normalize_to_unit_box(
            v_out, ref=v_in, margin_lo=self.margin_lo, margin_hi=self.margin_hi)

        x, bins = _pack(faces_to_array(v_out_n, f_out),
                        self.num_bins, self.order, self.state)
        # The condition is always Morton-sorted: it is read by cross-attention,
        # which is permutation invariant, so the order costs nothing -- but a
        # sorted condition makes the U-Net's condition encoder see the same
        # locality its main path does.
        cond, _ = _pack(faces_to_array(v_in_n, f_in),
                        self.num_bins, "morton", "continuous")

        item = {
            "x": torch.from_numpy(x),
            "cond": torch.from_numpy(cond),
            "id": self.ids[index],
            "center": torch.tensor(center, dtype=torch.float32),
            "scale": torch.tensor(scale, dtype=torch.float32),
        }
        if bins is not None:
            item["x_bins"] = torch.from_numpy(bins)
        return item


def _pad_stack(seqs, width, fill_coord=0.0):
    """``list[[F,10]]`` to ``([B,10,width], [B,width] bool)``, True = real."""
    out = torch.zeros((len(seqs), width, N_CHANNELS), dtype=torch.float32)
    out[:, :, :9] = fill_coord
    out[:, :, 9] = ABSENT
    mask = torch.zeros((len(seqs), width), dtype=torch.bool)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
        mask[i, : len(s)] = True
    return out.permute(0, 2, 1).contiguous(), mask


def mesh_set_collate_fn(batch, multiple_of=8):
    """Right-pad a batch of face sets to a common, U-Net-divisible width.

    Padded to the batch maximum rounded up, not to a corpus-wide constant: the
    face-count distribution is long-tailed, so a fixed width would spend most
    of every batch denoising padding. `multiple_of` exists because three
    stride-2 downsamples need a length divisible by 8; the transformer denoiser
    does not care and is unaffected by the rounding.

    Args:
        batch: list of `MeshSetDataset` items.
        multiple_of: pad width is rounded up to this. 8 for the 3-level U-Net.

    Returns:
        dict: channels-first tensors plus boolean masks (True = real face).
    """
    def width(key):
        longest = max(len(item[key]) for item in batch)
        return int(np.ceil(max(longest, 1) / multiple_of) * multiple_of)

    x, x_mask = _pad_stack([item["x"] for item in batch], width("x"))
    cond, cond_mask = _pad_stack([item["cond"] for item in batch], width("cond"))
    out = {"x": x, "x_mask": x_mask, "cond": cond, "cond_mask": cond_mask,
           "ids": [item["id"] for item in batch]}
    for key in ("center", "scale"):
        out[key] = torch.stack([item[key] for item in batch])
    if "x_bins" in batch[0]:
        w = x.shape[-1]
        bins = torch.zeros((len(batch), w, 9), dtype=torch.long)
        for i, item in enumerate(batch):
            bins[i, : len(item["x_bins"])] = item["x_bins"]
        out["x_bins"] = bins.permute(0, 2, 1).contiguous()
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_mesh_set_dataset.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/dataset/mesh_set_dataset.py tests/test_mesh_set_dataset.py
git commit -m "feat(mesh-diff): face-set dataset with canonical rotation and Morton order"
```

---

### Task 2: Config schema and the compatibility gate

**Files:**
- Create: `tests/test_mesh_diffusion_compat.py`
- Modify: `src/utils/config.py` (add `MeshDiffusionConfig` after `MeshVQVAEConfig`, register on `Config`)
- Modify: `src/train.py:29-34`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `MeshDiffusionConfig` dataclass; `validate_combination(cfg) -> None` (module-level in `src/utils/config.py`), raising `ValueError` on an illegal arm and emitting `logger.warning` on a wasteful one.

- [ ] **Step 1: Write the failing test**

```python
"""The compatibility gate. Every rejection here is a run that would either
crash hours in or train to a loss floor it cannot get under.

Cheap and pure: no model, no data, no trainer.
"""
import pytest
from omegaconf import OmegaConf

from src.utils.config import Config, validate_combination


def _cfg(**kw):
    cfg = OmegaConf.structured(Config)
    cfg.config_set = "mesh_diffusion"
    for k, v in kw.items():
        OmegaConf.update(cfg, f"mesh_diffusion.{k}", v, force_add=True)
    return cfg


def test_default_arm_is_legal():
    validate_combination(_cfg())  # morton / sinusoidal / mse / unet / ddpm / noise


def test_unordered_with_mse_raises():
    with pytest.raises(ValueError, match="hungarian"):
        validate_combination(_cfg(order="none", pos_embed="none", loss="mse"))


def test_unordered_with_pos_embed_raises():
    with pytest.raises(ValueError, match="pos_embed"):
        validate_combination(
            _cfg(order="none", pos_embed="sinusoidal", loss="hungarian"))


def test_no_pos_embed_with_mse_raises():
    with pytest.raises(ValueError, match="hungarian"):
        validate_combination(_cfg(order="morton", pos_embed="none", loss="mse"))


def test_unet_requires_sorted_order():
    with pytest.raises(ValueError, match="unet"):
        validate_combination(_cfg(denoiser="unet", order="none",
                                  pos_embed="none", loss="hungarian"))


def test_flow_requires_velocity_target():
    with pytest.raises(ValueError, match="velocity"):
        validate_combination(_cfg(process="flow", target="noise"))
    validate_combination(_cfg(process="flow", target="velocity"))


def test_d3pm_requires_bins_and_original():
    with pytest.raises(ValueError, match="state: bins"):
        validate_combination(_cfg(process="d3pm", target="original",
                                  loss="ce", state="quantized"))
    with pytest.raises(ValueError, match="original"):
        validate_combination(_cfg(process="d3pm", target="noise",
                                  loss="ce", state="bins"))
    validate_combination(_cfg(process="d3pm", target="original",
                              loss="ce", state="bins"))


def test_gaussian_process_rejects_the_integer_state():
    with pytest.raises(ValueError, match="no Gaussian path"):
        validate_combination(_cfg(process="ddpm", target="original",
                                  loss="ce", state="bins"))


def test_ce_needs_an_alphabet_and_the_x0_target():
    with pytest.raises(ValueError, match="discrete alphabet"):
        validate_combination(_cfg(loss="ce", state="continuous",
                                  target="original"))
    with pytest.raises(ValueError, match="signal-free"):
        validate_combination(_cfg(loss="ce", state="quantized", target="noise"))
    validate_combination(_cfg(loss="ce", state="quantized", target="original"))


def test_flow_with_ce_is_rejected():
    with pytest.raises(ValueError, match="incompatible"):
        validate_combination(_cfg(process="flow", target="velocity",
                                  loss="ce", state="quantized"))


def test_onehot_with_a_regression_loss_is_rejected():
    with pytest.raises(ValueError, match="indicator channels"):
        validate_combination(_cfg(loss="mse", state="onehot", target="original"))


def test_ddpm_rejects_velocity():
    with pytest.raises(ValueError, match="target"):
        validate_combination(_cfg(process="ddpm", target="velocity"))


def test_sorted_plus_hungarian_warns_but_passes(caplog):
    validate_combination(_cfg(order="morton", pos_embed="sinusoidal",
                              loss="hungarian", denoiser="transformer"))
    assert any("wasteful" in r.message.lower() or "wasteful" in r.getMessage().lower()
               for r in caplog.records)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mesh_diffusion_compat.py -v`
Expected: FAIL, `ImportError: cannot import name 'validate_combination'`

- [ ] **Step 3: Add the dataclass to `src/utils/config.py`**

Insert after `MeshVQVAEConfig` (which ends at line 308, before `EarlyStoppingConfig`):

```python
@dataclass
class ScaffoldConfig:
    """LOD1-derived legal region, applied during reverse diffusion.

    MeshWeaver's scaffold masking (arXiv 2606.04688, section 3.2 fix 3), which
    the 2026-08-24 research scan flags as the item most likely to transfer. In
    an autoregressive model it is a logit mask; here it is a projection, and it
    costs one operation per reverse step rather than one per token.
    """
    enabled: bool = False
    # Occupancy grid resolution, in normalized box units. 1/32 of the box is
    # ~0.5 m on a 16 m building -- coarse enough that a legal roof plane is not
    # carved into disconnected cells, fine enough to exclude open air.
    voxel: float = 1.0 / 32
    # Grow the LOD1 occupancy by this many voxels before testing. LOD2 is NOT
    # contained in LOD1 -- the ridge rises above it -- so a zero dilation would
    # project every ridge vertex back down onto the LOD1 roof.
    dilate: int = 3
    # Only apply below this fraction of the trajectory. Above it x_t is nearly
    # pure noise and every coordinate is out of bounds, so projecting would
    # overwrite the denoiser rather than guide it.
    apply_below_t: float = 0.5


@dataclass
class MeshDiffusionConfig:
    """Non-autoregressive whole-mesh diffusion. Read under
    `config_set: mesh_diffusion`.

    Mesh *data* still comes from `mesh_data` -- the same dataset_dir, LOD names,
    num_bins, margins and max_faces as the autoregressive branch, so the two are
    comparable. This block holds only what is specific to the diffusion model.

    The seven axes below are meant to be mixed; `validate_combination` in this
    module is the authority on which mixtures are legal.
    """
    # --- representation ---------------------------------------------------
    # "morton": faces sorted by Z-order of their quantized centroid.
    # "none":   file order, only meaningful with a permutation-equivariant model.
    order: str = "morton"
    # What `x` is. Independent of how it is corrupted (`process`) and how it is
    # read out (`loss`) -- see plan D10, which exists because the reference
    # implementation ties all three together and it is easy to copy that.
    #   "continuous" -- raw coordinates in [-0.5, 0.5]. 10 channels.
    #   "quantized"  -- coordinates snapped to the num_bins grid, still floats.
    #                   10 channels. Output leaves the grid unless loss is ce.
    #   "onehot"     -- 9 * num_bins indicator channels plus presence. What the
    #                   reference actually diffuses. Wide but cheap: one 1x1
    #                   conv at the input, ~7 MB of activations at F=200, B=8.
    #   "bins"       -- 9 integer indices. The only state a categorical process
    #                   can corrupt.
    state: str = "continuous"
    # D3PM only. "uniform" corrupts a bin to a uniformly random one; "gaussian"
    # corrupts it to a nearby one (Austin et al. arXiv:2107.03006 section 3.2).
    # Coordinate bins are ordinal, so these are genuinely different processes
    # and not two spellings of the same default -- see plan D11.
    transition: str = "gaussian"
    # Width of the discretized-Gaussian transition at t = 1, in bins. Only read
    # under transition: gaussian.
    transition_sigma: float = 16.0
    # Categorical readout over a Gaussian process only. After each reverse step
    # the predicted x0 is projected back onto the alphabet: "hard" takes the
    # argmax one-hot (Diffusion-LM's clamping trick, Li et al.
    # arXiv:2205.14217 section 4.2), "soft" keeps the softmax. Hard is what
    # makes a continuous process over a discrete alphabet actually land on
    # grid points; soft is the ablation that shows whether it matters.
    x0_clamp: str = "hard"

    # --- denoiser ---------------------------------------------------------
    denoiser: str = "unet"           # "unet" | "transformer"
    d_model: int = 256               # transformer width / U-Net base channels
    n_head: int = 8
    num_layers: int = 8              # transformer only
    dropout: float = 0.1
    # "sinusoidal" needs a meaningful face order; "none" makes the transformer
    # permutation-equivariant, which is the only correct setting for order:none.
    pos_embed: str = "sinusoidal"
    time_dim: int = 128
    # Probability of dropping the LOD1 condition during training, enabling
    # classifier-free guidance at sampling. 0 disables CFG.
    cond_dropout: float = 0.1

    # --- process ----------------------------------------------------------
    process: str = "ddpm"            # "ddpm" | "flow" | "d3pm"
    target: str = "noise"            # "noise" | "original" | "velocity"
    noise_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    # Reverse steps at eval. Full-length reverse diffusion per validation
    # building is what makes this callback expensive; 50 DDIM steps is the
    # standard trade and is what every reported number should be measured at.
    eval_steps: int = 50
    # Guidance weight at sampling. 1.0 is plain conditional; >1 sharpens
    # toward the condition at the cost of diversity. Requires cond_dropout > 0.
    guidance: float = 1.0

    # --- loss -------------------------------------------------------------
    loss: str = "mse"                # "mse" | "hungarian" | "ce"
    # Weight on the presence channel relative to the nine coordinate channels.
    # Presence decides face count, which the geometric metrics are far more
    # sensitive to than to a fraction of a bin on one vertex.
    presence_weight: float = 1.0
    # Hungarian only: relative weight of presence inside the matching cost.
    match_presence_weight: float = 1.0

    # --- sampling and write-back -----------------------------------------
    scaffold: ScaffoldConfig = field(default_factory=ScaffoldConfig)
    # Snap to the num_bins grid before welding. Without it `weld` merges by
    # exact float equality and never fires, and every shared vertex stays
    # split -- a crack at every edge.
    snap_before_weld: bool = True
    # Faces whose area is below this (in normalized box units squared) are
    # dropped after welding. A diffusion sample can put all three corners in
    # one bin; such a face contributes nothing and breaks winding repair.
    min_face_area: float = 1e-6

    # --- eval -------------------------------------------------------------
    every_n_epochs: int = 5
    n_val: int = 16
    n_test: int = 64
    eval_batch_size: int = 8
    n_points: int = 4096
    taus: List[float] = field(default_factory=lambda: [0.25, 0.5])
    voxel_m: float = 0.25
    save_samples: int = 8
```

Register on `Config` (after the `mesh_vqvae` line, around line 431):

```python
    mesh_diffusion: MeshDiffusionConfig = field(default_factory=MeshDiffusionConfig)
```

And extend the `config_set` docstring comment on `Config` with:

```python
    #   "mesh_diffusion" — mesh_data + mesh_diffusion, non-autoregressive
    #                 whole-mesh diffusion / flow matching
```

- [ ] **Step 4: Add `validate_combination` at the end of `src/utils/config.py`**

```python
def validate_combination(cfg):
    """Reject arm combinations that cannot work, warn about ones that waste.

    Called once at startup, before a datamodule or a model is built, because
    every rejection here is otherwise a run that either crashes hours in or
    trains to a loss floor no amount of epochs gets under.

    Args:
        cfg: the resolved root `Config` (or an OmegaConf view of one).

    Raises:
        ValueError: the combination is unlearnable or structurally impossible.
    """
    d = cfg.mesh_diffusion
    sorted_order = d.order == "morton"
    has_pe = d.pos_embed == "sinusoidal"

    if d.order not in ("morton", "none"):
        raise ValueError(f"mesh_diffusion.order must be 'morton' or 'none', got {d.order!r}.")
    if d.pos_embed not in ("sinusoidal", "none"):
        raise ValueError(
            f"mesh_diffusion.pos_embed must be 'sinusoidal' or 'none', got {d.pos_embed!r}.")

    # D4. Slot-to-slot MSE asks the model which output slot a face belongs in.
    # Without positional information it cannot answer, and without a canonical
    # order there is no right answer to give. Either fix works; neither alone.
    if d.loss == "mse" and not (sorted_order and has_pe):
        raise ValueError(
            "loss: mse requires order: morton AND pos_embed: sinusoidal. With "
            f"order={d.order!r}, pos_embed={d.pos_embed!r} the target is a "
            "permutation the model cannot see, which is unlearnable rather "
            "than merely hard. Use loss: hungarian instead.")
    if not sorted_order and has_pe:
        raise ValueError(
            "pos_embed: sinusoidal with order: none embeds file order, which "
            "carries no geometric information. Set pos_embed: none.")

    # D5. Stride-2 convolution along the face axis is only meaningful if
    # adjacent slots are spatially adjacent.
    if d.denoiser == "unet" and not sorted_order:
        raise ValueError(
            "denoiser: unet requires order: morton -- its downsampling convolves "
            "along the face axis, which is arbitrary under order: none. Use "
            "denoiser: transformer for unordered sets.")
    if d.denoiser not in ("unet", "transformer"):
        raise ValueError(
            f"mesh_diffusion.denoiser must be 'unet' or 'transformer', got {d.denoiser!r}.")

    allowed_targets = {"ddpm": {"noise", "original"},
                       "flow": {"velocity"},
                       "d3pm": {"original"}}
    if d.process not in allowed_targets:
        raise ValueError(
            f"mesh_diffusion.process must be one of {sorted(allowed_targets)}, "
            f"got {d.process!r}.")
    if d.target not in allowed_targets[d.process]:
        raise ValueError(
            f"process: {d.process} allows target in "
            f"{sorted(allowed_targets[d.process])}, got {d.target!r}. "
            + ("Flow matching regresses the velocity x1 - x0; there is no "
               "epsilon to predict." if d.process == "flow" else
               "D3PM is parameterised by x0 only."))

    if d.loss not in ("mse", "hungarian", "ce"):
        raise ValueError(
            f"mesh_diffusion.loss must be 'mse', 'hungarian' or 'ce', got {d.loss!r}.")
    if d.state not in ("continuous", "quantized", "onehot", "bins"):
        raise ValueError(
            "mesh_diffusion.state must be 'continuous', 'quantized', 'onehot' "
            f"or 'bins', got {d.state!r}.")

    # D10. State and process are separate axes, but not every pair is a real
    # object: a Gaussian path needs a continuous state, a categorical chain
    # needs an alphabet.
    gaussian = d.process in ("ddpm", "flow")
    if gaussian and d.state == "bins":
        raise ValueError(
            f"state: bins has no Gaussian path -- process: {d.process} would "
            "add real noise to integer indices. Use state: onehot for a "
            "Gaussian process over the alphabet, or process: d3pm.")
    if d.process == "d3pm" and d.state != "bins":
        raise ValueError(
            f"process: d3pm requires state: bins, got {d.state!r}. The "
            "categorical chain corrupts indices, not coordinates.")
    if d.transition not in ("uniform", "gaussian"):
        raise ValueError(
            f"mesh_diffusion.transition must be 'uniform' or 'gaussian', "
            f"got {d.transition!r}.")

    if d.loss == "ce":
        # The alphabet has to exist before it can be read out.
        if d.state == "continuous":
            raise ValueError(
                "loss: ce needs a discrete alphabet; state: continuous has "
                "none. Use state: quantized (coordinate channels, categorical "
                "head) or state: onehot (the reference's scheme).")
        # The reference permits denoiser_output: noise with cross_entropy,
        # which supervises on argmax of a Gaussian noise tensor -- an arbitrary
        # index carrying no signal. It trains, and it trains to nothing.
        if d.target != "original":
            raise ValueError(
                f"loss: ce requires target: original, got {d.target!r}. Under "
                "target: noise the CE label is the argmax of a noise tensor, "
                "which is signal-free.")
        if d.process == "flow":
            raise ValueError(
                "process: flow is incompatible with loss: ce -- flow matching "
                "regresses a velocity, and there is no categorical quantity in "
                "that parameterisation to apply cross-entropy to. Reaching one "
                "needs a second x0 head, which is a different model.")
    elif d.state == "onehot":
        raise ValueError(
            f"state: onehot with loss: {d.loss!r} regresses {9} x num_bins "
            "indicator channels as if they were coordinates. Use loss: ce, or "
            "state: quantized for a regression readout on a snapped target.")
    if d.x0_clamp not in ("hard", "soft"):
        raise ValueError(
            f"mesh_diffusion.x0_clamp must be 'hard' or 'soft', got {d.x0_clamp!r}.")

    if d.guidance != 1.0 and d.cond_dropout <= 0:
        raise ValueError(
            "guidance != 1.0 needs cond_dropout > 0: classifier-free guidance "
            "requires an unconditional branch, which only exists if the "
            "condition was dropped during training.")

    if sorted_order and d.loss == "hungarian":
        logger.warning(
            "order: morton with loss: hungarian is legal but wasteful -- the "
            "matcher re-derives a correspondence the sort already fixed. Kept "
            "because it is the controlled A/B against loss: mse.")
```

Add at the top of `src/utils/config.py` if not already present:

```python
import logging

logger = logging.getLogger(__name__)
```

- [ ] **Step 5: Wire the config set in `src/train.py`**

Add immediately after the `mesh_vqvae` branch (line 31-32):

```python
    if cfg.config_set == "mesh_diffusion":
        from src.train_mesh_diffusion import train_mesh_diffusion
        return train_mesh_diffusion(cfg)
```

And extend the `ValueError` message on line 33-34 to list `'mesh_diffusion'`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_mesh_diffusion_compat.py -v`
Expected: 13 passed

Then confirm nothing else regressed:

Run: `python -m pytest tests/ -q -x`
Expected: the pre-existing suite still passes (`train_mesh_diffusion` does not exist yet, but nothing imports it until `config_set: mesh_diffusion` is used).

- [ ] **Step 7: Commit**

```bash
git add src/utils/config.py src/train.py tests/test_mesh_diffusion_compat.py
git commit -m "feat(mesh-diff): config schema and compatibility gate"
```

---

### Task 3: Losses — masked MSE, presence, and the Hungarian matcher

Both losses are written here even though Phase A only exercises MSE: they share the presence term and the mask convention, and splitting them across phases would mean writing that shared half twice.

**Files:**
- Create: `src/models/mesh_set_losses.py`
- Test: `tests/test_mesh_set_losses.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `masked_mse_loss(pred, target, mask, presence_weight=1.0) -> Tensor` — `pred`/`target` `[B,10,F]`, `mask` `[B,F]` bool.
  - `hungarian_match(pred, target, mask, match_presence_weight=1.0) -> LongTensor [B,F]` — for each batch item, the target index assigned to each prediction slot.
  - `hungarian_loss(pred, target, mask, presence_weight=1.0, match_presence_weight=1.0) -> Tensor`
  - `discrete_ce_loss(logits, target_bins, mask, presence_logits, presence_target, presence_weight=1.0) -> Tensor` — `logits` `[B,9,K,F]`.

- [ ] **Step 1: Write the failing test**

```python
"""Loss contracts for the face-set diffusion branch.

The properties that matter: padding never contributes a coordinate gradient,
presence contributes everywhere, and the Hungarian loss is invariant to a
permutation of the prediction slots while the MSE loss is not. That last pair
is the whole reason both exist.
"""
import numpy as np
import pytest
import torch

from src.models.mesh_set_losses import (
    discrete_ce_loss,
    hungarian_loss,
    hungarian_match,
    masked_mse_loss,
)


def _batch(n_real=3, width=8, batch=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    target = torch.zeros(batch, 10, width)
    target[:, :9] = torch.rand(batch, 9, width, generator=g) - 0.5
    mask = torch.zeros(batch, width, dtype=torch.bool)
    mask[:, :n_real] = True
    target[:, 9] = torch.where(mask, 0.5, -0.5)
    target[:, :9] = target[:, :9] * mask[:, None, :]
    return target, mask


def test_masked_mse_ignores_padded_coordinates():
    target, mask = _batch()
    pred = target.clone()
    base = masked_mse_loss(pred, target, mask)
    pred[:, :9, mask[0].sum():] += 100.0        # garbage in padding only
    assert torch.isclose(masked_mse_loss(pred, target, mask), base)
    assert base.item() == pytest.approx(0.0, abs=1e-7)


def test_masked_mse_counts_padded_presence():
    target, mask = _batch()
    pred = target.clone()
    pred[:, 9, mask[0].sum():] = 0.5            # claims padding is a real face
    assert masked_mse_loss(pred, target, mask).item() > 0.0


def test_presence_weight_scales_only_presence():
    target, mask = _batch()
    pred = target.clone()
    pred[:, 9] += 0.1
    a = masked_mse_loss(pred, target, mask, presence_weight=1.0)
    b = masked_mse_loss(pred, target, mask, presence_weight=2.0)
    assert b.item() == pytest.approx(2.0 * a.item(), rel=1e-5)


def test_mse_is_not_permutation_invariant():
    target, mask = _batch(n_real=4, width=8, batch=1)
    perm = torch.tensor([3, 2, 1, 0, 4, 5, 6, 7])
    shuffled = target[:, :, perm]
    assert masked_mse_loss(shuffled, target, mask).item() > 1e-4


def test_hungarian_is_permutation_invariant():
    target, mask = _batch(n_real=4, width=8, batch=1)
    perm = torch.tensor([3, 2, 1, 0, 4, 5, 6, 7])
    shuffled = target[:, :, perm]
    loss = hungarian_loss(shuffled, target, mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_hungarian_match_recovers_the_permutation():
    target, mask = _batch(n_real=4, width=8, batch=1)
    perm = torch.tensor([3, 2, 1, 0, 4, 5, 6, 7])
    assignment = hungarian_match(target[:, :, perm], target, mask)
    # Prediction slot i holds target perm[i], so it must be assigned perm[i].
    assert assignment[0, :4].tolist() == perm[:4].tolist()


def test_hungarian_equals_mse_when_already_aligned():
    target, mask = _batch(n_real=4, width=8, batch=2, seed=3)
    pred = target + 0.01
    assert hungarian_loss(pred, target, mask).item() == pytest.approx(
        masked_mse_loss(pred, target, mask).item(), rel=1e-4)


def test_hungarian_handles_an_all_padding_row():
    target, mask = _batch(n_real=4, width=8, batch=2)
    mask[1] = False
    target[1, 9] = -0.5
    loss = hungarian_loss(target.clone(), target, mask)
    assert torch.isfinite(loss)


def test_discrete_ce_ignores_padded_bins():
    b, k, w = 1, 128, 8
    bins = torch.randint(0, k, (b, 9, w))
    mask = torch.zeros(b, w, dtype=torch.bool)
    mask[:, :3] = True
    logits = torch.zeros(b, 9, k, w)
    logits.scatter_(2, bins.unsqueeze(2), 20.0)        # confident and correct
    presence_logits = torch.where(mask, 10.0, -10.0)[:, None].squeeze(1)
    loss_a = discrete_ce_loss(logits, bins, mask, presence_logits, mask.float())
    logits[:, :, :, 3:] = 0.0                          # wreck the padded slots
    loss_b = discrete_ce_loss(logits, bins, mask, presence_logits, mask.float())
    assert loss_a.item() == pytest.approx(loss_b.item(), rel=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mesh_set_losses.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'src.models.mesh_set_losses'`

- [ ] **Step 3: Write the implementation**

```python
"""Losses over padded face sets.

Two regimes, and the choice between them is forced by the model, not by taste
(see the plan's D4). With a canonical face order the model can be told which
slot to fill and a slot-to-slot MSE is correct. Without one -- which is the
only honest setting for a permutation-equivariant model -- the target is a
*set*, and the loss has to find the correspondence first. That is the DETR
construction (Carion et al., arXiv:2005.12872 section 3.1), with the presence
channel playing the role of their no-object class.

Convention throughout: tensors are [B, C, F] channels-first with C = 10
(9 coordinates + presence), and `mask` is [B, F] bool with True at real faces.
Coordinate terms are masked; the presence term never is, because predicting
"absent" at a padded slot is exactly what the model has to learn.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

N_COORD = 9
PRESENCE = 9


def _masked_mean(per_element, mask):
    """Mean of ``[B, C, F]`` over real faces only. ``clamp`` guards empty rows."""
    m = mask[:, None, :].to(per_element.dtype)
    return (per_element * m).sum() / (m.sum() * per_element.shape[1]).clamp(min=1.0)


def masked_mse_loss(pred, target, mask, presence_weight=1.0):
    """Slot-to-slot squared error, coordinates masked, presence not.

    Args:
        pred, target: ``[B, 10, F]``.
        mask: ``[B, F]`` bool, True at real faces.
        presence_weight: multiplier on the presence channel's contribution.
            Face count moves the geometric metrics far more than a fraction of
            a bin on one vertex does, so this is a real knob and not cosmetic.

    Returns:
        Tensor: scalar.
    """
    coord = _masked_mean((pred[:, :N_COORD] - target[:, :N_COORD]) ** 2, mask)
    presence = ((pred[:, PRESENCE] - target[:, PRESENCE]) ** 2).mean()
    return coord + presence_weight * presence


@torch.no_grad()
def hungarian_match(pred, target, mask, match_presence_weight=1.0):
    """Optimal assignment of prediction slots to target slots, per batch item.

    Cost is mean absolute error over the nine coordinates plus a weighted
    presence term. L1 rather than L2 inside the matcher on DETR's grounds: the
    squared cost lets one far-off coordinate dominate an otherwise good match,
    and the matcher's job is correspondence, not calibration.

    Padded *target* slots are still matched -- something has to absorb the
    surplus prediction slots, and the surplus must learn to predict absence.

    Args:
        pred, target: ``[B, 10, F]``.
        mask: ``[B, F]`` bool. Unused for the assignment itself; kept in the
            signature so callers cannot accidentally pass an unmasked pair to
            the loss and a masked one to the matcher.
        match_presence_weight: presence weight inside the matching cost only.

    Returns:
        LongTensor: ``[B, F]`` where entry ``i`` is the target index assigned to
        prediction slot ``i``.
    """
    b, _, f = pred.shape
    # [B, F_pred, F_tgt]: broadcast prediction slots against target slots.
    p = pred.permute(0, 2, 1)          # [B, F, 10]
    t = target.permute(0, 2, 1)
    coord_cost = (p[:, :, None, :N_COORD] - t[:, None, :, :N_COORD]).abs().mean(-1)
    pres_cost = (p[:, :, None, PRESENCE] - t[:, None, :, PRESENCE]).abs()
    cost = (coord_cost + match_presence_weight * pres_cost).cpu().numpy()

    out = torch.empty((b, f), dtype=torch.long)
    for i in range(b):
        rows, cols = linear_sum_assignment(cost[i])
        out[i, torch.from_numpy(rows)] = torch.from_numpy(cols)
    return out.to(pred.device)


def hungarian_loss(pred, target, mask, presence_weight=1.0,
                   match_presence_weight=1.0):
    """Set loss: match first, then the same masked MSE on the matched pairs.

    The matcher runs under `no_grad` -- the assignment is a discrete decision
    and is treated as a constant, exactly as DETR does. Gradients flow only
    through the reordered squared error.

    Args:
        pred, target: ``[B, 10, F]``.
        mask: ``[B, F]`` bool, True at real *target* faces.
        presence_weight: as `masked_mse_loss`.
        match_presence_weight: presence weight inside the matching cost.

    Returns:
        Tensor: scalar.
    """
    assignment = hungarian_match(pred, target, mask, match_presence_weight)
    idx = assignment[:, None, :].expand(-1, target.shape[1], -1)
    # Reorder the *target* onto the prediction slots, and the mask with it, so
    # the masked mean below counts the same real faces it always did.
    target_m = torch.gather(target, 2, idx)
    mask_m = torch.gather(mask, 1, assignment)
    return masked_mse_loss(pred, target_m, mask_m, presence_weight)


def discrete_ce_loss(logits, target_bins, mask, presence_logits,
                     presence_target, presence_weight=1.0):
    """Cross-entropy over coordinate bins, plus binary presence.

    The D3PM arm's x0-parameterisation: nine independent categorical heads over
    `num_bins` classes each, one per coordinate channel.

    Args:
        logits: ``[B, 9, K, F]`` unnormalised bin scores.
        target_bins: ``[B, 9, F]`` int64 in ``[0, K)``.
        mask: ``[B, F]`` bool, True at real faces.
        presence_logits: ``[B, F]`` unnormalised.
        presence_target: ``[B, F]`` float in {0, 1}.
        presence_weight: multiplier on the presence term.

    Returns:
        Tensor: scalar.
    """
    b, c, k, f = logits.shape
    ce = F.cross_entropy(
        logits.permute(0, 2, 1, 3).reshape(b, k, c * f),
        target_bins.reshape(b, c * f),
        reduction="none",
    ).reshape(b, c, f)
    coord = _masked_mean(ce, mask)
    presence = F.binary_cross_entropy_with_logits(
        presence_logits, presence_target.to(presence_logits.dtype))
    return coord + presence_weight * presence
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_mesh_set_losses.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/models/mesh_set_losses.py tests/test_mesh_set_losses.py
git commit -m "feat(mesh-diff): masked MSE, Hungarian set loss, discrete CE"
```

---

### Task 4: Shared denoiser modules and the conditional U-Net

**Files:**
- Create: `src/models/mesh_set_modules.py`, `src/models/mesh_set_unet.py`
- Test: `tests/test_mesh_set_denoisers.py` (U-Net cases only; the transformer cases are added in Task 9)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `timestep_embedding(t, dim) -> Tensor [B, dim]` — `t` float in `[0,1]`.
  - `SinusoidalFacePositions(d_model)`; `forward(length, device) -> [length, d_model]`
  - `FaceEncoder(in_ch, d_model)`; `forward(cond [B,10,Fc]) -> [B, d_model, Fc]`
  - `SelfAttention1d(channels, n_head)`; `forward(x [B,C,F], key_padding_mask [B,F] bool True=real) -> [B,C,F]`
  - `CrossAttention1d(channels, cond_dim, n_head)`; `forward(x, cond, cond_mask) -> [B,C,F]`
  - `DoubleConv(in_ch, out_ch, residual=False)`, `Down(in_ch, out_ch, emb_dim)`, `Up(in_ch, out_ch, emb_dim)`
  - `ConditionalMeshUNet(in_ch=10, out_ch=10, base=64, cond_dim=256, time_dim=128, n_head=8, dropout=0.1, out_bins=None)`; `forward(x [B,10,F], t [B] float in [0,1], cond [B,10,Fc] or None, mask [B,F], cond_mask [B,Fc]) -> [B, out_ch, F]` or, when `out_bins` is set, `([B,9,K,F], [B,F])`.

- [ ] **Step 1: Write the failing test**

```python
"""Denoiser shape and masking contracts.

No learning here -- these are the invariants that make a wrong wiring fail in
seconds instead of after an epoch: shapes survive a round trip, padded faces
cannot influence real ones, and dropping the condition is a legal call rather
than a crash.
"""
import pytest
import torch

from src.models.mesh_set_modules import timestep_embedding
from src.models.mesh_set_unet import ConditionalMeshUNet


def _inputs(b=2, f=16, fc=8):
    x = torch.randn(b, 10, f)
    t = torch.rand(b)
    cond = torch.randn(b, 10, fc)
    mask = torch.zeros(b, f, dtype=torch.bool)
    mask[:, : f - 3] = True
    cond_mask = torch.ones(b, fc, dtype=torch.bool)
    return x, t, cond, mask, cond_mask


def test_timestep_embedding_shape_and_range():
    emb = timestep_embedding(torch.rand(5), 128)
    assert emb.shape == (5, 128)
    assert emb.abs().max() <= 1.0


def test_unet_preserves_shape():
    net = ConditionalMeshUNet(base=16, cond_dim=32, time_dim=32, n_head=2)
    x, t, cond, mask, cond_mask = _inputs()
    assert net(x, t, cond, mask, cond_mask).shape == x.shape


def test_unet_requires_length_divisible_by_eight():
    net = ConditionalMeshUNet(base=16, cond_dim=32, time_dim=32, n_head=2)
    x, t, cond, mask, cond_mask = _inputs(f=13)
    with pytest.raises(ValueError, match="divisible by 8"):
        net(x, t, cond, mask, cond_mask)


def test_unet_accepts_a_dropped_condition():
    net = ConditionalMeshUNet(base=16, cond_dim=32, time_dim=32, n_head=2)
    x, t, _, mask, _ = _inputs()
    assert net(x, t, None, mask, None).shape == x.shape


def test_padded_faces_do_not_change_real_outputs():
    torch.manual_seed(0)
    net = ConditionalMeshUNet(base=16, cond_dim=32, time_dim=32, n_head=2).eval()
    x, t, cond, mask, cond_mask = _inputs()
    with torch.no_grad():
        a = net(x, t, cond, mask, cond_mask)
        x2 = x.clone()
        x2[:, :, ~mask[0]] = 99.0            # scribble on the padding
        b = net(x2, t, cond, mask, cond_mask)
    # Convolution has a receptive field, so only assert on faces far from the
    # padded tail; attention is the part that must be exactly masked.
    assert torch.allclose(a[:, :, :4], b[:, :, :4], atol=1e-4)


def test_unet_discrete_head_shapes():
    net = ConditionalMeshUNet(base=16, cond_dim=32, time_dim=32, n_head=2,
                              out_bins=128)
    x, t, cond, mask, cond_mask = _inputs()
    logits, presence = net(x, t, cond, mask, cond_mask)
    assert logits.shape == (2, 9, 128, 16)
    assert presence.shape == (2, 16)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mesh_set_denoisers.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'src.models.mesh_set_modules'`

- [ ] **Step 3: Write `src/models/mesh_set_modules.py`**

```python
"""Blocks shared by both face-set denoisers.

Ported from the conditional 1-D U-Net in maxim-mat/trace-denoise-refactor
(`src/modules/`), with two deliberate changes:

  * conditioning is cross-attention, never the elementwise sum that repo uses.
    That sum requires the condition and the target to have equal, aligned
    length; LOD1 is ~12 triangles against LOD2's ~100, so there is no alignment
    to exploit. Cross-attention is also MeshWeaver's fix (ii) for exactly the
    architecture this branch is competing with.
  * the padding-mask convention is True = real (matching this repo's face-set
    collate), and it is inverted once at each `nn.MultiheadAttention` call
    rather than at every call site.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    """Fourier features for a diffusion timestep (Vaswani, arXiv:1706.03762 3.5).

    Args:
        t: ``[B]`` float in ``[0, 1]``. Normalised, not an integer index --
            that is what lets one denoiser serve DDPM, D3PM and flow matching,
            whose native time variables are otherwise incomparable.
        dim: embedding width. Must be even.

    Returns:
        Tensor: ``[B, dim]``.
    """
    if dim % 2:
        raise ValueError(f"timestep_embedding dim must be even, got {dim}.")
    half = dim // 2
    # Scaled to the usual 0..1000 band so the frequencies land where the
    # standard schedules put them, whatever the process calls its own time.
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
    args = (t.float() * 1000.0)[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class SinusoidalFacePositions(nn.Module):
    """Fixed positional encoding over the face axis. No parameters, no ceiling.

    Only meaningful when the face order carries information -- i.e. under
    `order: morton`. `validate_combination` is what enforces that.
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, length, device):
        """``[length, d_model]``."""
        pos = torch.arange(length, device=device).float()
        return timestep_embedding(pos / max(length - 1, 1), self.d_model)


def _mha_mask(key_padding_mask):
    """True = real (ours) to True = ignore (torch's). ``None`` passes through."""
    return None if key_padding_mask is None else ~key_padding_mask


class SelfAttention1d(nn.Module):
    """Pre-norm multi-head self-attention over the face axis, channels-first."""

    def __init__(self, channels, n_head=8):
        super().__init__()
        n_head = min(n_head, max(1, channels // 8))
        self.mha = nn.MultiheadAttention(channels, n_head, batch_first=True)
        self.ln = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels), nn.Linear(channels, channels),
            nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x, key_padding_mask=None):
        """``[B, C, F] -> [B, C, F]``. ``key_padding_mask`` is [B, F], True = real."""
        h = x.permute(0, 2, 1)
        hn = self.ln(h)
        attn, _ = self.mha(hn, hn, hn, key_padding_mask=_mha_mask(key_padding_mask),
                           need_weights=False)
        h = h + attn
        h = h + self.ff(h)
        return h.permute(0, 2, 1)


class CrossAttention1d(nn.Module):
    """Face-axis cross-attention onto an encoded condition of any length.

    This is the conditioning mechanism, and it is the reason the condition need
    not be length-matched to the target. It is also MeshWeaver's fix (ii):
    local geometric context at every layer instead of one global prefix.
    """

    def __init__(self, channels, cond_dim, n_head=8):
        super().__init__()
        n_head = min(n_head, max(1, channels // 8))
        self.mha = nn.MultiheadAttention(channels, n_head, batch_first=True,
                                         kdim=cond_dim, vdim=cond_dim)
        self.ln_q = nn.LayerNorm(channels)
        self.ln_kv = nn.LayerNorm(cond_dim)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels), nn.Linear(channels, channels),
            nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x, cond, cond_mask=None):
        """``x [B,C,F]``, ``cond [B,D,Fc]`` -> ``[B,C,F]``. ``cond=None`` is a no-op."""
        if cond is None:
            return x
        h = x.permute(0, 2, 1)
        kv = self.ln_kv(cond.permute(0, 2, 1))
        attn, _ = self.mha(self.ln_q(h), kv, kv,
                           key_padding_mask=_mha_mask(cond_mask),
                           need_weights=False)
        h = h + attn
        h = h + self.ff(h)
        return h.permute(0, 2, 1)


class FaceEncoder(nn.Module):
    """The LOD1 condition as a sequence of per-face embeddings.

    A two-layer MLP applied per face, not a convolution: the condition is read
    only through cross-attention, which is permutation invariant, so there is
    nothing for a convolution's locality to buy here.
    """

    def __init__(self, in_ch=10, d_model=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, d_model, 1), nn.GELU(),
            nn.Conv1d(d_model, d_model, 1))

    def forward(self, cond):
        """``[B, 10, Fc] -> [B, d_model, Fc]``. ``None`` passes through."""
        return None if cond is None else self.net(cond)


class DoubleConv(nn.Module):
    """Two GroupNorm-GELU convolutions, optionally residual."""

    def __init__(self, in_ch, out_ch, mid_ch=None, residual=False):
        super().__init__()
        self.residual = residual
        mid_ch = mid_ch or out_ch
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, mid_ch), mid_ch), nn.GELU(),
            nn.Conv1d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch))

    def forward(self, x):
        return F.gelu(x + self.net(x)) if self.residual else self.net(x)


class Down(nn.Module):
    """Stride-2 downsample along the face axis, with an additive time embedding."""

    def __init__(self, in_ch, out_ch, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_ch, in_ch, residual=True),
            DoubleConv(in_ch, out_ch))
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch))

    def forward(self, x, t):
        x = self.net(x)
        return x + self.emb(t)[:, :, None]


class Up(nn.Module):
    """Nearest-neighbour upsample, skip concat, additive time embedding."""

    def __init__(self, in_ch, out_ch, emb_dim=128):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.net = nn.Sequential(
            DoubleConv(in_ch, in_ch, residual=True),
            DoubleConv(in_ch, out_ch, mid_ch=in_ch // 2))
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch))

    def forward(self, x, skip, t):
        x = torch.cat([skip, self.up(x)], dim=1)
        x = self.net(x)
        return x + self.emb(t)[:, :, None]


def downsample_mask(mask):
    """``[B, F] -> [B, F//2]``. A coarse face is real if either child was.

    ``any`` rather than ``all``: a half-padded pair still carries real geometry,
    and masking it out would delete that geometry from every coarser level.
    """
    return mask.reshape(mask.shape[0], -1, 2).any(dim=-1)
```

- [ ] **Step 4: Write `src/models/mesh_set_unet.py`**

```python
"""Attention-augmented 1-D U-Net over a spatially sorted face set.

The face axis is treated as a sequence, which is only legitimate because
`MeshSetDataset` sorts faces by the Morton code of their centroid -- adjacent
slots are then spatially adjacent, and a stride-2 convolution pools a
neighbourhood rather than an arbitrary subset. `validate_combination` refuses
this denoiser under `order: none` for exactly that reason.
"""
import torch
import torch.nn as nn

from src.models.mesh_set_modules import (
    CrossAttention1d,
    DoubleConv,
    Down,
    FaceEncoder,
    SelfAttention1d,
    Up,
    downsample_mask,
    timestep_embedding,
)


class ConditionalMeshUNet(nn.Module):
    """Denoiser: ``(x_t, t, LOD1) -> prediction`` over a padded face set.

    Args:
        in_ch, out_ch: channel counts. 10 = 9 coordinates + presence.
        base: channel width at full resolution; doubles twice going down.
        cond_dim: width of the encoded LOD1 condition.
        time_dim: timestep embedding width.
        n_head: attention heads (clamped down on narrow levels).
        dropout: applied to the bottleneck only.
        out_bins: when set, the head emits ``[B, 9, out_bins, F]`` bin logits
            plus ``[B, F]`` presence logits instead of ``[B, 10, F]`` -- the
            D3PM arm. ``None`` for every continuous process.
    """

    def __init__(self, in_ch=10, out_ch=10, base=64, cond_dim=256,
                 time_dim=128, n_head=8, dropout=0.1, out_bins=None):
        super().__init__()
        self.time_dim = time_dim
        self.out_bins = out_bins
        self.cond_encoder = FaceEncoder(in_ch, cond_dim)

        c1, c2, c3 = base, base * 2, base * 4
        self.inc = DoubleConv(in_ch, c1)
        self.down1, self.sa1 = Down(c1, c2, time_dim), SelfAttention1d(c2, n_head)
        self.ca1 = CrossAttention1d(c2, cond_dim, n_head)
        self.down2, self.sa2 = Down(c2, c3, time_dim), SelfAttention1d(c3, n_head)
        self.ca2 = CrossAttention1d(c3, cond_dim, n_head)
        self.down3, self.sa3 = Down(c3, c3, time_dim), SelfAttention1d(c3, n_head)
        self.ca3 = CrossAttention1d(c3, cond_dim, n_head)

        self.bot = nn.Sequential(
            DoubleConv(c3, c3 * 2), nn.Dropout(dropout), DoubleConv(c3 * 2, c3))

        self.up1, self.sa4 = Up(c3 + c3, c2, time_dim), SelfAttention1d(c2, n_head)
        self.ca4 = CrossAttention1d(c2, cond_dim, n_head)
        self.up2, self.sa5 = Up(c2 + c2, c1, time_dim), SelfAttention1d(c1, n_head)
        self.ca5 = CrossAttention1d(c1, cond_dim, n_head)
        self.up3, self.sa6 = Up(c1 + c1, c1, time_dim), SelfAttention1d(c1, n_head)

        head_ch = out_ch if out_bins is None else 9 * out_bins + 1
        self.outc = nn.Conv1d(c1, head_ch, 1)
        # Zero-init the head so the model starts as the identity on x_t rather
        # than injecting noise of its own on step 0 (Nichol & Dhariwal's
        # zero-module trick, arXiv:2102.09672).
        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)

    def forward(self, x, t, cond=None, mask=None, cond_mask=None):
        """``x [B,10,F]``, ``t [B]`` in ``[0,1]``, ``cond [B,10,Fc]`` or None."""
        f = x.shape[-1]
        if f % 8:
            raise ValueError(
                f"Face-set length {f} is not divisible by 8; the U-Net "
                "downsamples three times. mesh_set_collate_fn pads to a "
                "multiple of 8 -- a caller bypassed it.")
        if mask is None:
            mask = torch.ones(x.shape[0], f, dtype=torch.bool, device=x.device)

        temb = timestep_embedding(t, self.time_dim)
        c = self.cond_encoder(cond)
        m1 = mask
        m2 = downsample_mask(m1)
        m3 = downsample_mask(m2)
        m4 = downsample_mask(m3)

        x1 = self.inc(x)
        x2 = self.ca1(self.sa1(self.down1(x1, temb), m2), c, cond_mask)
        x3 = self.ca2(self.sa2(self.down2(x2, temb), m3), c, cond_mask)
        x4 = self.ca3(self.sa3(self.down3(x3, temb), m4), c, cond_mask)

        x4 = self.bot(x4)

        h = self.ca4(self.sa4(self.up1(x4, x3, temb), m3), c, cond_mask)
        h = self.ca5(self.sa5(self.up2(h, x2, temb), m2), c, cond_mask)
        h = self.sa6(self.up3(h, x1, temb), m1)
        out = self.outc(h)

        if self.out_bins is None:
            return out
        logits = out[:, : 9 * self.out_bins].reshape(
            x.shape[0], 9, self.out_bins, f)
        return logits, out[:, -1]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_mesh_set_denoisers.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add src/models/mesh_set_modules.py src/models/mesh_set_unet.py tests/test_mesh_set_denoisers.py
git commit -m "feat(mesh-diff): shared denoiser blocks and the conditional mesh U-Net"
```

---

### Task 5: The process interface and Gaussian diffusion (DDPM + DDIM)

**Files:**
- Create: `src/models/mesh_processes.py`
- Test: `tests/test_mesh_processes.py` (Gaussian cases only; flow and D3PM cases are added in Tasks 11 and 12)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `BaseProcess(nn.Module)` with abstract `sample_t(n, device) -> Tensor [B] float in [0,1]`, `corrupt(x0, t) -> (x_t, aux)`, `target_for(x0, x_t, t, aux) -> Tensor`, `step(x_t, t, t_prev, model_out) -> Tensor`, `prior(shape, device) -> Tensor`, and concrete `sample(denoiser_fn, shape, device, n_steps, callback=None) -> Tensor`.
  - `GaussianProcess(noise_steps=1000, beta_start=1e-4, beta_end=0.02, target="noise")`
  - `create_process(cfg) -> BaseProcess` — factory keyed on `cfg.mesh_diffusion.process`.

- [ ] **Step 1: Write the failing test**

```python
"""Diffusion process contracts.

The properties worth a test are the ones a shape check will not catch: the
noising marginal has the variance the schedule promises, x0- and
epsilon-parameterisations are algebraically the same object, and a reverse
trajectory driven by a perfect oracle lands back on the data.
"""
import pytest
import torch

from src.models.mesh_processes import GaussianProcess


def test_corrupt_has_the_scheduled_marginal():
    p = GaussianProcess(noise_steps=1000)
    x0 = torch.zeros(4096, 10, 8)
    t = torch.full((4096,), 0.5)
    x_t, eps = p.corrupt(x0, t)
    idx = p.to_index(t)[0]
    expected = (1 - p.alpha_hat[idx]).sqrt()
    assert x_t.std().item() == pytest.approx(expected.item(), rel=0.05)


def test_corrupt_at_t_zero_is_nearly_the_data():
    p = GaussianProcess(noise_steps=1000)
    x0 = torch.randn(64, 10, 8)
    x_t, _ = p.corrupt(x0, torch.zeros(64))
    assert (x_t - x0).abs().mean().item() < 0.05


def test_target_for_noise_and_original_agree():
    p_eps = GaussianProcess(target="noise")
    p_x0 = GaussianProcess(target="original")
    x0 = torch.randn(16, 10, 8)
    t = torch.rand(16)
    torch.manual_seed(1)
    x_t, aux = p_eps.corrupt(x0, t)
    eps = p_eps.target_for(x0, x_t, t, aux)
    x0_hat = p_x0.target_for(x0, x_t, t, aux)
    idx = p_eps.to_index(t)
    a = p_eps.alpha_hat[idx][:, None, None]
    # x_t = sqrt(a) x0 + sqrt(1-a) eps must hold for both readings.
    assert torch.allclose(a.sqrt() * x0_hat + (1 - a).sqrt() * eps, x_t, atol=1e-5)


def test_oracle_reverse_recovers_the_data():
    torch.manual_seed(0)
    p = GaussianProcess(noise_steps=1000, target="original")
    x0 = torch.randn(8, 10, 8) * 0.3

    def oracle(x_t, t):
        return x0                      # a perfect x0 predictor

    out = p.sample(oracle, x0.shape, x0.device, n_steps=50)
    assert (out - x0).abs().mean().item() < 0.1


def test_sample_is_seed_reproducible():
    p = GaussianProcess(noise_steps=100, target="noise")

    def oracle(x_t, t):
        return torch.zeros_like(x_t)

    torch.manual_seed(7)
    a = p.sample(oracle, (2, 10, 8), torch.device("cpu"), n_steps=10)
    torch.manual_seed(7)
    b = p.sample(oracle, (2, 10, 8), torch.device("cpu"), n_steps=10)
    assert torch.allclose(a, b)


def test_ddim_step_count_is_honoured():
    p = GaussianProcess(noise_steps=1000)
    seen = []

    def oracle(x_t, t):
        seen.append(float(t[0]))
        return torch.zeros_like(x_t)

    p.sample(oracle, (1, 10, 8), torch.device("cpu"), n_steps=25)
    assert len(seen) == 25
    assert seen == sorted(seen, reverse=True)      # time runs backwards


def test_sample_callback_sees_every_step():
    p = GaussianProcess(noise_steps=100)
    steps = []
    p.sample(lambda x, t: torch.zeros_like(x), (1, 10, 8),
             torch.device("cpu"), n_steps=10,
             callback=lambda t, x: steps.append(t))
    assert len(steps) == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mesh_processes.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'src.models.mesh_processes'`

- [ ] **Step 3: Write the implementation**

```python
"""Noising processes for the face-set branch.

One interface, three implementations, so the denoiser and the training loop
are written once and the process is a config switch. The split is the one that
worked in maxim-mat/trace-denoise-refactor (`src/diffusion/base_diffusion.py`):
the process owns the schedule and the reverse loop, the denoiser owns the
network, and neither knows what the other is.

Time is always a float in [0, 1] at the interface (plan D7). DDPM and D3PM
carry an integer index internally and convert; flow matching is natively
continuous and does not.
"""
import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseProcess(nn.Module, ABC):
    """Forward corruption and reverse sampling for one diffusion formulation.

    Subclasses implement four things: how to draw a training time, how to
    corrupt, what the regression target is, and how to take one reverse step.
    `sample` is shared and never overridden.

    `nn.Module` rather than a plain class so Lightning moves the schedule
    buffers to the accelerator with the model.
    """

    @abstractmethod
    def sample_t(self, n, device):
        """``[n]`` training times in ``[0, 1]``."""

    @abstractmethod
    def corrupt(self, x0, t):
        """``(x_t, aux)``. ``aux`` is whatever `target_for` needs -- the drawn
        noise, typically -- and is never inspected by the caller."""

    @abstractmethod
    def target_for(self, x0, x_t, t, aux):
        """What the denoiser is asked to regress at this ``t``."""

    @abstractmethod
    def step(self, x_t, t, t_prev, model_out):
        """One reverse step, ``x_t -> x_{t_prev}``. ``t`` are floats in [0,1]."""

    @abstractmethod
    def prior(self, shape, device):
        """A draw from the terminal distribution the reverse loop starts at."""

    def timesteps(self, n_steps):
        """``[(t, t_prev), ...]`` descending, ``n_steps`` long, ending at 0."""
        edges = torch.linspace(1.0, 0.0, n_steps + 1)
        return list(zip(edges[:-1].tolist(), edges[1:].tolist()))

    def sample(self, denoiser_fn, shape, device, n_steps=50, callback=None):
        """Run the reverse trajectory and return the final state.

        Args:
            denoiser_fn: ``(x_t [B,C,F], t [B] float) -> prediction``. The
                caller closes over the condition, the masks and any guidance,
                so this class never learns what a condition is.
            shape: ``(B, C, F)``.
            device: where to allocate.
            n_steps: reverse steps. Fewer than `noise_steps` is a strided
                (DDIM-style) trajectory, which is the only affordable setting
                for a per-epoch eval.
            callback: optional ``(t, x_t)`` per step, for scaffold projection
                and trajectory logging.

        Returns:
            Tensor: ``[B, C, F]``.
        """
        x = self.prior(shape, device)
        for t, t_prev in self.timesteps(n_steps):
            t_batch = torch.full((shape[0],), t, device=device)
            with torch.no_grad():
                out = denoiser_fn(x, t_batch)
            x = self.step(x, t, t_prev, out)
            if callback is not None:
                x = callback(t_prev, x)
        return x


class GaussianProcess(BaseProcess):
    """DDPM forward process with a DDIM (deterministic, strided) reverse.

    Ho et al. arXiv:2006.11239 for the schedule, Song et al. arXiv:2010.02502
    for the reverse. DDIM rather than ancestral sampling because the eval
    budget is the binding constraint: a per-epoch callback over 16 buildings at
    1000 ancestral steps is 16,000 forward passes, and eta=0 lets 50 steps
    stand in with no retraining.

    Args:
        noise_steps: schedule resolution. Time in ``[0,1]`` is discretised onto
            this many rungs.
        beta_start, beta_end: linear schedule endpoints.
        target: "noise" (epsilon-prediction, the default and what nearly every
            reported result uses) or "original" (x0-prediction, which is better
            conditioned at low t and is what D3PM and most set models use).
    """

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02,
                 target="noise"):
        super().__init__()
        if target not in ("noise", "original"):
            raise ValueError(
                f"GaussianProcess target must be 'noise' or 'original', got {target!r}.")
        self.noise_steps = noise_steps
        self.target = target
        beta = torch.linspace(beta_start, beta_end, noise_steps)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha_hat", torch.cumprod(1.0 - beta, dim=0))

    def to_index(self, t):
        """Float time in ``[0,1]`` to an integer schedule rung."""
        idx = (t.clamp(0.0, 1.0) * (self.noise_steps - 1)).round().long()
        return idx.clamp(0, self.noise_steps - 1)

    def _abar(self, t, ndim=3):
        idx = self.to_index(t)
        a = self.alpha_hat[idx]
        return a.reshape(-1, *([1] * (ndim - 1)))

    def sample_t(self, n, device):
        return torch.rand(n, device=device)

    def corrupt(self, x0, t):
        a = self._abar(t, x0.dim())
        eps = torch.randn_like(x0)
        return a.sqrt() * x0 + (1 - a).sqrt() * eps, eps

    def target_for(self, x0, x_t, t, aux):
        return aux if self.target == "noise" else x0

    def to_x0(self, x_t, t, model_out):
        """Whatever the denoiser predicted, read as an ``x0`` estimate."""
        if self.target == "original":
            return model_out
        a = self._abar(torch.as_tensor(t, device=x_t.device).expand(x_t.shape[0]),
                       x_t.dim())
        return (x_t - (1 - a).sqrt() * model_out) / a.sqrt().clamp(min=1e-8)

    def step(self, x_t, t, t_prev, model_out):
        """Deterministic DDIM step (eta = 0)."""
        device = x_t.device
        n = x_t.shape[0]
        a_t = self._abar(torch.full((n,), t, device=device), x_t.dim())
        a_p = self._abar(torch.full((n,), t_prev, device=device), x_t.dim())
        x0 = self.to_x0(x_t, t, model_out)
        eps = ((x_t - a_t.sqrt() * x0) / (1 - a_t).sqrt().clamp(min=1e-8))
        return a_p.sqrt() * x0 + (1 - a_p).sqrt() * eps

    def prior(self, shape, device):
        return torch.randn(shape, device=device)


def create_process(cfg):
    """Build the process named by ``cfg.mesh_diffusion.process``.

    A plain switch, matching `src.train.train`'s handling of `config_set`:
    there are three of these and they all live in this file.
    """
    d = cfg.mesh_diffusion
    if d.process == "ddpm":
        return GaussianProcess(d.noise_steps, d.beta_start, d.beta_end, d.target)
    if d.process == "flow":
        from src.models.mesh_processes import FlowMatchingProcess
        return FlowMatchingProcess()
    if d.process == "d3pm":
        from src.models.mesh_processes import DiscreteProcess
        return DiscreteProcess(d.noise_steps, cfg.mesh_data.num_bins,
                               transition=d.transition,
                               sigma_max=d.transition_sigma)
    raise ValueError(f"Unknown process: {d.process!r}.")
```

Note: `create_process` refers to `FlowMatchingProcess` and `DiscreteProcess`, which land in Tasks 11 and 12. Until then those two branches raise `ImportError`, which is correct behaviour — `validate_combination` has already accepted the config, and a missing implementation should be loud. Leave the imports inside the branches so Phase A does not need them.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_mesh_processes.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/models/mesh_processes.py tests/test_mesh_processes.py
git commit -m "feat(mesh-diff): process interface and Gaussian diffusion with DDIM reverse"
```

---

### Task 6: Lightning module and training entry point

This is the task that makes the branch runnable. It wires dataset, denoiser, process and loss together and adds a smoke config.

**Files:**
- Create: `src/models/mesh_diffusion_module.py`, `src/train_mesh_diffusion.py`, `configs/mesh-diff-smoke.yaml`
- Test: `tests/test_mesh_diffusion_smoke.py`

**Interfaces:**
- Consumes: `MeshSetDataset`, `mesh_set_collate_fn` (Task 1); `validate_combination`, `MeshDiffusionConfig` (Task 2); `masked_mse_loss`, `hungarian_loss`, `discrete_ce_loss` (Task 3); `ConditionalMeshUNet` (Task 4); `create_process`, `GaussianProcess` (Task 5).
- Produces:
  - `create_denoiser(cfg, in_ch=10, out_bins=None) -> nn.Module`
  - `MeshDiffusionModule(cfg)`; attributes `.process`, `.denoiser`, `.state`, `.categorical`; methods `training_step`, `validation_step`, `configure_optimizers`, `_state_shape(batch) -> tuple`, and `generate(batch, n_steps=None, scaffold=None) -> Tensor [B,10,F]`.
  - `train_mesh_diffusion(cfg)` in `src/train_mesh_diffusion.py`.
  - `MeshSetDataModule` (defined inside `src/train_mesh_diffusion.py`, since nothing else uses it): a `LightningDataModule` splitting `MeshSetDataset` by `training.train_val_test_split` with `cfg.seed`.

- [ ] **Step 1: Write the failing test**

```python
"""End-to-end smoke: the Phase A arm must train a step and sample a mesh.

Synthetic on-disk data is out of scope here -- these tests build a
`MeshDiffusionModule` directly and feed it a hand-made batch, which is the
smallest thing that proves the four subsystems are wired to each other.
"""
import pytest
import torch
from omegaconf import OmegaConf

from src.models.mesh_diffusion_module import MeshDiffusionModule
from src.utils.config import Config


def _cfg(**kw):
    cfg = OmegaConf.structured(Config)
    cfg.config_set = "mesh_diffusion"
    cfg.mesh_data.num_bins = 128
    cfg.mesh_diffusion.d_model = 32
    cfg.mesh_diffusion.time_dim = 32
    cfg.mesh_diffusion.n_head = 2
    cfg.mesh_diffusion.num_layers = 2
    cfg.mesh_diffusion.noise_steps = 100
    cfg.mesh_diffusion.eval_steps = 5
    for k, v in kw.items():
        OmegaConf.update(cfg, f"mesh_diffusion.{k}", v)
    return cfg


def _batch(b=2, f=16, fc=8):
    x = torch.zeros(b, 10, f)
    x[:, :9] = torch.rand(b, 9, f) - 0.5
    mask = torch.zeros(b, f, dtype=torch.bool)
    mask[:, : f - 4] = True
    x[:, 9] = torch.where(mask, 0.5, -0.5)
    cond = torch.zeros(b, 10, fc)
    cond[:, :9] = torch.rand(b, 9, fc) - 0.5
    cond[:, 9] = 0.5
    return {"x": x, "x_mask": mask, "cond": cond,
            "cond_mask": torch.ones(b, fc, dtype=torch.bool),
            "center": torch.zeros(b, 3), "scale": torch.ones(b, 3),
            "ids": ["a", "b"][:b]}


def test_training_step_produces_a_finite_scalar_with_gradients():
    m = MeshDiffusionModule(_cfg())
    loss = m.training_step(_batch(), 0)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_generate_returns_the_batch_shape():
    m = MeshDiffusionModule(_cfg()).eval()
    out = m.generate(_batch(), n_steps=3)
    assert out.shape == (2, 10, 16)
    assert torch.isfinite(out).all()


def test_condition_dropout_still_produces_a_loss():
    m = MeshDiffusionModule(_cfg(cond_dropout=1.0))   # always dropped
    assert torch.isfinite(m.training_step(_batch(), 0))


def test_guidance_changes_the_sample():
    torch.manual_seed(0)
    m = MeshDiffusionModule(_cfg(cond_dropout=0.1, guidance=1.0)).eval()
    torch.manual_seed(0)
    a = m.generate(_batch(), n_steps=3)
    m.guidance = 3.0
    torch.manual_seed(0)
    b = m.generate(_batch(), n_steps=3)
    assert not torch.allclose(a, b)


def test_x0_target_arm_also_trains():
    m = MeshDiffusionModule(_cfg(target="original"))
    assert torch.isfinite(m.training_step(_batch(), 0))


def test_illegal_arm_is_rejected_at_construction():
    with pytest.raises(ValueError, match="hungarian"):
        MeshDiffusionModule(_cfg(order="none", pos_embed="none", loss="mse"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mesh_diffusion_smoke.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'src.models.mesh_diffusion_module'`

- [ ] **Step 3: Write `src/models/mesh_diffusion_module.py`**

```python
"""Lightning module for non-autoregressive whole-mesh generation.

Holds the four swappable pieces together and owns nothing else: the dataset
decides the representation, the denoiser the network, the process the
schedule, and this module only routes between them and logs.

Loss is measured on a single corruption step, which is cheap enough to run
every batch. Geometry is measured by `MeshSetEvalCallback`, which runs the full
reverse trajectory and is therefore gated to every N epochs -- the same split
`MeshEvalCallback` uses on the autoregressive branch, and for the same reason.
"""
import logging

import torch
import torch.nn as nn
import lightning as L
from omegaconf import OmegaConf

from src.models.mesh_processes import create_process
from src.models.mesh_set_losses import (
    discrete_ce_loss,
    hungarian_loss,
    masked_mse_loss,
)
from src.models.mesh_set_unet import ConditionalMeshUNet
from src.utils.config import validate_combination

logger = logging.getLogger(__name__)


def create_denoiser(cfg, in_ch=10, out_bins=None):
    """Build the denoiser named by ``cfg.mesh_diffusion.denoiser``.

    Args:
        in_ch: input channels. 10 for every state but ``onehot``, which is
            ``9 * num_bins + 1`` (Task 11). Parameterised from the start so
            adding that arm is a call-site change, not a signature change.
        out_bins: set to ``num_bins`` for a categorical readout, ``None`` for a
            regression one.
    """
    d = cfg.mesh_diffusion
    if d.denoiser == "unet":
        return ConditionalMeshUNet(
            in_ch=in_ch, base=d.d_model // 4, cond_dim=d.d_model,
            time_dim=d.time_dim, n_head=d.n_head, dropout=d.dropout,
            out_bins=out_bins)
    if d.denoiser == "transformer":
        from src.models.mesh_set_transformer import MeshSetTransformer
        return MeshSetTransformer(
            in_ch=in_ch, d_model=d.d_model, n_head=d.n_head,
            num_layers=d.num_layers, dropout=d.dropout,
            pos_embed=d.pos_embed, time_dim=d.time_dim, out_bins=out_bins)
    raise ValueError(f"Unknown denoiser: {d.denoiser!r}.")


class MeshDiffusionModule(L.LightningModule):
    """Conditional face-set diffusion: LOD1 in, LOD2 out, one trajectory.

    Args:
        cfg: the resolved root `Config`. Validated here rather than in
            `train_mesh_diffusion` so a module built from a checkpoint in a
            notebook gets the same gate a training run does.
    """

    def __init__(self, cfg):
        super().__init__()
        validate_combination(cfg)
        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True))
        d = cfg.mesh_diffusion
        self.cfg_d = d
        self.num_bins = cfg.mesh_data.num_bins
        self.lr = cfg.training.lr
        self.lr_scheduler = cfg.training.lr_scheduler
        self.cond_dropout = d.cond_dropout
        self.guidance = d.guidance
        self.eval_steps = d.eval_steps
        self.loss_name = d.loss
        self.presence_weight = d.presence_weight
        self.match_presence_weight = d.match_presence_weight

        # The readout decides the head, not the process: a Gaussian process
        # with loss: ce also needs bin logits (Task 11, plan D10).
        self.state = d.state
        self.x0_clamp = d.x0_clamp
        self.categorical = d.loss == "ce"
        in_ch = 9 * self.num_bins + 1 if d.state == "onehot" else 10
        self.denoiser = create_denoiser(
            cfg, in_ch=in_ch, out_bins=self.num_bins if self.categorical else None)
        self.process = create_process(cfg)

    # -- plumbing ---------------------------------------------------------

    def _denoise(self, x, t, cond, mask, cond_mask):
        return self.denoiser(x, t, cond, mask, cond_mask)

    def _loss(self, pred, target, batch, x0=None):
        mask = batch["x_mask"]
        if self.loss_name == "mse":
            return masked_mse_loss(pred, target, mask, self.presence_weight)
        if self.loss_name == "hungarian":
            return hungarian_loss(pred, target, mask, self.presence_weight,
                                  self.match_presence_weight)
        if self.loss_name == "ce":
            logits, presence_logits = pred
            return discrete_ce_loss(
                logits, batch["x_bins"], mask, presence_logits,
                mask.float(), self.presence_weight)
        raise ValueError(f"Unknown loss: {self.loss_name!r}.")

    def _shared_step(self, batch, stage):
        x0 = batch["x"]
        cond = batch["cond"]
        cond_mask = batch["cond_mask"]
        # Classifier-free guidance needs an unconditional branch to exist, and
        # it only exists if training sometimes saw no condition. Dropped per
        # *batch*, not per sample, which is what the reference implementations
        # do and what keeps the cross-attention path a single code path.
        if stage == "train" and torch.rand(1).item() < self.cond_dropout:
            cond, cond_mask = None, None

        t = self.process.sample_t(x0.shape[0], x0.device)
        x_t, aux = self.process.corrupt(x0, t)
        pred = self._denoise(x_t, t, cond, batch["x_mask"], cond_mask)
        target = self.process.target_for(x0, x_t, t, aux)
        loss = self._loss(pred, target, batch, x0=x0)

        self.log(f"{stage}_loss", loss, on_step=(stage == "train"),
                 on_epoch=True, prog_bar=True, batch_size=x0.shape[0])
        # Coordinate error in bins, logged separately from the objective:
        # under target=noise the loss is in epsilon units and is not comparable
        # across timesteps, so it says nothing about geometry on its own.
        if self.loss_name != "ce":
            with torch.no_grad():
                x0_hat = self.process.to_x0(x_t, float(t[0]), pred) \
                    if hasattr(self.process, "to_x0") else pred
                err = (x0_hat[:, :9] - x0[:, :9]).abs()
                m = batch["x_mask"][:, None, :].float()
                bins = (err * m).sum() / m.sum().clamp(min=1) * (self.num_bins - 1)
            self.log(f"{stage}_coord_err_bins", bins, on_epoch=True,
                     batch_size=x0.shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if self.lr_scheduler == "none":
            return opt
        if self.lr_scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.trainer.max_epochs if self.trainer else 100,
                eta_min=0.01 * self.lr)
            return {"optimizer": opt, "lr_scheduler": sched}
        raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler!r}.")

    # -- sampling ---------------------------------------------------------

    def _state_shape(self, batch):
        """Shape the process allocates its prior at.

        Not `batch["x"].shape`: the channel count follows `state`, and the
        `onehot` arm (Task 11) is 9 * num_bins + 1 channels wide while `bins`
        (Task 12) is 9. Centralised here so adding either is a one-line change
        rather than a hunt through the sampler.
        """
        b, _, f = batch["x"].shape
        if self.state == "bins":
            return (b, 9, f)
        if self.state == "onehot":
            return (b, 9 * self.num_bins + 1, f)
        return (b, 10, f)

    @torch.no_grad()
    def generate(self, batch, n_steps=None, scaffold=None):
        """Run the full reverse trajectory from the LOD1 condition.

        Args:
            batch: a collated batch; only `cond`, `cond_mask` and the shape of
                `x` are read. The target `x` is never looked at.
            n_steps: reverse steps. Defaults to `mesh_diffusion.eval_steps`.
            scaffold: optional ``(t, x) -> x`` projection, supplied by
                `MeshSetEvalCallback` when `scaffold.enabled`. Kept as a
                parameter rather than read from config so a caller can score
                the same checkpoint with and without it.

        Returns:
            Tensor: ``[B, 10, F]`` in the LOD1-normalized frame.
        """
        cond, cond_mask = batch["cond"], batch["cond_mask"]
        mask = batch["x_mask"]
        shape = self._state_shape(batch)

        def denoiser_fn(x_t, t):
            out = self._denoise(x_t, t, cond, mask, cond_mask)
            if self.guidance == 1.0:
                return out
            # Classifier-free guidance (Ho & Salimans, arXiv:2207.12598):
            # push away from the unconditional prediction. Two forward passes
            # per step, so it doubles the eval cost -- which is why it defaults
            # to 1.0 and is a per-experiment choice.
            uncond = self._denoise(x_t, t, None, mask, None)
            return uncond + self.guidance * (out - uncond)

        return self.process.sample(denoiser_fn, shape, batch["x"].device,
                                   n_steps=n_steps or self.eval_steps,
                                   callback=scaffold)
```

- [ ] **Step 4: Write `src/train_mesh_diffusion.py`**

```python
"""Entry point for `config_set: mesh_diffusion`.

Deliberately parallel to `src/train_mesh.py` -- same save_dir layout, same
`create_loggers` / `create_callbacks` helpers, same split semantics -- so a
diffusion run and an autoregressive run land side by side in wandb and can be
read against each other without a translation step.
"""
import logging
from functools import partial
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, random_split

from src.dataset.mesh_set_dataset import MeshSetDataset, mesh_set_collate_fn
from src.models.mesh_diffusion_module import MeshDiffusionModule
from src.utils.config import Config, validate_combination
from src.utils.setup_utils import create_callbacks, create_loggers

logger = logging.getLogger(__name__)


class MeshSetDataModule(L.LightningDataModule):
    """Train/val/test split over one `MeshSetDataset`.

    Split is by `training.train_val_test_split` under a generator seeded with
    `cfg.seed`, matching `MeshDataModule`, so the same building lands in the
    same split on both branches and a cross-branch comparison is honest.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size
        self.collate = partial(mesh_set_collate_fn, multiple_of=8)
        self.dataset = None
        self.train_dataset = self.val_dataset = self.test_dataset = None

    def setup(self, stage=None):
        d, md = self.cfg.mesh_diffusion, self.cfg.mesh_data
        self.dataset = MeshSetDataset(
            dataset_dir=md.dataset_dir, lod_in=md.lod_in, lod_out=md.lod_out,
            num_bins=md.num_bins, margin_lo=list(md.margin_lo),
            margin_hi=list(md.margin_hi), max_faces=md.max_faces,
            max_files=md.max_files, order=d.order, state=d.state)
        fracs = list(self.cfg.training.train_val_test_split)
        gen = torch.Generator().manual_seed(self.cfg.seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, fracs, generator=gen)

    def _loader(self, ds, shuffle):
        md = self.cfg.mesh_data
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          collate_fn=self.collate, num_workers=md.num_workers,
                          persistent_workers=md.persistent_workers and md.num_workers > 0)

    def train_dataloader(self):
        return self._loader(self.train_dataset, True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, False)


def train_mesh_diffusion(cfg: Config):
    """Train the non-autoregressive mesh diffusion branch."""
    validate_combination(cfg)
    L.seed_everything(cfg.seed, workers=True)
    if not cfg.mesh_data.dataset_dir:
        raise ValueError(
            "config_set: mesh_diffusion needs mesh_data.dataset_dir. It shares "
            "the autoregressive branch's data block on purpose -- the two must "
            "read the same corpus to be comparable.")

    save_dir = Path(cfg.logging.save_dir, cfg.logging.experiment_name,
                    cfg.logging.run_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    datamodule = MeshSetDataModule(cfg)
    datamodule.setup()
    logger.info("Mesh pairs: train=%d, val=%d, test=%d",
                len(datamodule.train_dataset), len(datamodule.val_dataset),
                len(datamodule.test_dataset))
    logger.info("Arm: order=%s pos_embed=%s loss=%s denoiser=%s process=%s "
                "target=%s scaffold=%s",
                cfg.mesh_diffusion.order, cfg.mesh_diffusion.pos_embed,
                cfg.mesh_diffusion.loss, cfg.mesh_diffusion.denoiser,
                cfg.mesh_diffusion.process, cfg.mesh_diffusion.target,
                cfg.mesh_diffusion.scaffold.enabled)

    model = MeshDiffusionModule(cfg)
    logger.info("Denoiser: %s, %.1fM parameters", cfg.mesh_diffusion.denoiser,
                sum(p.numel() for p in model.denoiser.parameters()) / 1e6)

    callbacks = create_callbacks(cfg, save_dir)
    if cfg.mesh_diffusion.every_n_epochs > 0:
        from src.eval.mesh_set_eval import MeshSetEvalCallback
        callbacks.append(MeshSetEvalCallback(cfg, save_dir, seed=cfg.seed))

    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        callbacks=callbacks,
        logger=create_loggers(cfg, save_dir) or False,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.resume_from)
    trainer.test(model, datamodule=datamodule)
    return model
```

Note: `MeshSetEvalCallback` arrives in Task 8. Until then set `mesh_diffusion.every_n_epochs: 0` in the smoke config so the import is not reached.

- [ ] **Step 5: Write `configs/mesh-diff-smoke.yaml`**

```yaml
# Phase A smoke: does the diffusion branch run at all?
# Usage: python main.py --train --config configs/mesh-diff-smoke.yaml
#
# 20 files, 1 epoch, no logger, no eval callback. Not a result -- the only
# question it answers is whether dataset, denoiser, process and loss are wired
# to each other.
mode: train
config_set: mesh_diffusion
seed: 42

data:
  dataset_dir: data/The Hague/mini_cleaner

mesh_data:
  dataset_dir: data/The Hague/mini_cleaner
  lod_in: LOD1
  lod_out: LOD2
  num_bins: 128
  margin_lo: [0.0, 0.0, 0.0]
  margin_hi: [0.0, 0.0, 0.1]
  max_faces: 200
  max_files: 20
  num_workers: 0
  persistent_workers: false

mesh_diffusion:
  order: morton
  pos_embed: sinusoidal
  loss: mse
  denoiser: unet
  d_model: 128
  time_dim: 64
  n_head: 4
  process: ddpm
  target: noise
  noise_steps: 1000
  eval_steps: 10
  every_n_epochs: 0        # eval callback lands in Task 8

training:
  batch_size: 4
  accumulate_grad_batches: 1
  train_val_test_split: [0.8, 0.1, 0.1]
  lr: 1e-4
  max_epochs: 1
  gradient_clip_val: 1.0
  lr_scheduler: "none"
  early_stopping:
    enabled: false
  checkpoint:
    enabled: false

trainer:
  accelerator: auto
  devices: 1
  precision: "32"
  log_every_n_steps: 1

logging:
  loggers: []
  project_name: lod-generation
  experiment_name: smoke
  run_name: mesh-diff-smoke
  save_dir: outputs

resume_from: null
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_mesh_diffusion_smoke.py -v`
Expected: 6 passed. `test_guidance_changes_the_sample` and the transformer path are not exercised yet beyond construction.

- [ ] **Step 7: Commit**

```bash
git add src/models/mesh_diffusion_module.py src/train_mesh_diffusion.py configs/mesh-diff-smoke.yaml tests/test_mesh_diffusion_smoke.py
git commit -m "feat(mesh-diff): lightning module, datamodule and training entry point"
```

---

### Task 7: Face set back to a mesh — snap, weld, repair

**Files:**
- Create: `src/models/mesh_set_postprocess.py`
- Test: `tests/test_mesh_set_postprocess.py`

**Interfaces:**
- Consumes: `weld`, `drop_duplicate_faces`, `n_degenerate` from `src/eval/mesh_postprocess.py`; `fix_winding`, `quantize`, `dequantize` from `src/dataset/mesh_dataset.py`.
- Produces: `faces_to_mesh(x, num_bins=128, snap=True, min_area=1e-6) -> (verts [V,3], faces [F,3], stats dict)` where `x` is `[10, F]` (one sample, channels-first) or `[F, 10]`.

- [ ] **Step 1: Write the failing test**

```python
"""Face set to welded mesh.

The assertion that matters: without the grid snap, welding cannot fire, so
every shared vertex stays split and the mesh is a pile of loose triangles. That
is the crack-at-every-edge failure this step exists to prevent, and it is
invisible in chamfer distance -- only the vertex count and watertightness show
it.
"""
import numpy as np
import pytest
import torch

from src.eval.mesh_metrics import is_watertight_mesh
from src.models.mesh_set_postprocess import faces_to_mesh

NUM_BINS = 128


def _cube_faces():
    """A unit cube's 12 triangles, as explicit per-face corners [12,3,3]."""
    v = np.array([[x, y, z] for x in (-0.4, 0.4) for y in (-0.4, 0.4)
                  for z in (-0.4, 0.4)], dtype=float)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 7, 5], [4, 6, 7],
                  [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v[f]


def _to_x(tri, n_pad=4):
    """[F,3,3] plus padding -> the [10, F] channel layout a sampler emits."""
    f = len(tri)
    x = np.zeros((10, f + n_pad), dtype=np.float32)
    x[:9, :f] = tri.reshape(f, 9).T
    x[9, :f] = 0.5
    x[9, f:] = -0.5
    return torch.from_numpy(x)


def test_padding_is_dropped_by_presence():
    verts, faces, stats = faces_to_mesh(_to_x(_cube_faces(), n_pad=6), NUM_BINS)
    assert len(faces) == 12
    assert stats["n_dropped_absent"] == 6


def test_snap_welds_the_cube_to_eight_vertices():
    verts, faces, _ = faces_to_mesh(_to_x(_cube_faces()), NUM_BINS, snap=True)
    assert len(verts) == 8
    assert is_watertight_mesh(faces)


def test_without_snap_jitter_prevents_welding():
    rng = np.random.default_rng(0)
    tri = _cube_faces() + rng.normal(0, 1e-5, size=(12, 3, 3))
    _, _, stats = faces_to_mesh(_to_x(tri), NUM_BINS, snap=False)
    assert stats["n_verts"] == 36          # nothing merged: 12 faces x 3 corners
    verts, faces, stats = faces_to_mesh(_to_x(tri), NUM_BINS, snap=True)
    assert stats["n_verts"] == 8           # the snap put them back on the grid


def test_degenerate_faces_are_dropped():
    tri = _cube_faces()
    tri[0] = tri[0, 0]                     # all three corners coincide
    _, faces, stats = faces_to_mesh(_to_x(tri), NUM_BINS)
    assert stats["n_dropped_degenerate"] == 1
    assert len(faces) == 11


def test_accepts_face_major_layout_too():
    x = _to_x(_cube_faces())
    a = faces_to_mesh(x, NUM_BINS)[1]
    b = faces_to_mesh(x.T, NUM_BINS)[1]
    assert np.array_equal(a, b)


def test_empty_sample_returns_empty_arrays():
    x = torch.full((10, 8), -0.5)          # nothing present
    verts, faces, stats = faces_to_mesh(x, NUM_BINS)
    assert len(verts) == 0 and len(faces) == 0
    assert stats["n_faces"] == 0


def test_stats_report_duplicate_removal():
    tri = np.concatenate([_cube_faces(), _cube_faces()[:2]])   # 2 exact dupes
    _, faces, stats = faces_to_mesh(_to_x(tri), NUM_BINS)
    assert stats["n_dropped_duplicate"] == 2
    assert len(faces) == 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mesh_set_postprocess.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'src.models.mesh_set_postprocess'`

- [ ] **Step 3: Write the implementation**

```python
"""A sampled face set back to an indexed triangle mesh.

The autoregressive branch gets vertex sharing free: its tokens *are* bin
indices, so two faces that name the same corner produce byte-identical
coordinates and `weld` merges them. A continuous sampler produces two floats
that differ in the eighth decimal, `weld` merges nothing, and every edge of the
mesh is a crack. The snap onto the `num_bins` grid is what restores the
autoregressive branch's guarantee -- and it costs nothing in accuracy, since
half a bin is already the tokenizer's own error floor.

Order matters and is the reference's (see `src/eval/mesh_postprocess`): weld
before dropping duplicates, because two faces are only duplicates once their
corners are the same vertex.
"""
import logging

import numpy as np
import torch

from src.dataset.mesh_dataset import NUM_BINS, dequantize, fix_winding, quantize
from src.eval.mesh_postprocess import drop_duplicate_faces, weld

logger = logging.getLogger(__name__)


def _as_face_major(x):
    """``[10, F]`` or ``[F, 10]`` to ``[F, 10]`` numpy."""
    a = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    if a.ndim != 2:
        raise ValueError(f"Expected a 2-D sample, got shape {a.shape}.")
    if a.shape[0] == 10 and a.shape[1] != 10:
        return a.T
    if a.shape[1] == 10:
        return a
    raise ValueError(
        f"Neither axis of {a.shape} is the 10-channel axis; a sample must be "
        "[10, F] or [F, 10].")


def _triangle_areas(tri):
    """``[F,3,3] -> [F]`` via half the cross-product norm."""
    if len(tri) == 0:
        return np.zeros(0)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)


def faces_to_mesh(x, num_bins=NUM_BINS, snap=True, min_area=1e-6):
    """One sampled face set to ``(verts, faces, stats)``.

    Args:
        x: ``[10, F]`` or ``[F, 10]``. Channels 0-8 are the three corners in
            the normalized box; channel 9 is presence, and a face survives when
            it is positive.
        num_bins: grid to snap onto. Must match the dataset's, or the snap
            moves geometry rather than de-duplicating it.
        snap: round coordinates onto the bin grid before welding. Turning this
            off is a diagnostic, not a mode: see the module docstring.
        min_area: faces below this (normalized box units squared) are dropped.
            A diffusion sample can land all three corners in one bin; such a
            face contributes no surface and makes `fix_winding` pick a normal
            from a zero-length cross product.

    Returns:
        tuple: ``(verts [V,3], faces [F,3] int64, stats dict)``. `stats` carries
        the four drop counts and the final vertex/face counts, which is what
        makes a bad sample legible -- a mesh with the right chamfer and 400
        vertices where 60 belong is a different failure from a wrong shape.
    """
    a = _as_face_major(x)
    stats = {"n_slots": len(a)}

    present = a[:, 9] > 0.0
    tri = a[present, :9].reshape(-1, 3, 3).astype(float)
    stats["n_dropped_absent"] = int((~present).sum())

    if snap and len(tri):
        tri = dequantize(quantize(tri.reshape(-1, 3), num_bins), num_bins)
        tri = tri.reshape(-1, 3, 3)

    keep = _triangle_areas(tri) > min_area
    stats["n_dropped_degenerate"] = int((~keep).sum())
    tri = tri[keep]

    if len(tri) == 0:
        stats.update(n_dropped_duplicate=0, n_verts=0, n_faces=0)
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), stats

    verts = tri.reshape(-1, 3)
    faces = np.arange(len(verts), dtype=np.int64).reshape(-1, 3)
    verts, faces = weld(verts, faces)

    before = len(faces)
    faces = drop_duplicate_faces(faces)
    stats["n_dropped_duplicate"] = before - len(faces)

    faces = fix_winding(verts, faces)
    stats.update(n_verts=len(verts), n_faces=len(faces))
    return verts, faces, stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_mesh_set_postprocess.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/models/mesh_set_postprocess.py tests/test_mesh_set_postprocess.py
git commit -m "feat(mesh-diff): snap-weld-repair a sampled face set into a mesh"
```

---

### Task 8: Geometric eval callback

**Files:**
- Create: `src/eval/mesh_set_eval.py`
- Test: extend `tests/test_mesh_diffusion_smoke.py` with the two cases below

**Interfaces:**
- Consumes: `MeshDiffusionModule.generate` (Task 6); `faces_to_mesh` (Task 7); `mesh_metrics` from `src/eval/mesh_metrics.py`; `mesh_to_cityjson`, `save_to_file` from `src/post_process/post_process.py`; `write_obj` from `src/dataset/mesh_dataset.py`.
- Produces:
  - `run_mesh_set_eval(model, dataset, indices, cfg, seed=1234, save_dir=None, scaffold=None, gt_ceiling=True) -> dict`
  - `MeshSetEvalCallback(cfg, save_dir, seed=1234)`

**Do not touch `src/eval/mesh_eval.py`.** This is a parallel module by design: `run_mesh_eval` is built around `model.generate(cond, ...)` returning a token sequence and `model.decode_tokens`, neither of which exists here, and refactoring it to inject both would put a HIGH-blast-radius edit on the autoregressive branch's critical path for no gain.

- [ ] **Step 1: Write the failing test (append to `tests/test_mesh_diffusion_smoke.py`)**

```python
def test_run_mesh_set_eval_returns_paired_metrics(tmp_path):
    """The eval must produce metrics from a model that has learned nothing.

    A random model's chamfer is meaningless; that it is a finite float, keyed
    the way the AR branch keys its metrics, is not.
    """
    import numpy as np
    from src.eval.mesh_set_eval import run_mesh_set_eval

    class _FakeDataset:
        """Two identical cubes, in the shape MeshSetDataset returns."""

        def __init__(self):
            v = np.array([[x, y, z] for x in (-0.4, 0.4) for y in (-0.4, 0.4)
                          for z in (-0.4, 0.4)], dtype=float)
            f = np.array([[0, 1, 3], [0, 3, 2], [4, 7, 5], [4, 6, 7],
                          [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                          [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
            self.tri = v[f]
            self.ids = ["cube0", "cube1"]
            self.pairs = [((v, f), (v, f))] * 2

        def __len__(self):
            return 2

        def mesh_pair(self, i):
            return self.pairs[i]

        def __getitem__(self, i):
            x = torch.zeros(12, 10)
            x[:, :9] = torch.from_numpy(self.tri.reshape(12, 9)).float()
            x[:, 9] = 0.5
            return {"x": x, "cond": x.clone(), "id": self.ids[i],
                    "center": torch.zeros(3), "scale": torch.ones(3)}

    m = MeshDiffusionModule(_cfg()).eval()
    out = run_mesh_set_eval(m, _FakeDataset(), [0, 1], _cfg(),
                            save_dir=tmp_path, seed=0)
    assert "chamfer_m" in out and np.isfinite(out["chamfer_m"])
    assert "n_faces" in out and "gt_chamfer_m" in out
    assert list(tmp_path.glob("*.obj"))


def test_eval_callback_is_gated_by_epoch():
    from src.eval.mesh_set_eval import MeshSetEvalCallback

    cfg = _cfg()
    cfg.mesh_diffusion.every_n_epochs = 5
    cb = MeshSetEvalCallback(cfg, save_dir=None)
    assert not cb._due(epoch=0) and not cb._due(epoch=3)
    assert cb._due(epoch=4) and cb._due(epoch=9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mesh_diffusion_smoke.py -k eval -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'src.eval.mesh_set_eval'`

- [ ] **Step 3: Write the implementation**

```python
"""Free-running whole-mesh generation, scored against the paired LOD2.

A sibling of `src/eval/mesh_eval.py`, not a refactor of it: that module is
built around a token sequence and `decode_tokens`, and this branch produces a
face array. Everything downstream of "we have a mesh in metres" is shared --
`mesh_metrics`, the CityJSON write-back, the ground-truth ceiling.

The ceiling row is not optional. A chamfer of 0.04 m says nothing until you
know whether the tokenizer's own round-trip floor at this bin count is 0.01 or
0.15, and here there is a second floor on top of it: `faces_to_mesh` snaps to
the same grid, so the ground truth through the same snap is the best this
branch could possibly score.
"""
import logging
from pathlib import Path

import lightning as L
import numpy as np
import torch

from src.dataset.mesh_dataset import write_obj
from src.dataset.mesh_set_dataset import mesh_set_collate_fn
from src.eval.mesh_metrics import mesh_metrics
from src.models.mesh_set_postprocess import faces_to_mesh
from src.post_process.post_process import mesh_to_cityjson, save_to_file

logger = logging.getLogger(__name__)


def eval_indices(n, count, seed):
    """``count`` positions drawn without replacement, reproducibly."""
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n, size=min(count, n), replace=False).tolist())


def _aggregate(rows):
    """Mean of each key over rows, ignoring NaN, plus the sample count."""
    if not rows:
        return {}
    keys = {k for row in rows for k in row}
    out = {}
    for k in sorted(keys):
        vals = [row[k] for row in rows if k in row and np.isfinite(row[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    out["n_eval"] = float(len(rows))
    return out


def run_mesh_set_eval(model, dataset, indices, cfg, seed=1234, save_dir=None,
                      scaffold=None, gt_ceiling=True):
    """Sample a mesh per LOD1 condition and score it against its true LOD2.

    Args:
        model: a `MeshDiffusionModule`.
        dataset: `MeshSetDataset` or a `Subset` of one.
        indices: positions within `dataset` to score.
        cfg: the root `Config`; `mesh_diffusion` and `mesh_data` are read.
        seed: threaded into the index draw and torch's generator, so a rerun of
            the same checkpoint gives the same numbers.
        save_dir: where .obj and .city.json samples are written. None writes none.
        scaffold: optional ``(t, x) -> x`` projection passed to `generate`.
        gt_ceiling: also score the ground truth through `faces_to_mesh`, which
            is the floor every generated number must be read against.

    Returns:
        dict: metric name to float, averaged over the sampled buildings.
        Generated metrics are unprefixed; the ceiling is prefixed ``gt_``.
    """
    if len(indices) == 0:
        return {}
    d = cfg.mesh_diffusion
    num_bins = cfg.mesh_data.num_bins
    was_training = model.training
    model.eval()
    torch.manual_seed(seed)

    rows, gt_rows, written = [], [], 0
    for start in range(0, len(indices), d.eval_batch_size):
        chunk = indices[start:start + d.eval_batch_size]
        items = [dataset[int(i)] for i in chunk]
        batch = mesh_set_collate_fn(items, multiple_of=8)
        device = next(model.parameters()).device
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        with torch.no_grad():
            sampled = model.generate(batch, n_steps=d.eval_steps,
                                     scaffold=scaffold)

        for k, i in enumerate(chunk):
            centre = batch["center"][k].cpu().numpy()
            scale = batch["scale"][k].cpu().numpy()
            gen_v, gen_f, stats = faces_to_mesh(
                sampled[k], num_bins, snap=d.snap_before_weld,
                min_area=d.min_face_area)
            gen = (gen_v * scale + centre, gen_f)

            _, gt_raw = _resolve(dataset, int(i))
            row = {}
            if len(gen_f):
                row.update(mesh_metrics(gen, gt_raw, taus=list(d.taus),
                                        n_points=d.n_points, seed=seed,
                                        voxel_m=d.voxel_m))
            else:
                # An all-absent sample is a real failure mode of a presence
                # channel, and silently dropping it would flatter the average.
                row["chamfer_m"] = float("nan")
            row["n_faces"] = float(stats["n_faces"])
            row["n_verts"] = float(stats["n_verts"])
            row["empty"] = float(len(gen_f) == 0)
            for key in ("n_dropped_absent", "n_dropped_degenerate",
                        "n_dropped_duplicate"):
                row[key] = float(stats[key])
            rows.append(row)

            if gt_ceiling:
                ceil_v, ceil_f, _ = faces_to_mesh(
                    batch["x"][k], num_bins, snap=d.snap_before_weld,
                    min_area=d.min_face_area)
                ceil = (ceil_v * scale + centre, ceil_f)
                if len(ceil_f):
                    gt_rows.append(mesh_metrics(ceil, gt_raw, taus=list(d.taus),
                                                n_points=d.n_points, seed=seed,
                                                voxel_m=d.voxel_m))

            if save_dir is not None and written < d.save_samples and len(gen_f):
                out = Path(save_dir)
                out.mkdir(parents=True, exist_ok=True)
                name = batch["ids"][k]
                write_obj(out / f"{name}_gen.obj", *gen)
                write_obj(out / f"{name}_gt.obj", *gt_raw)
                try:
                    save_to_file(mesh_to_cityjson(*gen), out / f"{name}_gen.city.json")
                except Exception:
                    # Write-back is a convenience; a sample that cannot be
                    # expressed as CityJSON must not take the metrics down.
                    logger.warning("CityJSON write-back failed for %s", name,
                                   exc_info=True)
                written += 1

    if was_training:
        model.train()
    out = _aggregate(rows)
    out.update({f"gt_{k}": v for k, v in _aggregate(gt_rows).items()})
    return out


def _resolve(dataset, i):
    """``(index_in_base, raw_gt_mesh)`` through a `Subset` if there is one."""
    base, idx = dataset, i
    while hasattr(base, "dataset"):
        idx = base.indices[idx]
        base = base.dataset
    return idx, base.mesh_pair(idx)[1]


class MeshSetEvalCallback(L.Callback):
    """Runs `run_mesh_set_eval` every N epochs and logs the aggregate.

    Gated rather than folded into `validation_step` because the full reverse
    trajectory is `eval_steps` forward passes per building -- 50 by default,
    and 100 with classifier-free guidance on. That is affordable every fifth
    epoch on 16 buildings and unaffordable every epoch on the val split.
    """

    def __init__(self, cfg, save_dir, seed=1234):
        self.cfg = cfg
        self.d = cfg.mesh_diffusion
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.seed = seed
        self._scaffold = None      # populated in Task 14

    def _due(self, epoch):
        n = self.d.every_n_epochs
        return n > 0 and (epoch + 1) % n == 0

    def _run(self, trainer, pl_module, split, count, out_dir):
        dataset = getattr(trainer.datamodule, f"{split}_dataset")
        if dataset is None or len(dataset) == 0:
            return
        indices = eval_indices(len(dataset), count, self.seed)
        metrics = run_mesh_set_eval(
            pl_module, dataset, indices, self.cfg, seed=self.seed,
            save_dir=out_dir, scaffold=self._scaffold)
        if metrics:
            pl_module.log_dict({f"{split}_{k}": v for k, v in metrics.items()},
                               prog_bar=False, sync_dist=True)
            logger.info("%s mesh eval: %s", split,
                        {k: round(v, 4) for k, v in metrics.items()})

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not self._due(trainer.current_epoch):
            return
        self._run(trainer, pl_module, "val", self.d.n_val, None)

    def on_test_epoch_end(self, trainer, pl_module):
        out = self.save_dir / "samples" if self.save_dir else None
        self._run(trainer, pl_module, "test", self.d.n_test, out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_mesh_diffusion_smoke.py -v`
Expected: 8 passed

- [ ] **Step 5: Run the real smoke config**

Set `mesh_diffusion.every_n_epochs: 1` in `configs/mesh-diff-smoke.yaml`, then:

Run: `python main.py --train --config configs/mesh-diff-smoke.yaml`
Expected: one epoch completes, the eval callback logs a finite `val_chamfer_m` and a `val_gt_chamfer_m` below it, and no exception. Revert `every_n_epochs` to `1` in the committed file (the smoke config is allowed to be slow; it reads 20 files).

- [ ] **Step 6: Commit**

```bash
git add src/eval/mesh_set_eval.py tests/test_mesh_diffusion_smoke.py configs/mesh-diff-smoke.yaml
git commit -m "feat(mesh-diff): paired geometric eval callback with ground-truth ceiling"
```

**Phase A is complete here.** The branch trains, samples, welds and scores. Everything after this adds an axis to the experiment grid.

---

# Phase B — the remaining axes

Each task here adds one column to the experiment grid and is independently reviewable. They have no dependencies on each other, only on Phase A.

### Task 9: Pure transformer denoiser (and with it, the unordered arm)

The Hungarian loss already exists (Task 3) but has no model that can use it: the U-Net requires a sorted order. This task supplies the permutation-equivariant denoiser that makes `order: none` reachable.

**Files:**
- Create: `src/models/mesh_set_transformer.py`
- Test: extend `tests/test_mesh_set_denoisers.py`

**Interfaces:**
- Consumes: `SelfAttention1d`, `CrossAttention1d`, `FaceEncoder`, `SinusoidalFacePositions`, `timestep_embedding` (Task 4).
- Produces: `MeshSetTransformer(in_ch=10, out_ch=10, d_model=256, n_head=8, num_layers=8, dropout=0.1, pos_embed="sinusoidal", time_dim=128, out_bins=None)` with the same `forward(x, t, cond, mask, cond_mask)` signature as `ConditionalMeshUNet`.

- [ ] **Step 1: Write the failing test (append to `tests/test_mesh_set_denoisers.py`)**

```python
from src.models.mesh_set_transformer import MeshSetTransformer


def _tf(pos_embed="none", **kw):
    return MeshSetTransformer(d_model=32, n_head=2, num_layers=2,
                              time_dim=32, pos_embed=pos_embed, **kw)


def test_transformer_preserves_shape():
    x, t, cond, mask, cond_mask = _inputs()
    assert _tf()(x, t, cond, mask, cond_mask).shape == x.shape


def test_transformer_accepts_any_length():
    net = _tf()
    x, t, cond, mask, cond_mask = _inputs(f=13)      # not a multiple of 8
    assert net(x, t, cond, mask, cond_mask).shape == x.shape


def test_no_pe_transformer_is_permutation_equivariant():
    torch.manual_seed(0)
    net = _tf(pos_embed="none").eval()
    x, t, cond, mask, cond_mask = _inputs(f=16)
    mask[:] = True                                    # no padding to confuse it
    perm = torch.randperm(16)
    with torch.no_grad():
        a = net(x, t, cond, mask, cond_mask)[:, :, perm]
        b = net(x[:, :, perm], t, cond, mask[:, perm], cond_mask)
    assert torch.allclose(a, b, atol=1e-5)


def test_pe_transformer_is_not_permutation_equivariant():
    torch.manual_seed(0)
    net = _tf(pos_embed="sinusoidal").eval()
    x, t, cond, mask, cond_mask = _inputs(f=16)
    mask[:] = True
    perm = torch.randperm(16)
    with torch.no_grad():
        a = net(x, t, cond, mask, cond_mask)[:, :, perm]
        b = net(x[:, :, perm], t, cond, mask[:, perm], cond_mask)
    assert not torch.allclose(a, b, atol=1e-5)


def test_transformer_is_invariant_to_condition_order():
    """Cross-attention reads the condition as a set; the LOD1 face order must
    not change the output, which is what makes a length-mismatched condition
    legitimate in the first place."""
    torch.manual_seed(0)
    net = _tf().eval()
    x, t, cond, mask, cond_mask = _inputs(fc=8)
    perm = torch.randperm(8)
    with torch.no_grad():
        a = net(x, t, cond, mask, cond_mask)
        b = net(x, t, cond[:, :, perm], mask, cond_mask[:, perm])
    assert torch.allclose(a, b, atol=1e-5)


def test_transformer_discrete_head_shapes():
    x, t, cond, mask, cond_mask = _inputs()
    logits, presence = _tf(out_bins=128)(x, t, cond, mask, cond_mask)
    assert logits.shape == (2, 9, 128, 16)
    assert presence.shape == (2, 16)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mesh_set_denoisers.py -k transformer -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'src.models.mesh_set_transformer'`

- [ ] **Step 3: Write the implementation**

```python
"""Permutation-equivariant transformer denoiser over a face set.

With `pos_embed: none` this network cannot tell one output slot from another,
which is the point: a mesh is a *set* of faces, and any model that reads a
canonical order is reading a convention rather than geometry. The price is that
a slot-to-slot loss becomes meaningless -- there is no slot -- so this
configuration is only legal with `loss: hungarian`, which `validate_combination`
enforces.

No causal mask anywhere. The whole mesh is processed and emitted at once, so
none of the autoregressive machinery -- KV cache, beam search, exposure bias --
applies or is needed.
"""
import torch
import torch.nn as nn

from src.models.mesh_set_modules import (
    CrossAttention1d,
    FaceEncoder,
    SelfAttention1d,
    SinusoidalFacePositions,
    timestep_embedding,
)


class _Block(nn.Module):
    """Self-attention over faces, cross-attention to LOD1, additive time.

    The time embedding is added to every token rather than modulating the norms
    (DiT's adaLN-Zero). ponytail: additive is the smaller thing that works and
    matches the conv path's `Down`/`Up`; upgrade to adaLN-Zero if the time
    signal turns out to be too weak to steer the late steps.
    """

    def __init__(self, d_model, n_head, time_dim, dropout):
        super().__init__()
        self.sa = SelfAttention1d(d_model, n_head)
        self.ca = CrossAttention1d(d_model, d_model, n_head)
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, temb, cond, mask, cond_mask):
        x = x + self.emb(temb)[:, :, None]
        x = self.sa(x, mask)
        x = self.ca(x, cond, cond_mask)
        return self.drop(x)


class MeshSetTransformer(nn.Module):
    """Denoiser: ``(x_t, t, LOD1) -> prediction``, no convolution, no order.

    Args:
        in_ch, out_ch: 10 = 9 coordinates + presence.
        d_model, n_head, num_layers, dropout: the usual transformer knobs.
        pos_embed: "sinusoidal" makes the model order-aware, which is only
            meaningful under `order: morton`; "none" makes it permutation
            equivariant, which requires `loss: hungarian`.
        time_dim: timestep embedding width.
        out_bins: when set, emits ``[B,9,K,F]`` bin logits and ``[B,F]``
            presence logits for the D3PM arm instead of ``[B,10,F]``.
    """

    def __init__(self, in_ch=10, out_ch=10, d_model=256, n_head=8,
                 num_layers=8, dropout=0.1, pos_embed="sinusoidal",
                 time_dim=128, out_bins=None):
        super().__init__()
        if pos_embed not in ("sinusoidal", "none"):
            raise ValueError(
                f"pos_embed must be 'sinusoidal' or 'none', got {pos_embed!r}.")
        self.time_dim = time_dim
        self.out_bins = out_bins
        self.pos = SinusoidalFacePositions(d_model) if pos_embed == "sinusoidal" else None

        self.inp = nn.Conv1d(in_ch, d_model, 1)
        self.cond_encoder = FaceEncoder(in_ch, d_model)
        self.blocks = nn.ModuleList(
            [_Block(d_model, n_head, time_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

        head_ch = out_ch if out_bins is None else 9 * out_bins + 1
        self.outc = nn.Conv1d(d_model, head_ch, 1)
        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)

    def forward(self, x, t, cond=None, mask=None, cond_mask=None):
        """``x [B,10,F]``, ``t [B]`` in ``[0,1]``, ``cond [B,10,Fc]`` or None."""
        b, _, f = x.shape
        if mask is None:
            mask = torch.ones(b, f, dtype=torch.bool, device=x.device)

        h = self.inp(x)
        if self.pos is not None:
            h = h + self.pos(f, x.device).T[None]
        temb = timestep_embedding(t, self.time_dim)
        c = self.cond_encoder(cond)

        for block in self.blocks:
            h = block(h, temb, c, mask, cond_mask)

        h = self.norm(h.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.outc(h)
        if self.out_bins is None:
            return out
        logits = out[:, : 9 * self.out_bins].reshape(b, 9, self.out_bins, f)
        return logits, out[:, -1]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_mesh_set_denoisers.py -v`
Expected: 12 passed

- [ ] **Step 5: Verify the unordered arm trains end to end**

Add to `tests/test_mesh_diffusion_smoke.py`:

```python
def test_unordered_hungarian_arm_trains():
    m = MeshDiffusionModule(_cfg(order="none", pos_embed="none",
                                 loss="hungarian", denoiser="transformer"))
    loss = m.training_step(_batch(), 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.parameters())
```

Run: `python -m pytest tests/test_mesh_diffusion_smoke.py -v`
Expected: 9 passed

- [ ] **Step 6: Commit**

```bash
git add src/models/mesh_set_transformer.py tests/test_mesh_set_denoisers.py tests/test_mesh_diffusion_smoke.py
git commit -m "feat(mesh-diff): permutation-equivariant transformer denoiser"
```

---

### Task 10: Flow matching

**Files:**
- Modify: `src/models/mesh_processes.py` (append `FlowMatchingProcess`)
- Test: extend `tests/test_mesh_processes.py`

**Interfaces:**
- Consumes: `BaseProcess` (Task 5).
- Produces: `FlowMatchingProcess(sigma_min=0.0)` implementing the `BaseProcess` interface, plus a `to_x0(x_t, t, model_out)` with the same meaning as `GaussianProcess`'s so `MeshDiffusionModule`'s coordinate-error log works unchanged.

- [ ] **Step 1: Write the failing test (append to `tests/test_mesh_processes.py`)**

```python
from src.models.mesh_processes import FlowMatchingProcess


def test_flow_corrupt_interpolates_linearly():
    p = FlowMatchingProcess()
    x0 = torch.ones(4, 10, 8)
    torch.manual_seed(0)
    x_t, noise = p.corrupt(x0, torch.full((4,), 0.25))
    # Convention here: t = 1 is data, t = 0 is noise, matching the reverse loop
    # in BaseProcess.sample which runs t from 1 down to 0.
    assert torch.allclose(x_t, 0.25 * x0 + 0.75 * noise, atol=1e-6)


def test_flow_target_is_the_velocity():
    p = FlowMatchingProcess()
    x0 = torch.randn(4, 10, 8)
    torch.manual_seed(0)
    x_t, noise = p.corrupt(x0, torch.rand(4))
    v = p.target_for(x0, x_t, torch.rand(4), noise)
    assert torch.allclose(v, x0 - noise, atol=1e-6)


def test_flow_oracle_recovers_the_data():
    torch.manual_seed(0)
    p = FlowMatchingProcess()
    x0 = torch.randn(8, 10, 8) * 0.3
    noise_holder = {}

    def oracle(x_t, t):
        # A perfect velocity field for the straight path from the prior we drew.
        return x0 - noise_holder["z"]

    def prior_spy(shape, device):
        noise_holder["z"] = torch.randn(shape, device=device)
        return noise_holder["z"]

    p.prior = prior_spy
    out = p.sample(oracle, x0.shape, x0.device, n_steps=20)
    assert (out - x0).abs().mean().item() < 1e-4


def test_flow_step_count_is_honoured():
    p = FlowMatchingProcess()
    seen = []
    p.sample(lambda x, t: (seen.append(float(t[0])), torch.zeros_like(x))[1],
             (1, 10, 8), torch.device("cpu"), n_steps=12)
    assert len(seen) == 12


def test_flow_to_x0_is_consistent_with_the_path():
    p = FlowMatchingProcess()
    x0 = torch.randn(4, 10, 8)
    torch.manual_seed(0)
    t = torch.full((4,), 0.4)
    x_t, noise = p.corrupt(x0, t)
    v = x0 - noise
    assert torch.allclose(p.to_x0(x_t, 0.4, v), x0, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mesh_processes.py -k flow -v`
Expected: FAIL, `ImportError: cannot import name 'FlowMatchingProcess'`

- [ ] **Step 3: Append the implementation to `src/models/mesh_processes.py`**

```python
class FlowMatchingProcess(BaseProcess):
    """Conditional flow matching along the straight path (Lipman et al.,
    arXiv:2210.02747), with an Euler integrator.

    Same denoiser, different target: instead of a noise level indexed into a
    schedule, time is continuous and the network regresses the constant
    velocity ``x0 - z`` of the straight line from prior to data. There is no
    beta schedule to tune, and the trajectory is straight by construction, so
    few-step sampling degrades far more gracefully than a strided DDPM does.

    Time convention: ``t = 1`` is data, ``t = 0`` is the prior, matching
    `BaseProcess.sample`, which runs ``t`` from 1 down to 0. That is the
    reverse of the usual flow-matching paper convention and is chosen so all
    three processes share one loop.

    Args:
        sigma_min: floor on the prior's contribution at ``t = 1``. 0 gives the
            exact straight path; a small positive value keeps the target
            distribution absolutely continuous, which matters for likelihood
            evaluation and not at all for sampling.
    """

    def __init__(self, sigma_min=0.0):
        super().__init__()
        self.sigma_min = sigma_min
        # No buffers, but keep the nn.Module contract so `create_process`
        # returns something Lightning treats identically to GaussianProcess.
        self.register_buffer("_unused", torch.zeros(1), persistent=False)

    def _coef(self, t, ndim):
        t = t.reshape(-1, *([1] * (ndim - 1)))
        return t, (1.0 - (1.0 - self.sigma_min) * t)

    def sample_t(self, n, device):
        return torch.rand(n, device=device)

    def corrupt(self, x0, t):
        z = torch.randn_like(x0)
        a, b = self._coef(t, x0.dim())
        return a * x0 + b * z, z

    def target_for(self, x0, x_t, t, aux):
        return (1.0 - self.sigma_min) * x0 - (1.0 - self.sigma_min) * aux \
            if self.sigma_min else x0 - aux

    def to_x0(self, x_t, t, model_out):
        """Read a velocity as an ``x0`` estimate: ``x0 = x_t + (1 - t) v``.

        Exact on the straight path, and it is what makes the coordinate-error
        log in `MeshDiffusionModule` comparable across all three processes.
        """
        tt = torch.as_tensor(t, device=x_t.device, dtype=x_t.dtype)
        return x_t + (1.0 - tt) * model_out

    def step(self, x_t, t, t_prev, model_out):
        """One Euler step. ``t_prev < t`` in this convention, so the increment
        is negative and the integrator walks *back* toward the prior; the
        reverse loop's descending timesteps invert that into progress."""
        return x_t + (t_prev - t) * model_out

    def prior(self, shape, device):
        return torch.randn(shape, device=device)

    def timesteps(self, n_steps):
        """Ascending here, not descending: flow matching integrates from the
        prior at ``t = 0`` toward the data at ``t = 1``, which is the opposite
        direction to a denoising trajectory."""
        edges = torch.linspace(0.0, 1.0, n_steps + 1)
        return list(zip(edges[:-1].tolist(), edges[1:].tolist()))
```

Note the two convention flips that must land together: `timesteps` ascends, and `step` uses `t_prev - t`, which is now positive. Get one without the other and the sampler integrates away from the data — the `test_flow_oracle_recovers_the_data` test is what catches it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_mesh_processes.py -v`
Expected: 12 passed

- [ ] **Step 5: Verify the flow arm trains (append to `tests/test_mesh_diffusion_smoke.py`)**

```python
def test_flow_arm_trains_and_samples():
    m = MeshDiffusionModule(_cfg(process="flow", target="velocity"))
    assert torch.isfinite(m.training_step(_batch(), 0))
    m.eval()
    assert m.generate(_batch(), n_steps=4).shape == (2, 10, 16)
```

Run: `python -m pytest tests/test_mesh_diffusion_smoke.py -v`
Expected: 10 passed

- [ ] **Step 6: Commit**

```bash
git add src/models/mesh_processes.py tests/test_mesh_processes.py tests/test_mesh_diffusion_smoke.py
git commit -m "feat(mesh-diff): conditional flow matching with an Euler integrator"
```

---

### Task 11: Gaussian process, categorical readout

The reference implementation's actual scheme, and the one I had mischaracterised: a Gaussian forward process over a discrete alphabet, read out with cross-entropy. It is not D3PM and it is not continuous regression, and it gets grid-exact output — hence exact welding — without a categorical chain.

Two states share this task because they share every line except the input width: `quantized` diffuses the 9 coordinate channels and puts a 128-way head on each; `onehot` diffuses `9 x 128` indicator channels, which is what the reference does.

**Files:**
- Modify: `src/dataset/mesh_set_dataset.py` (`discrete: bool` becomes `state: str`)
- Modify: `src/models/mesh_diffusion_module.py` (state routing, the clamping trick)
- Test: extend `tests/test_mesh_set_dataset.py` and `tests/test_mesh_diffusion_smoke.py`

**Interfaces:**
- Consumes: `GaussianProcess` (Task 5), `discrete_ce_loss` (Task 3), both denoisers' `out_bins` head (Tasks 4, 9).
- Produces:
  - No new dataset API: `MeshSetDataset(..., state=...)` and `_pack(..., state=...)` already ship in Task 1. This task only consumes them.
  - `MeshDiffusionModule._to_state(batch) -> Tensor`, `._from_state(x) -> Tensor [B,10,F]`, `._clamp_x0(logits) -> Tensor`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mesh_diffusion_smoke.py`:

```python
def _batch_with_bins(b=2, f=16, num_bins=128):
    batch = _batch(b=b, f=f)
    batch["x_bins"] = torch.randint(0, num_bins, (b, 9, f))
    return batch


def test_quantized_ce_arm_trains_and_lands_on_the_grid():
    """state: quantized, Gaussian noise, categorical readout -- the reference's
    scheme on coordinate channels."""
    from src.dataset.mesh_dataset import quantize

    m = MeshDiffusionModule(_cfg(state="quantized", loss="ce",
                                 process="ddpm", target="original"))
    assert torch.isfinite(m.training_step(_batch_with_bins(), 0))
    m.eval()
    out = m.generate(_batch_with_bins(), n_steps=4)
    assert out.shape == (2, 10, 16)
    coords = out[:, :9].detach().numpy()
    # Hard clamping must leave every coordinate on a grid point.
    import numpy as np
    from src.dataset.mesh_dataset import dequantize
    assert np.allclose(coords, dequantize(quantize(coords, 128), 128), atol=1e-6)


def test_onehot_ce_arm_trains():
    """state: onehot -- 9 x 128 indicator channels, exactly what the reference
    diffuses."""
    m = MeshDiffusionModule(_cfg(state="onehot", loss="ce",
                                 process="ddpm", target="original"))
    assert torch.isfinite(m.training_step(_batch_with_bins(), 0))


def test_onehot_denoiser_input_width_is_nine_times_bins_plus_one():
    m = MeshDiffusionModule(_cfg(state="onehot", loss="ce",
                                 process="ddpm", target="original"))
    x = m._to_state(_batch_with_bins())
    assert x.shape == (2, 9 * 128 + 1, 16)


def test_soft_clamp_leaves_the_grid():
    import numpy as np
    from src.dataset.mesh_dataset import dequantize, quantize

    m = MeshDiffusionModule(_cfg(state="quantized", loss="ce", process="ddpm",
                                 target="original", x0_clamp="soft")).eval()
    coords = m.generate(_batch_with_bins(), n_steps=4)[:, :9].detach().numpy()
    assert not np.allclose(coords, dequantize(quantize(coords, 128), 128), atol=1e-6)


def test_ce_with_noise_target_is_rejected():
    with pytest.raises(ValueError, match="signal-free"):
        MeshDiffusionModule(_cfg(state="quantized", loss="ce",
                                 process="ddpm", target="noise"))


def test_module_construction_rejects_flow_with_ce():
    """Same rule as the compat suite's `test_flow_with_ce_is_rejected`, checked
    through the constructor -- the path a training run actually takes."""
    with pytest.raises(ValueError, match="incompatible"):
        MeshDiffusionModule(_cfg(state="quantized", loss="ce",
                                 process="flow", target="velocity"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_mesh_set_dataset.py tests/test_mesh_diffusion_smoke.py -v`
Expected: FAIL — `_pack() got an unexpected keyword argument 'state'`, and `MeshDiffusionModule` has no `_to_state`.

- [ ] **Step 3: Confirm the dataset already emits this state**

`state` landed in `MeshSetDataset` and `_pack` in Task 1, and `train_mesh_diffusion` already passes `state=d.state`. Nothing to write here — verify it:

Run: `python -m pytest tests/test_mesh_set_dataset.py -k state -v`
Expected: 2 passed (`test_state_quantized_snaps_the_continuous_channels`, `test_state_continuous_emits_no_bins`)

If those fail, Task 1 was implemented against the pre-revision spec; fix `_pack` there rather than patching around it here.

- [ ] **Step 4: Route the state in `MeshDiffusionModule`**

Task 6 already set `self.state`, `self.categorical`, `self.x0_clamp` and sized `in_ch`/`out_bins` from them, so `__init__` needs no change. Add the three helpers that use those flags:

```python
    def _to_state(self, batch):
        """The tensor the *process* corrupts, for this arm's state.

        `continuous` and `quantized` corrupt the [B,10,F] float layout directly
        -- they differ only in whether the target was snapped, which happened in
        the dataset. `onehot` expands the bins here rather than in the
        dataloader; see `_pack`'s docstring for why.
        """
        if self.state in ("continuous", "quantized"):
            return batch["x"]
        if self.state == "bins":
            return batch["x_bins"]
        b, _, f = batch["x"].shape
        oh = torch.zeros(b, 9, self.num_bins, f, device=batch["x"].device)
        oh.scatter_(2, batch["x_bins"].unsqueeze(2), 1.0)
        # Centred to +-0.5, matching the coordinate and presence channels, so
        # one Gaussian noise level is correctly scaled for every channel.
        oh = oh - 0.5
        return torch.cat([oh.reshape(b, 9 * self.num_bins, f),
                          batch["x"][:, 9:10]], dim=1)

    def _clamp_x0(self, logits):
        """Predicted bin logits back to a state the next reverse step can take.

        Hard clamping (`argmax` to a one-hot) is Diffusion-LM's trick
        (arXiv:2205.14217 section 4.2) and is what keeps a *continuous* process
        over a *discrete* alphabet landing on grid points instead of drifting
        between them. `soft` keeps the softmax and is the ablation that shows
        whether the trick is load-bearing here.
        """
        probs = torch.softmax(logits, dim=2)              # [B,9,K,F]
        if self.x0_clamp == "hard":
            idx = probs.argmax(dim=2, keepdim=True)
            probs = torch.zeros_like(probs).scatter_(2, idx, 1.0)
        if self.state == "onehot":
            b, c, k, f = probs.shape
            return probs - 0.5, probs.argmax(dim=2)
        # `quantized`: collapse the distribution to a single coordinate. Under
        # hard clamping this is the bin centre; under soft it is the posterior
        # mean, which is what makes the two visibly different at sample time.
        centres = torch.linspace(-0.5, 0.5, self.num_bins,
                                 device=probs.device).view(1, 1, -1, 1)
        return (probs * centres).sum(dim=2), probs.argmax(dim=2)

    def _from_state(self, x, bins=None):
        """Whatever the sampler produced, back to the ``[B,10,F]`` eval layout."""
        if self.state in ("continuous", "quantized"):
            return x
        from src.dataset.mesh_dataset import dequantize
        coords = torch.from_numpy(
            dequantize(bins.detach().cpu().numpy(), self.num_bins)).float()
        out = torch.zeros(bins.shape[0], 10, bins.shape[-1], device=bins.device)
        out[:, :9] = coords.to(bins.device)
        return out
```

In `_shared_step`, corrupt the state and supervise on the bins:

```python
        x0 = self._to_state(batch)
        t = self.process.sample_t(x0.shape[0], x0.device)
        x_t, aux = self.process.corrupt(x0, t)
        pred = self._denoise(x_t, t, cond, batch["x_mask"], cond_mask)
        if self.categorical:
            loss = self._loss(pred, None, batch)
        else:
            loss = self._loss(pred, self.process.target_for(x0, x_t, t, aux), batch)
```

In `generate`, the denoiser's categorical output has to become a state again before the next step. Wrap the process's `to_x0` for this arm:

```python
        def denoiser_fn(x_t, t):
            out = self._denoise(x_t, t, cond, mask, cond_mask)
            if self.guidance != 1.0:
                uncond = self._denoise(x_t, t, None, mask, None)
                out = _guide(out, uncond, self.guidance)
            if not self.categorical or self.state == "bins":
                return out
            # Categorical readout over a Gaussian process: the process expects
            # an x0-shaped tensor, so project the logits back to the state.
            logits, presence_logits = out
            state_x0, self._last_bins = self._clamp_x0(logits)
            if self.state == "onehot":
                b, _, f = x_t.shape
                state_x0 = state_x0.reshape(b, 9 * self.num_bins, f)
            return torch.cat([state_x0, presence_logits.unsqueeze(1)], dim=1)
```

with `_guide` a two-line module-level helper that handles both the tensor and the `(logits, presence)` tuple, and the process built with `target="original"` — which `validate_combination` has already forced for `loss: ce`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_mesh_set_dataset.py tests/test_mesh_diffusion_smoke.py tests/test_mesh_diffusion_compat.py -v`
Expected: 10 + 16 + 13 = 39 passed

- [ ] **Step 6: Commit**

```bash
git add src/dataset/mesh_set_dataset.py src/models/mesh_diffusion_module.py src/train_mesh_diffusion.py tests/test_mesh_set_dataset.py tests/test_mesh_diffusion_smoke.py
git commit -m "feat(mesh-diff): Gaussian process with categorical readout (quantized and onehot states)"
```

---

### Task 12: Discrete diffusion over coordinate bins (D3PM)

The arm that makes welding exact: coordinates never leave the `num_bins` grid, so two faces naming the same corner produce identical floats and `weld` fires without needing the snap.

**Read D11 before starting.** The Levi-graph branch's D3PM failure does not transfer as a prediction: its alphabet was nominal class labels, this one is ordinal coordinate bins. That difference is exactly what the `transition` axis exists to exploit — a discretized-Gaussian transition corrupts a coordinate to a *nearby* bin, which is the process the data actually has. Ship both transitions and treat the pair as a controlled A/B. The per-noise-bucket logging in Step 6 is there so the answer arrives in an epoch, not so you can watch a failure repeat.

**Files:**
- Modify: `src/models/mesh_processes.py` (append `DiscreteProcess`)
- Test: extend `tests/test_mesh_processes.py`

**Interfaces:**
- Consumes: `cosine_beta_schedule_discrete` from `src/models/noise.py`; `BaseProcess` (Task 5).
- Produces: `DiscreteProcess(noise_steps=1000, num_bins=128)` implementing the `BaseProcess` interface over integer bins. `corrupt` takes and returns integer tensors `[B,9,F]`; `step` consumes `(logits [B,9,K,F], presence_logits [B,F])`.

- [ ] **Step 1: Write the failing test (append to `tests/test_mesh_processes.py`)**

```python
from src.models.mesh_processes import DiscreteProcess


def test_discrete_corrupt_leaves_most_bins_alone_at_low_t():
    p = DiscreteProcess(noise_steps=1000, num_bins=128)
    bins = torch.randint(0, 128, (256, 9, 8))
    noisy, _ = p.corrupt(bins, torch.full((256,), 0.02))
    assert (noisy == bins).float().mean().item() > 0.9


@pytest.mark.parametrize("transition", ["uniform", "gaussian"])
def test_discrete_terminal_marginal_is_the_sampler_prior(transition):
    """Both transitions must end uniform, or the reverse loop starts from a
    distribution the forward process never produces. For `gaussian` this is the
    (1 - alpha_bar)^2 mixture weight doing its job, and it is the single
    easiest thing to get wrong in that transition."""
    p = DiscreteProcess(noise_steps=1000, num_bins=32, transition=transition)
    bins = torch.zeros(8192, 9, 4, dtype=torch.long)
    noisy, _ = p.corrupt(bins, torch.ones(8192))
    counts = torch.bincount(noisy.reshape(-1), minlength=32).float()
    assert counts.std().item() / counts.mean().item() < 0.15


def test_gaussian_transition_moves_bins_locally_mid_trajectory():
    """The whole reason the gaussian transition exists: at moderate noise a
    coordinate should land NEAR where it was, not anywhere on the grid."""
    torch.manual_seed(0)
    uni = DiscreteProcess(noise_steps=1000, num_bins=128, transition="uniform")
    gau = DiscreteProcess(noise_steps=1000, num_bins=128, transition="gaussian",
                          sigma_max=16.0)
    bins = torch.full((4096, 9, 4), 64, dtype=torch.long)
    t = torch.full((4096,), 0.4)
    d_uni = (uni.corrupt(bins, t)[0] - 64).abs().float().mean().item()
    d_gau = (gau.corrupt(bins, t)[0] - 64).abs().float().mean().item()
    assert d_gau < 0.5 * d_uni


def test_discrete_corrupt_stays_in_range():
    p = DiscreteProcess(noise_steps=100, num_bins=128)
    noisy, _ = p.corrupt(torch.randint(0, 128, (32, 9, 8)), torch.rand(32))
    assert noisy.min() >= 0 and noisy.max() < 128


def test_discrete_prior_is_uniform():
    p = DiscreteProcess(noise_steps=100, num_bins=128)
    x = p.prior((4096, 9, 8), torch.device("cpu"))
    counts = torch.bincount(x.reshape(-1), minlength=128).float()
    assert counts.std().item() / counts.mean().item() < 0.1


def test_discrete_oracle_reverse_recovers_the_bins():
    torch.manual_seed(0)
    p = DiscreteProcess(noise_steps=1000, num_bins=32)
    bins = torch.randint(0, 32, (4, 9, 8))
    presence = torch.ones(4, 8)

    def oracle(x_t, t):
        logits = torch.full((4, 9, 32, 8), -10.0)
        logits.scatter_(2, bins.unsqueeze(2), 10.0)
        return logits, torch.full((4, 8), 10.0)

    out = p.sample(oracle, (4, 9, 8), torch.device("cpu"), n_steps=50)
    agree = (out[0] == bins).float().mean().item()
    assert agree > 0.95


def test_discrete_sample_returns_bins_and_presence():
    p = DiscreteProcess(noise_steps=100, num_bins=32)
    out = p.sample(lambda x, t: (torch.zeros(2, 9, 32, 8), torch.zeros(2, 8)),
                   (2, 9, 8), torch.device("cpu"), n_steps=5)
    bins, presence = out
    assert bins.shape == (2, 9, 8) and bins.dtype == torch.long
    assert presence.shape == (2, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mesh_processes.py -k discrete -v`
Expected: FAIL, `ImportError: cannot import name 'DiscreteProcess'`

- [ ] **Step 3: Append the implementation to `src/models/mesh_processes.py`**

```python
class DiscreteProcess(BaseProcess):
    """D3PM over coordinate bins (Austin et al., arXiv:2107.03006), x0-parameterised.

    Nine independent categorical chains per face -- one per coordinate channel
    -- plus a two-state chain for presence.

    Two transitions, and the choice is the arm (plan D11):

      * ``uniform`` -- a corrupted bin is redrawn uniformly. The standard
        text/graph setting, and the right one when the alphabet is *nominal*.
      * ``gaussian`` -- a corrupted bin moves to a nearby one, with a
        discretized-Gaussian kernel whose width grows along the schedule
        (their section 3.2, ``D3PM-gauss``). Coordinate bins are *ordinal*:
        bin 40 and bin 41 are half a centimetre apart, and a uniform transition
        discards that structure at every step. This is the better-matched
        process for quantized geometry and is the default here.

    Why this arm exists at all: bins never leave the grid, so two faces naming
    the same corner emit byte-identical coordinates and `weld` merges them with
    no snap. It shares that property with the ``state: onehot`` arm in Task 11;
    what it does *not* share is the corruption, which is the thing under test.

    Args:
        noise_steps: schedule resolution.
        num_bins: alphabet size per coordinate channel.
        transition: "uniform" or "gaussian".
        sigma_max: discretized-Gaussian width at t = 1, in bins. Only read
            under ``transition="gaussian"``. 16 on a 128-bin grid is an eighth
            of the range, which reaches near-uniform by the end of the schedule
            while staying local for most of it.
    """

    def __init__(self, noise_steps=1000, num_bins=128, transition="gaussian",
                 sigma_max=16.0):
        super().__init__()
        if transition not in ("uniform", "gaussian"):
            raise ValueError(
                f"transition must be 'uniform' or 'gaussian', got {transition!r}.")
        self.noise_steps = noise_steps
        self.num_bins = num_bins
        self.transition = transition
        self.sigma_max = sigma_max
        # One nu per chain, all 1.0: the per-feature exponent exists for the
        # Levi branch, where nodes and edges corrupt at different rates. Here
        # every chain is the same kind of variable.
        betas = cosine_beta_schedule_discrete(noise_steps, [1.0])[:, 0]
        alpha_bar = np.cumprod(1.0 - betas)
        self.register_buffer("alpha_bar", torch.tensor(alpha_bar, dtype=torch.float32))
        # Offsets used by the Gaussian transition, precomputed once.
        self.register_buffer(
            "_offsets", torch.arange(-num_bins + 1, num_bins, dtype=torch.float32))

    def _gaussian_offsets(self, t, shape, device):
        """Signed bin displacements drawn from the discretized Gaussian at t.

        Width interpolates from ~0 at t = 0 to `sigma_max` at t = 1, so a
        coordinate wanders locally early and reaches the whole grid late. The
        draw is truncated by the reflect-free clamp in `corrupt`, which is the
        one place the boundary is handled.
        """
        a = self.alpha_bar[self.to_index(t)]
        sigma = ((1.0 - a) * self.sigma_max).reshape(-1, *([1] * (len(shape) - 1)))
        return torch.round(torch.randn(shape, device=device) * sigma).long()

    def to_index(self, t):
        idx = (t.clamp(0.0, 1.0) * (len(self.alpha_bar) - 1)).round().long()
        return idx.clamp(0, len(self.alpha_bar) - 1)

    def _keep_prob(self, t, shape):
        """P(a bin survives to time t) under the uniform transition."""
        a = self.alpha_bar[self.to_index(t)]
        return a.reshape(-1, *([1] * (len(shape) - 1)))

    def sample_t(self, n, device):
        return torch.rand(n, device=device)

    def corrupt(self, x0, t):
        """``x0 [B,9,F]`` int64 bins -> ``(x_t [B,9,F], None)``.

        Sampled from the marginal q(x_t | x0) directly rather than by walking
        the chain. Under `uniform` the marginal is exactly "keep with
        probability alpha_bar, else redraw uniformly", which is O(1) in t.
        Under `gaussian` it is a single displacement drawn at the width the
        schedule has reached, which is the marginal of the composed kernel up
        to the boundary clamp -- exact in the interior, and the interior is
        where every coordinate that matters lives.
        """
        if self.transition == "uniform":
            keep = self._keep_prob(t, x0.shape)
            redraw = torch.randint_like(x0, 0, self.num_bins)
            stay = torch.rand(x0.shape, device=x0.device) < keep
            return torch.where(stay, x0, redraw), None

        # Three-way mixture, and the weights are not arbitrary: the terminal
        # distribution has to BE the sampler's prior, or the reverse loop
        # starts somewhere the forward process never reaches. `p_uniform` is
        # (1 - alpha_bar)^2 so it goes to 1 exactly as alpha_bar goes to 0,
        # which makes the t = 1 marginal uniform and `prior` correct for both
        # transitions. The local component peaks in the middle of the
        # trajectory -- which is where ordinal structure is worth having, since
        # that is the stretch on which the model learns to refine a coordinate
        # rather than to invent one.
        a = self._keep_prob(t, x0.shape)
        p_uniform = (1.0 - a) ** 2
        u = torch.rand(x0.shape, device=x0.device)
        shifted = (x0 + self._gaussian_offsets(t, x0.shape, x0.device))
        # Clamp, not wrap: bin 0 and bin 127 are opposite ends of a building,
        # not neighbours, and a wrapping kernel would teach the model that the
        # floor is adjacent to the roof.
        shifted = shifted.clamp(0, self.num_bins - 1)
        uniform = torch.randint_like(x0, 0, self.num_bins)
        out = torch.where(u < a, x0,
                          torch.where(u < a + p_uniform, uniform, shifted))
        return out, None

    def target_for(self, x0, x_t, t, aux):
        return x0

    def step(self, x_t, t, t_prev, model_out):
        """Ancestral step: sample x0 from the predicted posterior, re-noise to
        ``t_prev``.

        The exact reverse posterior for a uniform chain would weight q(s|t,x0)
        against the predicted p(x0); this samples x0 first and re-noises, which
        is the same expectation with one extra sampling step and none of the
        [B, F, K, K] intermediate that the exact form needs at K = 128. ponytail:
        exact posterior via `compute_batched_over0_posterior_distribution` if
        the sample quality turns out to be limited by this and not by the model.
        """
        logits, presence_logits = model_out
        b, c, k, f = logits.shape
        probs = torch.softmax(logits, dim=2).permute(0, 1, 3, 2).reshape(-1, k)
        x0 = torch.multinomial(probs, 1).reshape(b, c, f)
        if t_prev <= 0.0:
            return x0, presence_logits
        t_batch = torch.full((b,), t_prev, device=logits.device)
        x_prev, _ = self.corrupt(x0, t_batch)
        return x_prev, presence_logits

    def prior(self, shape, device):
        return torch.randint(0, self.num_bins, shape, device=device)

    def sample(self, denoiser_fn, shape, device, n_steps=50, callback=None):
        """As `BaseProcess.sample`, but the state is integer bins and the last
        step's presence logits are carried out alongside them."""
        x = self.prior(shape, device)
        presence = None
        for t, t_prev in self.timesteps(n_steps):
            t_batch = torch.full((shape[0],), t, device=device)
            with torch.no_grad():
                out = denoiser_fn(x, t_batch)
            x, presence = self.step(x, t, t_prev, out)
            if callback is not None:
                x = callback(t_prev, x)
        return x, presence
```

Add at the top of `src/models/mesh_processes.py`:

```python
import numpy as np

from src.models.noise import cosine_beta_schedule_discrete
```

- [ ] **Step 4: Extend the state routing for integer bins**

Task 11 already built `_to_state` / `_clamp_x0` / `_from_state` and the
`self.state` / `self.categorical` flags. **Do not add a `self.discrete` flag** —
`state == "bins"` is the condition, and a second flag for the same fact drifts
out of sync with the first. Three small extensions:

`_to_state` already returns `batch["x_bins"]` for `state: "bins"` (Task 11). What
it does not do is hand the *denoiser* an integer tensor, which no convolution
will accept. The conversion belongs in the denoiser call, not the process, so
add to `MeshDiffusionModule`:

```python
    def _model_input(self, x_t):
        """The process's state as something a denoiser can consume.

        Only `bins` needs this: its state is int64 indices, and the network
        reads the same [B, 10, F] float layout every other arm does. Presence
        is left at 0 -- at input time it is unknown, and the model predicts it.
        """
        if self.state != "bins":
            return x_t
        from src.dataset.mesh_dataset import dequantize
        coords = torch.from_numpy(
            dequantize(x_t.detach().cpu().numpy(), self.num_bins)).float()
        out = torch.zeros(x_t.shape[0], 10, x_t.shape[-1], device=x_t.device)
        out[:, :9] = coords.to(x_t.device)
        return out
```

and route both call sites through it. In `_shared_step`:

```python
        pred = self._denoise(self._model_input(x_t), t, cond,
                             batch["x_mask"], cond_mask)
```

and inside `generate`'s `denoiser_fn`:

```python
        def denoiser_fn(x_t, t):
            out = self._denoise(self._model_input(x_t), t, cond, mask, cond_mask)
            ...
```

The `if not self.categorical or self.state == "bins": return out` guard Task 11
put in `denoiser_fn` is what keeps the categorical-over-Gaussian projection from
firing here: `DiscreteProcess.step` consumes the `(logits, presence)` tuple
directly, because a categorical chain's reverse step *is* a draw from those
logits. Leave it as written.

Finally, `generate` must unpack `DiscreteProcess.sample`'s tuple. Extend Task
11's tail:

```python
        out = self.process.sample(denoiser_fn, shape, batch["x"].device,
                                  n_steps=n_steps or self.eval_steps,
                                  callback=scaffold)
        if self.state != "bins":
            return self._from_state(out)
        bins, presence_logits = out
        x = self._from_state(None, bins=bins)
        x[:, 9] = torch.where(presence_logits > 0, 0.5, -0.5)
        return x
```

`_state_shape` already ships in Task 6 and already returns `(b, 9, f)` for
`state: "bins"`, so nothing here reaches backwards into an earlier task.

- [ ] **Step 5: Add the per-noise-bucket accuracy log (D11)**

In `_shared_step`, for `state: "bins"` only:

```python
        if self.state == "bins":
            with torch.no_grad():
                logits = pred[0]
                correct = (logits.argmax(dim=2) == x0).float()
                m = batch["x_mask"][:, None, :].float()
                acc = (correct * m).sum() / (m.sum() * 9).clamp(min=1)
                # Bucketed by noise level, because the aggregate hides exactly
                # what D11 is about: heads that are accurate at low t and carry
                # nothing at high t still average to a healthy-looking number,
                # and the uniform-vs-gaussian transition A/B is precisely a
                # question about the high-noise buckets.
                bucket = int(min(float(t.mean()), 0.999) * 4)
            self.log(f"{stage}_bin_acc", acc, on_epoch=True,
                     batch_size=x0.shape[0])
            self.log(f"{stage}_bin_acc_t{bucket}", acc, on_epoch=True,
                     batch_size=x0.shape[0])
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_mesh_processes.py -v`
Expected: 20 passed (19 test functions; `test_discrete_terminal_marginal_is_the_sampler_prior` is parametrized over both transitions)

Add to `tests/test_mesh_diffusion_smoke.py`:

```python
def test_discrete_arm_trains_and_samples():
    cfg = _cfg(process="d3pm", target="original", loss="ce", state="bins")
    m = MeshDiffusionModule(cfg)
    batch = _batch()
    batch["x_bins"] = torch.randint(0, 128, (2, 9, 16))
    assert torch.isfinite(m.training_step(batch, 0))
    m.eval()
    out = m.generate(batch, n_steps=4)
    assert out.shape == (2, 10, 16)
    assert set(out[:, 9].unique().tolist()) <= {0.5, -0.5}
```

Run: `python -m pytest tests/test_mesh_diffusion_smoke.py -v`
Expected: 17 passed

- [ ] **Step 7: Commit**

```bash
git add src/models/mesh_processes.py src/models/mesh_diffusion_module.py tests/test_mesh_processes.py tests/test_mesh_diffusion_smoke.py
git commit -m "feat(mesh-diff): D3PM over coordinate bins with per-timestep accuracy logging"
```

---

### Task 13: Scaffold masking in the sampler

**Files:**
- Create: `src/models/mesh_scaffold.py`
- Modify: `src/eval/mesh_set_eval.py` (populate `MeshSetEvalCallback._scaffold`)
- Test: `tests/test_mesh_scaffold.py`

**Interfaces:**
- Consumes: `_raster` from `src/eval/mesh_metrics.py`; `quantize`, `dequantize` from `src/dataset/mesh_dataset.py`.
- Produces:
  - `lod1_scaffold(cond [B,10,Fc], cond_mask [B,Fc], voxel=1/32, dilate=3) -> BoolTensor [B, G, G, G]`
  - `make_projector(scaffold, cfg) -> callable (t, x) -> x` for continuous processes
  - `make_bin_masker(scaffold, num_bins) -> callable (t, bins) -> bins` for the discrete process

- [ ] **Step 1: Write the failing test**

```python
"""Scaffold masking: MeshWeaver's fix (3), as a sampler-side projection.

LOD1 already bounds where an LOD2 vertex can legally sit. These tests pin the
three things that make that bound useful rather than harmful: it must contain
the LOD1 surface, it must be dilated enough to admit the ridge that rises above
LOD1, and it must not fire early, when x_t is still noise.
"""
import numpy as np
import pytest
import torch

from src.models.mesh_scaffold import lod1_scaffold, make_bin_masker, make_projector


def _prism_cond(b=1, fc=8):
    """A box occupying the lower half of the normalized cube."""
    v = np.array([[x, y, z] for x in (-0.3, 0.3) for y in (-0.3, 0.3)
                  for z in (-0.4, 0.0)], dtype=float)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 7, 5], [4, 6, 7],
                  [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6]])
    tri = v[f]
    cond = torch.zeros(b, 10, fc)
    cond[:, :9, : len(tri)] = torch.from_numpy(tri.reshape(len(tri), 9)).float().T
    cond[:, 9, : len(tri)] = 0.5
    mask = torch.zeros(b, fc, dtype=torch.bool)
    mask[:, : len(tri)] = True
    return cond, mask


def test_scaffold_contains_the_lod1_surface():
    cond, mask = _prism_cond()
    sc = lod1_scaffold(cond, mask, voxel=1 / 32, dilate=0)
    g = sc.shape[-1]
    # A point on the LOD1 wall must be inside.
    idx = ((torch.tensor([0.3, 0.0, -0.2]) + 0.5) * (g - 1)).round().long()
    assert sc[0, idx[0], idx[1], idx[2]]


def test_dilation_admits_the_ridge_above_lod1():
    cond, mask = _prism_cond()
    tight = lod1_scaffold(cond, mask, voxel=1 / 32, dilate=0)
    loose = lod1_scaffold(cond, mask, voxel=1 / 32, dilate=3)
    assert loose.sum() > tight.sum()
    g = loose.shape[-1]
    ridge = ((torch.tensor([0.0, 0.0, 0.05]) + 0.5) * (g - 1)).round().long()
    assert not tight[0, ridge[0], ridge[1], ridge[2]]
    assert loose[0, ridge[0], ridge[1], ridge[2]]


def test_projector_is_a_noop_above_the_threshold():
    cond, mask = _prism_cond()
    sc = lod1_scaffold(cond, mask)
    proj = make_projector(sc, apply_below_t=0.5)
    x = torch.randn(1, 10, 8) * 3.0
    assert torch.allclose(proj(0.9, x), x)


def test_projector_pulls_outliers_inside_below_the_threshold():
    cond, mask = _prism_cond()
    sc = lod1_scaffold(cond, mask, dilate=3)
    proj = make_projector(sc, apply_below_t=0.5)
    x = torch.zeros(1, 10, 8)
    x[:, :9] = 5.0                          # far outside the box
    x[:, 9] = 0.5
    out = proj(0.1, x)
    assert out[:, :9].abs().max().item() <= 0.5 + 1e-6
    assert not torch.allclose(out[:, :9], x[:, :9])


def test_projector_leaves_presence_alone():
    cond, mask = _prism_cond()
    proj = make_projector(lod1_scaffold(cond, mask), apply_below_t=0.5)
    x = torch.zeros(1, 10, 8)
    x[:, 9] = -0.3
    assert torch.allclose(proj(0.1, x)[:, 9], x[:, 9])


def test_projector_leaves_already_legal_points_untouched():
    cond, mask = _prism_cond()
    sc = lod1_scaffold(cond, mask, dilate=3)
    proj = make_projector(sc, apply_below_t=0.5)
    x = torch.zeros(1, 10, 8)
    x[:, :9] = -0.2                          # inside the prism
    assert torch.allclose(proj(0.1, x)[:, :9], x[:, :9], atol=1e-6)


def test_bin_masker_moves_illegal_bins_and_keeps_legal_ones():
    cond, mask = _prism_cond()
    sc = lod1_scaffold(cond, mask, dilate=3)
    masker = make_bin_masker(sc, num_bins=128, apply_below_t=0.5)
    bins = torch.full((1, 9, 8), 127, dtype=torch.long)   # corner of the box
    out = masker(0.1, bins)
    assert out.shape == bins.shape and out.dtype == torch.long
    assert (out != bins).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mesh_scaffold.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'src.models.mesh_scaffold'`

- [ ] **Step 3: Write the implementation**

```python
"""LOD1 as a legal region for LOD2 vertices, enforced during sampling.

MeshWeaver (arXiv:2606.04688) section 3.2 fix (3): mask the logits of empty
voxels so every predicted vertex is anchored near the input surface. The
2026-08-24 research scan flags this as the item most likely to transfer here,
because LOD1 already bounds where an LOD2 vertex can legally sit.

In an autoregressive model it is a logit mask applied once per token. In
diffusion it is a projection applied once per reverse step, which is both
cheaper and stronger: it acts on the whole mesh at every noise level rather
than on one coordinate at a time with no chance to revise.

The dilation is not a tuning knob to leave at zero. LOD2 is *not* contained in
LOD1 -- the ridge rises above what is only the median roof height, which is the
same fact `margin_hi` exists for -- so an undilated scaffold would project every
ridge vertex back down onto the LOD1 roof and flatten exactly the geometry the
model is being asked to invent.
"""
import logging

import numpy as np
import torch

from src.dataset.mesh_dataset import dequantize, quantize

logger = logging.getLogger(__name__)


def _rasterize(tri, grid):
    """``[F,3,3]`` in ``[-0.5,0.5]`` to a ``[G,G,G]`` bool occupancy grid.

    Marks the voxel of every corner and of the edge midpoints. Not a
    conservative triangle rasteriser: the scaffold is dilated anyway, and a
    dilation of 3 voxels covers a triangle whose edges are shorter than 6
    voxels, which every LOD1 prism face is at G = 32.
    """
    occ = np.zeros((grid, grid, grid), dtype=bool)
    if len(tri) == 0:
        return occ
    pts = [tri[:, i] for i in range(3)]
    pts += [(tri[:, i] + tri[:, (i + 1) % 3]) / 2 for i in range(3)]
    pts += [tri.mean(axis=1)]
    p = np.concatenate(pts, axis=0)
    idx = np.clip(np.rint((p + 0.5) * (grid - 1)).astype(int), 0, grid - 1)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ


def _dilate(occ, r):
    """Cubic dilation by ``r`` voxels, via repeated 6-neighbour ``or``."""
    out = occ.copy()
    for _ in range(r):
        acc = out.copy()
        for axis in range(3):
            acc |= np.roll(out, 1, axis=axis)
            acc |= np.roll(out, -1, axis=axis)
        out = acc
    return out


def lod1_scaffold(cond, cond_mask, voxel=1.0 / 32, dilate=3):
    """Occupancy grid of the dilated LOD1 surface, per batch item.

    Args:
        cond: ``[B, 10, Fc]`` LOD1 face set in the normalized box.
        cond_mask: ``[B, Fc]`` bool, True at real faces.
        voxel: grid spacing in normalized box units; the grid is ``1/voxel``
            cells per axis.
        dilate: growth in voxels. See the module docstring -- 0 is wrong.

    Returns:
        BoolTensor: ``[B, G, G, G]`` on ``cond``'s device.
    """
    b, _, fc = cond.shape
    grid = int(round(1.0 / voxel))
    out = np.zeros((b, grid, grid, grid), dtype=bool)
    c = cond.detach().cpu().numpy()
    m = cond_mask.detach().cpu().numpy()
    for i in range(b):
        tri = c[i, :9, m[i]].T.reshape(-1, 3, 3) if m[i].any() else np.zeros((0, 3, 3))
        out[i] = _dilate(_rasterize(tri, grid), dilate)
    return torch.from_numpy(out).to(cond.device)


def _legal_points(scaffold_i, grid):
    """``[N, 3]`` centre coordinates of the occupied voxels, in ``[-0.5,0.5]``."""
    idx = torch.nonzero(scaffold_i, as_tuple=False).float()
    return idx / (grid - 1) - 0.5


def make_projector(scaffold, apply_below_t=0.5):
    """Continuous-state projection callback for `BaseProcess.sample`.

    Args:
        scaffold: ``[B, G, G, G]`` bool from `lod1_scaffold`.
        apply_below_t: only project once ``t`` is below this. Above it the
            state is nearly pure noise and every coordinate is out of bounds,
            so projecting would overwrite the denoiser rather than guide it.

    Returns:
        callable: ``(t, x [B,10,F]) -> x``. Presence (channel 9) is never
        touched -- it is not a coordinate and has no legal region.
    """
    grid = scaffold.shape[-1]
    legal = [_legal_points(scaffold[i], grid) for i in range(scaffold.shape[0])]

    def project(t, x):
        if t >= apply_below_t:
            return x
        out = x.clone()
        for i, pts in enumerate(legal):
            if len(pts) == 0:
                continue
            v = out[i, :9].T.reshape(-1, 3)            # [F*3, 3]
            d = torch.cdist(v, pts.to(v.device))       # [F*3, N]
            nearest = pts.to(v.device)[d.argmin(dim=1)]
            # Only move what is actually outside: a legal vertex snapped to a
            # voxel centre would be quantized twice, once here and once in
            # faces_to_mesh, for no reason.
            outside = d.min(dim=1).values > (1.5 / (grid - 1))
            v = torch.where(outside[:, None], nearest, v)
            out[i, :9] = v.reshape(-1, 3, 3).reshape(-1, 9).T
        return out

    return project


def make_bin_masker(scaffold, num_bins=128, apply_below_t=0.5):
    """Discrete-state analogue: move illegal bins to the nearest legal one.

    The logit mask MeshWeaver describes would be the exact form, but the
    denoiser's nine heads are independent per axis while legality is a joint
    property of the triple -- masking each axis separately admits combinations
    the scaffold forbids. Projecting the sampled triple is the honest version.

    Args:
        scaffold: ``[B, G, G, G]`` bool.
        num_bins: coordinate alphabet size.
        apply_below_t: as `make_projector`.

    Returns:
        callable: ``(t, bins [B,9,F]) -> bins``.
    """
    grid = scaffold.shape[-1]
    legal = [_legal_points(scaffold[i], grid) for i in range(scaffold.shape[0])]

    def project(t, bins):
        if t >= apply_below_t:
            return bins
        out = bins.clone()
        for i, pts in enumerate(legal):
            if len(pts) == 0:
                continue
            coords = torch.from_numpy(
                dequantize(out[i].detach().cpu().numpy(), num_bins)).float()
            v = coords.T.reshape(-1, 3)
            d = torch.cdist(v, pts)
            outside = d.min(dim=1).values > (1.5 / (grid - 1))
            v = torch.where(outside[:, None], pts[d.argmin(dim=1)], v)
            snapped = quantize(v.reshape(-1, 9).numpy(), num_bins)
            out[i] = torch.from_numpy(snapped).to(out.device).T
        return out

    return project
```

- [ ] **Step 4: Wire it into the eval callback**

In `src/eval/mesh_set_eval.py`, inside `run_mesh_set_eval`, build the scaffold per batch when it is enabled and the caller did not supply one:

```python
        step_hook = scaffold
        if step_hook is None and d.scaffold.enabled:
            from src.models.mesh_scaffold import (
                lod1_scaffold, make_bin_masker, make_projector)
            grid = lod1_scaffold(batch["cond"], batch["cond_mask"],
                                 voxel=d.scaffold.voxel, dilate=d.scaffold.dilate)
            step_hook = (make_bin_masker(grid, num_bins, d.scaffold.apply_below_t)
                         if d.process == "d3pm"
                         else make_projector(grid, d.scaffold.apply_below_t))
        with torch.no_grad():
            sampled = model.generate(batch, n_steps=d.eval_steps, scaffold=step_hook)
```

The scaffold is per batch and not per run, so it cannot be built in `MeshSetEvalCallback.__init__`. Leave `self._scaffold = None` there; it now means "let `run_mesh_set_eval` decide", which is what the `None` default already does.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_mesh_scaffold.py tests/test_mesh_diffusion_smoke.py -v`
Expected: 7 + 17 = 24 passed

- [ ] **Step 6: Commit**

```bash
git add src/models/mesh_scaffold.py src/eval/mesh_set_eval.py tests/test_mesh_scaffold.py
git commit -m "feat(mesh-diff): LOD1 scaffold projection during reverse diffusion"
```

---

# Phase C — the experiment grid

### Task 14: Arm configs and a staged runner

The full cross product across all five axes is in the hundreds, most of it uninformative or illegal. This is a staged grid instead: **14 runs** that answer four questions in order, each stage fixing the previous stage's winner.

## The arms, explicitly

Every row is legal under `validate_combination`. `NFE` is forward passes per sampled mesh at `eval_steps: 50` — the column that makes a speed claim against the autoregressive branch honest, since an AR sample is ~1800 sequential passes.

**Stage 1 — denoiser and loss regime.** Held fixed: `state: continuous`, `process: ddpm`, `target: noise`, scaffold off.

| arm | `order` | `pos_embed` | `loss` | `denoiser` | question |
|---|---|---|---|---|---|
| `a1-unet-mse` | morton | sinusoidal | mse | unet | reference; closest to the trace-denoiser port |
| `a2-tf-mse` | morton | sinusoidal | mse | transformer | is the Morton sort's locality real, or is the convolution noise? |
| `a3-tf-hungarian` | none | none | hungarian | transformer | is a set formulation worth a Hungarian solve per sample? |

`a4-unet-hungarian` does not exist: the U-Net needs an order (D5), and sorted + Hungarian is the warn-but-legal cell — run it only if `a3` beats `a2` and you need to separate "set loss" from "no order".

**Stage 2 — state, corruption and readout.** This is the stage the revision opens up, and the one worth the most runs. Held fixed at stage 1's winner for `order` / `pos_embed` / `denoiser`; the table shows them at `a1`'s values, which is what ships.

| arm | `state` | `process` | `loss` | `target` | `transition` | on-grid out | question |
|---|---|---|---|---|---|---|---|
| `b1-x0` | continuous | ddpm | mse | original | — | no | is x0-prediction better conditioned than epsilon here? |
| `b2-flow` | continuous | flow | mse | velocity | — | no | do straight paths beat a strided DDPM at 50 NFE? |
| `b3-quant-mse` | quantized | ddpm | mse | original | — | no | **control**: does snapping the *target* alone buy anything, with the readout unchanged? |
| `b4-quant-ce` | quantized | ddpm | ce | original | — | **yes** | the reference's scheme on 9 coordinate channels with 128-way heads |
| `b5-onehot-ce` | onehot | ddpm | ce | original | — | **yes** | the reference's scheme exactly: Gaussian noise on 9x128 indicator channels |
| `b6-d3pm-uniform` | bins | d3pm | ce | original | uniform | **yes** | categorical corruption, nominal-alphabet transition |
| `b7-d3pm-gauss` | bins | d3pm | ce | original | gaussian | **yes** | categorical corruption, **ordinal**-alphabet transition (D11) |

The four `on-grid` arms all get exact welding — `snap_before_weld` becomes a no-op for them — so `n_verts` and `watertight_gen` should separate them cleanly from `b1`/`b2`/`b3`. That is the shared payoff; what the stage actually measures is which corruption process learns a 128-way coordinate posterior best.

Read `b3` against `b4` before reading either against anything else: `b3` and `b4` share an identical target and differ only in readout, so their gap **is** the value of the categorical head, with quantization held constant. Without `b3` you cannot tell whether `b4`'s numbers came from the snap or the softmax.

Read `b4` against `b5` for whether the one-hot state earns its 1153 input channels. Read `b6` against `b7` for D11's claim that ordinality matters; if `b7` does not beat `b6`, the ordinal argument is wrong for this data and that is a publishable-sized negative result on its own.

**Stage 3 — sampling-time levers.** On stage 2's winner.

| arm | change | question |
|---|---|---|
| `c1-scaffold` | `scaffold.enabled: true` | does the LOD1 legal region help, as the research scan predicts? |
| `c2-guidance` | `guidance: 3.0` | are samples plausible meshes but not plausible answers to *this* LOD1? |
| `c3-clamp-soft` | `x0_clamp: soft` | **only if stage 2's winner is `b4` or `b5`**: is the clamping trick load-bearing, or does a soft posterior mean do as well? |

`c3` is skipped when the winner is `b6`/`b7` — a categorical chain has no clamp to ablate — and when it is `b1`/`b2`/`b3`, which have no alphabet.

**Stage 4 — the comparison that matters.** No new training. Score the stage-3 winner and the five `mesh-v3`/`mesh-v4` autoregressive checkpoints on the same test indices with the same `mesh_metrics` call, and fill in Task 15's table. Include the LOD1 passthrough from `src/eval/lod1_baseline.py` as the floor.

Count: 3 + 7 + 2-or-3 + 0 = **12 to 13 training runs**, plus one scoring pass.

**Files:**
- Create: `configs/mesh-diff-base.yaml` and 8 arm configs
- Modify: `run_experiments.sh`
- Test: `tests/test_mesh_diffusion_compat.py` (add the config-sweep case)

**Interfaces:**
- Consumes: everything from Phases A and B.
- Produces: no Python API. The deliverable is a runnable queue.

- [ ] **Step 1: Write the failing test (append to `tests/test_mesh_diffusion_compat.py`)**

```python
def test_every_shipped_diffusion_config_is_a_legal_arm():
    """Every configs/mesh-diff-*.yaml must survive the gate.

    This is the cheapest possible guard against the failure the gate exists
    for: a config committed months ago, launched overnight, and rejected at
    startup after the GPU was already reserved.
    """
    from pathlib import Path
    from omegaconf import OmegaConf

    paths = sorted(Path("configs").glob("mesh-diff-*.yaml"))
    assert paths, "no mesh-diff configs found"
    for path in paths:
        cfg = OmegaConf.merge(OmegaConf.structured(Config),
                              OmegaConf.load(path))
        validate_combination(cfg)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mesh_diffusion_compat.py -k shipped -v`
Expected: FAIL, `AssertionError: no mesh-diff configs found` (only `mesh-diff-smoke.yaml` exists, and the glob will pick it up — if it passes on the smoke config alone, the assertion still holds; the test becomes meaningful once Step 3 lands)

- [ ] **Step 3: Write `configs/mesh-diff-base.yaml`**

Every arm config is this file plus a handful of overrides. Do not duplicate the shared block into each arm; each arm file carries only what it changes, and the runner passes both.

```yaml
# Shared settings for every mesh-diff arm. Not runnable alone -- it sets no
# mesh_diffusion axis. Merge an arm config over it:
#   python main.py --train --config configs/mesh-diff-base.yaml \
#       --config-overlay configs/mesh-diff-a1-unet-mse.yaml
#
# Dataset, split, effective batch, schedule and seed are identical across all
# arms and identical to the mesh-v3-*-sin autoregressive arms, so a diffusion
# number and an AR number on the same corpus are directly comparable. That
# comparability is the point of this file existing.
mode: train
config_set: mesh_diffusion
seed: 42

data:
  dataset_dir: data/The Hague/mini_cleaner

mesh_data:
  dataset_dir: data/The Hague/mini_cleaner
  lod_in: LOD1
  lod_out: LOD2
  num_bins: 128
  margin_lo: [0.0, 0.0, 0.0]
  margin_hi: [0.0, 0.0, 0.1]
  # 200, matching every mesh-v3-*-sin arm. Here it is purely a corpus filter:
  # nothing in this branch has a position table, and the batch pads to its own
  # maximum, so max_faces decides only which buildings are in the dataset.
  max_faces: 200
  max_files: null
  num_workers: 10
  persistent_workers: false

mesh_diffusion:
  d_model: 256
  n_head: 8
  num_layers: 8
  dropout: 0.1
  time_dim: 128
  cond_dropout: 0.1
  noise_steps: 1000
  eval_steps: 50
  guidance: 1.0
  presence_weight: 1.0
  match_presence_weight: 1.0
  snap_before_weld: true
  min_face_area: 1.0e-6
  scaffold:
    enabled: false
    voxel: 0.03125
    dilate: 3
    apply_below_t: 0.5
  every_n_epochs: 10
  n_val: 16
  n_test: 64
  eval_batch_size: 8
  n_points: 4096
  taus: [0.25, 0.5]
  voxel_m: 0.25
  save_samples: 8

training:
  batch_size: 8
  accumulate_grad_batches: 2   # effective batch 16, matching the AR arms
  train_val_test_split: [0.8, 0.1, 0.1]
  # 1e-4 flat. Not the AR arms' 5e-5: that value was chosen for a from-scratch
  # decoder at 1e-3-scale gradients, and a diffusion denoiser regressing a
  # bounded target is a different optimisation. 1e-4 is the standard DDPM
  # starting point and is the value to change first if training is unstable.
  lr: 1.0e-4
  max_epochs: 400
  gradient_clip_val: 1.0
  lr_scheduler: "none"
  lr_decay_steps: 50
  lr_decay_rate: 0.5
  # val_loss, NOT the geometric metrics: those exist only every 10 epochs, so
  # they cannot be monitored. And NOT val_token_acc -- there are no tokens on
  # this branch. Read the mode: min, because this loss is a regression error.
  early_stopping:
    enabled: true
    monitor: val_loss
    patience: 25
    mode: min
  checkpoint:
    enabled: true
    monitor: val_loss
    mode: min
    save_top_k: 3
    filename: "{epoch:02d}-{val_loss:.4f}"

trainer:
  accelerator: auto
  devices: 1
  precision: "32"
  log_every_n_steps: 50

logging:
  loggers:
    - wandb
  project_name: lod-generation
  experiment_name: initial-runs
  save_dir: outputs
  wandb_entity: maxim-lod
  wandb_offline: false
  log_model: false

resume_from: null
```

- [ ] **Step 4: Write the arm configs**

Each is a short overlay over `mesh-diff-base.yaml`, carrying only the axes it sets. Create all thirteen. Every file's `mesh_diffusion:` block is given in full below — nothing here is a placeholder, and an unedited stage-2 or stage-3 launch is a valid run (it just answers the question with `a1`'s stage-1 settings rather than the measured winner).

**Stage 1.** `configs/mesh-diff-a1-unet-mse.yaml`:

```yaml
# Stage 1, arm 1: sorted faces, slot-to-slot MSE, conv U-Net.
# The closest thing to the trace-denoiser architecture this branch is ported
# from, and the reference every other arm is read against.
mesh_diffusion: {order: morton, pos_embed: sinusoidal, loss: mse,
                 denoiser: unet, state: continuous, process: ddpm, target: noise}
logging: {run_name: mesh-diff-a1-unet-mse}
```

`configs/mesh-diff-a2-tf-mse.yaml`:

```yaml
# Stage 1, arm 2: same data, same loss, transformer instead of the U-Net.
# Isolates the denoiser. If the U-Net wins, the Morton sort is carrying real
# locality; if the transformer wins, the convolution was reading noise.
mesh_diffusion: {order: morton, pos_embed: sinusoidal, loss: mse,
                 denoiser: transformer, state: continuous, process: ddpm, target: noise}
logging: {run_name: mesh-diff-a2-tf-mse}
```

`configs/mesh-diff-a3-tf-hungarian.yaml`:

```yaml
# Stage 1, arm 3: unordered faces, permutation-equivariant transformer, set
# loss. The honest formulation -- a mesh is a set -- at the cost of a Hungarian
# solve per sample. Against a2 it isolates order+loss with the denoiser fixed.
# No U-Net counterpart exists: the convolution needs an order (plan D5).
mesh_diffusion: {order: none, pos_embed: none, loss: hungarian,
                 denoiser: transformer, state: continuous, process: ddpm, target: noise}
logging: {run_name: mesh-diff-a3-tf-hungarian}
```

**Stage 2.** All seven ship with stage 1 set to `a1`'s axes; edit `order` / `pos_embed` / `denoiser` in all seven together once stage 1 has been read.

`configs/mesh-diff-b1-x0.yaml`:

```yaml
# Stage 2, arm 1: x0-prediction instead of epsilon. Better conditioned at low
# noise, and the parameterisation every set-based generative model uses.
mesh_diffusion: {order: morton, pos_embed: sinusoidal, loss: mse, denoiser: unet,
                 state: continuous, process: ddpm, target: original}
logging: {run_name: mesh-diff-b1-x0}
```

`configs/mesh-diff-b2-flow.yaml`:

```yaml
# Stage 2, arm 2: conditional flow matching. Straight paths, no beta schedule,
# and few-step sampling that degrades gracefully -- which matters because
# eval_steps IS the eval budget.
mesh_diffusion: {order: morton, pos_embed: sinusoidal, loss: mse, denoiser: unet,
                 state: continuous, process: flow, target: velocity}
logging: {run_name: mesh-diff-b2-flow}
```

`configs/mesh-diff-b3-quant-mse.yaml`:

```yaml
# Stage 2, arm 3: THE CONTROL. Target snapped to the 128 grid, readout still a
# regression. Identical target to b4, identical readout to b1. Its gap to b4 is
# the value of the categorical head with quantization held constant; without
# this arm you cannot tell whether b4's numbers came from the snap or the
# softmax. Output still leaves the grid, so welding still needs snap_before_weld.
mesh_diffusion: {order: morton, pos_embed: sinusoidal, loss: mse, denoiser: unet,
                 state: quantized, process: ddpm, target: original}
logging: {run_name: mesh-diff-b3-quant-mse}
```

`configs/mesh-diff-b4-quant-ce.yaml`:

```yaml
# Stage 2, arm 4: Gaussian noise, categorical readout, on the nine coordinate
# channels. This is what maxim-mat/trace-denoise-refactor actually does --
# `noise_data` is Gaussian, the loss is CrossEntropy on argmax -- ported to
# coordinates rather than one-hot. Output lands on the grid, so welding is
# exact and snap_before_weld is a no-op.
mesh_diffusion: {order: morton, pos_embed: sinusoidal, loss: ce, denoiser: unet,
                 state: quantized, process: ddpm, target: original, x0_clamp: hard}
logging: {run_name: mesh-diff-b4-quant-ce}
```

`configs/mesh-diff-b5-onehot-ce.yaml`:

```yaml
# Stage 2, arm 5: the reference's scheme exactly -- Gaussian noise on 9 x 128
# indicator channels. 1153 input channels; one 1x1 conv absorbs them, and the
# activations are ~7 MB at F=200, B=8. Against b4 this asks whether diffusing
# in the simplex-like space earns its width over diffusing the scalar.
mesh_diffusion: {order: morton, pos_embed: sinusoidal, loss: ce, denoiser: unet,
                 state: onehot, process: ddpm, target: original, x0_clamp: hard}
logging: {run_name: mesh-diff-b5-onehot-ce}
```

`configs/mesh-diff-b6-d3pm-uniform.yaml`:

```yaml
# Stage 2, arm 6: categorical corruption with a uniform transition -- the
# standard D3PM setting, and the right one for a NOMINAL alphabet.
# Watch val_bin_acc_t3 (the high-noise bucket) from epoch 1.
mesh_diffusion: {order: morton, pos_embed: sinusoidal, loss: ce, denoiser: unet,
                 state: bins, process: d3pm, target: original, transition: uniform}
logging: {run_name: mesh-diff-b6-d3pm-uniform}
```

`configs/mesh-diff-b7-d3pm-gauss.yaml`:

```yaml
# Stage 2, arm 7: categorical corruption with a discretized-Gaussian
# transition (Austin et al. arXiv:2107.03006 section 3.2). Coordinate bins are
# ORDINAL -- bin 40 and 41 are half a centimetre apart -- so this is the
# better-matched process, and b6 vs b7 is the test of that claim (plan D11).
# If b7 does not beat b6, the ordinality argument is wrong for this data, which
# is a result rather than a failure.
mesh_diffusion: {order: morton, pos_embed: sinusoidal, loss: ce, denoiser: unet,
                 state: bins, process: d3pm, target: original,
                 transition: gaussian, transition_sigma: 16.0}
logging: {run_name: mesh-diff-b7-d3pm-gauss}
```

**Stage 3.** All three ship at `a1`+`b1` axes; edit to stage 2's winner before launching.

`configs/mesh-diff-c1-scaffold.yaml`:

```yaml
# Stage 3, arm 1: scaffold masking on. MeshWeaver's fix (3) -- the research
# scan's top transfer candidate -- as a projection onto the dilated LOD1
# occupancy at every reverse step below t = 0.5.
mesh_diffusion:
  order: morton
  pos_embed: sinusoidal
  loss: mse
  denoiser: unet
  state: continuous
  process: ddpm
  target: noise
  scaffold: {enabled: true, dilate: 3, apply_below_t: 0.5}
logging: {run_name: mesh-diff-c1-scaffold}
```

`configs/mesh-diff-c2-guidance.yaml`:

```yaml
# Stage 3, arm 2: classifier-free guidance at 3.0. Doubles eval cost (two
# forward passes per reverse step) and is the standard lever when samples are
# plausible meshes but not plausible answers to THIS LOD1. Needs
# cond_dropout > 0, which the base config already sets.
mesh_diffusion: {order: morton, pos_embed: sinusoidal, loss: mse, denoiser: unet,
                 state: continuous, process: ddpm, target: noise, guidance: 3.0}
logging: {run_name: mesh-diff-c2-guidance}
```

`configs/mesh-diff-c3-clamp-soft.yaml`:

```yaml
# Stage 3, arm 3: CONDITIONAL. Run only if stage 2's winner is b4 or b5 -- the
# arms where a continuous process reads out categorically and the clamping
# trick (Li et al. arXiv:2205.14217 section 4.2) is what keeps samples on the
# grid. `soft` replaces the argmax one-hot with the posterior mean, so output
# leaves the grid and snap_before_weld starts mattering again. If this scores
# the same as b4/b5, the trick is not load-bearing here.
# Skip entirely when the winner is b6/b7 (no clamp exists in a categorical
# chain) or b1/b2/b3 (no alphabet).
mesh_diffusion: {order: morton, pos_embed: sinusoidal, loss: ce, denoiser: unet,
                 state: quantized, process: ddpm, target: original, x0_clamp: soft}
logging: {run_name: mesh-diff-c3-clamp-soft}
```

- [ ] **Step 5: Confirm `main.py` supports config overlays, or drop to a single-file form**

`src/utils/initialization.py::load_config` takes one `--config` plus dotted overrides; it has no `--config-overlay`. **Do not add one.** Instead the runner passes the arm's axes as dotted overrides on the command line, which `load_config` already supports:

Verify:

Run: `python main.py --train --config configs/mesh-diff-smoke.yaml mesh_diffusion.denoiser=transformer mesh_diffusion.order=none mesh_diffusion.pos_embed=none mesh_diffusion.loss=hungarian training.max_epochs=1`
Expected: the run starts and logs `Arm: order=none pos_embed=none loss=hungarian denoiser=transformer ...`

If that works, the eight arm files above are documentation of the grid and the runner is the thing that executes it. Keep the files — they are how an arm is reproduced six months from now — and have the runner read the axes out of each file with a two-line merge rather than inventing a CLI flag:

```bash
python - "$cfg" <<'PY' > /tmp/mesh_diff_merged.yaml
import sys
from omegaconf import OmegaConf
print(OmegaConf.to_yaml(OmegaConf.merge(
    OmegaConf.load("configs/mesh-diff-base.yaml"), OmegaConf.load(sys.argv[1]))))
PY
python main.py --train --config /tmp/mesh_diff_merged.yaml
```

- [ ] **Step 6: Extend `run_experiments.sh`**

Add a second queue and a selector, leaving the existing `DEFAULT_CONFIGS` untouched so the AR arms still launch the way they do today:

```bash
# The mesh-diffusion grid, in stages. Stage 1 decides denoiser and loss;
# stage 2 decides the objective on stage 1's winner; stage 3 adds the two
# sampling-time levers. Run one stage, read it, then edit the next stage's
# configs to match the winner -- they ship set to arm a1's axes, not to a
# placeholder, so an unedited stage-2 launch is a valid run rather than a
# crash. It just answers a slightly different question.
DIFF_STAGE1=(
  configs/mesh-diff-a1-unet-mse.yaml
  configs/mesh-diff-a2-tf-mse.yaml
  configs/mesh-diff-a3-tf-hungarian.yaml
)
# Cheapest first inside each stage, so a config-level mistake surfaces before
# the long runs. Within stage 2 that means the continuous arms before the
# 1153-channel one-hot arm.
DIFF_STAGE2=(
  configs/mesh-diff-b1-x0.yaml
  configs/mesh-diff-b2-flow.yaml
  configs/mesh-diff-b3-quant-mse.yaml
  configs/mesh-diff-b4-quant-ce.yaml
  configs/mesh-diff-b6-d3pm-uniform.yaml
  configs/mesh-diff-b7-d3pm-gauss.yaml
  configs/mesh-diff-b5-onehot-ce.yaml
)
# c3 is conditional on stage 2's winner being b4 or b5 -- see its config header.
DIFF_STAGE3=(
  configs/mesh-diff-c1-scaffold.yaml
  configs/mesh-diff-c2-guidance.yaml
  configs/mesh-diff-c3-clamp-soft.yaml
)
```

and a `--diff-stage N` option that selects one of the three and routes each config through the base-merge above. Keep `--keep-going` on by default, for the same reason it already is: these are independent arms and a failure in one must not cost the others their overnight slot.

- [ ] **Step 7: Smoke every arm before committing GPU time**

Run: `SMOKE=1 ./run_experiments.sh --diff-stage 1` then stages 2 and 3.
Expected: all eight configs resolve, build a model, and complete one epoch on 20 files.

- [ ] **Step 8: Run tests**

Run: `python -m pytest tests/ -q`
Expected: the whole suite passes, including `test_every_shipped_diffusion_config_is_a_legal_arm` over all nine `mesh-diff-*.yaml` files.

- [ ] **Step 9: Commit**

```bash
git add configs/mesh-diff-*.yaml run_experiments.sh tests/test_mesh_diffusion_compat.py
git commit -m "feat(mesh-diff): staged experiment grid and runner"
```

---

### Task 15: Research note

The plan produces runs; this produces the thing that makes them readable next to the autoregressive branch.

**Files:**
- Create: `docs/research/2026-08-25-mesh-set-diffusion-notes.md`

- [ ] **Step 1: Write the note**

It must contain, and nothing else is required:

1. **The framing gap.** No paper in the 2026-08-24 scan does non-autoregressive whole-mesh generation for buildings. State that plainly, and state that it means nobody has de-risked it either.
2. **What the scan does support**, with the numbers: BuildAnyPoint section 3.1 (CD 0.107 -> 0.034, 127 -> 70 faces, a diffusion prior on The Hague/Rotterdam LoD2); MeshWeaver limitation (ii), which names prefix-conditioned autoregression — the mesh-v3 branch — as the thing to fix, and cross-attention at every layer as the fix this branch implements by construction.
3. **What becomes irrelevant**, which is a real result and should be stated as one: the entire compression-ratio table (section 2, section 3.3), beam search and backtracking (section 3.4), `mask_invalid`, and exposure bias. None of them has a counterpart here.
4. **What this branch gives up**, stated as plainly: connectivity is not modelled, so vertex sharing is reconstructed by welding rather than generated; face count is a learned threshold on a presence channel rather than an EOS token; and there is no equivalent of the AR branch's exact tokenizer inverse — `gt_chamfer_m` is the ceiling and it is not zero.
5. **The comparison table to fill in**, with the AR baseline rows already present: `mesh-v3-coord-sin`, `mesh-v3-amt-sin`, `mesh-v3-opt-coord-sin`, `mesh-v3-opt-amt-sin`, `mesh-v4-scratch-sin`, and the LOD1 passthrough baseline from `src/eval/lod1_baseline.py`. Columns: `chamfer_m`, `gt_chamfer_m`, `fscore_25cm`, `vol_iou`, `n_faces`, `watertight_gen`, wall-clock per epoch, and NFEs per sample. That last column is the one that makes the comparison fair: an AR sample is ~1800 sequential forward passes, a 50-step diffusion sample is 50, and any speed claim that omits it is not a claim.
6. **The open confound**, restated: the dataset issue recorded in the mesh-v3 analysis gates this branch exactly as it gates scaffold masking and BPT indexing in the scan's section 4. A diffusion arm is a larger change than either, so it inherits the gate rather than escaping it.

- [ ] **Step 2: Commit**

```bash
git add docs/research/2026-08-25-mesh-set-diffusion-notes.md
git commit -m "docs: research note framing the mesh-set diffusion branch"
```

---

## Self-Review

Run before declaring the plan done. Findings are fixed inline, not re-reviewed.

**Spec coverage.** Every item the user named maps to a task: canonical order with PE (Tasks 1, 4, 9), Hungarian loss (Task 3), U-Net with attention (Task 4), pure transformer (Task 9), post-hoc welding (Task 7), discrete noise schedule on the quantized grid (Task 12, both transitions), Gaussian noise with a categorical readout as the reference actually does it (Task 11), x0 and epsilon prediction (Task 5), flow matching (Task 10), within-batch padding to a U-Net-processable length (Task 1, D6), scaffold integrated into sampling (Task 13), mix-and-match experiments (Task 14, 13 explicit arms in four stages). The research-doc question is answered in Task 15.

**Type consistency.** `mask` is `[B, F]` bool with True = real everywhere; `x` is `[B, 10, F]` in every module except `DiscreteProcess`, whose state is `[B, 9, F]` int64 and whose `sample` returns a tuple — the one asymmetry, handled explicitly in `MeshDiffusionModule.generate` (Task 12 Step 4). `faces_to_mesh` accepts both `[10, F]` and `[F, 10]` and is tested for it. Both denoisers share `forward(x, t, cond, mask, cond_mask)` and both return `[B,10,F]` or `(logits, presence)`.

**Test inventory.** 95 test functions across 8 files: `test_mesh_set_dataset` 10, `test_mesh_diffusion_compat` 14, `test_mesh_set_losses` 9, `test_mesh_set_denoisers` 12, `test_mesh_processes` 19 (20 collected, one parametrized), `test_mesh_diffusion_smoke` 17, `test_mesh_scaffold` 7, `test_mesh_set_postprocess` 7. Every `Expected: N passed` line in the plan is the running total for that file at that point, not the delta.

**Known gaps, deliberately left.**
- `run_experiments.sh` Step 6 describes the selector rather than showing it in full; the config arrays and the merge snippet are shown, the argument parsing is mechanical and follows the existing `--smoke` handling in that file.
- Task 15's note is specified by content, not written out — it depends on numbers that do not exist until the runs finish.







