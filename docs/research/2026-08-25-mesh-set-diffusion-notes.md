# Mesh-set diffusion: what this branch is for, and what it gives up

Companion to `docs/superpowers/plans/2026-08-25-mesh-set-diffusion.md` (the
implementation) and `docs/research/2026-08-24-preliminary-future-plan.md` (the
scan this reads against). Written when the code landed, before any arm had run
to convergence — the comparison table in section 5 is the deliverable this note
exists to be filled into.

## 1. The framing gap

**No paper in the 2026-08-24 scan does non-autoregressive whole-mesh generation
for buildings.** Every mesh generator in section 1 of that scan — MeshAnything
V2, BPT, TreeMeshGPT, Meshtron, ARMesh, MeshWeaver, and BuildAnyPoint's
downstream stage — is autoregressive over a token sequence. The diffusion models
in the building literature operate on point clouds, voxels, or graphs, and hand
their output to an AR mesher.

That means nobody has de-risked this. There is no reported number to beat, no
architecture to port, and no published failure to learn from. The upside is that
a positive result is genuinely new; the honest downside is that the usual
reassurance — "this works elsewhere, it should work here" — is not available,
and the arms in section 4 have to earn their conclusions from this corpus alone.

## 2. What the scan does support

Two findings survive the change of paradigm, and they are why this branch exists
rather than a fourth AR variant.

**BuildAnyPoint (arXiv:2602.23645), section 3.1 of the scan.** Their ablation
holds the mesh generator fixed (MeshAnything V2) and changes only what it is
conditioned on:

| conditioning | #V | #F | failure | CD |
|---|---|---|---|---|
| raw input point cloud | 78 | 127 | <1% | 0.107 |
| diffusion-recovered point cloud | 38 | 70 | 0% | **0.034** |

Three times the chamfer accuracy and half the faces, from inserting a *diffusion
prior over building geometry* ahead of the mesher. Our LOD1 condition is already
clean, so their exact pipeline does not transfer — but this is the largest
measured effect in the building-mesh literature, and it is attributable to a
generative prior regularising an ill-posed step. This branch asks whether that
prior can produce the mesh itself instead of a point cloud for something else to
mesh.

**MeshWeaver (arXiv:2606.04688), section 3.2 of the scan.** Its limitation (ii)
names *our* mesh-v3 architecture: generation conditioned on a global shape
embedding, with static vocabulary embeddings and no local geometric context.
Their fix is cross-attention to local geometry at every prediction step. This
branch implements that by construction — `CrossAttention1d` at every U-Net level
and in every transformer block, never a prefix and never an elementwise sum
(plan D1, and the module docstring in `src/models/mesh_set_modules.py`). Their
fix (3), scaffold masking, ships as `src/models/mesh_scaffold.py`.

## 3. What becomes irrelevant — and that is a result

A substantial fraction of the AR branch's engineering has no counterpart here.
This is worth stating positively rather than as an omission, because it is the
clearest measure of how different the two branches are:

- **The entire compression-ratio table** (scan sections 2 and 3.3). Sequence
  length is not a cost in this branch — a mesh is a `[F, 10]` array, and F is
  the face count, not 9× it. `coord` vs `amt` (1.00 vs 0.53 on our corpus), BPT
  block indexing, Mesh-Silksong, EdgeRunner: none of them is a lever here,
  because there is no sequence to compress.
- **Beam search and constrained decoding with backtracking** (scan 3.4, ARMesh).
  There is no left-to-right decision to search over or to back out of.
- **`mask_invalid`.** The grammar it enforces — which token may follow which —
  does not exist when the whole mesh is emitted at once.
- **Exposure bias.** There is no teacher forcing, so there is no train/test
  mismatch of that kind. This is the branch's structural answer to the second of
  the three causes recorded in the levi-1 diagnosis.
- **KV caching.** Nothing is decoded incrementally.

## 4. What this branch gives up

Stated as plainly as the above, because these are real and they are not
temporary:

- **Connectivity is not modelled.** The representation is a face *set* with
  vertices repeated per face (plan D1). Vertex sharing is reconstructed
  post-hoc by welding, not generated. The AR tokenizers exist precisely to solve
  connectivity, and this branch dodges the problem rather than solving it.
- **Welding needs a snap to fire at all.** Two continuous floats never coincide,
  so `weld` merges nothing without `snap_before_weld` (plan D8). The four
  on-grid arms (b4, b5, b6, b7) get exact welding for free; the continuous ones
  are relying on a quantization step to recover a guarantee the AR branch has by
  construction.
- **Face count is a learned threshold, not an EOS token.** It is the sign of a
  diffused presence channel (plan D2). A miscalibrated presence head produces a
  mesh with the wrong number of faces and no error signal saying so — which is
  why `n_faces`, `n_verts` and `empty` are logged alongside chamfer.
- **There is no exact tokenizer inverse.** The AR branch's `coord` tokenizer
  round-trips exactly; here the ground truth pushed through `faces_to_mesh` is
  itself lossy. `gt_chamfer_m` is that ceiling, it is measured every eval, and
  **it is not zero** — on the 20-file smoke corpus it sits at ~0.016 m. Every
  generated number must be read against that row, never against 0.

## 5. The comparison table

To be filled in after stage 4 of the grid (no new training: score the stage-3
winner and the AR checkpoints on the same test indices with the same
`mesh_metrics` call). The AR rows and the passthrough floor are named here so
the comparison cannot quietly drop the unflattering ones.

`NFE` is the column that makes any speed claim honest: an autoregressive sample
is ~1800 *sequential* forward passes, a 50-step diffusion sample is 50 (100 with
classifier-free guidance on). A wall-clock claim that omits it is not a claim.

| model | chamfer_m | gt_chamfer_m | fscore_25cm | vol_iou | n_faces | watertight_gen | s/epoch | NFE/sample |
|---|---|---|---|---|---|---|---|---|
| LOD1 passthrough (`src/eval/lod1_baseline.py`) | | — | | | | | — | 0 |
| mesh-v3-coord-sin | | | | | | | | ~1800 |
| mesh-v3-amt-sin | | | | | | | | ~950 |
| mesh-v3-opt-coord-sin | | | | | | | | ~1800 |
| mesh-v3-opt-amt-sin | | | | | | | | ~950 |
| mesh-v4-scratch-sin | | | | | | | | ~1800 |
| mesh-diff (stage-3 winner) | | | | | | | | 50 |

## 5b. File order already carries most of the locality the U-Net needs

The plan excluded a conv U-Net over unordered faces (its D5) on the grounds
that "with `order: none` the axis is arbitrary and the convolution is noise".
Measured on 150 `mini_cleaner` LOD2 buildings, normalized to the unit box, that
premise is false:

| face order | mean consecutive-centroid step | consecutive pairs sharing a corner |
|---|---|---|
| file order | 0.3618 | 72.4% |
| Morton | 0.3161 | 78.5% |
| random permutation | 0.5125 | 40.9% |

CityJSON groups faces by surface, so the triangles of one wall arrive
consecutively. File order sits far closer to Morton than to a random
permutation; the sort buys ~13% on the step and ~6 points of adjacency, not the
difference between signal and noise.

The image analogy that prompted this is worth stating precisely, because it is
mostly right. An image *does* have a canonical ordering — row-major on a grid —
it is simply free, because the array index is a bijection with spatial
position. A face set has no such structure, so any 1-D convolution over it
requires serialising a set, and that serialisation is a choice. But the choice
already made by the file is not arbitrary.

That separates two properties the plan conflated. **Locality** — do adjacent
indices mean adjacent geometry? File order: yes. **Canonicality** — is the order
a function of the geometry rather than of the writer? File order: no. The
convolution needs the first; only the slot-to-slot *loss* needs the second.

Consequences, all shipped: D5 is now a warning rather than a rejection;
`a4-unet-hungarian` joins stage 1 and prices canonicality directly against a1;
and the `pos_embed` half of D4 was narrowed to the transformer, since
`ConditionalMeshUNet` has no positional input at all — a convolution is
translation-equivariant along the face axis, so requiring `pos_embed` of it was
requiring a no-op. Setting it under `denoiser: unet` now warns.

## 5a. Two things the grid design gets wrong, and what was done about them

Recorded because both are properties of the *experiment*, not of the model, and
both would otherwise be invisible in the results.

**The grid is greedy, and the axes are not separable.** Stage 1 ranks
`(denoiser, loss)` under exactly one objective — `continuous`/`ddpm`/`ε` — and
stage 2 assumes that ranking transfers to six others. Nothing guarantees it.
One failure is structural and certain: `loss: ce` is scored slot-to-slot, so it
requires `order: morton` (plan D4), and the unordered set regime therefore
**cannot reach b4–b7 at all**. If a3 wins stage 1, stage 2 either abandons the
winner or loses its four on-grid arms. Two more are plausible but unmeasured:
the denoiser comparison is made at 10 input channels and applied at 1153 (b5),
and `eval_steps: 50` is fixed for every arm although it is the entire eval
budget and the processes degrade differently under it.

Mitigation is a **revisit pass** (`--diff-stage revisit`), not a full factorial:
re-run stage 1's alternatives under stage 2's winner. Two runs against the ~8 a
full cross would add. `r2-loss` is conditional on the winner's loss not being
`ce`, for the reason above.

**`presence_weight: 1.0` did not mean the same thing across arms.** The two
loss terms start at scales set by the target and the readout. Measured at init
on `mini_cleaner`, with the zero-init head predicting 0 and E[x0²] = 0.0958:

| regime | coord | presence | presence/coord | weight for parity |
|---|---|---|---|---|
| `target: noise` | 1.0000 | 1.0000 | 1.000 | 1.000 |
| `target: original` | 0.0958 | 0.2500 | 2.610 | 0.383 |
| `target: velocity` | 1.0958 | 1.2500 | 1.141 | 0.877 |
| `loss: ce` | ln 128 = 4.852 | ln 2 = 0.693 | 0.143 | 7.000 |

Left flat, **b3 and b4 — the grid's cleanest control, sharing a target and
differing only in readout — would have differed by 18× in effective presence
weight**, landing on face count, which is what the geometric metrics are most
sensitive to. Every arm now sets its regime's parity value;
`test_shipped_configs_equalise_presence_weight` pins it. `target: noise` is
already at parity, so the a-block and c1/c2 are unchanged at 1.0. Re-derive the
0.0958 if the corpus changes.

## 6. The open confound

The dataset issue recorded in the mesh-v3 analysis gates this branch exactly as
it gates scaffold masking and BPT block indexing in section 4 of the scan: an
A/B against a moving corpus is unreadable. A diffusion arm is a *larger* change
than either of those, not a smaller one, so it inherits that gate rather than
escaping it. Every number this branch produces before the corpus is settled is a
plumbing check, not a result — including the smoke numbers quoted in section 4.

## 7. Two things worth watching from epoch 1

- **`val_bin_acc_t3`** on the b6/b7 pair. Plan D11 argues that coordinate bins
  are an *ordinal* alphabet, unlike the Levi branch's nominal class labels, and
  that a discretized-Gaussian transition is therefore the better-matched
  corruption. The high-noise bucket is where that claim lives. If b7 does not
  beat b6, the ordinality argument is wrong for this data — which is a result,
  not a failure, and the reason both transitions ship rather than one default.
- **`n_verts` against `n_faces`.** A sample with a plausible chamfer and three
  times the vertices it should have is a welding failure, not a geometry
  failure, and chamfer alone will not distinguish them.
