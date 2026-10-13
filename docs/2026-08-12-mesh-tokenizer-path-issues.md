  # Known issues on the mesh tokenizer data path

Date: 2026-08-12
Status: preferred direction is the MeshAnything V2 transition (last section),
which supersedes Issues 1 and 3. Issue 2 closed (faithful to V1), Issue 4 open.
Scope: `src/dataset/mesh_dataset.py`, `src/dataset/mesh_vqvae_datamodule.py`,
`src/models/mesh_vqvae.py`, `src/models/mesh_transformer.py`

Findings from tracing the full path, source to decoded mesh:

```
CityJSON -> parse_cityjson_file_to_meshes -> normalize_to_unit_box (LOD1 frame)
  -> quantize (128 bins) -> canonicalize -> 9 tokens/face
  -> [stage 1] _FaceView            -> MeshVQVAE encode/quantize -> codes
  -> [stage 2] MeshTransformerModule._prepare -> codes -> transformer
  -> decode: codes -> per-coordinate logits -> argmax -> canonicalize -> dequantize
```

Reference checked throughout: MeshAnything (Chen et al., arXiv:2406.10163) and
its released V1 code at `buaacyw/MeshAnything`.

## Not an issue: `MeshDataset.__getitem__` re-tokenizing every access

Recorded so it does not get re-litigated. The transform is a pure function of
`self.pairs[i]` (deterministic numpy: `np.unique`, `lexsort`, no RNG, no state),
so re-running it per epoch cannot drift the targets. Measured on
`data/The Hague/mini`, 20041 buildings, `max_faces=200`:

| check | result |
| --- | --- |
| `__getitem__` deterministic over 200 items | True |
| coordinate `tokenize` -> `detokenize` round trip | 0 / 200 mismatches |
| stage-1 `_FaceView` coords == stage-2 `_faces` coords | 32 / 32 identical |
| codes for a mesh, solo vs inside a padded batch | 0.00% disagreement |

The third row is the one that mattered: the two stages reach `[F, 9]` by
different code paths (`_FaceView.reshape(-1, 9)` vs `_faces()` with its
clamp/trim/pad), and they agree bit for bit, so the VQ-VAE is used on exactly
what it was trained on.

Determinism is *load-bearing*, not incidental — see the `train()` override at
`src/models/mesh_transformer.py:373`. Lightning's per-epoch `model.train()` was
re-enabling VQ-VAE dropout, and because `_prepare` runs the tokenizer inside
`training_step`, that resampled the target labels every epoch while validation
scored clean ones. Anything that makes this path stochastic reintroduces that
bug. Keep it deterministic or precompute.

### `detokenize` is not on the VQ-VAE's train or encode path

A reasonable worry is that the dataset's fixed scheme has to be *undone* before
the VQ-VAE can learn its own, making the coordinate tokenizer pure overhead.
It does not. Instrumenting both `detokenize` functions (including the copy
`mesh_transformer` imports into its own namespace) and counting calls:

| path | coord `detokenize` | `MeshVQVAE.detokenize` |
| --- | --- | --- |
| stage-1 `training_step` (fwd + bwd) | 0 | 0 |
| stage-2 `_prepare` | 0 | 0 |
| `decode_tokens`, one mesh | 0 | 1 |

There is nothing to undo because the VQ-VAE consumes bin ids natively:
`coord_embed = nn.Embedding(num_bins, d_model)` indexes the dataset's integers
directly, and `head = Linear(d_model, 9 * num_bins)` emits logits over the same
grid. Stage-1 batches are `int64` in `[0, 127]`, never metres.

The two schemes are stacked, not competing: the dataset fixes the *alphabet*
(continuous coordinates to 128 symbols per axis, plus canonical ordering); the
VQ-VAE learns a *vocabulary* over that alphabet. MeshGPT and MeshAnything both
discretize before the autoencoder for the same reason.

One real constraint does follow from the fixed scheme, short of a bug: the
canonical *face order* is imposed, and `encode` adds `pos_embed(arange(F))`, so
face position is meaningful input the VQ-VAE cannot learn around. Matches the
reference; noted in case set-invariance ever becomes desirable.

---

## Issue 1 — the VQ-VAE decoder cracks meshes (open, highest value)

`MeshVQVAE.quantize` gathers one code per *unique* vertex, so every face
touching a vertex carries identical codes — this holds, 300/300 buildings. But
`MeshVQVAE.decode` predicts 9 coordinate logits per face **independently**, so
nothing forces two faces to argmax onto the same grid point. `detokenize` then
merges vertices by exact equality (`canonicalize`), and split vertices do not
merge — the mesh opens at the seam.

Measured, 300 random buildings, checkpoint
`outputs/initial-runs/mesh-vqvae-4/checkpoints/last.ckpt`:

| signal | value |
| --- | --- |
| buildings where the decoder splits >=1 shared vertex | 21% |
| vertices split, averaged over buildings | 1.7% |
| vertex inflation after round trip | mean 1.020x, p50 1.00, p95 1.13, max 1.28 |
| buildings returning watertight | 81% |

**This is a floor, not a generation error.** The measurement fed the decoder
*ground-truth* codes straight from `MeshVQVAE.tokenize`. So even with a perfect
stage-2 transformer, 19% of buildings come back cracked. The coordinate
tokenizer has no such floor — its round trip is exact (0/200 mismatches), and
any non-watertightness there comes from the model predicting wrong geometry, not
from the representation.

The docstring on `MeshVQVAE.quantize` claims the gather means "`canonicalize`'s
exact-equality merge cannot crack the mesh". That is true of the codes and false
of the decoder output; correct it when fixing this.

### Why the decoder can contradict its own codes

The encoder pools across faces — `quantize` does a `scatter_reduce` mean over
every face touching a vertex, so vertex identity is enforced *going in*. The
decoder has no counterpart: `face_proj_down` concatenates three vertex features
back into one face token, and the output head `Linear(d_model, 9 * num_bins)`
scores each face's nine coordinates independently. Vertex sharing is asserted by
the codes and merely *hoped for* at the output.

The `ponytail:` comment already on `src/models/mesh_vqvae.py` predicted exactly
this — "a linear split, not MeshGPT's cross-face vertex sharing — add that if
vertices shared between faces start disagreeing visibly." They have. This debt
item is now due.

### What the reference does

MeshAnything does **not** solve this in the model. `NoiseResistantDecoder`
returns a triangle soup — `continuous_coors` is `[b, nf, 3, 3]`, per-face
vertices with no topology at all. The weld happens in `main.py`, outside the
network:

```python
vertices  = recon_mesh.reshape(-1, 3)            # fully exploded, 3 verts per face
triangles = np.arange(len(vertices)).reshape(-1, 3)
scene_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, ...)
scene_mesh.merge_vertices()
scene_mesh.update_faces(scene_mesh.unique_faces())
scene_mesh.fix_normals()
```

`trimesh.merge_vertices()` defaults to `tol.merge = 1e-8`, and undiscretized
coordinates are multiples of 1/128 ~ 0.0078, so it is an **exact** merge in
effect — the same rule `canonicalize` already applies here. The reference would
crack the same way; its metrics (Chamfer, Edge Chamfer, Normal Consistency) are
surface-sampling measures invariant to welding, and it never reports
watertightness. This project does (`mesh_eval.py: watertight_rate`), which is
why the problem surfaced here and not in the paper.

### Approach A — tolerance weld (lazy)

Merge vertices within one bin instead of requiring exact equality, at the point
`MeshVQVAE.detokenize` calls `canonicalize`.

- **Where:** `src/models/mesh_vqvae.py`, `detokenize` only. The coordinate
  tokenizer's `detokenize` must keep its exact merge — it is provably lossless
  there and a tolerance would only introduce error.
- **Cost:** a few lines. No retrain, stage-1 checkpoint stays valid, stage 2
  unaffected.
- **Mechanism:** snap decoded coordinates to a representative before
  `np.unique`, e.g. cluster grid points whose L-inf distance is <= 1 bin, or
  round to a coarser grid for the *merge key* only while keeping the fine
  coordinate for output.
- **Ceiling:** cannot distinguish a crack from two genuinely distinct vertices
  one bin apart. On buildings that is rare (a 1-bin feature is ~0.16 m on a
  20 m building) but not impossible — thin roof details are the risk. Gate it on
  `watertight_rate` going up *without* `chamfer_m` going up.
- **Also:** picking a representative is a choice. Lowest-index vertex is the
  cheapest and is deterministic under the existing canonical order; the centroid
  is smoother but breaks exactness for uncracked vertices, so prefer the former.

### Approach B — decode per unique vertex (principled)

Make the decoder emit one coordinate per *unique vertex* rather than per face
slot, so sharing is structural and cracking is impossible by construction.

- **Where:** `MeshVQVAE.decode` and `head`. Instead of `[B, F, 9, bins]` over
  face slots, produce `[B, V, 3, bins]` over the unique vertices that
  `vertex_ids` already computes at encode time, then gather back per face for
  the loss.
- **Cost:** changes the decoder's output shape and the `recon_ce` reduction;
  requires a stage-1 retrain. `vertex_ids` already returns `([B, F, 3] indices,
  [B] counts)`, so the plumbing exists — the work is in `decode`, `head`, and
  the loss in `MeshVQVAEModule`.
- **Payoff:** removes the floor entirely rather than papering over it, and
  removes a redundancy — a vertex shared by k faces is currently predicted k
  times, so the head spends capacity re-deriving a value it already committed
  to. Should show up as better `recon_ce` at equal codebook size.
- **Risk:** a departure from the reference, which decodes per face
  (`project_down_codebook = Linear(codebook_dim * 3, n_embd)` then 9 logits).
  Document it as deliberate. Also note the decoder no longer sees the face as a
  unit, so face-level context has to come from the transformer stack rather than
  the output head — that is the part most likely to cost quality.

**Recommendation:** A first, as a measurement. It is cheap and it tells you how
much of the `watertight_rate` gap is mechanical cracking versus genuinely wrong
geometry. If A closes most of the gap, B is the real fix and worth the retrain;
if it barely moves, the gap is generation quality and B would be wasted effort.

---

## Issue 2 — the 128-bin ceiling (closed: faithful to the paper)

Quantization happens in `MeshDataset.__getitem__`, before either model sees
data, and `MeshVQVAE.decode` predicts logits over those *same* 128 bins, so the
VQ-VAE cannot beat the coordinate tokenizer's measured 0.107 m chamfer, only add
to it.

Checked against the reference: this is exactly what MeshAnything does.

```python
self.discrete_num = 128
self.to_coor_logits = nn.Sequential(
    nn.Linear(self.n_embd, self.discrete_num * 9),
    Rearrange('... (v c) -> ... v c', v = 9)
)
```

Same bin count, same per-face coordinate classification. The paper states it:
meshes "are discretized and input as a sequence of triangle faces", the decoder
predicts "the logits for each vertex's coordinates", trained with "cross-entropy
loss on the predicted vertex coordinate logits".

**So this is not a defect and deviating would be the departure.** Reopen only if
stage-2 geometry error approaches 0.107 m and the grid becomes the binding
constraint. If that happens, note that raising `num_bins` costs sequence length
on the coordinate path but is nearly free on the code path, so the two stages
need not share a value forever.

### One place this project is already more correct

MeshAnything's inverse map is off by one:

```python
def undiscretize(t, low, high, num_discrete):
    t = t.float()
    t /= num_discrete            # divides by 128, not 127
    return t * (high - low) + low
```

Bin 127 maps to `127/128 - 0.5 = 0.4922`, so the top of the unit box is
unreachable and the map is not the exact inverse of their discretization. This
project's `dequantize` divides by `num_bins - 1`, which *is* the exact inverse of
its `quantize` — hence the 0/200 round-trip result above. Deliberate
improvement; keep it.

---

## Issue 3 — codebook capacity spent on faces that are never quantized (open)

`_FaceView` serves both LODs (index `2i` is LOD1, `2i+1` is LOD2), so the
codebook trains on LOD1 wall quads and ground planes. But stage 2 only ever
*encodes* LOD1 — `MeshTransformerModule._prepare` calls `vqvae.encode` for the
condition and `vqvae.tokenize` only for the target. So roughly half the
codebook's training signal comes from a distribution that never consumes a code
at inference.

Deliberate, and justified in the `mesh_vqvae_datamodule` docstring: a codebook
trained on LOD2 only would see LOD1 faces as out-of-distribution exactly when
stage 2 asks the encoder to embed the condition. That argument covers the
*encoder*; it does not obviously cover the *quantizer*.

**What to try:** train the encoder on both LODs but restrict the quantizer /
codebook update to LOD2 faces. Judge on `val_codes_used` and `val_codebook_ppl`
alongside `val_recon_ce` — if LOD1 is currently occupying a meaningful share of
the 1024 entries, freeing them should show up as better LOD2 reconstruction at
the same codebook size.

---

## Issue 4 — no mesh post-processing at all (open, cheap)

`grep -rn "merge_vertices|unique_faces|fix_normals|trimesh" src/` returns
nothing. The reference's `main.py` runs three cleanup steps after decoding that
this project does not:

| step | what it fixes | have it? |
| --- | --- | --- |
| `merge_vertices()` | welds coincident vertices | yes, via `canonicalize` |
| `update_faces(unique_faces())` | drops duplicate faces | **no** |
| `fix_normals()` | repairs inconsistent winding | **no** |

Duplicate faces and inconsistent winding both feed `watertight_rate` and
`volumetric_iou`, and `canonicalize` deliberately does not dedupe faces (it
preserves `F` so the `[F, 9]` reshape stays valid across both stages — do not
change that; do the dedupe after decode instead).

Cheap, applies to both tokenizer paths, and independent of Issue 1 — worth doing
first just to establish how much of the watertight gap is bookkeeping.

### Implemented as a measurement, not a pipeline change (2026-08-12)

`src/eval/mesh_postprocess.py` reproduces all three reference steps on numpy
arrays. trimesh is **not** added as a dependency: `canonicalize` already is the
weld, and the other two are a face dedupe and a winding propagation over the
face-adjacency graph. Nothing in training, evaluation or inference imports it —
it is called only from the two notebook panels below.

- `weld` -> `canonicalize` (exact merge; trimesh's `tol.merge` of 1e-8 against a
  1/128 grid is exact in effect too, so this matches rather than approximates).
- `drop_duplicate_faces` -> `unique_faces`, grouping on sorted rows so a face
  and its reversal count as duplicates.
- `fix_winding` -> `fix_normals`: BFS over face adjacency flipping disagreeing
  neighbours per connected component, then one global flip if the signed volume
  is negative. Matches trimesh's default `multibody=False`, which is what
  `main.py` calls.

Degenerate faces are counted, never removed — the reference calls `unique_faces`,
not `nondegenerate_faces`, and on this corpus degeneracy comes from the 128-bin
grid rather than from the model.

Panels added to `notebooks/test-mesh.ipynb` (VQ-VAE path, `results` is a
5-tuple, scored against `truth`) and `notebooks/test-mesh-normal.ipynb`
(coordinate path, 4-tuple, scored against `gt`). The second carries an assertion
that cleanup is a no-op on `gt`, since the coordinate round trip is provably
lossless — it is a live control on `mesh_postprocess` itself.

`python -m src.eval.mesh_postprocess` runs a cube-based self-check covering all
three repairs plus the empty-mesh case.

**Preliminary reading, not yet the answer.** On the VQ-VAE *round trip* (120
buildings, ground-truth codes) the cleanup is nearly inert: watertight 73% ->
73%, mean face delta -0.03, only 2% of meshes losing a duplicate face. That is
expected — `canonicalize` already welded, and the round trip introduces few
duplicate or miswound faces. The open question is *generated* meshes, where
`test-mesh-normal.ipynb` already shows `watertight_gen: 0.0` against
`watertight_gt: 1.0` on all four sampled buildings. Run both panels with a
larger `N_SHOW` before deciding whether to wire this into `mesh_eval`.

---

## Direction — transition to MeshAnything V2 (preferred)

MeshAnything V2 (Chen et al., arXiv:2408.02555) supersedes the V1 design this
project follows, and its central decision resolves Issue 1 by removing its
cause. Quoting §3.2:

> "Following [4], we discard the VQ-VAE and directly use the discretized
> coordinates from Seq_V as token indices."

[4] is MeshXL (arXiv:2405.20853), which "propose[s] directly using the
discretized coordinates of the vertex as the token index, bypassing the need for
VQ-VAE as in [19]" ([19] = MeshGPT). **V2 reverted to the coordinate tokenizer
this project started with.** The compression the VQ-VAE was supposed to buy is
obtained from tokenization instead.

That matters here because the VQ-VAE never bought compression in this project
either — `mesh_vqvae.py`'s own header says so: quantization is per vertex, "3 x
depth = 9 tokens per face, the same count as raw coordinates -- the codebook
buys a *learned vocabulary*, not compression." It was adopted for a decoder that
tolerates imperfect codes. V2's answer to the same problem is a shorter, more
regular sequence rather than a learned vocabulary.

### Components

**1. Drop the VQ-VAE.** Removes Issue 1's floor outright (no per-face decoder,
so no split vertices), removes Issue 3 entirely (no codebook to misallocate),
and restores the exact detokenizer measured at 0/200 round-trip mismatches.

**2. AMT on raw discretized coordinates.** Algorithm 1 of the V2 paper:
represent each face by a *single* new vertex whenever it is adjacent to the
previous one, emitting `&` to restart when no adjacent unvisited face exists.
Measured on this corpus (500 random LOD2 meshes, canonical order from
`canonicalize`, faithful implementation of Algorithm 1):

| signal | this corpus | paper (Objaverse) |
| --- | --- | --- |
| vertex-entry ratio (paper's S Ratio) | 0.525 | ~0.49 |
| token ratio including `&` | 0.551 | — |
| breaks per mesh | 5.7 (0.227 per face) | — |
| meshes emitted as a single strip | 1% | — |
| token ratio p25 / p50 / p75 | 0.519 / 0.561 / 0.583 | — |

**AMT transfers.** 0.55 against the paper's 0.49 — LOD2 roofs are less
well-connected than artist meshes (0.23 breaks per face, only 1% single-strip),
but the compression still lands near the reference. Roughly half the sequence
means roughly a quarter of the attention cost, which is the constraint
`mesh4-no-nr-train.yaml` is currently managing with `batch_size: 16`.

AMT also gives *structural* vertex sharing: a shared vertex is written once and
referenced implicitly, so two faces cannot disagree about its coordinate. This
is strictly stronger than MeshGPT's feature-space sharing and stronger than
anything Approach A or B achieves.

**3. Masking invalid predictions** (V2 §3.2, from PolyGen). Mask logits at
inference so structurally illegal tokens cannot be sampled — V2 enforces "at
least three vertices must be generated before allowing any interruptions."
Directly applicable: `MeshTransformerModule.generate` currently has no such
guard, so it can terminate mid-face, which `detokenize` then silently discards
(`tokens[: len(tokens) - len(tokens) % 9]`).

**4. Unfreezing the point encoder — not applicable here.** V1 froze a pretrained
Michelangelo point-cloud encoder and trained only a linear projection; V2 unfroze
it because "its accuracy [was] insufficient for handling complex meshes with up
to 1600 faces." This project has no such component. The LOD1 condition is a mesh
in the same coordinate frame, embedded by the *same* `token_embed` table as the
target and processed by the same stack — trained from scratch, always trainable.
Verified: the only `requires_grad_(False)` in `mesh_transformer.py` is the
VQ-VAE (lines 338-340), and on the coordinate path there is none.

Worth noting the converse, though: the *current* VQ-VAE path **is** structurally
MeshAnything V1's frozen-encoder-plus-projection — `vqvae.encode(...)` under
`no_grad` feeding `cond_proj = nn.Linear(cond_dim, d_model)`. So V2's lesson does
apply to this project, and dropping the VQ-VAE *is* the fix. Nothing extra to do.

**5. Face count condition** (optional). V2 conditions on an approximate target
face count, dropped 10% of the time for robustness. Value here is unclear — at
sampling time the LOD2 face count is unknown, though the LOD1 count is a
plausible proxy. Defer until the core transition is measured.

### What this discards

Stage 1 in full, plus `outputs/initial-runs/mesh-vqvae-4`. The VQ-VAE was
introduced because `mesh-1` (the coordinate path) converged to a systematically
simplified mesh — `merged_face_ratio` 0.83, per
`2026-08-08-mesh-vqvae-design.md`. **AMT is not a known fix for that**; it
addresses sequence length and regularity, not the model's tendency to simplify.
Treat "shorter, more regular sequences also reduce simplification" as a
hypothesis to test, not an established result.

### Honest read on AMT's quality claim

From V2's own tokenization ablation (Table 2, OPT-125M, <=400 faces):

| method | CD | ECD | NC | S Ratio | Perplexity |
| --- | --- | --- | --- | --- | --- |
| Baseline (3 verts/face) | 2.478 | **18.21** | 0.893 | 1.000 | **1.150** |
| AMT | **2.348** | 19.33 | 0.904 | 0.492 | 1.363 |
| AMT (Swap) | 2.517 | 19.86 | **0.913** | **0.455** | 1.416 |

CD improves ~5%, ECD gets *worse*, perplexity gets notably worse. The paper's
own framing is the right one: AMT "shortens the token length without
compromising mesh quality." **Adopt it for the length, not for the accuracy.**

Also from that table: `Unsort` scores CD 8.151 against baseline's 2.478.
Canonical ordering is worth ~3x — whatever else changes, `canonicalize` stays.

### Implemented (2026-08-12), toggleable and default-off

Strictly additive: `vocab_size(num_bins)` still returns `num_bins + 3`, existing
checkpoints still load, and every existing config resolves to the old behaviour.

| piece | where | toggle |
| --- | --- | --- |
| `BREAK` token (`num_bins + 3`) | `mesh_dataset.py` | only under AMT |
| `amt_tokenize` / `amt_detokenize` (Algorithm 1 and its inverse) | `mesh_dataset.py` | `mesh_data.tokenization: amt` |
| `fix_winding` / `signed_volume` | `mesh_dataset.py` | always available |
| `invalid_logits_mask` | `mesh_transformer.py` | `mesh_model.mask_invalid: true` |
| pass-through | `mesh_datamodule.py`, `train_mesh.py`, `config.py` | — |
| worked example | `configs/mesh-v2-train.yaml` | — |

**Winding is recovered, not stored.** AMT records that three vertices form a
triangle but not which way round; the inverse always reads `(previous two,
new)`, which is the reverse of the original face on a consistently wound mesh.
`amt_detokenize` therefore runs `fix_winding` — which is exactly why the
reference runs `fix_normals()` after decoding. `fix_winding` moved from
`src/eval/mesh_postprocess.py` into `mesh_dataset.py` for this: it is now part
of a tokenizer inverse, not just eval cleanup, and importing eval from dataset
would have been a cycle. `mesh_postprocess` re-exports it, so Issue 4's panels
are unchanged.

Measured on `data/The Hague/mini` (`max_files=6`, 1670 buildings):

| signal | value |
| --- | --- |
| target sequence length, AMT / coord | 0.542 |
| longest segment | 1630 -> 928 tokens (0.57x) |
| buildings decoding to the *same mesh* under both tokenizers | **1670 / 1670** |

That last row is the correctness result: AMT plus winding repair is not merely
close, it is exact on every building in the sample.

AMT and the VQ-VAE are mutually exclusive and the model raises rather than
silently ignoring one — AMT rewrites the coordinate sequence, the VQ-VAE
replaces it with codes.

Masking rules implemented (`invalid_logits_mask`), all keyed off position within
the current face/strip: BOS and PAD never legal; EOS only on a completed face
(coord) or a closed 3-vertex strip (AMT); BREAK only on a vertex boundary with a
strip standing, which also forbids two breaks in a row. Coordinates are never
masked, so the mask cannot deadlock sampling — asserted directly.

Tests: `tests/test_mesh_amt.py`, 29 cases, TDD (each watched failing first).
Pre-existing unrelated failures remain: `test_mesh_tokenizer::test_margin_is_per_axis`
(numpy 2 removed `ndarray.ptp`) and four in `test_mesh_code_transformer`
(`transformers` not installed) — both confirmed identical on stashed code.

### Risks

- AMT needs reliable face adjacency, which here depends on `canonicalize`'s
  128-bin vertex merge. Degenerate faces from quantization (two vertices
  collapsing onto one grid point) have no well-defined third vertex; the
  measurement above skips them, and a real implementation must too.
- 0.227 breaks per face means the `&` token is frequent — roughly one in four
  faces. V2 notes AMT degrades toward the baseline as connectivity falls, so the
  regularity benefit is weaker here than on artist meshes.
- Detokenization becomes stateful (reverse Algorithm 1) rather than a pure
  reshape, so the exactness currently guaranteed by construction has to be
  re-established by test.

---

## Suggested order

1. Issue 4 — cheapest, no retrain, and its cleanup steps are needed under any
   tokenizer.
2. **The V2 transition** — drop the VQ-VAE, AMT on discretized coordinates, mask
   invalid predictions. Supersedes Issues 1 and 3 rather than fixing them.
3. Issue 1 Approach A — only as a stopgap if the VQ-VAE path must be kept
   working while the transition lands.
4. Issue 1 Approach B, Issue 3 — drop if the transition goes ahead; both exist
   only to repair the VQ-VAE.
5. Issue 2 — closed; reopen only if geometry error approaches the bin grid.
