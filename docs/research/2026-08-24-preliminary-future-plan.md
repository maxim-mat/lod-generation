# Buildings-as-meshes with transformers: what the field does that we don't

Preliminary literature scan, 2026-08-24. No code changed.
Scope: papers from 2024-06 onward, found via alphaXiv; implementation claims
checked against GitHub where code exists.
Status: research notes, not a plan of record. Nothing here is scheduled.

---

## 1. The landscape

Two literatures that barely touch:

* **Autoregressive mesh transformers** (MeshGPT, MeshAnything V1/V2, EdgeRunner,
  BPT, TreeMeshGPT, Meshtron, ARMesh, MeshWeaver) work on ShapeNet / Objaverse /
  Toys4K -- organic, 1K-16K faces, no semantics, no georeference.
* **Building reconstruction** (PBWR, BWFormer, EdgeDiff, Point2Roof, ArcPro,
  Point2Building, Delaunay Canopy) works on airborne LiDAR and outputs
  *wireframes or primitives*, not triangle meshes, and mostly not
  autoregressively.

**LOD1 -> LOD2 as autoregressive mesh generation appears unoccupied.** I found no
paper doing it. That is the useful headline: the framing is novel, so the risk is
that nobody has de-risked it either.

### The one paper sitting on top of us

**BuildAnyPoint** -- arXiv 2602.23645, HKUST(GZ) + XJTU + HKUST, 2026-02-27.
Project page: https://ai4city-hkust.github.io/BuildAnyPoint/

Trains an **OPT-350M backbone with MeshAnything V2 Adjacent Mesh Tokenization**
on **The Hague and Rotterdam LoD2 buildings** (Building-PCC benchmark, Gao /
Peters / Stoter, 50k instances). Same city, same LoD2 source, same backbone, same
tokenizer as us. The task differs: point cloud -> mesh, not LOD1 -> LOD2. Their
setup (OPT-350M, 4x A40, 520k iterations, total batch 8, frozen point encoder) is
the closest thing to a reference budget we have.

### Also in scope, not AR-mesh

* **CM2LoD3** -- arXiv 2508.15672, TUM. LoD2 -> LoD3 via ray-to-model conflict
  maps plus a U-Net. CityGML-native: writes openings back as `bldg:Window` /
  `bldg:Door` and adds reversed-winding interior rings to the wall polygon. Code:
  https://github.com/InFraHank/CM2LoD3
  Relevant to us as a CityGML write-back reference, and because their interior
  rings are the same construct as our face-ring bug.
* **BuildingWorld** -- arXiv 2511.06337. ~5M LoD2 models, 44 cities, five
  continents, plus simulated ALS via Helios++. Far larger and more diverse than
  The Hague alone; the answer if corpus diversity becomes the bottleneck.
* **Edge Prediction for Roof Wireframe Reconstruction** (2606.02406) and
  **WireframeDETR** (2606.14811) -- S23DR 2026 challenge entries, roof wireframes.

---

## 2. The tokenization metric, defined

This is the number quoted in section 3, and it needs stating precisely because it
is easy to compare the wrong things.

**Compression ratio = L / (9N)**, where `L` is the tokenized sequence length and
`N` the face count. Lower is better. Definition from MeshWeaver section 4.3.

The denominator is the naive coordinate tokenizer: 3 vertices x 3 axes = 9 tokens
per face, every vertex repeated for every face that uses it, no sharing. **That is
exactly our `coord` path**, so our coordinate tokenizer *is* the unit -- it scores
1.00 by construction, not by measurement.

What the ratio does and does not capture:

* It measures **sequence length only**. It says nothing about reconstruction
  fidelity, vocabulary size, or whether the scheme preserves winding.
* It is **corpus-dependent**. AMT's length depends on face *adjacency*, so the
  same tokenizer scores differently on different meshes. MeshWeaver lists
  MeshAnything V2 at 0.46; we measure our AMT implementation at 0.53 on
  `mini_cleaner`. That gap is the corpus, not a bug.
* It ignores the **softmax width trade**. BPT reaches 0.26 with a vocabulary of
  roughly 5.1k ids (8^3 blocks + 16^3 offsets + a special-block base) against our
  131. Fewer, wider tokens is not automatically cheaper.

### Where we sit

Measured in this repo, from the mesh-v3 run configs (resolved `max_seq_len` at
`max_faces: 200`):

| path | tokens for 200 faces | ratio |
|---|---|---|
| `coord` (default) | 1801 = 9x200 + 1 | **1.00** (the unit) |
| `amt` | 947 | **0.53** |

### The field, per MeshWeaver Table 1

| method | ratio |
|---|---|
| EdgeRunner | 0.47 |
| MeshAnything V2 | 0.46 |
| DeepMesh | 0.28 |
| Nautilus | 0.27 |
| **BPT** | **0.26** |
| TreeMeshGPT | 0.22 |
| Mesh-Silksong | 0.22 |
| MeshWeaver | **0.18** |

ARMesh reports absolute counts instead (their Table 4, bench category):
EdgeRunner 5840 tokens/shape, BPT 3235, ARMesh 2556 -- while spending *more*
tokens per element (9.1 per vertex against BPT's 2.73 per triangle), because it
emits far fewer elements.

**Caveat that matters for us:** every one of these is measured on 1K-16K-face
organic meshes. Our buildings are ~100 planar, mostly axis-aligned faces. Schemes
tuned for dense organic connectivity may not transfer proportionally, and some
(half-edge traversal) require manifold input.

---

## 3. Tricks we do not use, ranked by applicability

### 3.1 Condition on a learned intermediate, not the raw input (BuildAnyPoint)

Their ablation, both arms MeshAnything V2, only the conditioning changes:

| conditioning | #V | #F | failure rate | CD |
|---|---|---|---|---|
| raw input point cloud | 78 | 127 | <1% | 0.107 |
| diffusion-recovered point cloud | 38 | 70 | 0% | **0.034** |

3x chamfer and half the faces. Our LOD1 condition is already clean, so this does
not transfer directly -- but the mechanism (a generative prior that regularises an
ill-posed step before the AR model ever sees it) is the largest measured effect in
the building literature.

Two smaller things from the same paper we also lack:

* They **condition on inferred vertex normals**; their VAE learns normals
  specifically to support AR mesh inference. Our LOD1 prefix is token ids only.
* They **freeze a point encoder pretrained on large object datasets**. We train
  from scratch or from OPT language weights.

Their future-work section names the prior we deliberately normalise away: height
and geographic coordinate embeddings.

### 3.2 Geometry-aware conditioning instead of a prefix (MeshWeaver, 2606.04688)

MeshWeaver names our architecture as its limitation (ii): generation conditioned
on a global shape embedding with static vocabulary embeddings and no local
geometric context. Three fixes, all ablated as significant:

1. **Voxel features as vertex embeddings** rather than a static lookup table. Our
   `token_embed` is one fixed table shared across every building.
2. **Cross-attention to local geometry** at each prediction step, not only a
   prefix.
3. **Scaffold masking** -- mask the logits of empty voxels so every predicted
   vertex is anchored near the input surface.

**(3) is the one to try first here.** LOD1 already bounds where an LOD2 vertex can
legally sit, and the mechanism is identical to our existing `mask_invalid`, just
applied to geometry instead of grammar. It needs no new architecture: a mask over
coordinate-bin logits derived from the LOD1 box.

### 3.3 Block-wise coordinate compression (BPT)

Verified against the released implementation, `model/serializaiton.py` at
https://github.com/Tencent-Hunyuan/bpt

```python
block_id  = coords // offset_size     # block_size=8, offset_size=16 -> 128 grid
offset_id = coords %  offset_size     # exactly our num_bins=128
...
elif sequence[i, 0] == cur_block_id:
    codes.append(sequence[i, 1])      # same block -> emit the offset only
```

Two independent ideas: (a) split each coordinate into a coarse block id and a fine
offset, and elide the block id whenever it repeats; (b) patchify faces around the
vertex touching the most unvisited faces, so a patch centre is named once and its
ring follows.

Their 128 grid is our `num_bins`, exactly. Buildings are spatially compact, so
consecutive vertices land in the same block constantly -- the elision should pay
off *more* on LOD2 roofs than on organic shapes. Costs a ~5.1k vocabulary.

One caveat their own code states, bearing on a choice we already made: with
`fix_orient=True` the normals come out correct, but the comment notes this may
make learning harder. Our rotate-don't-sort decision is the same trade,
independently acknowledged by the reference.

**ARMesh additionally runs BPE over mesh tokens** -- vocabulary 16,384, a further
2-3x length reduction. Orthogonal to the tokenizer, so it composes with either of
our existing paths.

### 3.4 Constrained decoding with backtracking (ARMesh, 2509.20824)

Our `mask_invalid` is forward-only: once the model has painted itself into a
corner, no legal token remains and it cannot recover. ARMesh hardcodes a validity
predicate over the prefix and runs **depth-first traversal with backtracking** --
if it cannot proceed, it returns to a parent and tries an alternative. The beam
search added 2026-08-24 is half of that infrastructure already.

### 3.5 Next-LOD ordering (ARMesh -- the conceptual twin)

ARMesh reverses mesh simplification (GSlim over simplicial complexes, which unlike
QSlim survives non-manifold input) to get a coarse-to-fine refinement sequence,
then generates level by level. Their Table 4, at 10% of AR steps: COV 41.07
against EdgeRunner 9.43 and BPT 15.15 -- partial generations are good
*approximations* rather than partial meshes, and early stopping yields any LOD.

For a project whose whole subject is level of detail, this is the natural framing.
Related: **VertexRegen** (2508.09062, continuous LOD via progressive meshes),
**SubdivAR** (2606.27088, next-scale subdivision), **MeshFIM** (2605.08744,
fill-in-the-middle for local edits instead of full regeneration).

### 3.6 Training and data

* MeshWeaver: **subvolume pruning** at training time, and a **cross-attention KV
  cache** (we now have the self-attention one).
* BuildingWorld: **simulated ALS** and a procedural synthetic city for
  augmentation. Their PBWR benchmark shows a simulation-trained model reaching
  EF1 0.74 on real point clouds against 0.76 for a real-trained one -- the
  sim-to-real gap is small in this domain.

---

## 4. What I would actually do

Scaffold masking (3.2 item 3) and BPT block indexing (3.3) are the two with
released code, a clear mechanism, and a plausible fit to buildings. Everything
else is a research project rather than a change.

Neither should be attempted before the dataset confound recorded in the mesh-v3
analysis is fixed -- an A/B against a moving corpus is unreadable.

## 5. What I did not verify

* Released code was confirmed only for **BPT** and **CM2LoD3**. ARMesh and
  MeshWeaver have project pages; I did not confirm a code drop for either.
* No compression ratio in section 2 was re-measured on our corpus. Only our own
  `coord` (1.00) and `amt` (0.53) numbers come from this repo.
* I did not read Mesh-Silksong, Nautilus, DeepMesh, Meshtron or EdgeRunner
  directly -- their ratios are as reported by MeshWeaver.
