# Orthophoto conditioning for the mesh transformer: prefix tokens vs. staged cross-attention

Research note for the aerial-view phase. **Nothing here is built yet.** It records
the two ways considered for adding a top-down orthophoto to the LOD1 -> LOD2
model, what each costs in this codebase, and the literature behind each. The
implementation spec comes after a choice is made.

Baseline throughout is `configs/mesh-v5-full-opt-coord.yaml`:
- the OPT-125m architecture with random init (`opt_pretrained: false`);
- coordinate tokenization, 128 bins, sinusoidal positions;
- the LOD1 condition as a token prefix;
- selection on `val_gen_chamfer_m`.

Both approaches encode the image with a frozen DINOv3. They differ only in how
the image features reach the decoder.

## 1. What the image can and cannot add

A top-down orthophoto shows roof *structure*: ridge and valley lines, facet
boundaries (visible through shading), dormers, flat versus pitched roofs,
courtyards, and footprint detail that the LOD1 extrusion has smoothed away. It
does not show *heights*. Those still come from the LOD1 (its extrusion is the
median roof height) and from priors the model has learned.

Khomiakov et al. (arXiv:2303.11215) show that this is enough to generate
plausible 3D roof meshes from a single aerial image on Swiss data. They also
report that the model is fairly robust to lower image resolution, and that it
fails on crops containing several buildings and on complicated roof topology.
Both failure modes matter here: The Hague is full of row houses.

## 2. Shared front end (needed by both approaches)

Everything in this section is independent of the fusion choice. It is also
probably worth more than the fusion choice. MM1 (arXiv:2403.09611) found image
resolution and visual token count to be the dominant factors, with the
connector design (average pooling, attention pooling, C-Abstractor) of
comparatively negligible importance. Note that MM1 compares connectors
*within* a prefix architecture, not prefix against cross-attention.

### 2.1 Imagery

- **Source:** Beeldmateriaal Nederland via PDOK, CC-BY 4.0. There are two
  national flights a year: a leaf-off spring flight at 7.5 cm (served as "8 cm")
  and a leaf-on summer flight at 25 cm. **Use the spring flight**, because fewer
  roofs are hidden by trees.
- **Year:** match the flight year to the LOD2 source. Source A (the
  Gemeente Den Haag model) was produced from a 2021 flight. A mismatched year
  pairs some buildings with roofs that have since changed.
- **Open risk: building lean.** The national mosaic is built from the central
  parts of individual orthophotos to keep displacement small. No official spec
  found says it is a *true* orthophoto. If it is not, high-rise roofs are offset
  from their footprints, which breaks the alignment in 2.2 for exactly the
  tallest buildings. **Check this by eye on a handful of towers before building
  anything.**

### 2.2 Crop aligned to the mesh frame

`normalize_to_unit_box` already defines the frame the tokens live in. With the
v5 margins (x and y are 0; only +z is 0.1), the x/y part of that frame is:

- centre: the LOD1 bounding-box centre (cx, cy);
- side length: `base`, the box's longest extent over **x, y and z**;
- normalized coordinate: `u = (x - cx) / base`, which lies in [-0.5, 0.5].

Crop the orthophoto to the square [cx ± (base/2 + b)] × [cy ± (base/2 + b)],
with a buffer b of a few metres (Khomiakov et al. use 3 m). Resample it to a
fixed size: 256 px at patch size 16 gives a 16 × 16 = 256-token grid.

Each patch then covers a known (u, v) region of the same frame the vertex
tokens are quantized in.

### 2.3 Patch coordinate embedding

Add to every image token an embedding of its patch-centre (u, v): a sinusoidal
code of the continuous value, projected to the model width. Use a continuous
code rather than the bin lookup, because patches in the buffer fall outside
[-0.5, 0.5].

This lets the decoder relate "a vertex at bin (37, 80)" to the patch above it
without learning the mapping implicitly. Precedent:
- PolyGen (arXiv:2002.10880) adds coordinate embeddings to its image feature
  maps before flattening them.
- PixARMesh (arXiv:2603.05888) gets its single-view gains from DINOv2 features
  aligned to points by pixel.

**Optional:** a learned per-patch flag marking whether the patch lies inside the
LOD1 footprint, for the row-house case.

### 2.4 Encoder: DINOv3, frozen

DINOv3 (arXiv:2508.10104) ships distilled ViT-L/16 models trained on two
datasets:

| Variant | Hugging Face id | Trained on | Paper finding |
|---|---|---|---|
| Web | `facebook/dinov3-vitl16-pretrain-lvd1689m` | LVD-1689M, web images | Best on semantic geospatial tasks |
| Sat | `facebook/dinov3-vitl16-pretrain-sat493m` | SAT-493M, Maxar orthoimagery at 0.6 m | Best on metric tasks; canopy height MAE 2.2 vs 2.4 m for the 7B web model |

Our 8 cm imagery sits between the two. With the encoder frozen, swapping them is
a config change, so treat it as an ablation rather than a decision.

- **Cost:** ViT-L at 256 px is about 163 GFLOPs per image (DINOv3 Fig. 16a).
  That is small next to one OPT-125m pass over a ~3k-token sequence. Running it
  on the fly is fine; precomputing ViT-L features for 259k buildings at fp16
  would be ~130 GB.
- **Output:** 1024-d patch tokens (ViT-L).
- **Licence:** both variants are gated on Hugging Face under Meta's custom
  DINOv3 licence, not Apache. DINOv2 is the Apache-2.0 fallback.

### 2.5 Image dropout

During training, replace the image tokens with a single learned "no image"
token in about 10% of samples.

- It keeps the model usable where imagery is missing or occluded.
- It turns "does the image help" into a controlled comparison inside one model.

This is the same idea as classifier-free guidance's condition dropout.

## 3. Approach A: image patches as prefix tokens

### 3.1 Mechanism

`MeshOPTTransformer.forward` already builds the decoder input as `inputs_embeds`:
the embedded LOD1 tokens and the embedded target, concatenated, plus a learned
`segment_embed` that tells the two halves apart. The image becomes a third
segment:

```
[ image patches (256) | LOD1 condition | BOS + target ]
  segment 2             segment 0        segment 1
```

The image tokens are DINOv3 features passed through a 2-layer MLP projector
(1024 -> 768 for OPT-125m), plus the coordinate embedding from 2.3. Logits are
still read only over the target span. The change is about 20 lines inside
`MeshOPTTransformer` and touches no Hugging Face internals.

### 3.2 Training

Fine-tune everything (the projector plus all of OPT) from the v5 checkpoint,
with DINOv3 frozen.

**Caveat:** OPT's positions run straight through the whole sequence, so
inserting 256 tokens shifts every LOD1 and target position. Under sinusoidal
positions there is no table to overrun, but v5's attention patterns were learned
at the old offsets. The first steps therefore start from a *perturbed* v5, not
from v5 itself. Fine-tuning everything should recover, but the run has no clean
"identical to v5 at step 0" point.

### 3.3 Cost

- About 256 extra tokens per sample. The v5 smoke batch had 720 condition and
  1,190 target tokens, so that is roughly 13% more sequence.
- At generation time the image tokens join the prefix and the KV cache absorbs
  them.
- No new attention parameters; only the projector.
- **Optional:** a Perceiver resampler down to 64 tokens. Idefics2 found it both
  cheaper and better, but pooling discards the explicit spatial layout that 2.3
  provides, so start without it.

### 3.4 Evidence for this approach

- **Idefics2, "What matters when building vision-language models?"
  (arXiv:2405.02246), Finding 2 / Table 3.** A controlled comparison of the two
  fusion styles. With the backbones *trained* (via LoRA), prefix tokens scored
  69.5 against 67.3 for cross-attention, even though the cross-attention model
  had more parameters. We train OPT fully, so we are in this regime.
- **Idefics2, Finding 3.** Fully unfreezing the backbones under the prefix
  architecture caused loss divergences that a lower lr did not fix; LoRA did.
  This is a real risk for Approach A: start from a conservative lr and keep
  LoRA in reserve.
- **MeshAnything (arXiv:2406.10163).** The model our OPT path follows
  (`ShapeOPT`) conditions by prefix: a fixed 257-token point-cloud prefix ahead
  of the mesh tokens. That is the same mechanism v5 already uses for its LOD1.
- **MM1 (arXiv:2403.09611).** A decoder-only (prefix) model throughout. Among
  pooling connectors, the design mattered little relative to image resolution
  and token count. This supports the cheapest projector that keeps all the
  visual tokens, rather than a learned resampler.

## 4. Approach B: gated cross-attention, frozen first, then joint

### 4.1 Mechanism

Wrap each `decoder.layers[i]` in a small module that applies a gated
cross-attention to the hidden states, then calls the original layer unchanged:

```
h = h + tanh(alpha_i) * CrossAttn(LayerNorm(h), image_tokens)   # alpha_i = 0 at init
h = original_layer(h, ...)                                         # untouched
```

- The image tokens are the same projector output as in Approach A (2.3 and 2.4
  still apply).
- Optionally, add a gated feed-forward after each cross-attention, as Flamingo
  does.
- This replaces modules in the decoder's layer list; it does not subclass
  `OPTDecoderLayer`. That keeps the project's rule of not depending on
  Hugging Face internals.
- **Sequence:** stays `[LOD1 | target]`, so v5's positions are untouched.
- **Generation:** compute the image keys and values once per sample and reuse
  them every step.
- **Size:** roughly 2.4M parameters per cross-attention block at d = 768, so
  ~28M over 12 layers, or roughly double that with the gated feed-forward.
  That compares with OPT-125m's 125M, which is frozen in stage A.
- **Exact start:** because `tanh(0) = 0`, the model at step 0 **is v5 exactly**.

### 4.2 Stage A: v5 frozen; train only the cross-attention and projector

- **Frozen:** OPT (the v5 weights) and DINOv3.
- **Trained:** the cross-attention blocks, gates, projector and coordinate
  embedding.
- **Selection:** early stopping on `val_gen_chamfer_m`, as now.
- **Diagnostics to log:**
  - |tanh(alpha_i)| for each layer;
  - the shuffled-image gap on validation (see section 6).

### 4.3 Gate to stage B

Move on when *all* of these hold:

1. **Stage A has plateaued:** early stopping fired.
2. **The gates opened:** |tanh(alpha_i)| is clearly non-zero in at least some
   layers. If every gate stays near zero, the model found nothing usable in the
   image. Unfreezing OPT will not fix that; look upstream at crop alignment,
   building lean and the encoder variant.
3. **The shuffled-image gap is real:** chamfer distance with a wrong building's
   image is worse than with the right one, by more than the measured
   per-object metric noise (`docs/posthoc-eval-runbook.md`).

### 4.4 Stage B: joint fine-tuning

- **Unfrozen:** OPT, with the cross-attention also still trainable.
- **Learning rates:** separate parameter groups. OPT gets a low lr, around
  1e-5 to 1e-4; the new parameters keep stage A's lr. Use a warmup.
- **Still frozen:** DINOv3.
- **Fallback:** if joint training diverges, LoRA on OPT (Idefics2 Finding 3).

Run stage A and stage B as **two separate runs**, with stage B resuming from
stage A's best checkpoint under its own config, not as a mid-run switch:
- each stage selects its own best checkpoint;
- each stage is reproducible on its own;
- the gain from stage A to stage B is a clean comparison.

### 4.5 Why this order and not the reverse

The reverse schedule was also considered: train the main model with the
cross-attention frozen, then freeze the main model and train the
cross-attention. It fails both ways:

- **Zero-gated cross-attention, frozen:** contributes exactly nothing, so
  phase 1 is just more image-free training of a model that v5 already is.
- **Random ungated cross-attention, frozen:** the decoder learns to read a fixed
  random projection, which phase 2 then changes under a frozen reader.

The working order is the standard one: **train the new parts against the frozen
pretrained model first, then fine-tune everything.** v5 is already the "main
model trained" phase.

### 4.6 Evidence for this approach

- **Flamingo (arXiv:2204.14198).** Introduces gated cross-attention layers
  between frozen language-model layers, with a tanh gate initialised at zero so
  the frozen model's behaviour is preserved at the start of training. Only the
  new layers (and its Perceiver resampler) are trained. This is exactly the
  stage A mechanism.
- **Idefics2 (arXiv:2405.02246), Finding 2 / Table 3, the other half.** With the
  backbones *frozen*, cross-attention scored 66.7 against 60.3 for prefix tokens.
  Stage A is in this regime, which is why the staged plan favours cross-attention
  even though Approach A wins once everything is trained.
- **BLIP-2 (arXiv:2301.12597).** Bridges a frozen image encoder and a frozen
  language model with a small trained module, in stages. It is precedent that a
  frozen generator can be steered by a trained adapter alone.
- **LLaVA (arXiv:2304.08485).** The two-stage order recommended here: stage 1
  trains only the vision-to-language projector with both backbones frozen;
  stage 2 unfreezes the language model and trains end to end.
- **PolyGen (arXiv:2002.10880).** Conditions its autoregressive mesh model on
  images (and voxels) by cross-attention from the decoder into encoder feature
  maps, with coordinate embeddings added to the features.
- **Khomiakov et al. (arXiv:2303.11215).** PolyGen with a ResNet image encoder
  and cross-attention, generating 3D roof meshes from single aerial
  orthoimages. It is the closest prior work to this task, though small-scale
  and trained from scratch.
- **Point2Building (arXiv:2403.02136).** Autoregressive building meshes from
  airborne LiDAR, conditioned by cross-attention onto sparse-CNN features.
  Building-domain precedent for cross-attention conditioning.

## 5. Side by side

| | A: prefix tokens | B: gated cross-attention, staged |
|---|---|---|
| Code change | ~20 lines in `MeshOPTTransformer`; third segment | Layer wrappers, gates, image KV cache; more code |
| Hugging Face internals touched | None | None (wraps layers, no subclassing) |
| Sequence length | +256 (~13%) | Unchanged |
| New parameters | Projector only | Projector + ~28-56M cross-attention |
| Starting point | Perturbed v5 (positions shift) | **Exactly v5** (zero gates) |
| Training | One run, everything trainable | Two runs: frozen-backbone stage A, then joint stage B |
| Literature, frozen backbone | Weaker (Idefics2: 60.3) | Stronger (Idefics2: 66.7) |
| Literature, trained backbone | Stronger (Idefics2: 69.5) | Weaker (Idefics2: 67.3) |
| Built-in diagnostic | Shuffled-image gap only | Gate magnitudes + shuffled-image gap |
| Main risk | Divergence when fully unfrozen (Idefics2 Finding 3) | Gates stay shut (image not used) |

The Idefics2 numbers are vision-language benchmark averages at 7-8B scale, not
mesh results. They show which direction the trade-off goes, not how big it will
be here.

## 6. Shared evaluation protocol

Run the same protocol for either approach, scored with post-hoc eval on the
same test split as v5:

1. **Paired against v5:** chamfer, Hausdorff, symmetric-difference volume and
   roof chamfer, per object.
2. **Shuffled image:** give each building another building's crop. The gap from
   the true-image score is the measured contribution of the image. **This is
   the headline result, not the absolute chamfer.**
3. **No image:** the null token from 2.5. Should land near v5.
4. **Stratified results:**
   - by building height, where lean shows up;
   - by detached versus row house, where crop contamination shows up;
   - by roof type (flat versus pitched), since flat roofs leave little for the
     image to add.

## 7. Recommendation and open decisions

- **Build the shared front end (section 2) first.** Both approaches need it, and
  it carries most of the expected gain.
- **If the staged plan stands, Approach B is the better fit.** Its frozen stage
  is the regime where cross-attention wins, and the zero-gated start makes
  stage A a clean measurement of what the image adds to v5.
- **Approach A is cheap enough to run as a control arm,** reusing the same front
  end. That turns the fusion question into a measured result on this corpus
  rather than a borrowed one.

Decisions needed before the implementation spec:

1. Verify building lean on a few towers in the chosen flight (2.1).
2. Flight year (2021 to match Source A?) and spring 7.5 cm imagery.
3. DINOv3 web or sat: pick one to start, ablate the other.
4. Crop size and buffer (256 px, b = 3 m proposed).
5. Footprint flag (2.3): include from the start or keep as an ablation.
6. Whether to run A as a control arm alongside B.

## References

- Alayrac et al., *Flamingo: a Visual Language Model for Few-Shot Learning*, arXiv:2204.14198
- Khomiakov et al., *Learning to Generate 3D Representations of Building Roofs Using Single-View Aerial Imagery*, arXiv:2303.11215
- Laurençon et al., *What matters when building vision-language models?* (Idefics2), arXiv:2405.02246
- Li et al., *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*, arXiv:2301.12597
- Liu et al., *Visual Instruction Tuning* (LLaVA), arXiv:2304.08485
- Liu et al., *Point2Building: Reconstructing Buildings from Airborne LiDAR Point Clouds*, arXiv:2403.02136
- McKinzie et al., *MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training*, arXiv:2403.09611
- Nash et al., *PolyGen: An Autoregressive Generative Model of 3D Meshes*, arXiv:2002.10880
- Chen et al., *MeshAnything: Artist-Created Mesh Generation with Autoregressive Transformers*, arXiv:2406.10163
- Siméoni et al., *DINOv3*, arXiv:2508.10104
- Zhang et al., *PixARMesh: Autoregressive Mesh-Native Single-View Scene Reconstruction*, arXiv:2603.05888
- Data: Beeldmateriaal Nederland orthophotos via PDOK (CC-BY 4.0); DINOv3 weights on Hugging Face (gated, DINOv3 licence)
