# Aligned image prefix for the v6 orthophoto model — design

Date: 2026-11-02. Status: approved (attention option (a) chosen 2026-11-02).

## 1. Why

The v6 prefix arm (`configs/mesh-v6-ortho-prefix.yaml`, W&B `7swbxcwh`) was
initialised from v5 (`epoch=20-val_gen_chamfer_m=0.0859.ckpt`) but got nothing
from that init:

- After 10 epochs it reached val_loss 0.227 and val token accuracy 0.938. v5
  ended at 0.180 / 0.961.
- Its best `val_gen_chamfer_m` was 0.097 (epoch 5), and it ended at 0.106. v5's
  best was 0.086 on the same random split.
- So, with images, the model is still worse than image-free v5.

Three things the current prefix path does break the init from step 0:

1. **Positions shift.** `MeshOPTTransformer.forward` puts 256 DINOv3 tokens in
   front of the sequence. OPT derives position ids from a cumsum over the
   attention mask, so every mesh token moves by 256 positions. v5 never saw
   those positions.
2. **No-image is not v5's input.** Under image dropout and under
   `image_mode: none`, the 256 slots are filled with copies of one learned null
   vector, not removed.
3. **Attention is diluted.** 256 randomly projected tokens take softmax mass in
   every layer.

Separately, the image has no fine spatial link to the mesh. The only place
image and mesh meet is through attention over 16×16 patches. Each patch is
1.2·L/16 wide, about 1.5 m on a building whose longest LOD1 side is L = 20 m,
which is about 9.5 voxel bins.

## 2. Goals

- **G1. Clean start from v5.** With the image masked, the model's input and
  attention pattern are exactly v5's, so logits equal v5's. With the image
  present, the only step-0 deviation from v5 is the 64 prefix tokens.
- **G2. A per-vertex, bin-aligned image signal.** It reaches about 0.2–0.4 m at
  typical building sizes, not the 1.5–3 m of today's patches.
- **G3. Every piece can be ablated from config**, independently where that
  makes sense.
- **G4. Full backward compatibility.** Every existing checkpoint loads and
  produces bit-identical logits, including `7swbxcwh`, the original v5, and the
  xattn checkpoints. Old configs reproduce old models.

**Non-goals**
- Retraining v5-tile.
- Changing the xattn arm, apart from B being usable with it (§3.4).
- AMT or VQ-VAE tokenizers.
- A Perceiver or Q-Former resampler.
- Learnable attention-key gating.
- Direct bidirectional attention among LOD1 tokens. The indirect path through
  image tokens is accepted (§3.5).

**Anti-goals** (failures even if training runs)
- Any change to image-free or old-checkpoint logits.
- B silently misaligned: a flipped axis, or the margin mis-mapped, so features
  come from the wrong ground cell.
- Cached generation diverging from uncached.

## 3. What gets built

All new behaviour is behind `mesh_model` config fields. Each field is also a
`MeshTransformerModule` hyperparameter whose default reproduces today's model
exactly. So an old checkpoint, whose `hyper_parameters` lack these keys, loads
into today's model.

### 3.1 Separate positions (fix 1)

Field: `image_separate_positions: bool = False`.

When true and `image_fusion: prefix`:

- Mesh tokens (condition + target) get position ids from a cumsum over the
  **mesh-only** attention mask. These are exactly v5's ids, padding invariance
  included.
- Image tokens get **no 1-D position**. Their position embedding is the zero
  vector, so they carry only the encoder's existing 2-D (u,v) sin/cos code.
  - Mechanism: pass explicit `position_ids` to OPT, then have
    `OPTSinusoidalPositions` return zero for a dedicated sentinel id.
  - transformers 5.10.2 forwards `position_ids` from `OPTForCausalLM` to
    `decoder.embed_positions`; checked.
  - The existing `-1 → clamp(0)` handling of pad slots must stay as it is
    (backward compatibility), so use a different sentinel, e.g. `-2`.
- This requires `pos_embed: sinusoidal`; a learned table has no "no position".
  Raise at construction otherwise.
- The cached path (`new_cache` / `cached_logits`) must pass explicit
  `position_ids` too, continuing the mesh-only count across steps.

### 3.2 No image means masked (fix 2)

Field: `image_mask_invalid: bool = False`. Requires `image_separate_positions`;
raise otherwise, because with shared positions masking would move mesh
positions.

When true:

- A row whose `image_valid` is False (training dropout, `image_mode: none`, or
  the `generate(image_valid=...)` argument) gets attention mask 0 on all of its
  image tokens.
- Its vertex features are zero. §3.4 does this for every invalid row anyway,
  whatever this flag says.
- With §3.1, that row's computation is then exactly v5's (G1).
- **Fully masked query rows (my call).** An image query row whose keys are all
  masked is NaN under a hand-built 4-D mask with SDPA, and the NaN spreads. So a
  masked image query keeps its own diagonal key. Its output is never read by
  mesh rows, because their image keys are masked.
- The `null` parameter stays in the module so state dicts keep their shape. It
  is simply unused in this mode.

### 3.3 Pooled prefix (fix 3)

Field: `image_prefix_grid: Optional[int] = None`. None means the native grid
(16×16 = 256 tokens); `8` means 64 tokens.

- Average-pool the DINOv3 patch features from the native grid to the target
  grid **before** `proj`.
- Compute the (u,v) code at the pooled cell centres: `pixel_to_uv` on the pooled
  grid with the same β.
- The native grid must be divisible by the target grid; raise otherwise.
- Pooling applies to the prefix tokens only. §3.4 always samples the unpooled
  map.

### 3.4 Per-vertex image features ("B")

Field: `image_vertex_features: bool = False`. Usable with `prefix` or `xattn`
fusion. Coordinate tokenization only (`tokenization: coord`, `tokenizer:
coord`); raise otherwise.

**Fine map.** A small CNN, trained from scratch, reads the same mesh-order
uint8 crop the DINOv3 encoder gets.
- Normalise with the encoder's mean/std.
- Output a stride-4 feature map: 64×64 for a 256-px crop.
- Suggested layout: conv3×3 s2 → GELU → conv3×3 s2 → GELU → conv3×3 s1, about
  64 channels. The exact layout is the builder's call.
- At stride 4 the map is 1.2·L/64 per cell: 0.19 m at L = 10 m, 0.375 m at
  20 m. A voxel bin is L/127.

**Coarse map.** The unpooled DINOv3 patch features, 16×16×1024, in mesh order.
This is the same tensor the prefix path computes before pooling. Compute it
once per batch and reuse it.

**Lookup.**
- For a vertex with x-bin `bx` and y-bin `by`:
  - u = bx/(num_bins−1) − 0.5 and v = by/(num_bins−1) − 0.5. This is
    `dequantize`.
  - This is valid only because x/y margins are 0, which holds in every v5/v6
    config. The module does not receive `margin_lo`/`margin_hi`.
  - **My call:** `train_mesh` (and posthoc's `build_split`) refuses
    `image_vertex_features` when x or y margins are non-zero, rather than adding
    margin hyperparameters to the module.
- Sample both maps bilinearly at (u, v) with `grid_sample`:
  - `align_corners=False`, grid coordinates `(u/(0.5+β), v/(0.5+β))`.
  - The crop spans [−0.5−β, 0.5+β] on both axes, and mesh-order rows run along
    +y.
- Concatenate the two samples, then apply LayerNorm → MLP → width
  `word_embed_proj_dim`.
- Zero-initialise the final layer, so B contributes exactly 0 at step 0.

**Which tokens get it.**
- Every coordinate token whose vertex's x and y are both known **at that
  position**, i.e. the y-token and the z-token of each vertex.
- In the condition, token i has axis i % 3.
- In the target input `[BOS, t0, t1, …]`, token i ≥ 1 is coordinate i−1.
- x-tokens, BOS, EOS and PAD get nothing.
- The y-token's output predicts the vertex's z, so the height prediction sees
  the image right under the vertex.
- The feature is **added to the input embedding**, alongside the segment
  embedding.

**Validity (my call).**
- B is zero for every row with `image_valid` False, regardless of
  `image_mask_invalid` and of the fusion type. Otherwise dropout and the
  `image_mode: none` control would leak the crop through B, under xattn and in
  the "`image_mask_invalid` off" ablation.
- `ImageContext` therefore carries `valid`. Today validity is used up inside
  the encoder.

**Cache.** `cached_logits` already embeds the whole target each step and keeps
only the new rows. Compute B over the whole target and slice the same way.
Training, uncached generation and cached generation then agree by construction.

**Location.** Put the CNN and the projection inside `OrthoImageEncoder`, e.g.
`image_encoder.pixel_cnn` and `image_encoder.vertex_proj`. They are then
automatically:
- covered by `IMAGE_PARAM_PREFIXES`, so they train under `train_scope:
  image_only` and get the image learning rate;
- allowed as missing keys in `load_weights_from`;
- kept in checkpoints. Only `backbone.*` is dropped.

### 3.5 Image-token attention (fix 5)

Field: `image_prefix_attention: str = "causal"`, either `causal` or
`bidirectional`. Requires `image_fusion: prefix` and
`image_separate_positions`.

| query \ key | image | LOD1 cond | target |
|---|---|---|---|
| image | all (both directions) | all | none |
| LOD1 cond | all | causal (direct keys) | none |
| target | all | all | causal |

Pad keys are masked as usual, and §3.2's masked image keys stay masked.

**Consequence.**
- With the image masked, every LOD1 and target row is exactly v5. The image
  rows' own outputs are never read, so they don't matter.
- With the image present, the condition is effectively bidirectional from
  layer 2 on. Image tokens read all of LOD1, and LOD1 token i reads image
  tokens, so it sees later LOD1 tokens through them. This is accepted (user
  decision, 2026-11-02). The whole condition is known at sampling time, so
  nothing leaks from the target, and the cache is unaffected.

**Mechanism.** Pass a 4-D attention mask. transformers 5.10.2's
`_preprocess_mask_arguments` returns a prepared 4-D mask as-is. The builder
still has to verify the expected dtype and convention (bool keep-mask or
additive float) for the attention implementation OPT uses here.
- In the cached path, the first call carries the whole prefix plus BOS and
  needs the 4-D mask.
- Later single-token steps attend to every non-pad past key, so a plain
  2-D/4-D keep-row is enough.

## 4. Backward compatibility (G4), concretely

- Old checkpoints (`7swbxcwh`, v5, xattn): `hyper_parameters` lack the new keys,
  so they get the defaults and today's code path. Their `state_dict`s load
  unchanged: no renamed or removed parameters, and `null` and `image_segment`
  are kept.
- The image-free model (`uses_image` False) runs none of the new code.
- `load_weights_from(v5)` into a new-mode model: the new parameters are all
  under `image_encoder.`, so they are allowed missing keys.
- `posthoc_eval` and `notebooks/test-mesh-ortho.ipynb` call
  `generate(image=..., image_valid=...)` / `_prepare` unchanged, so they work
  for both old and new checkpoints.
- `src/inference.py` still refuses every image-conditioned checkpoint (lines
  97–99). That is out of scope here and stays unchanged.

## 5. Config and run

- New `MeshModelConfig` fields, with the defaults above:
  `image_separate_positions`, `image_mask_invalid`, `image_prefix_grid`,
  `image_vertex_features`, `image_prefix_attention`, plus the CNN width if the
  builder makes it configurable.
- Validate them in config/module construction with clear errors.
- New config `configs/mesh-v6-ortho-prefix-aligned.yaml`, a copy of
  `mesh-v6-ortho-prefix.yaml` with all five on and `image_prefix_grid: 8`:
  - same random split and same `init_from` (v5 epoch 20);
  - run name `mesh-v6-ortho-prefix-aligned-random`;
  - otherwise the same training settings, so it is directly comparable with
    `7swbxcwh`.
- A matching `-smoke` config on mini_cleaner, 1 epoch.
- Runbook: add the arm, its smoke criterion, and the ablation recipe.
- **Ablations (my call).** These are documented as CLI overrides of the aligned
  config, not as separate config files. Which ones to run, and for how long, is
  your decision. Each switches off one piece of the full model:

  | ablation | override | removes |
  |---|---|---|
  | −B | `mesh_model.image_vertex_features=false` | per-vertex lookup |
  | −bidirectional | `mesh_model.image_prefix_attention=causal` | §3.5 |
  | −pooling | `mesh_model.image_prefix_grid=null` | back to 256 tokens |
  | −masking | `mesh_model.image_mask_invalid=false` | dropout and `none` go back to the null token |

  `image_separate_positions` cannot be switched off alone, because masking and
  bidirectional attention both require it. Its ablation is the old prefix run
  `7swbxcwh` itself, as long as that comparison is read as "all fixes vs none".
- **Smoke criterion (my call).** The smoke run on mini_cleaner is a plumbing
  check only:
  - it finishes an epoch;
  - losses stay finite;
  - `val_loss` falls.

  The v5-equivalence gate is test 2 in §6, which is exact. There is no v5
  number on mini_cleaner to compare a chamfer against.

## 6. Verification (tests first; tiny DINOv3 test config, no download, no full dataset)

1. **Golden backward compatibility.** Before any edit, record logits from
   today's code for small seeded models (image-free; `prefix`; `xattn`) on a
   fixed batch, and commit them as a fixture. After the change, the defaults
   reproduce them bit-for-bit (or to ≤1e-6 if the device forces it).
2. **v5 equivalence.** Load the same weights into an image-free model and into
   a new-mode model (§3.1 + §3.2 + §3.5, B on). With all `image_valid` False,
   the target logits that `forward` returns are equal (atol ~1e-5). Also the
   cached generation is token-identical.
3. **B starts at zero.** With the same weights, B on vs off gives identical
   logits at init.
4. **Cache.** Cached vs uncached generation is token-identical for each flag
   combination. Extend `tests/test_kv_cache.py`.
5. **Attention pattern.** Check the 4-D mask itself against the §3.5 table.
   Behaviourally:
   - perturbing a later LOD1 token changes image-token outputs;
   - with the image masked, it does not change earlier LOD1 outputs;
   - perturbing a target token never changes earlier target or any condition
     outputs.
6. **Alignment.** A synthetic crop with a distinct patch at a known (u,v): the
   CNN path and the DINOv3 path sample it at the bin that dequantizes to that
   (u,v). Include a +y check, so a row flip fails the test.
7. **Positions.** Mesh position ids are the same with and without the prefix,
   and the same under condition padding.
8. **Config validation.** Each invalid combination raises.
9. **Hyperparameter round trip.** A new-mode checkpoint saves the flags. An old
   one (keys absent) loads with the defaults.

## 7. Left to the builder

- The CNN's exact layers and width, and the vertex-MLP width.
- The sentinel value for "no position".
- Whether B also feeds the xattn arm in its first config (it is allowed; no
  xattn config needs to turn it on).
- How the 4-D mask is built, within the semantics of §3.5.

## 8. Blast radius (gitnexus, index one commit behind HEAD; the notebook only)

- `MeshOPTTransformer`: LOW (2 direct, 5 total; lower bound, through the
  `MeshEmbedding` dispatch boundary).
- `OrthoImageEncoder`: LOW (1 direct).
- `OPTSinusoidalPositions`: LOW (2 direct).
- In practice `MeshOPTTransformer.forward` / `cached_logits` are the hot path of
  every OPT run, training, eval, posthoc and inference included. G4 and test 1
  are what protect them, not the LOW rating. Re-run `impact` on the specific
  methods before editing them.
