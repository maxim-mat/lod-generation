# Aligned image prefix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Five config-ablatable changes to the v6 prefix path:
- separate positions;
- masking instead of the null token;
- 8×8 pooling;
- per-vertex image features (B);
- bidirectional image-token attention.

Each one's default reproduces today's logits bit-for-bit.

**Architecture:** All new behaviour is keyed off an extended `image` dict that `MeshTransformerModule` hands to `MeshOPTTransformer`.
- When `separate_positions` is on, the network passes OPT explicit `position_ids` and a 4-D boolean attention mask. Two pure helpers in a new `src/models/prefix_layout.py` build them.
- B lives in `OrthoImageEncoder`: a small stride-4 CNN, a bilinear sampler and a zero-initialised projection. Its maps travel on `ImageContext`.
- The defaults run exactly today's code path.

**Tech Stack:** PyTorch, transformers 5.10.2 (OPT, DINOv3ViT), Lightning, pytest.

**Spec:** `docs/superpowers/specs/2026-11-02-aligned-prefix-design.md`

## Global Constraints

- **Defaults.** Every new hyperparameter defaults to today's behaviour:
  - `image_separate_positions=False`
  - `image_mask_invalid=False`
  - `image_prefix_grid=None`
  - `image_vertex_features=False`
  - `image_prefix_attention="causal"`
  - `image_vertex_width=64`
- **Old checkpoints.** No existing parameter is renamed, removed or reshaped. `image_encoder.null` and `image_segment` stay.
- **Defaults construct nothing new.** With the defaults, no new `nn.Module` or `nn.Parameter` is created. Otherwise the RNG order changes and the golden fixture breaks.
- **Parameter names.** New parameters live under `image_encoder.`: `image_encoder.pixel_cnn.*` and `image_encoder.vertex_proj.*`.
- **Sentinel.** The no-position id is `-2` (`NO_POSITION`). The existing `-1 → clamp(0)` pad behaviour of `OPTSinusoidalPositions` is unchanged.
- **Mask builder for every new mode.** Whenever `separate_positions` is on, the full-sequence call uses the 4-D mask from `prefix_attention_mask`, even with causal attention. That is how fully masked image rows keep their diagonal key.
- **Tests.**
  - Tests use the tiny OPT/DINOv3 configs already in `tests/test_ortho_fusion.py`: hidden 32, 2 layers, `S=64`, so the native grid is 4.
  - No dataset loading, no downloads.
  - Pooling tests therefore use `image_prefix_grid=2`.
- **Before editing symbols.** Run `impact({target, direction: "upstream"})` before editing each named symbol, and stop on HIGH or CRITICAL.
- **Before committing.**
  - Run `detect_changes({scope: "all"})` before each commit.
  - Commit only this task's paths: `git add <paths>`, then `git commit <paths>`. The user's pre-staged files must not ride along (`.superpowers/`, `configs/mesh-v2-2-train.yaml`, `configs/mesh2-2train.yaml`, `configs/mesh-v6-ortho-prefix-smoke.yaml`, `docs/*.pptx`, `src/*_explorer.py`).
  - Do not push.
- **Commit messages** end with `Claude-Session: https://claude.ai/code/session_01DohCuvBB8nsE5SQdWuzZFb`.

## Review Focus

1. **Beam search in new mode.** `_beam_search` calls `image_ctx.repeat(beam)` and `reorder_cache`, so `valid` and `maps` must follow the beams. The cache's new keys must be reordered too. Expect beam cached == uncached. Test in Task 5.
2. **Condition padding in new mode.** A right-padded condition must give the same target logits as the unpadded one. Positions skip pad slots, and pad keys are masked. Test in Task 4.
3. **`image_mode: shuffled`, i.e. `valid` all True with another row's image.** B must read the given image, not the row's own; nothing keys B to the building id. Covered by Task 5's "B changes with the image" test.
4. **The mixed-validity batch.** Rows with valid True and False in one batch: the invalid row equals v5, and the valid row differs. Test in Task 4.
5. **`load_from_checkpoint` of a new-mode checkpoint.** The flags are restored from `hyper_parameters`, so the rebuilt model has the B parameters and the state dict loads strictly. Test in Task 6.

---

### Task 1: Golden backward-compatibility fixture (before any source edit)

**Files:**
- Create: `tests/test_golden_logits.py`
- Create: `tests/fixtures/golden_logits.pt` (written by the test file's `__main__`)

**Interfaces:**
- Produces: `build_cases() -> dict[str, dict[str, Tensor]]`, keyed `"image_free"`, `"prefix"` and `"xattn"`. Each case holds:
  - `"logits"`: `forward` output on the fixed batch;
  - `"greedy"`: 12-step cached greedy tokens.

- [ ] **Step 1: Write `tests/test_golden_logits.py`.**
  - Copy `net()`/`batch()` from `tests/test_ortho_fusion.py` locally; don't import across test files.
  - `build_cases()`:
    - for each fusion in `(None, "prefix", "xattn")`: `net(fusion, seed=0)`;
    - for xattn, `gates.fill_(0.7)`;
    - `batch(seed=1)`; image valid `[True, False]`, so the null path is pinned too;
    - record `forward(cond, tgt, image_ctx=...)` logits;
    - record greedy via `new_cache` / `cached_logits` (12 steps, argmax).
  - `test_defaults_reproduce_golden_logits()`:
    - load the fixture;
    - assert `torch.equal` on the greedy tokens;
    - assert `torch.allclose(atol=1e-6, rtol=0)` on the logits for every case.
  - `if __name__ == "__main__": torch.save(build_cases(), FIXTURE)`.
- [ ] **Step 2: Record the fixture on the unmodified tree.** Run `python -m tests.test_golden_logits`. Expect `tests/fixtures/golden_logits.pt` to exist.
- [ ] **Step 3: Run the test.** Run `pytest tests/test_golden_logits.py -v`. Expect PASS.
- [ ] **Step 4: Commit.** `test: golden logits fixture for the image paths`.

---

### Task 2: Mask and position helpers, and the no-position sentinel

**Files:**
- Create: `src/models/prefix_layout.py`
- Modify: `src/models/mesh_transformer.py` (`OPTSinusoidalPositions.forward`, ~line 121)
- Test: `tests/test_prefix_layout.py`

**Interfaces:**
- Produces:
  - `NO_POSITION = -2`
  - `mesh_position_ids(n_img: int, mesh_keep: BoolTensor[B, M]) -> LongTensor[B, n_img + M]`
    - The first `n_img` entries are `NO_POSITION`.
    - Then `cumsum(mesh_keep) * mesh_keep - 1`, so pads are `-1`.
  - `prefix_attention_mask(img_keep: Bool[B,P], cond_keep: Bool[B,C], tgt_keep: Bool[B,T], bidirectional: bool) -> Bool[B, 1, P+C+T, P+C+T]`, where True means attend. Rules:
    - `allowed = (k <= q) & keep[k]`;
    - if bidirectional, also `(q < P) & (k < P + C) & keep[k]`;
    - always `q == k` for `q < P`, the diagonal.
- `OPTSinusoidalPositions.forward`: output rows where `position_ids == NO_POSITION` are zero. Every other id behaves as today.

- [ ] **Step 1: Write the failing tests in `tests/test_prefix_layout.py`.**
  - `test_mesh_positions_skip_pads_and_ignore_the_prefix`:
    - `mesh_keep=[[1,1,0,1]]`, `n_img=2` → `[[-2,-2,0,1,-1,2]]`;
    - with `n_img=0` → `[[0,1,-1,2]]`.
  - `test_causal_mask_matches_the_table`, with P=2, C=2, T=2, all keep, bidirectional False: the mask equals `tril(ones(6,6))`.
  - `test_bidirectional_mask_matches_the_table`, same sizes, bidirectional True:
    - rows 0–1 (image) are True on columns 0–3 and False on 4–5;
    - rows 2–3 (cond) are True on 0–1; row 2 is False on column 3 (direct cond attention is causal);
    - rows 4–5 equal tril.
  - `test_masked_image_keeps_only_its_diagonal`, `img_keep=[[0,0]]`:
    - column 0–1 entries are False everywhere except `[0,0]` and `[1,1]`;
    - no row is all-False.
  - `test_pad_keys_are_masked`: with `cond_keep=[[1,0]]`, column `P+1` is all False. Condition slots get no diagonal exemption.
  - `test_no_position_sentinel_is_a_zero_vector`:
    - `OPTSinusoidalPositions(8)(mask, position_ids=tensor([[-2, -1, 0, 3]]))`;
    - row 0 is all zeros;
    - row 1 equals row 2 (`-1` still clamps to 0);
    - row 3 is non-zero.
- [ ] **Step 2: Run them.** Run `pytest tests/test_prefix_layout.py -v`. Expect FAIL with `ModuleNotFoundError: src.models.prefix_layout`.
- [ ] **Step 3: Implement.**
  - Write `prefix_layout.py`, with a module docstring pointing at spec §3.1/§3.5.
  - Run impact on `OPTSinusoidalPositions`.
  - Add the sentinel zeroing: `masked_fill` after `self.embed(...)`.
- [ ] **Step 4: Run the tests.** Run `pytest tests/test_prefix_layout.py tests/test_golden_logits.py -v`. Expect PASS.
- [ ] **Step 5: Commit.** `feat(models): prefix position/mask helpers and no-position sentinel`.

---

### Task 3: Encoder — pooling, masking-aware tokens, vertex maps and sampling; `ImageContext` fields

**Files:**
- Modify: `src/models/ortho_encoder.py` (`OrthoImageEncoder.__init__`, `forward`)
- Modify: `src/models/ortho_fusion.py` (`ImageContext`)
- Test: `tests/test_ortho_encoder.py` (append)

**Interfaces:**
- `ImageContext` gains:
  - `valid: Optional[Tensor] = None`, a [B] bool;
  - `maps: Optional[tuple] = None`, a pair `(fine [B,Cf,Hf,Wf], coarse [B,C,G,G])` in mesh order (row index along +y).
  - `select`, `repeat` and `caching` carry both, indexing or `repeat_interleave`-ing dim 0.
  - `kv` stays the second positional field, so `ImageContext(tokens)` and `ImageContext(tokens, {})` callers are unchanged.
- `OrthoImageEncoder.__init__(out_dim, name=None, config=None, image_size=256, beta=0.1, n_freq=8, prefix_grid=None, vertex_dim=None, vertex_width=64)`
  - `prefix_grid`: pool to this grid before `proj`; the `uv` buffer is built on it. `ValueError` if `self.grid % prefix_grid`.
  - `vertex_dim`: not None builds `pixel_cnn` and `vertex_proj`, output width `vertex_dim`.
    - `pixel_cnn = Sequential(Conv2d(3,w,3,2,1), GELU, Conv2d(w,w,3,2,1), GELU, Conv2d(w,w,3,1,1))`, with `w = vertex_width`.
    - `vertex_proj = Sequential(LayerNorm(w + C), Linear(w + C, vertex_dim), GELU, Linear(vertex_dim, vertex_dim))`, with the last Linear's weight and bias zeroed.
- `OrthoImageEncoder.encode(image, valid=None, use_null=True) -> ImageContext`
  - Computes `features()` once.
  - `tokens` come from pooled features (if `prefix_grid`) through `proj` + `coord(uv)`. The null token replaces invalid rows only when `use_null`.
  - `maps` is set only when `vertex_dim`; it holds `(pixel_cnn(normalised mesh-order crop), unpooled features as [B,C,G,G])`.
  - `valid` is always set; all True when None is given.
- `forward(image, valid=None)` stays, returning `encode(image, valid).tokens`, which keeps the existing tests and the golden fixture.
- `sample_map(fmap: Tensor[B,C,H,W], u: Tensor[B,L], v: Tensor[B,L], beta: float) -> Tensor[B,L,C]`
  - Uses `F.grid_sample(fmap, grid, mode="bilinear", align_corners=False)` with `grid[...,0] = u/(0.5+beta)` (x indexes W, the columns) and `grid[...,1] = v/(0.5+beta)` (y indexes H, the rows).
  - It is a module-level function.
- `OrthoImageEncoder.vertex_features(maps, bx: Long[B,L], by: Long[B,L], num_bins: int) -> Tensor[B,L,vertex_dim]`
  - Computes `u = bx/(num_bins-1) - 0.5` and `v` likewise.
  - Samples both maps, concatenates, and applies `vertex_proj`.
- The CNN input is the same normalisation as `features()` minus its `.flip(-2)`: mesh order, not north-up.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_ortho_encoder.py`).
  - `test_sample_map_reads_a_ramp_back_exactly`:
    - fmap `[1,2,8,8]`, where channel 0 holds the cell-centre u of column j and channel 1 the cell-centre v of row i, both from `pixel_to_uv(i, j, 8, 0.1)`;
    - for bins `bx, by` in an interior set, e.g. `[(20,100),(64,64),(100,30)]` at `num_bins=128`, sampling at the dequantized (u, v) returns (u, v) within 1e-5;
    - a row-flip bug swaps v's sign, so the `(20,100)` case fails.
  - `test_pooled_prefix_has_grid_squared_tokens`:
    - `tiny(prefix_grid=2)` on 64-px crops gives 4 tokens;
    - `prefix_grid=3` raises `ValueError`.
  - `test_encode_without_null_leaves_invalid_rows_unreplaced`:
    - `encode(img, valid=[False], use_null=False).tokens` equals `encode(img, [True]).tokens`;
    - `.valid` is `[False]`.
  - `test_vertex_features_start_at_zero`:
    - `tiny(vertex_dim=32)`;
    - `vertex_features(encode(img).maps, bx, by, 128)` is all zeros;
    - `pixel_cnn.0.weight` exists in `state_dict()`.
  - `test_fine_map_is_in_mesh_order`:
    - replace `enc.pixel_cnn` with `nn.AvgPool2d(4)`;
    - the crop is bright on mesh-order rows 0–3 and columns 0–3, dark elsewhere;
    - fine-map cell `[0,0]` is the brightest cell, i.e. no flip.
  - `test_image_context_select_and_repeat_carry_valid_and_maps`: shapes and the order of `valid` follow `select([1,0])` and `repeat(3)`.
- [ ] **Step 2: Run them.** Run `pytest tests/test_ortho_encoder.py -v`. Expect the new tests to FAIL (`TypeError: unexpected keyword 'prefix_grid'`, …).
- [ ] **Step 3: Implement.** Run impact on `OrthoImageEncoder` and `ImageContext` first.
- [ ] **Step 4: Run the tests.** Run `pytest tests/test_ortho_encoder.py tests/test_ortho_fusion.py tests/test_golden_logits.py -v`. Expect PASS.
- [ ] **Step 5: Commit.** `feat(ortho): pooled prefix, vertex feature maps and bilinear sampler`.

---

### Task 4: Network — separate positions, masking, bidirectional attention

**Files:**
- Modify: `src/models/mesh_transformer.py`: `MeshOPTTransformer.__init__`, `encode_image`, `forward`, `new_cache`, `cached_logits`, `reorder_cache`
- Test: `tests/test_aligned_prefix.py` (create)

**Interfaces:**
- Consumes: Task 2 helpers; Task 3 `encode(..., use_null)` and the `ImageContext.valid` field.
- The `image` dict accepts these optional keys, read with `.get`:
  - `separate_positions` (False)
  - `mask_invalid` (False)
  - `prefix_grid` (None)
  - `prefix_attention` ("causal")
  - `vertex_features` (False)
  - `vertex_width` (64)
- `MeshOPTTransformer._check_image_options(image, pos_embed, tokenization, codebook)` is a staticmethod called in `__init__`. It raises `ValueError` when:
  - `separate_positions` is on and fusion ≠ prefix or `pos_embed` ≠ sinusoidal;
  - `mask_invalid` is on without `separate_positions`;
  - `prefix_attention` is not in {causal, bidirectional};
  - `bidirectional` is on without prefix + `separate_positions`;
  - `prefix_grid` is set and fusion ≠ prefix;
  - `vertex_features` is on and tokenization ≠ coord or `codebook` is not None.
- `__init__` passes `prefix_grid=image.get("prefix_grid")` to `OrthoImageEncoder`. It sets `self.separate_positions`, `self.mask_invalid` and `self.bidirectional` on every network; they are all False for an image-free one.
- `encode_image(image, valid=None)`:
  - returns `self.image_encoder.encode(image, valid, use_null=not mask_invalid)`;
  - in the default mode this is identical to today.
- New-mode `forward`, when `separate_positions` is on:
  - `img_keep = ctx.valid` if `mask_invalid`, else all True, expanded to [B, P];
  - `mesh_keep = ~cat(cond_pad, tgt_pad[:, :-1])`, or all True;
  - call `self.opt(inputs_embeds=x, attention_mask=prefix_attention_mask(...), position_ids=mesh_position_ids(P, mesh_keep))`.
- The cache dict gains, in new mode only:
  - `"mesh_keep"`: Bool [B, n_cond + n_tgt], grown each step;
  - `"img_keep"`: Bool [B, P].
- `cached_logits`:
  - First call: the full 4-D mask and full position ids.
  - Later calls:
    - 2-D `attention_mask = cat(img_keep, mesh_keep).long()`;
    - `position_ids = mesh_position_ids(0, mesh_keep)[:, -n_new:]`.
- `reorder_cache` reorders both new keys when present.
- The default-mode code paths are untouched; branch on `self.separate_positions`.

- [ ] **Step 1: Verify the 4-D mask convention.**
  - Write `test_4d_bool_mask_equals_the_default_causal_mask` first.
  - On an image-free `net(None)`, compare two calls:
    - `opt(inputs_embeds=x, attention_mask=ones(B,L))`;
    - `opt(inputs_embeds=x, attention_mask=prefix_attention_mask(empty P, keep, empty T, False), position_ids=mesh_position_ids(0, keep))`.
  - The logits must be allclose at atol 1e-6.
  - Run it. If it fails because HF wants an additive float mask, convert in one place (`torch.where(mask, 0.0, finfo.min)` cast to the embed dtype) inside a `_hf_mask(mask, dtype)` helper in `prefix_layout.py`, and use that helper everywhere.
- [ ] **Step 2: Write the failing behaviour tests in `tests/test_aligned_prefix.py`.**
  - `net(**flags)` builds a prefix network whose `image` dict carries the flags (the copied helper from `test_ortho_fusion`).
  - `NEW = dict(separate_positions=True, mask_invalid=True, prefix_attention="bidirectional")`.
  - Tests:
    - `test_masked_image_reproduces_the_image_free_model` (spec test 2):
      - `base = net(None)`; `new = net("prefix", **NEW)`, loading `base.state_dict()` with `strict=False`;
      - with `valid=[False, False]`, `new(cond, tgt, image_ctx=ctx)` is allclose to `base(cond, tgt)` (atol 1e-5).
    - `test_mixed_validity_rows` (Review Focus 4): with `valid=[False, True]`, row 0 equals base and row 1 does not.
    - `test_padded_condition_gives_the_same_target_logits` (Review Focus 2):
      - same network, row 0's condition with 3 PAD tokens appended and `cond_pad_mask` set, vs unpadded;
      - target logits allclose (atol 1e-5).
    - Helper `_last_hidden(m, cond, tgt, ctx) -> Tensor[B, P+C+T-1, D]`: run `m(cond, tgt, image_ctx=ctx)` with a forward hook on `m.opt.model.decoder.layers[-1]` that captures `output[0]`. Image slots come first, then the condition.
    - `test_bidirectional_lets_image_tokens_see_later_lod1`: with valid True, perturbing `cond[:, -1]` changes `_last_hidden[:, :P]`.
    - `test_masked_image_keeps_lod1_causal`: with valid False, perturbing `cond[:, -1]` leaves `_last_hidden[:, P:P+n_cond-1]` unchanged (atol 1e-6).
    - `test_target_stays_causal`: perturbing `tgt[:, 10]` leaves `forward` logits `[:, :10]` unchanged, in every flag combination.
    - `test_mesh_positions_do_not_move_with_the_prefix` (spec test 7):
      - capture `position_ids` passed to `decoder.embed_positions` via a forward hook;
      - mesh positions equal `arange(n_cond + n_tgt - 1)`, and the image entries are `NO_POSITION`.
    - `test_cached_matches_uncached` (spec test 4): parametrized over the flag combos `{}`, `{"separate_positions": True}`, `NEW`, `{**NEW, "prefix_attention": "causal"}`, `{**NEW, "prefix_grid": 2}`, with valid `[True, False]`; greedy tokens are equal (reuse `_greedy` from `test_ortho_fusion`, copied).
    - `test_invalid_option_combinations_raise` (spec test 8): one `pytest.raises(ValueError)` per rule in `_check_image_options`.
- [ ] **Step 3: Run them.** Run `pytest tests/test_aligned_prefix.py -v`. Expect FAIL (`TypeError` or no behaviour change).
- [ ] **Step 4: Implement.** Run impact on `MeshOPTTransformer.forward`, `cached_logits` and `new_cache` first.
- [ ] **Step 5: Run the tests.** Run `pytest tests/test_aligned_prefix.py tests/test_ortho_fusion.py tests/test_kv_cache.py tests/test_golden_logits.py tests/test_mesh_transformer.py -v`. Expect PASS.
- [ ] **Step 6: Commit.** `feat(models): separate positions, masked image and bidirectional prefix attention`.

---

### Task 5: Network — per-vertex image features (B)

**Files:**
- Modify: `src/models/prefix_layout.py` (add `vertex_xy_bins`)
- Modify: `src/models/mesh_transformer.py` (`MeshOPTTransformer.__init__`, `forward`, `cached_logits`)
- Test: `tests/test_prefix_layout.py`, `tests/test_aligned_prefix.py`

**Interfaces:**
- `vertex_xy_bins(ids: Long[B,L], offset: int, num_bins: int) -> (bx: Long[B,L], by: Long[B,L], has: Bool[B,L])`
  - Position `p` is coordinate `c = p - offset`, axis `c % 3`.
  - For axis 1: `bx = ids[p-1]`, `by = ids[p]`.
  - For axis 2: `bx = ids[p-2]`, `by = ids[p-1]`.
  - `has = (c >= 0) & (axis >= 1) & (bx < num_bins) & (by < num_bins)`.
  - Elsewhere the bins are 0 and `has` is False.
  - Offset 0 is for the condition; offset 1 is for the target input `[BOS, t0, …]`.
- In `__init__`: when `vertex_features` is on, build the encoder with `vertex_dim=cfg.word_embed_proj_dim, vertex_width=image["vertex_width"]` (both fusions).
- `MeshOPTTransformer._vertex_embed(ids, offset, ctx) -> Tensor[B,L,D] | 0`
  - Computes `image_encoder.vertex_features(ctx.maps, bx, by, num_bins) * (has & ctx.valid[:, None])[..., None]`.
  - Returns plain `0` when B is off or `ctx` is None.
  - Added to `embed_cond(cond)` (offset 0) and to `embed_tokens(tgt[:, :-1])` (offset 1) in `forward`.
  - In `cached_logits` it is added to the whole-target embedding (offset 1) *before* the new rows are sliced, and to the condition on the first call.

- [ ] **Step 1: Write the failing tests.**
  - `test_vertex_xy_bins_condition_and_target`:
    - cond `[[1,2,3, 4,5,6, PAD,PAD,PAD]]` at offset 0:
      - `has = [0,1,1, 0,1,1, 0,0,0]`;
      - bx at positions 1, 2 = 1; by at positions 1, 2 = 2;
    - target input `[[BOS, 1,2,3, 4]]` at offset 1:
      - `has = [0, 0,1,1, 0]`;
      - position 2 gives bx=1, by=2.
  - `test_vertex_features_start_at_zero` (spec test 3): `net("prefix", **NEW, vertex_features=True)` loaded from `net("prefix", **NEW).state_dict()` (`strict=False`) gives identical logits.
  - `test_vertex_features_follow_the_image` (Review Focus 3):
    - after `vertex_proj[-1].weight.normal_()`, two different crops with valid True give different logits;
    - with valid False the logits are identical to each other.
  - `test_vertex_features_also_work_with_xattn`: `net("xattn", vertex_features=True)` runs and starts at zero (identical to the same weights with B off).
  - `test_cached_matches_uncached` gains `{**NEW, "vertex_features": True}`, with `vertex_proj[-1].weight.normal_()` applied so B is non-zero.
  - `test_beam_cached_matches_uncached` (Review Focus 1):
    - at module level, `MeshTransformerModule(..., image_fusion="prefix", image_separate_positions=True, image_mask_invalid=True, image_prefix_attention="bidirectional", image_vertex_features=True)`, with the final weights randomised;
    - `generate(beam_size=3, use_cache=True)` equals `use_cache=False`.
    - This test needs Task 6's hyperparameters, so it lands in Task 6, Step 1.
- [ ] **Step 2: Run them.** Run `pytest tests/test_prefix_layout.py tests/test_aligned_prefix.py -v`. Expect the new tests to FAIL.
- [ ] **Step 3: Implement.** Run impact on `MeshOPTTransformer.forward` first.
- [ ] **Step 4: Run the tests.** Run `pytest tests/test_prefix_layout.py tests/test_aligned_prefix.py tests/test_ortho_fusion.py tests/test_golden_logits.py -v`. Expect PASS.
- [ ] **Step 5: Commit.** `feat(models): per-vertex orthophoto features at the vertex's ground cell`.

---

### Task 6: Module hyperparameters, config, `train_mesh`, posthoc margin guard

**Files:**
- Modify: `src/models/mesh_transformer.py` (`MeshTransformerModule.__init__` and its docstring)
- Modify: `src/utils/config.py` (`MeshModelConfig`)
- Modify: `src/train_mesh.py` (`train_mesh`, plus a new `check_vertex_margins`)
- Modify: `src/eval/posthoc_eval.py` (`main`, after `load_model` at ~line 570)
- Test: `tests/test_ortho_module.py` (append), `tests/test_ortho_wiring.py` (append)

**Interfaces:**
- `MeshTransformerModule.__init__` gains keyword args with the Global Constraints defaults:
  - `image_separate_positions`
  - `image_mask_invalid`
  - `image_prefix_grid`
  - `image_vertex_features`
  - `image_prefix_attention`
  - `image_vertex_width`

  They are saved by `save_hyperparameters` and forwarded into the `image` dict.

  If any of them is non-default while `uses_image` is False, it raises `ValueError`.
- `MeshModelConfig` gets the same six fields, unprefixed by `mesh_model.`, with the same defaults. Each gets a one-line comment that cites the spec section.
- `check_vertex_margins(vertex_features: bool, margin_lo, margin_hi) -> None` in `src/train_mesh.py`. It raises `ValueError` if `vertex_features` is on and any of `margin_lo[0:2]` / `margin_hi[0:2]` is non-zero. The message says B's bin→pixel map assumes x/y margins of 0.
- Call it in `train_mesh` before building the model.
- Call it in posthoc `main` with `model.hparams.get("image_vertex_features", False)` and `ref_cfg.mesh_data` margins.

- [ ] **Step 1: Write the failing tests.**
  - In `test_ortho_module.py`:
    - `test_new_flags_round_trip_through_a_checkpoint` (spec test 9 and Review Focus 5):
      - save a new-mode `module(image_fusion="prefix", image_separate_positions=True, image_mask_invalid=True, image_prefix_grid=2, image_vertex_features=True, image_prefix_attention="bidirectional")` with `trainer`-free `torch.save({"state_dict", "hyper_parameters", "pytorch-lightning_version"})`;
      - rebuild with `MeshTransformerModule.load_from_checkpoint(path, image_encoder_config=dino, opt_config=opt)`;
      - the flags match, `load_state_dict(strict=True)` succeeds, and the logits are equal.
    - `test_old_checkpoint_without_the_keys_gets_defaults`:
      - the same, but pop the six keys from `hyper_parameters`;
      - the rebuilt network has `separate_positions` False and no `image_encoder.pixel_cnn`.
    - `test_new_flags_need_an_image_model`: `module(image=False, image_mask_invalid=True)` raises.
    - The beam test from Task 5.
  - In `test_ortho_wiring.py`:
    - `test_check_vertex_margins`: `check_vertex_margins(True, [0,0,0], [0,0,0.1])` passes; `[0.1,0,0]` raises; `(False, [0.1,0,0], …)` passes.
    - `test_config_defaults_keep_the_old_prefix`: a fresh `MeshModelConfig()` has the six defaults.
- [ ] **Step 2: Run them.** Run `pytest tests/test_ortho_module.py tests/test_ortho_wiring.py -v`. Expect FAIL.
- [ ] **Step 3: Implement.** Run impact on `MeshTransformerModule`, `MeshModelConfig`, `train_mesh` and `main` (posthoc) first.
- [ ] **Step 4: Run the full suite.** Run `pytest tests -q -x --ignore=tests/test_metric_sanity_real.py`. Expect all passed. The real-data sanity test is excluded because it reads the dataset.
- [ ] **Step 5: Commit.** `feat: aligned-prefix flags on the module, config and train/posthoc entry points`.

---

### Task 7: Configs and runbook

**Files:**
- Create: `configs/mesh-v6-ortho-prefix-aligned.yaml`
- Create: `configs/mesh-v6-ortho-prefix-aligned-smoke.yaml`
- Modify: `docs/ortho-conditioning-runbook.md`
- Test: `tests/test_ortho_wiring.py` (append)

**Interfaces:**
- The aligned config:
  - is a copy of `configs/mesh-v6-ortho-prefix.yaml`, with the header comment rewritten to point at the spec;
  - adds `image_separate_positions: true`, `image_mask_invalid: true`, `image_prefix_grid: 8`, `image_vertex_features: true` and `image_prefix_attention: bidirectional`;
  - sets `run_name: mesh-v6-ortho-prefix-aligned-random`;
  - leaves everything else identical, including `init_from`, `split: random`, the lr and `max_epochs`.
- The smoke config is a copy of the user's `configs/mesh-v6-ortho-prefix-smoke.yaml` with the same five flags. Read that file; don't modify or commit it.

- [ ] **Step 1: Write the failing tests.**
  - Add `("prefix-aligned", "random")` and `("prefix-aligned-smoke", "random")` to `test_shipped_v6_configs_load`'s parametrize list.
  - `test_aligned_differs_from_prefix_only_in_the_new_flags`:
    - load both configs;
    - `asdict` them;
    - the differing keys are exactly the five flags plus `logging.run_name`.
  - `test_aligned_smoke_matches_aligned_but_for_size`: the same five flags plus `init_from` are equal.
- [ ] **Step 2: Run them.** Run `pytest tests/test_ortho_wiring.py -v`. Expect FAIL with `FileNotFoundError`.
- [ ] **Step 3: Write both configs and the runbook section.** The runbook section covers:
  - the arm and its command;
  - the smoke criterion: one epoch finishes, losses are finite, and `val_loss` falls;
  - the ablation table of CLI overrides from spec §5;
  - that `7swbxcwh` is the "no fixes" comparison.
- [ ] **Step 4: Run the tests.** Run `pytest tests/test_ortho_wiring.py -v`. Expect PASS.
- [ ] **Step 5: Commit.** `feat(configs): aligned prefix arm, smoke config and runbook`. Also commit the spec and this plan here, if they're not already committed.
