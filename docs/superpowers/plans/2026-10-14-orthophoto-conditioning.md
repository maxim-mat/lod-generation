# Orthophoto Conditioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Condition the v5 OPT mesh transformer on the building's top-down orthophoto. The primary
approach adds gated cross-attention into a frozen DINOv3 encoding, trained in two stages:
frozen v5 first, then joint. Prefix tokens are a cheap control arm.

**Architecture:**
- **Image encoder:** a frozen DINOv3 ViT encodes the dataset's aligned crop, which `MeshDataset`
  already returns in the mesh frame. A trainable projector plus a 2-D coordinate embedding turn it into
  image tokens.
- **Cross-attention:** these image tokens enter OPT through one gated cross-attention block per decoder
  layer. The blocks are injected with forward pre-hooks, so OPT's modules and the v5 state-dict keys
  stay untouched. Zero-initialised gates make step 0 of stage A exactly v5.
- **Prefix control:** the same image tokens are prepended as a third sequence segment instead.
- **No imagery:** buildings without imagery, and image dropout, use a learned null token.

**Tech Stack:**
- PyTorch 2.9.1, Lightning;
- transformers 5.10.2 (`OPTForCausalLM`, `DINOv3ViTModel`, `AutoImageProcessor`);
- rasterio 1.4 (already in use);
- pytest.

**Spec:** `docs/research/2026-10-14-orthophoto-conditioning.md` (sections 2–7). Data pipeline:
`C:\Users\Maxim\.claude\plans\create-a-plan-to-iterative-swing.md` (implemented).

## What the real data changed (read before Task 1)

Measured from `data/The Hague/{cleaner,mini_cleaner}/ortho/ortho_manifest.json` and the
alignment check of 2026-10-14:

| Assumption in the research doc | Measured | Consequence in this plan |
|---|---|---|
| LOD2 mostly from 2022 point clouds (one tile) | `cleaner`: **2023 64.9%**, 2022 22.6%, 2020 10.0%, 2014 2.6% | **Decided 2026-10-14: 2022 + 2023 imagery.** Each building uses the year closest to its `b3_pw_datum` (2020/2014 → 2022), so 87.5% match exactly. Mosaics live in `ortho/<year>/<Source>/<name>.tif` with `ortho/year_assignment.json`; `MeshDataset` reads each building's own year. The evaluation still stratifies by `b3_pw_datum` (Task 8). |
| LOD2 may reach past the LOD1 box, needing a crop margin | `need_beta` p99 ≈ 1e-11; only 0.04% need > 0.05 | The margin is pure context. Keep `image_margin=0.1`; mosaics allow up to 0.25. |
| Every building has imagery | 500 `cleaner` buildings (0.19%) had coverage ≈ 0: three tiles at x = 123 km returned HTTP 403 | The block fallback (Task 1) recovered all three on the 2022+2023 download. **Decided 2026-10-14: buildings still without an orthophoto are skipped**, after the train/val/test split so v5's split is preserved (Task 2). |
| Parse reports missing tiles | 26 reported, only 3 truly missing: it counts tiles touched by 327 m read chunks | Fix the report (Task 1). |
| Crop is aligned | Footprints match roofs; ~0.5 m edge offsets; row-house crops are full of neighbours | Stratify the evaluation by height (lean) and keep the coordinate embedding so the model can localise its own roof (Tasks 3, 8). |
| DINOv3 output layout | transformers 5.10: `last_hidden_state = [CLS, registers…, patches row-major]`, `num_register_tokens` in config | The encoder skips `1 + num_register_tokens` and reshapes the grid (Task 3). |

## Global Constraints

- **Python and libraries:** Python 3.12 on Windows. `numpy==1.26.4` (scipy 1.13 needs numpy < 2.3);
  `rasterio>=1.4,<1.5`; `affine>=2.3,<3`. Add no other dependency (no Pillow, no shapely);
  `tests/test_requirements_complete.py` must stay green.
- **v5 must stay unchanged:** `ortho_dir: null` and `image_encoder: null` reproduce v5 exactly. Existing
  checkpoints keep loading with `MeshTransformerModule.load_from_checkpoint`. The defaults of every new
  kwarg keep old behaviour.
- **OPT internals:** never subclass or edit Hugging Face OPT internals. Cross-attention is injected by
  `register_forward_pre_hook(..., with_kwargs=True)` on `opt.model.decoder.layers[i]`.
- **DINOv3:** always frozen, always in eval mode, and never in checkpoints. Strip it with
  `register_state_dict_post_hook` and restore it with `register_load_state_dict_pre_hook`.
- **Image order:** the dataset image is `uint8 [B, 3, S, S]` in **mesh order**: row index along +y,
  column index along +x. DINOv3 must see it **north-up**, so flip the rows before the backbone and flip
  the patch grid back after.
- **Image-conditioned models need an OPT backbone:** `backbone: scratch` with an image encoder raises
  `ValueError`.
- **Tests:**
  - Tiny configs only: `OPTConfig(hidden_size=32, …)` and
    `DINOv3ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
    num_attention_heads=2, patch_size=16, image_size=64)`.
  - No hub downloads.
  - Never the real dataset, except the existing skip-if-absent sanity test.
- **One split scheme per comparison:** models are compared only if trained on the same split scheme
  (`training.split`: `random` or `tile`). `load_weights_from` refuses a checkpoint whose recorded
  `data_split` differs; a missing key counts as `random`.
- **Reproducibility artifacts are never trimmed:** seeds, manifests, the git commit, I/O error handling,
  and docstrings on functions shared across scripts.
- **Before editing:** run gitnexus `impact` on each symbol you are about to modify.
  `MeshTransformerModule.generate` and `run_mesh_eval` are expected to be HIGH, so
  stop and get the user's confirmation for each.
- **Before every commit:** gitnexus `detect_changes`.
- **Literature:** cite each method at the point of implementation: Flamingo arXiv:2204.14198, DINOv3
  arXiv:2508.10104, Idefics2 arXiv:2405.02246, LLaVA arXiv:2304.08485.

## Review Focus

1. **Skipping image-less buildings must not move the split.** Every kept building stays in the
   split it had in v5. Otherwise an image run initialised from v5 validates on buildings v5 trained on.
   Covered in Task 2 (split-membership test).
2. **Orientation into DINOv3:** a north-up input and a mesh-order output grid. A flip mistake is silent
   and halves what the model can learn. Covered by the fake-backbone test in Task 3.
3. **Loading v5 weights into the image model** (`init_from`): only image keys may be missing. Anything
   else (a renamed OPT key, a tokenization mismatch) must raise. Covered in Task 6.
4. **Beam search with images:** the per-beam image context must be reordered with the KV cache, so
   cached beams must equal uncached beams. Covered in Task 4 and Task 5.
5. **Initialising across split schemes:** v5 (random split) has trained on buildings inside the tile
   split's val/test tiles, so `init_from` a random-split checkpoint into a tile-split run must raise.
   Covered in Task 2a (`data_split` hparam) and Task 5 (`load_weights_from` check).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/ortho_tiles.py` (modify) | `sheets_for_tile` returns every candidate flight block |
| `src/ortho_download.py` (modify) | Try the candidate blocks in order when one returns 403 |
| `src/ortho_parse.py` (modify) | Report only the missing tiles a building window needs |
| `src/dataset/mesh_dataset.py`, `src/dataset/mesh_datamodule.py` (modify) | `has_image` from manifest coverage; `MeshDataset.image(i)`; `MeshDataModule` drops image-less buildings **after** the split |
| `src/dataset/splits.py` (create) | `random_split_indices` (v5's exact split) and `tile_split_indices` (whole 3DBAG tiles per split) |
| `src/analysis/spatial_correlation.py` (create) | ICC by tile, distance correlogram, nearest-train optimism of random vs tile split |
| `src/models/ortho_encoder.py` (create) | `OrthoImageEncoder`: frozen DINOv3, orientation, projector, coordinate embedding, null token, checkpoint hooks |
| `src/models/ortho_fusion.py` (create) | `ImageContext`, `CrossAttention`, `GatedCrossAttention` (forward pre-hooks, cached K/V) |
| `src/models/mesh_transformer.py` (modify) | `MeshOPTTransformer`: `encode_image`, image-aware forward/cache paths, prefix mode. `MeshTransformerModule`: kwargs, dropout, generate plumbing, train scope, lr groups, `load_weights_from`, gate logging |
| `src/utils/config.py`, `src/train_mesh.py` (modify) | Config fields and wiring, `init_from` |
| `src/eval/mesh_eval.py`, `src/eval/posthoc_eval.py` (modify) | Pass images to `generate`; post-hoc `image_mode` (`true`/`shuffled`/`none`) |
| `src/inference.py` (modify) | Refuse image-conditioned checkpoints (out of scope) |
| `src/analysis/ortho_strata.py` (create) | Join post-hoc `objects.csv` runs with `b3_pw_datum`/height and report per stratum |
| `configs/mesh-v6-ortho-*.yaml` (create) | Stage A, stage B, prefix control, mini smoke |
| `docs/ortho-conditioning-runbook.md` (create) | How to run and read the stages |

---

### Task 1: Data-pipeline fixes found in the real run

> **Status: DONE (2026-10-14)**, folded into the multi-year data rework.
> - `sheets_for_tile` and the 403 fallback are in `src/ortho_download.py`.
> - The window-only `missing_tiles` report is in `src/ortho_parse.py`.
> - Both scripts now take `--years`. Tests are in `tests/test_ortho_{tiles,download,parse}.py`.
>
> The steps below are kept as the record of what was built. Skip to Task 2.

**Files:**
- Modify: `src/ortho_tiles.py` (`sheet_for_tile`)
- Modify: `src/ortho_download.py` (`download_all`)
- Modify: `src/ortho_parse.py` (`mosaic_file`)
- Test: `tests/test_ortho_tiles.py`, `tests/test_ortho_download.py`, `tests/test_ortho_parse.py`

**Interfaces:**
- Produces: `sheets_for_tile(xmin, ymax, index) -> list[(perceel, blok)]`. The block containing the
  centre comes first, then every other block whose polygon contains a tile corner. `sheet_for_tile`
  stays as `(sheets_for_tile(...) or [None])[0]`.
- Produces: `mosaic_file(...)["missing_tiles"]` lists only tiles that intersect some window.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ortho_tiles.py
def test_sheets_for_tile_lists_every_block_touching_the_tile(tmp_path):
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"BLOK": "3", "PERCEEL": "Perceel 1"},
         "geometry": {"type": "MultiPolygon", "coordinates": [[[
             [80000, 450000], [81600, 450000], [81600, 460000], [80000, 460000], [80000, 450000]]]]}},
        {"type": "Feature", "properties": {"BLOK": "12", "PERCEEL": "Perceel 2"},
         "geometry": {"type": "MultiPolygon", "coordinates": [[[
             [81600, 450000], [90000, 450000], [90000, 460000], [81600, 460000], [81600, 450000]]]]}},
    ]}
    index = load_sheet_index(fc)
    assert sheets_for_tile(81000, 455000, index) == [(1, 3), (2, 12)]   # centre block first
    assert sheets_for_tile(83000, 455000, index) == [(2, 12)]
    assert sheets_for_tile(10000, 455000, index) == []
```

```python
# tests/test_ortho_download.py
def test_a_403_falls_back_to_the_next_candidate_block(tmp_path):
    from src.ortho_download import download_all
    from src.ortho_tiles import load_sheet_index, tile_url
    index = load_sheet_index({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"BLOK": "3", "PERCEEL": "Perceel 1"},
         "geometry": {"type": "MultiPolygon", "coordinates": [[[
             [80000, 450000], [81600, 450000], [81600, 460000], [80000, 460000], [80000, 450000]]]]}},
        {"type": "Feature", "properties": {"BLOK": "12", "PERCEEL": "Perceel 2"},
         "geometry": {"type": "MultiPolygon", "coordinates": [[[
             [81600, 450000], [90000, 450000], [90000, 460000], [81600, 460000], [81600, 450000]]]]}}]})
    good = tile_url(81000, 455000, 2, 12, YEAR)
    opener = FakeOpener({good: tile_bytes(81000, 455000)})       # the centre block URL is absent -> 403
    stats = download_all({(81000, 455000)}, index, tmp_path, YEAR, opener=opener,
                         sleep=lambda s: None, tile_px=TILE_PX)
    assert stats["downloaded"] == 1 and stats["failed"] == {}
```

`FakeOpener` must raise `urllib.error.HTTPError(url, 403, "Forbidden", {}, None)` for unknown URLs.
Update its `__call__`:

```python
        if req.full_url not in self.bodies:
            raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, None)
```

```python
# tests/test_ortho_parse.py
def test_missing_tiles_lists_only_tiles_a_window_needs(tmp_path):
    raw_dir = tmp_path / "raw"
    write_raw_tile(raw_dir, 80000, 451000)
    # One window inside tile 80000; the 4096 px chunk (2 km at GSD 0.5) also spans 81000,
    # which no window needs and which is absent.
    stats = mosaic_file({"a": (80100.0, 450100.0, 80200.0, 450200.0)}, raw_dir / str(YEAR),
                        tmp_path / "m.tif", YEAR, gsd=GSD)
    assert stats["missing_tiles"] == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ortho_tiles.py tests/test_ortho_download.py tests/test_ortho_parse.py -q -p no:warnings`

Expected:
- `ImportError: cannot import name 'sheets_for_tile'`;
- the download test fails with `failed` containing the tile;
- the parse test asserts a non-empty `missing_tiles`.

- [ ] **Step 3: Implement**

In `src/ortho_tiles.py`, replace `sheet_for_tile` with:

```python
def sheets_for_tile(xmin, ymax, index):
    """Candidate ``(perceel, blok)`` pairs for a tile, most likely first.

    The block containing the tile centre is the portal's own assignment; blocks
    containing only a corner are fallbacks for tiles on a block boundary, where
    the delivery folder does not always follow the centre (seen as HTTP 403).
    """
    def inside(exterior, holes, pt):
        return exterior.contains_point(pt) and not any(h.contains_point(pt) for h in holes)

    centre = (xmin + TILE_M / 2, ymax - TILE_M / 2)
    corners = [(xmin + 1, ymax - 1), (xmin + TILE_M - 1, ymax - 1),
               (xmin + 1, ymax - TILE_M + 1), (xmin + TILE_M - 1, ymax - TILE_M + 1)]
    first = [(p, b) for ext, holes, p, b in index if inside(ext, holes, centre)]
    rest = [(p, b) for ext, holes, p, b in index
            if (p, b) not in first and any(inside(ext, holes, c) for c in corners)]
    return first + rest


def sheet_for_tile(xmin, ymax, index):
    """``(perceel, blok)`` of the block containing the tile centre, or None."""
    return (sheets_for_tile(xmin, ymax, index) or [None])[0]
```

In `src/ortho_download.py` `download_all`, keep every candidate URL and try them in order:

```python
    jobs = {}
    for tx, ty in sorted(tiles):
        name = tile_name(tx, ty, year)
        sheets = sheets_for_tile(tx, ty, index)
        if not sheets:
            stats["not_in_index"].append(name)
            continue
        jobs[name] = ([tile_url(tx, ty, *s, year) for s in sheets],
                      Path(out_dir) / str(year) / name, (tx, ty))

    def fetch(urls, dest, origin):
        errors = []
        for url in urls:
            try:
                return download_tile(url, dest, origin, opener=opener, sleep=sleep,
                                     tile_px=tile_px)
            except RuntimeError as exc:
                errors.append(str(exc))
        raise RuntimeError(" | ".join(errors))

    with ThreadPoolExecutor(workers) as ex:
        futures = {ex.submit(fetch, *job): name for name, job in jobs.items()}
        # ... unchanged result loop ...
```

Import `sheets_for_tile` and drop the now-unused `sheet_for_tile` import if nothing else uses it.
`--check` and `--dry-run` keep using `sheet_for_tile`.

In `download_tile`, a 403 should not burn four retries per candidate. Treat
`urllib.error.HTTPError` with `code in (403, 404)` as final:

```python
        except urllib.error.HTTPError as exc:
            last = exc
            part.unlink(missing_ok=True)
            if exc.code in (403, 404):
                break
            if attempt + 1 < retries:
                sleep(backoff * 2 ** attempt)
        except (OSError, ValueError, KeyError, rasterio.errors.RasterioError) as exc:
            ...unchanged...
```

(Add `import urllib.error`. The `HTTPError` clause must come before the `OSError` one, because
`HTTPError` subclasses it.)

In `src/ortho_parse.py` `mosaic_file`, compute the needed tiles once and record a tile as missing only
if it is in that set:

```python
    needed = set()
    for box in boxes:
        needed |= tiles_for_bounds(box)
    ...
                            if not path.is_file():
                                if (tx, ty) in needed:
                                    missing.add(name)
                                continue
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ortho_tiles.py tests/test_ortho_download.py tests/test_ortho_parse.py -q -p no:warnings`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/ortho_tiles.py src/ortho_download.py src/ortho_parse.py tests/test_ortho_tiles.py tests/test_ortho_download.py tests/test_ortho_parse.py
git commit -m "fix(ortho): block fallback on 403, report only window-needed missing tiles"
```

(Re-running the download for the three 403 tiles, and re-parsing, is user-run data work in Task 9.)

---

### Task 2: Skip buildings without an orthophoto, without moving the v5 split

**Decision (user, 2026-10-14):** buildings without an orthophoto are **skipped**, not fed a null
token.

**Why the skip happens after the split:**
- Dropping them inside `MeshDataset` would change its length, and with it the seeded `random_split`.
- v6's val/test would then contain buildings v5 trained on, and stage A starts from v5's weights:
  train/test leakage.
- So `MeshDataset` keeps every building and only records `has_image`. `MeshDataModule` splits exactly
  as v5 does, then removes image-less buildings from each split.
- A kept building is therefore in the same split as in v5, and the two runs can be compared on
  identical buildings.

**What "without an orthophoto" means:** mosaic coverage below `image_min_coverage` (default 1.0).
Coverage comes from `ortho_manifest.json` → `totals.partial_coverage`, where an absent id means 1.0.
After the 2022+2023 re-download (the three 403 tiles were recovered), expect this to be a handful of
buildings at most; the parse manifest has the exact count.

**Not affected:** the model's null token (Task 3) stays. It is still used for training-time
`image_dropout` and the post-hoc `image_mode: none` control (Task 7), and it is fed through
`generate(..., image_valid=...)`, never through dataset items. `mesh_collate_fn` is **not** modified.

**Files:**
- Modify: `src/dataset/mesh_dataset.py` (`MeshDataset.__init__`, `_open_ortho`, new `image`)
- Modify: `src/dataset/mesh_datamodule.py` (`setup`)
- Modify: `src/utils/config.py` (`MeshDataConfig.image_min_coverage`)
- Modify: `src/train_mesh.py`, `src/eval/posthoc_eval.py` (`build_split`)
- Test: `tests/test_mesh_dataset_ortho.py`

**Interfaces:**
- Produces: `MeshDataset.has_image: list[bool]`. Every entry is True when `ortho_dir` is None.
- Produces: `MeshDataset.image(index) -> uint8 [3,S,S]`, the crop of building `index` in its own frame
  and its own imagery year. Task 7 uses it for shuffled-image evaluation.
- Produces: `MeshDataModule.{train,val,test}_dataset` are `Subset`s of the **unfiltered** split with
  image-less indices removed, and `MeshDataModule.skipped_no_image: dict[str, int]` per split.
- Produces: `MeshDataModule(..., image_min_coverage=1.0)` and
  `MeshDataConfig.image_min_coverage: float = 1.0`.

- [ ] **Step 1: Run impact** on `MeshDataModule.setup` and `MeshDataset.__init__`, then report it.

- [ ] **Step 2: Write the failing tests** (append to `tests/test_mesh_dataset_ortho.py`)

```python
def _with_coverage(subset, tmp_path, coverage):
    """A copy of the fixture subset whose manifest reports `coverage` ({id: fraction})."""
    root = tmp_path / "copy"
    shutil.copytree(subset, root)
    path = root / "ortho" / "ortho_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["totals"]["partial_coverage"] = coverage
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_dataset_keeps_every_building_and_flags_missing_imagery(subset, tmp_path):
    root = _with_coverage(subset, tmp_path, {"b": 0.0, "c": 0.97})
    ds = _ds(root, ortho_dir="ortho", image_size=S)
    assert ds.ids == ["a", "b", "c", "d"]                       # nothing dropped here
    assert dict(zip(ds.ids, ds.has_image)) == {"a": True, "b": False, "c": False, "d": True}
    loose = _ds(root, ortho_dir="ortho", image_size=S, image_min_coverage=0.9)
    assert dict(zip(loose.ids, loose.has_image)) == {"a": True, "b": False, "c": True, "d": True}
    assert "image_valid" not in ds[0]


def test_datamodule_skips_after_splitting_so_kept_buildings_keep_their_v5_split(subset, tmp_path):
    root = _with_coverage(subset, tmp_path, {"b": 0.0})
    kw = dict(lod_in="LOD1", lod_out="LOD2", batch_size=1, seed=7,
              train_val_test_split=(0.5, 0.25, 0.25))
    plain = MeshDataModule(root, **kw)
    plain.setup()
    with_img = MeshDataModule(root, ortho_dir="ortho", image_size=16, **kw)
    with_img.setup()
    for split in ("train", "val", "test"):
        before = [plain.full_dataset.ids[i] for i in getattr(plain, f"{split}_dataset").indices]
        after = [with_img.full_dataset.ids[i] for i in getattr(with_img, f"{split}_dataset").indices]
        assert after == [oid for oid in before if oid != "b"], split
    assert sum(with_img.skipped_no_image.values()) == 1


def test_image_accessor_returns_a_buildings_crop(subset):
    ds = _ds(subset, ortho_dir="ortho", image_size=S)
    assert torch.equal(ds.image(2), ds[2]["image"])
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_mesh_dataset_ortho.py -q -p no:warnings`
Expected: `AttributeError: 'MeshDataset' object has no attribute 'has_image'`, `TypeError` for the
unknown `image_min_coverage`, and `AttributeError` for `skipped_no_image` / `image`.

- [ ] **Step 4: Implement**

`MeshDataset.__init__`:
1. Add `image_min_coverage=1.0` after `image_margin`, and document it in the class docstring:
   "buildings whose mosaic coverage is below this get `has_image=False`; `MeshDataModule` drops them
   after splitting".
2. Before the `if ortho_dir is not None:` block: `self.has_image = [True] * len(ids)`.
3. Inside that block, after `self.ortho = ...`:

```python
            coverage = self._manifest_totals.get("partial_coverage", {})
            self.has_image = [coverage.get(i, 1.0) >= image_min_coverage for i in ids]
            if not all(self.has_image):
                logger.info("%d/%d buildings have no orthophoto (coverage < %.2f); "
                            "MeshDataModule drops them after splitting",
                            self.has_image.count(False), len(ids), image_min_coverage)
```

`_open_ortho`: replace `built = json.loads(...)["args"]` with

```python
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        built = manifest["args"]
        self._manifest_totals = manifest.get("totals", {})
```

New method:

```python
    def image(self, index):
        """Crop of building `index`, in its own frame and imagery year.

        For evaluation controls that pair a building with another's image.
        """
        item = self._tokens(index)
        return self.ortho.crop(self.image_files[index], item["center"].numpy(),
                               item["scale"].numpy())
```

`MeshDataModule`:
1. Add the kwarg `image_min_coverage=1.0`, store it and pass it to `MeshDataset`.
2. In `setup`, after `random_split`:

```python
        # Split first, then drop buildings without an orthophoto: the split
        # stays exactly v5's, so an image run initialised from v5 never
        # validates on buildings v5 trained on.
        self.skipped_no_image = {}
        if self.full_dataset.ortho is not None:
            keep = self.full_dataset.has_image
            for name in ("train", "val", "test"):
                subset = getattr(self, f"{name}_dataset")
                kept = [i for i in subset.indices if keep[i]]
                self.skipped_no_image[name] = len(subset.indices) - len(kept)
                setattr(self, f"{name}_dataset", Subset(self.full_dataset, kept))
            logger.info("Skipped buildings without an orthophoto: %s", self.skipped_no_image)
```

(Import `Subset` from `torch.utils.data` and add `logger = logging.getLogger(__name__)` with
`import logging`.)

**Thread `image_min_coverage` through:**
- `MeshDataConfig`: `image_min_coverage: float = 1.0`, with the comment "buildings whose mosaic covers
  less than this fraction of their crop are skipped (after the train/val/test split)".
- `train_mesh.py`: `image_min_coverage=cfg.mesh_data.image_min_coverage`.
- `posthoc_eval.build_split`: `image_min_coverage=m.image_min_coverage`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_mesh_dataset_ortho.py tests/test_mesh_tokenizer.py tests/test_mesh_eval.py -q -p no:warnings`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/dataset/mesh_dataset.py src/dataset/mesh_datamodule.py src/utils/config.py src/train_mesh.py src/eval/posthoc_eval.py tests/test_mesh_dataset_ortho.py
git commit -m "feat(data): skip buildings without an orthophoto after the split; MeshDataset.image"
```

---

### Task 2a: Geographic (tile-based) train/val/test split

**Why:**
- Neighbouring buildings are correlated: same street, same developer, same era, same roof style.
- The current split is a seeded random permutation of buildings. Almost every val/test building
  therefore has train neighbours from the same block, and the scores are optimistic about truly
  unseen areas. That isn't leakage in the strict sense; it's a representativeness problem.

**What changes:**
- **Unit:** a 3DBAG tile, i.e. one CityJSON file: ~0.5 km, ~1,000 buildings, 246 in `cleaner`.
- **Assignment:** each tile goes whole to train, val or test. Tiles are shuffled with the seed, then
  greedily assigned to whichever split is furthest below its target share of buildings.
- **Default stays `random`:** `random` reproduces the v5 split exactly, which is pinned by a test, so
  every existing config and checkpoint is unaffected.

**The consequence that matters:** models are comparable only when trained on the same split scheme.
v5 was trained on the random split, so its weights have seen buildings inside the tile split's
val/test tiles. Initialising an image model from v5 and evaluating on tile splits leaks. To make that
impossible to do by accident:
- the module records the split as a hyperparameter (`data_split`);
- `load_weights_from` (Task 5) refuses a checkpoint trained on a different scheme. A checkpoint
  without the key counts as `random`.

If Task 2b confirms the correlation, Task 9 retrains v5 on the tile split before stage A.

**Files:**
- Create: `src/dataset/splits.py`
- Modify: `src/dataset/mesh_dataset.py` (`MeshDataset.files`), `src/dataset/mesh_datamodule.py`
  (`split` kwarg, `setup`)
- Modify: `src/utils/config.py` (`TrainingConfig.split`), `src/train_mesh.py` (pass the split, write
  `split.json`, pass `data_split`), `src/eval/posthoc_eval.py` (`build_split`)
- Modify: `src/models/mesh_transformer.py` (`MeshTransformerModule(..., data_split="random")`, recorded only)
- Test: `tests/test_splits.py`

**Interfaces:**
- Produces in `src/dataset/splits.py`:
  - `random_split_indices(n, ratios, seed) -> list[list[int]]`, index-identical to what
    `torch.utils.data.random_split` gives `MeshDataModule` today;
  - `tile_split_indices(groups, ratios, seed) -> (list[list[int]], {group: split_index})`.
- Produces: `MeshDataset.files: list[str]`, the LOD2 file path relative to the LOD folder, per building.
- Produces: `MeshDataModule(..., split="random")` with `split in ("random", "tile")`, and
  `MeshDataModule.split_groups` ({tile: "train"|"val"|"test"}; empty for random).
- Produces: `TrainingConfig.split: str = "random"`.
- Produces: `MeshTransformerModule(..., data_split="random")`, saved in `hparams`.

- [ ] **Step 1: Run impact** on `MeshDataModule.setup`, `MeshDataset.__init__` and `MeshTransformerModule`,
  and report it.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_splits.py
"""Train/val/test splitting: random (v5's exact split) and geographic by tile."""
import collections

import pytest
import torch
from torch.utils.data import random_split

from src.dataset.splits import random_split_indices, tile_split_indices

RATIOS = (0.8, 0.1, 0.1)


def test_random_split_is_index_identical_to_torch_random_split():
    n, seed = 1037, 42
    sizes = [int(0.8 * n), int(0.1 * n)]
    sizes.append(n - sum(sizes))
    ref = random_split(range(n), sizes, generator=torch.Generator().manual_seed(seed))
    assert random_split_indices(n, RATIOS, seed) == [list(s.indices) for s in ref]


def _groups():
    # 120 tiles of uneven size, like 3DBAG tiles.
    sizes = [5 + (37 * t) % 90 for t in range(120)]
    return [f"tile-{t}" for t, k in enumerate(sizes) for _ in range(k)]


def test_tile_split_never_puts_one_tile_in_two_splits_and_covers_everything():
    groups = _groups()
    splits, assign = tile_split_indices(groups, RATIOS, seed=3)
    seen = sorted(i for s in splits for i in s)
    assert seen == list(range(len(groups)))
    for k, idx in enumerate(splits):
        assert {groups[i] for i in idx} == {g for g, s in assign.items() if s == k}


def test_tile_split_hits_the_target_shares_within_one_tile():
    groups = _groups()
    splits, _ = tile_split_indices(groups, RATIOS, seed=3)
    biggest = max(collections.Counter(groups).values())
    for share, idx in zip(RATIOS, splits):
        assert abs(len(idx) - share * len(groups)) <= biggest


def test_tile_split_is_deterministic_per_seed():
    groups = _groups()
    assert tile_split_indices(groups, RATIOS, 3) == tile_split_indices(groups, RATIOS, 3)
    assert tile_split_indices(groups, RATIOS, 3)[1] != tile_split_indices(groups, RATIOS, 4)[1]


def test_datamodule_tile_split_keeps_each_file_in_one_split(tmp_path):
    from src.dataset.mesh_datamodule import MeshDataModule
    from tests.test_ortho_tiles import write_tile

    root = tmp_path / "The Hague" / "mini_cleaner"
    for t in range(6):
        boxes = {f"t{t}-b{b}": (100 * t + 20 * b, 0, 100 * t + 20 * b + 10, 10, 5) for b in range(4)}
        for lod in ("LOD1", "LOD2"):
            write_tile(root / lod / "Source B" / f"10-{t}-0.city.json", boxes)
    dm = MeshDataModule(root, lod_in="LOD1", lod_out="LOD2", batch_size=1, seed=1,
                        train_val_test_split=(0.5, 0.25, 0.25), split="tile")
    dm.setup()
    files = dm.full_dataset.files
    per_split = {name: {files[i] for i in getattr(dm, f"{name}_dataset").indices}
                 for name in ("train", "val", "test")}
    assert not (per_split["train"] & per_split["val"] or per_split["train"] & per_split["test"]
                or per_split["val"] & per_split["test"])
    assert set(dm.split_groups.values()) <= {"train", "val", "test"}


def test_split_defaults_keep_v5_random():
    from pathlib import Path

    from src.utils.config import Config
    from src.utils.initialization import load_config
    assert Config().training.split == "random"
    assert load_config(Path("configs/mesh-v5-full-opt-coord.yaml"), []).training.split == "random"


def test_unknown_split_is_refused(tmp_path):
    from src.dataset.mesh_datamodule import MeshDataModule
    with pytest.raises(ValueError, match="split"):
        MeshDataModule(tmp_path, split="spatial")
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_splits.py -q -p no:warnings`
Expected: `ModuleNotFoundError: No module named 'src.dataset.splits'`.

- [ ] **Step 4: Implement** `src/dataset/splits.py`:

```python
"""Train/val/test index splits for `MeshDataModule`.

"random" reproduces the split every run up to v5 used (`torch.utils.data.random_split`
with a seeded generator), index for index. "tile" assigns whole 3DBAG tiles (one
CityJSON file each) to a split: neighbouring buildings are correlated (see
`src.analysis.spatial_correlation`), so a building-level split puts close
relatives of every test building into train and scores optimistically.
"""
import collections

import numpy as np
import torch


def _sizes(n, ratios):
    sizes = [int(ratios[0] * n), int(ratios[1] * n)]
    return sizes + [n - sum(sizes)]


def random_split_indices(n, ratios, seed):
    """``[train, val, test]`` index lists, identical to `torch.utils.data.random_split`."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    out, start = [], 0
    for size in _sizes(n, ratios):
        out.append(perm[start:start + size])
        start += size
    return out


def tile_split_indices(groups, ratios, seed):
    """Whole groups per split, balanced by item count.

    Groups are shuffled with `seed`, then each goes to the split furthest below
    its target share -- so no group straddles two splits and every split lands
    within one group's size of its target.

    Args:
        groups: one group label (tile) per item.
        ratios: (train, val, test) shares, summing to 1.

    Returns:
        tuple: ([train, val, test] index lists, {group: split index}).
    """
    sizes = collections.Counter(groups)
    names = sorted(sizes)
    np.random.default_rng(seed).shuffle(names)
    targets = [r * len(groups) for r in ratios]
    counts, assign = [0, 0, 0], {}
    for g in names:
        k = max(range(3), key=lambda s: (targets[s] - counts[s], -s))
        assign[g] = k
        counts[k] += sizes[g]
    out = [[], [], []]
    for i, g in enumerate(groups):
        out[assign[g]].append(i)
    return out, assign
```

`MeshDataset.__init__`: after `self.pairs = ...`, add
`self.files = [files_out[i] for i in ids]`, with the comment "LOD2 file per building: the geographic
split's unit".

`MeshDataModule`:
1. Add the kwarg `split="random"` and validate it in `__init__`:
   ```python
   if split not in ("random", "tile"):
       raise ValueError(f"Unknown split {split!r}; expected 'random' or 'tile'.")
   ```
2. In `setup`, replace the `random_split` call:

```python
        ratios = self.train_val_test_split
        if self.split == "tile":
            parts, assign = tile_split_indices(self.full_dataset.files, ratios, self.seed)
            names = ("train", "val", "test")
            self.split_groups = {g: names[k] for g, k in assign.items()}
        else:
            parts = random_split_indices(total, ratios, self.seed)
            self.split_groups = {}
        self.train_dataset, self.val_dataset, self.test_dataset = (
            Subset(self.full_dataset, p) for p in parts)
```

The Task 2 skip-after-split block follows this unchanged. Remove the now-unused `random_split` and
`torch.Generator` usage.

`TrainingConfig`: `split: str = "random"`, with the comment "'random' = per building (every run up to
v5); 'tile' = whole 3DBAG tiles per split, see src/analysis/spatial_correlation.py".

`train_mesh.py`:
- pass `split=cfg.training.split` to `MeshDataModule` and `data_split=cfg.training.split` to
  `MeshTransformerModule`;
- after `datamodule.setup()`, write the split as a reproducibility artifact:
  ```python
  _write_json(save_dir / "split.json",
              {"scheme": cfg.training.split, "seed": cfg.seed,
               "ratios": list(cfg.training.train_val_test_split),
               "sizes": {n: len(getattr(datamodule, f"{n}_dataset"))
                         for n in ("train", "val", "test")},
               "tiles": datamodule.split_groups})
  ```
  (Use `json.dumps(..., indent=2)` to `write_text`; don't import from `posthoc_eval`.)

`posthoc_eval.build_split`: pass `split=ref_cfg.training.split`.

`MeshTransformerModule.__init__`: add `data_split="random"`. It is stored by `save_hyperparameters()`
and has no other effect.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_splits.py tests/test_mesh_dataset_ortho.py tests/test_mesh_transformer.py -q -p no:warnings`
Expected: all pass. The random-split equality test proves v5's split is unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/dataset/splits.py src/dataset/mesh_dataset.py src/dataset/mesh_datamodule.py src/utils/config.py src/train_mesh.py src/eval/posthoc_eval.py src/models/mesh_transformer.py tests/test_splits.py
git commit -m "feat(data): geographic tile-based split; v5 random split reproduced exactly"
```

---

### Task 2b: Measure the spatial correlation the tile split assumes

**Goal:** a report that answers three questions with numbers, so the decision to retrain on the tile
split rests on evidence, not intuition:
1. Do buildings within a tile resemble each other more than buildings across tiles? Measured by the
   intraclass correlation (ICC(1)) of log footprint area, log LOD2 volume and log LOD2 face count, and
   the within-tile vs total standard deviation, against a tile-label permutation null.
2. Over what distance do neighbours resemble each other? A distance correlogram: the correlation
   between pairs of buildings binned by centroid distance (0–25 m … 2–4 km).
3. How optimistic is the random split? For each val/test building, its nearest train building under
   the random split vs under the tile split: the distance to it, and how similar its footprint and
   volume are.

**Metrics:**
- **Footprint area:** projected area of the LOD1 mesh's base (triangles lying at the lowest z). LOD1 is
  an extrusion, so its base is the footprint.
- **Volume:** |`signed_volume`| of the **LOD2** mesh only. `cleaner` guarantees closed shells.
- **Face count:** the number of **LOD2 triangles**. That is the unit the model generates (9 tokens each)
  and the unit `mesh_data.max_faces` filters on, so it is the closest of the three to what the model
  learns. CityJSON polygons are not counted.
- All three are log-transformed, because building sizes and complexity are heavy-tailed.

**Files:**
- Create: `src/analysis/spatial_correlation.py`
- Test: `tests/test_spatial_correlation.py`

**Interfaces:**
- Consumes: `random_split_indices`, `tile_split_indices` (Task 2a); `parse_cityjson_file_to_meshes`,
  `signed_volume` (`src/dataset/mesh_dataset.py`).
- Produces:
  - `building_metrics(lod1_path, lod2_path) -> {id: (x, y, footprint_m2, volume_m3, n_faces)}`
    (footprint from LOD1; volume and triangle count from LOD2);
  - `icc1(values, groups) -> float`;
  - `permutation_null(values, groups, n_perm, seed) -> np.ndarray`;
  - `correlogram(xy, values, edges, n_anchors, seed) -> DataFrame[lo, hi, n_pairs, corr]`;
  - `nearest_train(xy, values, train_idx, eval_idx) -> dict(median_dist_m, corr, median_abs_diff)`.
- Produces the CLI
  `python -m src.analysis.spatial_correlation "data/The Hague/cleaner" --out outputs/spatial_correlation`,
  which writes `report.json`, `report.md` and `buildings.csv`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_spatial_correlation.py
"""Spatial-correlation statistics on synthetic fields whose answer is known."""
import numpy as np
import pytest

from src.analysis.spatial_correlation import (building_metrics, correlogram, icc1,
                                              nearest_train, permutation_null)
from tests.test_ortho_tiles import write_tile


def test_building_metrics_reads_footprint_volume_and_centroid(tmp_path):
    for lod, h in (("LOD1", 6), ("LOD2", 6)):
        write_tile(tmp_path / lod / "t.city.json", {"a": (100, 200, 110, 208, h)})
    (x, y, area, vol, faces), = building_metrics(tmp_path / "LOD1" / "t.city.json",
                                                 tmp_path / "LOD2" / "t.city.json").values()
    assert (x, y) == pytest.approx((105.0, 204.0))
    assert area == pytest.approx(80.0) and vol == pytest.approx(480.0)
    assert faces == 12                    # six quads -> twelve LOD2 triangles


def test_icc_is_one_for_identical_members_and_zero_for_iid_noise():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(200), 25)
    clustered = rng.normal(size=200)[groups]
    assert icc1(clustered, groups) == pytest.approx(1.0, abs=1e-9)
    assert abs(icc1(rng.normal(size=groups.size), groups)) < 0.02


def test_permutation_null_centres_on_zero_and_rejects_real_clustering():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(100), 20)
    values = rng.normal(size=100)[groups] + rng.normal(size=groups.size)
    null = permutation_null(values, groups, n_perm=200, seed=2)
    assert abs(null.mean()) < 0.02
    assert (null >= icc1(values, groups)).mean() == 0.0


def test_correlogram_decays_with_distance_on_a_smooth_field():
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 4000, size=(6000, 2))
    values = np.sin(xy[:, 0] / 300) + np.cos(xy[:, 1] / 300) + 0.3 * rng.normal(size=6000)
    table = correlogram(xy, values, edges=[0, 50, 200, 1000, 3000], n_anchors=1500, seed=4)
    corr = table["corr"].to_numpy()
    assert corr[0] > 0.7 and corr[-1] < corr[0] - 0.5
    assert (table["n_pairs"] > 0).all()


def test_nearest_train_is_closer_and_more_similar_under_a_random_split():
    from src.dataset.splits import random_split_indices, tile_split_indices
    rng = np.random.default_rng(5)
    xy = rng.uniform(0, 4000, size=(8000, 2))
    values = np.sin(xy[:, 0] / 300) + 0.2 * rng.normal(size=8000)
    tiles = [f"{int(x // 500)}-{int(y // 500)}" for x, y in xy]
    r_train, r_val, r_test = random_split_indices(len(xy), (0.8, 0.1, 0.1), 0)
    (t_train, t_val, t_test), _ = tile_split_indices(tiles, (0.8, 0.1, 0.1), 0)
    rand = nearest_train(xy, values, r_train, r_val + r_test)
    tile = nearest_train(xy, values, t_train, t_val + t_test)
    assert rand["median_dist_m"] < tile["median_dist_m"]
    assert rand["corr"] > tile["corr"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_spatial_correlation.py -q -p no:warnings`
Expected: `ModuleNotFoundError: No module named 'src.analysis.spatial_correlation'`.

- [ ] **Step 3: Implement** `src/analysis/spatial_correlation.py`:

```python
"""Are neighbouring buildings alike? Evidence for (or against) a geographic split.

Three measurements on log footprint area (LOD1 base), log volume (LOD2) and log
face count (LOD2 triangles, the unit the model generates):
  * ICC(1) across 3DBAG tiles, with the within-tile / total SD ratio, against a
    null from shuffled tile labels;
  * a distance correlogram of building pairs;
  * nearest-train-neighbour distance and similarity for val/test buildings under
    the random split (every run up to v5) vs the tile split
    (`src.dataset.splits`). A random split that finds closer, more similar train
    neighbours is scoring on easier-than-real test data.

Usage:
    python -m src.analysis.spatial_correlation "data/The Hague/cleaner" \\
        --out outputs/spatial_correlation
"""
import argparse
import json
import logging
import multiprocessing as mp
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.dataset.mesh_dataset import parse_cityjson_file_to_meshes, signed_volume
from src.dataset.splits import random_split_indices, tile_split_indices
from src.filter_cityjson import _git_commit

logger = logging.getLogger(__name__)
EDGES_M = [0, 25, 50, 100, 250, 500, 1000, 2000, 4000]


def _base(verts, faces, tol=1e-3):
    """Projected area and area-weighted centroid of the triangles at the mesh's lowest z."""
    tri = verts[faces]
    flat = (np.abs(tri[:, :, 2] - verts[:, 2].min()) < tol).all(axis=1)
    t = tri[flat][:, :, :2]
    area = 0.5 * np.abs((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
                        - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1]))
    centroid = (t.mean(axis=1) * area[:, None]).sum(axis=0) / max(area.sum(), 1e-12)
    return float(area.sum()), centroid


def building_metrics(lod1_path, lod2_path):
    """``{id: (x, y, footprint_m2, volume_m3, n_faces)}`` for buildings present in both files.

    Footprint and centroid come from the LOD1 base; volume and triangle count from LOD2.
    """
    lod1 = parse_cityjson_file_to_meshes(lod1_path)
    lod2 = parse_cityjson_file_to_meshes(lod2_path)
    out = {}
    for oid, (v2, f2) in lod2.items():
        if oid not in lod1:
            continue
        area, (x, y) = _base(*lod1[oid])
        out[oid] = (float(x), float(y), area, abs(float(signed_volume(v2, f2))), int(len(f2)))
    return out


def icc1(values, groups):
    """One-way ANOVA ICC(1) with unequal group sizes (n0 correction)."""
    values = np.asarray(values, float)
    codes, groups = np.unique(groups, return_inverse=True)
    k, n = len(codes), len(values)
    sizes = np.bincount(groups)
    means = np.bincount(groups, weights=values) / sizes
    grand = values.mean()
    msb = (sizes * (means - grand) ** 2).sum() / (k - 1)
    msw = ((values - means[groups]) ** 2).sum() / (n - k)
    n0 = (n - (sizes ** 2).sum() / n) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


def permutation_null(values, groups, n_perm=200, seed=0):
    """ICC(1) under shuffled group labels: what 'no spatial structure' looks like."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    return np.array([icc1(values, rng.permutation(groups)) for _ in range(n_perm)])


def correlogram(xy, values, edges=EDGES_M, n_anchors=20000, seed=0):
    """Pearson correlation of building pairs, binned by centroid distance.

    Pairs are drawn from `n_anchors` random anchors to all buildings within
    the largest edge, so the cost stays linear in the anchor count.
    """
    xy, z = np.asarray(xy, float), np.asarray(values, float)
    z = (z - z.mean()) / z.std()
    rng = np.random.default_rng(seed)
    anchors = rng.choice(len(xy), size=min(n_anchors, len(xy)), replace=False)
    tree = cKDTree(xy)
    sums = np.zeros(len(edges) - 1)
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    for a, neighbours in zip(anchors, tree.query_ball_point(xy[anchors], r=edges[-1])):
        nb = np.asarray([j for j in neighbours if j != a], dtype=np.int64)
        if not len(nb):
            continue
        d = np.linalg.norm(xy[nb] - xy[a], axis=1)
        b = np.digitize(d, edges) - 1
        ok = (b >= 0) & (b < len(counts))
        np.add.at(sums, b[ok], z[a] * z[nb[ok]])
        np.add.at(counts, b[ok], 1)
    return pd.DataFrame({"lo": edges[:-1], "hi": edges[1:], "n_pairs": counts,
                         "corr": sums / np.maximum(counts, 1)})


def nearest_train(xy, values, train_idx, eval_idx):
    """Distance to, and similarity with, each eval building's nearest train building."""
    xy, values = np.asarray(xy, float), np.asarray(values, float)
    train_idx, eval_idx = np.asarray(train_idx), np.asarray(eval_idx)
    dist, j = cKDTree(xy[train_idx]).query(xy[eval_idx])
    mine, theirs = values[eval_idx], values[train_idx[j]]
    return {"median_dist_m": float(np.median(dist)),
            "corr": float(np.corrcoef(mine, theirs)[0, 1]),
            "median_abs_diff": float(np.median(np.abs(mine - theirs)))}


def _job(args):
    root, rel = args
    return rel, building_metrics(root / "LOD1" / rel, root / "LOD2" / rel)


def building_table(subset_root, workers=8):
    """One row per building: id, tile, x, y, footprint_m2, volume_m3, n_faces."""
    root = Path(subset_root)
    rels = sorted(p.relative_to(root / "LOD2") for p in (root / "LOD2").rglob("*.json"))
    rows = []
    with mp.Pool(workers) as pool:
        for done, (rel, metrics) in enumerate(pool.imap_unordered(_job, [(root, r) for r in rels]), 1):
            rows += [(oid, rel.as_posix(), *m) for oid, m in metrics.items()]
            if done % 25 == 0:
                logger.info("%d/%d files", done, len(rels))
    return pd.DataFrame(rows, columns=["id", "tile", "x", "y", "footprint_m2", "volume_m3",
                                       "n_faces"])


def report(table, ratios=(0.8, 0.1, 0.1), seed=42, n_perm=200):
    """All three measurements, for log footprint, log LOD2 volume and log LOD2 face count."""
    table = table[(table["footprint_m2"] > 0) & (table["volume_m3"] > 0)
                  & (table["n_faces"] > 0)].reset_index(drop=True)
    xy = table[["x", "y"]].to_numpy()
    r_train, r_val, r_test = random_split_indices(len(table), ratios, seed)
    (t_train, t_val, t_test), _ = tile_split_indices(table["tile"].tolist(), ratios, seed)
    out = {"n_buildings": len(table), "n_tiles": int(table["tile"].nunique()), "metrics": {}}
    for name in ("footprint_m2", "volume_m3", "n_faces"):
        v = np.log(table[name].to_numpy())
        within = table.assign(v=v).groupby("tile")["v"].std().mean()
        null = permutation_null(v, table["tile"].to_numpy(), n_perm, seed)
        out["metrics"][name] = {
            "icc1": icc1(v, table["tile"].to_numpy()),
            "icc1_null_p99": float(np.percentile(null, 99)),
            "within_tile_sd_over_total_sd": float(within / v.std()),
            "correlogram": correlogram(xy, v, seed=seed).to_dict(orient="records"),
            "nearest_train": {"random": nearest_train(xy, v, r_train, r_val + r_test),
                              "tile": nearest_train(xy, v, t_train, t_val + t_test)},
        }
    return out


def to_markdown(rep):
    lines = [f"# Spatial correlation ({rep['n_buildings']} buildings, {rep['n_tiles']} tiles)", ""]
    for name, m in rep["metrics"].items():
        nr, nt = m["nearest_train"]["random"], m["nearest_train"]["tile"]
        lines += [f"## log {name}", "",
                  f"- ICC(1) across tiles: **{m['icc1']:.3f}** (shuffled-label 99th pct "
                  f"{m['icc1_null_p99']:.3f}); within-tile SD / total SD "
                  f"{m['within_tile_sd_over_total_sd']:.3f}",
                  f"- Nearest train building of a val/test building, random vs tile split: "
                  f"median {nr['median_dist_m']:.0f} m vs {nt['median_dist_m']:.0f} m; "
                  f"correlation {nr['corr']:.3f} vs {nt['corr']:.3f}",
                  "", "| distance (m) | pairs | correlation |", "|---|---|---|"]
        lines += [f"| {c['lo']}-{c['hi']} | {c['n_pairs']} | {c['corr']:.3f} |"
                  for c in m["correlogram"]]
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-perm", type=int, default=200)
    parser.add_argument("--workers", type=int, default=min(8, mp.cpu_count()))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    table = building_table(args.subset, args.workers)
    rep = report(table, seed=args.seed, n_perm=args.n_perm)
    rep.update(created=datetime.now(timezone.utc).isoformat(), git_commit=_git_commit(),
               args={k: str(v) for k, v in vars(args).items()})
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "buildings.csv", index=False)
    (args.out / "report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    (args.out / "report.md").write_text(to_markdown(rep), encoding="utf-8")
    logger.info("wrote %s", args.out / "report.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_spatial_correlation.py -q -p no:warnings`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/analysis/spatial_correlation.py tests/test_spatial_correlation.py
git commit -m "feat(analysis): spatial correlation report (ICC by tile, correlogram, split optimism)"
```

---

### Task 3: `OrthoImageEncoder` (frozen DINOv3 → image tokens)

**Files:**
- Create: `src/models/ortho_encoder.py`
- Test: `tests/test_ortho_encoder.py`

**Interfaces:**
- Consumes: `src.dataset.ortho.pixel_to_uv(i, j, size, beta) -> (u, v)`.
- Produces:
  `OrthoImageEncoder(out_dim, name=None, config=None, image_size=256, beta=0.1, n_freq=8)`.
  - Exactly one of `name` (Hugging Face id) or `config` (`DINOv3ViTConfig`, the test hook) is given.
  - `features(image) -> float [B, G*G, C]`: frozen backbone patch features in **mesh order**.
  - `forward(image, valid=None) -> float [B, G*G, out_dim]`. Rows with `valid=False` are all equal to
    `self.null`.
  - `self.grid` (G).
  - `state_dict()` never contains `backbone.*` keys.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ortho_encoder.py
"""Frozen DINOv3 wrapper: orientation, null token, checkpoint hygiene. Tiny config, no hub."""
from types import SimpleNamespace

import pytest
import torch
from transformers import DINOv3ViTConfig

from src.models.ortho_encoder import OrthoImageEncoder

S, P = 64, 16
G = S // P


def tiny(**kw):
    torch.manual_seed(0)
    cfg = DINOv3ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=2, patch_size=P, image_size=S)
    return OrthoImageEncoder(out_dim=24, config=cfg, image_size=S, beta=0.1, **kw)


class FakeBackbone(torch.nn.Module):
    """Patch feature = mean of channel 0 over the patch, in the order it was fed."""
    config = SimpleNamespace(num_register_tokens=0, patch_size=P, hidden_size=1)

    def forward(self, pixel_values):
        self.seen = pixel_values
        pooled = torch.nn.functional.avg_pool2d(pixel_values[:, :1], P)       # [B,1,G,G]
        tokens = pooled.flatten(2).transpose(1, 2)                             # row-major
        cls = torch.zeros(tokens.shape[0], 1, 1)
        return SimpleNamespace(last_hidden_state=torch.cat([cls, tokens], dim=1))


def test_backbone_sees_north_up_and_features_come_back_in_mesh_order():
    enc = tiny()
    enc.backbone = FakeBackbone()
    enc.n_skip = 1
    img = torch.zeros(1, 3, S, S, dtype=torch.uint8)
    img[:, :, -P:, :] = 255                       # mesh order: last rows = +y = north
    feats = enc.features(img)                     # [1, G*G, 1]
    grid = feats.view(G, G)
    assert grid[-1].min() > grid[:-1].max()       # north patches stay at the +y end
    seen = enc.backbone.seen[0, 0]
    assert seen[:P].mean() > seen[P:].mean()      # ...but the backbone saw them on top


def test_invalid_rows_are_the_null_token_and_valid_rows_are_not():
    enc = tiny().eval()
    img = torch.randint(0, 255, (2, 3, S, S), dtype=torch.uint8)
    out = enc(img, valid=torch.tensor([True, False]))
    assert out.shape == (2, G * G, 24)
    assert torch.equal(out[1], enc.null.expand(G * G, -1))
    assert not torch.allclose(out[0], out[1])


def test_coordinate_embedding_distinguishes_patches_of_a_flat_image():
    enc = tiny().eval()
    out = enc(torch.full((1, 3, S, S), 128, dtype=torch.uint8))
    assert torch.unique(out[0], dim=0).shape[0] == G * G


def test_backbone_is_frozen_and_kept_out_of_checkpoints():
    enc = tiny()
    enc.train()
    assert not enc.backbone.training
    assert not any(p.requires_grad for p in enc.backbone.parameters())
    sd = enc.state_dict()
    assert sd and not any(k.startswith("backbone.") for k in sd)
    fresh = tiny()
    fresh.load_state_dict(sd, strict=True)        # backbone restored by the pre-hook
    assert torch.equal(fresh.null, enc.null)


def test_wrong_image_size_is_refused():
    with pytest.raises(ValueError, match="image_size"):
        tiny()(torch.zeros(1, 3, 2 * S, 2 * S, dtype=torch.uint8))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ortho_encoder.py -q -p no:warnings`
Expected: `ModuleNotFoundError: No module named 'src.models.ortho_encoder'`.

- [ ] **Step 3: Implement** `src/models/ortho_encoder.py`:

```python
"""Frozen DINOv3 encoding of the per-building orthophoto crop.

DINOv3 (Siméoni et al., arXiv:2508.10104) is used as a frozen feature extractor,
as the frozen-vision-encoder VLMs do (Flamingo arXiv:2204.14198, LLaVA
arXiv:2304.08485); only the projector, the coordinate embedding and the null
token train.

Orientation: the dataset's crop is in mesh order (row index along +y). DINOv3
was trained on north-up images, so rows are flipped before the backbone and the
patch grid is flipped back after it -- the tokens leave in mesh order again.

The backbone is excluded from `state_dict` and restored on load, so checkpoints
stay the size of the trainable model; it is rebuilt from the Hugging Face hub
(or the test config) at construction.
"""
import math

import numpy as np
import torch
import torch.nn as nn

from src.dataset.ortho import pixel_to_uv

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _uv_features(grid, beta, n_freq):
    """``[grid*grid, 4*n_freq]`` sin/cos code of each patch centre's (u, v), mesh order."""
    uv = np.array([pixel_to_uv(i, j, grid, beta) for i in range(grid) for j in range(grid)])
    freqs = (2.0 ** np.arange(n_freq)) * math.pi
    ang = uv[:, :, None] * freqs                                     # [P, 2, F]
    feats = np.concatenate([np.sin(ang), np.cos(ang)], axis=-1)      # [P, 2, 2F]
    return torch.tensor(feats.reshape(len(uv), -1), dtype=torch.float32)


class OrthoImageEncoder(nn.Module):
    """``uint8 [B, 3, S, S]`` mesh-order crops -> ``[B, (S/p)^2, out_dim]`` image tokens.

    Args:
        out_dim: token width the fusion expects (OPT hidden size for
            cross-attention, its embedding width for prefix tokens).
        name: Hugging Face id, e.g. ``facebook/dinov3-vitl16-pretrain-lvd1689m``.
        config: a `DINOv3ViTConfig` instead of `name` (tests; no download).
        image_size: crop side S; must be a multiple of the patch size.
        beta: the crop's margin beyond the unit box, for the coordinate code.
        n_freq: frequencies per axis of the coordinate code.
    """

    def __init__(self, out_dim, name=None, config=None, image_size=256, beta=0.1, n_freq=8):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel, DINOv3ViTModel

        if (name is None) == (config is None):
            raise ValueError("Give exactly one of `name` or `config`.")
        if config is None:
            self.backbone = AutoModel.from_pretrained(name)
            proc = AutoImageProcessor.from_pretrained(name)
            mean, std = proc.image_mean, proc.image_std
        else:
            self.backbone = DINOv3ViTModel(config)
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        cfg = self.backbone.config
        if image_size % cfg.patch_size:
            raise ValueError(f"image_size {image_size} is not a multiple of the "
                             f"patch size {cfg.patch_size}.")
        self.image_size = image_size
        self.grid = image_size // cfg.patch_size
        self.n_skip = 1 + int(getattr(cfg, "num_register_tokens", 0) or 0)

        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("uv", _uv_features(self.grid, beta, n_freq), persistent=False)
        self.proj = nn.Sequential(nn.Linear(cfg.hidden_size, out_dim), nn.GELU(),
                                  nn.Linear(out_dim, out_dim))
        self.coord = nn.Linear(4 * n_freq, out_dim)
        self.null = nn.Parameter(torch.randn(out_dim) * 0.02)

        self.register_state_dict_post_hook(_drop_backbone)
        self.register_load_state_dict_pre_hook(_restore_backbone)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()          # frozen means no dropout either
        return self

    def features(self, image):
        """Frozen backbone patch features ``[B, G*G, C]`` in mesh order."""
        if image.shape[-1] != self.image_size or image.shape[-2] != self.image_size:
            raise ValueError(f"Expected {self.image_size}px crops (image_size), got "
                             f"{tuple(image.shape[-2:])}.")
        x = image.float().div(255.0).flip(-2)                  # mesh order -> north-up
        x = (x - self.mean) / self.std
        with torch.no_grad():
            tokens = self.backbone(pixel_values=x).last_hidden_state[:, self.n_skip:]
        b, g = tokens.shape[0], self.grid
        return tokens.reshape(b, g, g, -1).flip(1).reshape(b, g * g, -1)   # back to mesh order

    def forward(self, image, valid=None):
        tokens = self.proj(self.features(image)) + self.coord(self.uv)
        if valid is not None:
            tokens = torch.where(valid.view(-1, 1, 1), tokens,
                                 self.null.to(tokens.dtype).expand_as(tokens))
        return tokens


def _drop_backbone(module, state_dict, prefix, local_metadata):
    for key in [k for k in state_dict if k.startswith(prefix + "backbone.")]:
        del state_dict[key]


def _restore_backbone(module, state_dict, prefix, local_metadata, strict, missing_keys,
                      unexpected_keys, error_msgs):
    for key, value in module.backbone.state_dict().items():
        state_dict.setdefault(prefix + "backbone." + key, value)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ortho_encoder.py -q -p no:warnings`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/ortho_encoder.py tests/test_ortho_encoder.py
git commit -m "feat(model): frozen DINOv3 ortho encoder with mesh-order tokens and null token"
```

---

### Task 4: Gated cross-attention into OPT (primary fusion)

**Files:**
- Create: `src/models/ortho_fusion.py`
- Modify: `src/models/mesh_transformer.py` (`MeshOPTTransformer.__init__`, `encode_image`, `forward`,
  `new_cache`, `cached_logits`, `reorder_cache`)
- Test: `tests/test_ortho_fusion.py`

**Interfaces:**
- Consumes: `OrthoImageEncoder` (Task 3).
- Produces in `ortho_fusion.py`:
  - `ImageContext(tokens, kv=None)` with `.select(order)` and `.repeat(n)`;
  - `CrossAttention(dim, n_head)`;
  - `GatedCrossAttention(layers, dim, n_head)` with `.gates` (zeros), `.active` and
    `.attending(ctx)` (a context manager).
- Produces on `MeshOPTTransformer(…, image=None)`:
  - `image` is `None` or a dict
    `{"fusion": "xattn"|"prefix", "name": str|None, "config": DINOv3ViTConfig|None,
    "image_size": int, "beta": float}`;
  - `encode_image(image, valid) -> ImageContext`;
  - `forward(cond, tgt, cond_pad_mask=None, tgt_pad_mask=None, image_ctx=None)`;
  - `new_cache(cond, cond_pad_mask=None, image_ctx=None)`.
  - With `image=None` every method behaves exactly as before.

- [ ] **Step 1: Run impact on `MeshOPTTransformer`** (`direction: "upstream"`) and report it.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_ortho_fusion.py
"""Image fusion into the OPT backbone: exact v5 at zero gates, cache equivalence."""
import torch
from transformers import DINOv3ViTConfig, OPTConfig

from src.dataset.mesh_dataset import specials, vocab_size
from src.models.mesh_transformer import MeshOPTTransformer

NUM_BINS, S = 16, 64
V = vocab_size(NUM_BINS)
BOS, EOS, PAD = specials(NUM_BINS)


def net(fusion=None, seed=0):
    torch.manual_seed(seed)
    opt = OPTConfig(vocab_size=V, hidden_size=32, word_embed_proj_dim=32, num_hidden_layers=2,
                    num_attention_heads=4, ffn_dim=64, max_position_embeddings=512, dropout=0.0)
    image = None if fusion is None else {
        "fusion": fusion, "name": None, "image_size": S, "beta": 0.1,
        "config": DINOv3ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                  num_attention_heads=2, patch_size=16, image_size=S)}
    return MeshOPTTransformer(vocab_size=V, opt_config=opt, dropout=0.0,
                              pos_embed="sinusoidal", max_seq_len=256, num_bins=NUM_BINS,
                              image=image).eval()


def batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    cond = torch.randint(0, NUM_BINS, (2, 18), generator=g)
    tgt = torch.cat([torch.full((2, 1), BOS), torch.randint(0, NUM_BINS, (2, 18), generator=g),
                     torch.full((2, 1), EOS)], dim=1)
    img = torch.randint(0, 255, (2, 3, S, S), generator=g, dtype=torch.uint8)
    return cond, tgt, img


def test_zero_gates_reproduce_the_image_free_model_exactly():
    base, fused = net(None), net("xattn")
    fused.load_state_dict(base.state_dict(), strict=False)      # same OPT weights
    cond, tgt, img = batch()
    ctx = fused.encode_image(img, torch.tensor([True, True]))
    with torch.no_grad():
        assert torch.equal(base(cond, tgt), fused(cond, tgt, image_ctx=ctx))


def test_open_gates_make_the_image_matter():
    fused = net("xattn")
    with torch.no_grad():
        fused.xattn.gates.fill_(1.0)
    cond, tgt, img = batch()
    ok = torch.tensor([True, True])
    with torch.no_grad():
        a = fused(cond, tgt, image_ctx=fused.encode_image(img, ok))
        b = fused(cond, tgt, image_ctx=fused.encode_image(img.flip(0), ok))
    assert not torch.allclose(a, b)


def test_old_state_dict_keys_are_untouched():
    base, fused = net(None), net("xattn")
    missing, unexpected = fused.load_state_dict(base.state_dict(), strict=False)
    assert unexpected == []
    assert all(k.startswith(("image_encoder.", "xattn.")) for k in missing)


def _greedy(m, cond, img, use_cache):
    """Greedy decode through the network's own cached / uncached paths."""
    ctx = m.encode_image(img, torch.tensor([True, True]))
    tgt = torch.full((2, 1), BOS)
    cache = m.new_cache(cond, None, image_ctx=ctx) if use_cache else None
    with torch.no_grad():
        for _ in range(12):
            if cache is None:
                step = torch.cat([tgt, torch.full_like(tgt[:, :1], PAD)], dim=1)
                logits = m(cond, step, image_ctx=ctx)[:, -1]
            else:
                logits = m.cached_logits(cache, tgt)
            tgt = torch.cat([tgt, logits.argmax(-1, keepdim=True)], dim=1)
    return tgt


def test_cached_decoding_matches_uncached_with_images():
    for fusion in ("xattn", "prefix"):
        m = net(fusion)
        if fusion == "xattn":
            with torch.no_grad():
                m.xattn.gates.fill_(0.7)
        cond, _, img = batch()
        assert torch.equal(_greedy(m, cond, img, False), _greedy(m, cond, img, True)), fusion


def test_reorder_cache_reorders_the_image_context():
    m = net("xattn")
    with torch.no_grad():
        m.xattn.gates.fill_(0.7)
    cond, _, img = batch()
    ctx = m.encode_image(img, torch.tensor([True, True]))
    cache = m.new_cache(cond, None, image_ctx=ctx)
    tgt = torch.full((2, 1), BOS)
    with torch.no_grad():
        m.cached_logits(cache, tgt)
        swapped = m.reorder_cache(cache, torch.tensor([1, 0]))
        nxt = torch.cat([tgt, torch.zeros(2, 1, dtype=torch.long)], dim=1)
        a = m.cached_logits(swapped, nxt)
        ref = m.new_cache(cond.flip(0), None, image_ctx=ctx.select(torch.tensor([1, 0])))
        m.cached_logits(ref, tgt)
        b = m.cached_logits(ref, nxt)
    torch.testing.assert_close(a, b)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ortho_fusion.py -q -p no:warnings`
Expected: `TypeError: ... unexpected keyword argument 'image'`.

- [ ] **Step 4: Implement** `src/models/ortho_fusion.py`:

```python
"""Gated cross-attention from OPT's decoder layers into image tokens.

Flamingo-style (Alayrac et al., arXiv:2204.14198): a cross-attention block per
decoder layer whose output is scaled by tanh(gate) with the gate initialised
at zero, so a model loaded from a text-only (here: image-free) checkpoint is
reproduced exactly at step 0 and the image's influence grows only as far as it
lowers the loss.

Injected by forward pre-hooks on the existing OPT layers rather than by
subclassing them: OPT's modules -- and therefore every v5 state-dict key --
stay untouched, and no Hugging Face internals are copied.
"""
import contextlib
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ImageContext:
    """Image tokens for one batch, plus per-layer cached K/V during decoding."""
    tokens: torch.Tensor                       # [B, P, D]
    kv: dict = field(default=None)            # layer -> (k, v); None = do not cache

    def select(self, order):
        kv = None if self.kv is None else {i: (k[order], v[order]) for i, (k, v) in self.kv.items()}
        return ImageContext(self.tokens[order], kv)

    def repeat(self, n):
        return ImageContext(self.tokens.repeat_interleave(n, dim=0))

    def caching(self):
        return ImageContext(self.tokens, {})


class CrossAttention(nn.Module):
    """Pre-norm multi-head attention from hidden states into image tokens."""

    def __init__(self, dim, n_head):
        super().__init__()
        if dim % n_head:
            raise ValueError(f"dim {dim} is not divisible by n_head {n_head}.")
        self.n_head = n_head
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, h, ctx, kv=None):
        b, l, d = h.shape
        heads = lambda x: x.view(b, -1, self.n_head, d // self.n_head).transpose(1, 2)
        q = heads(self.q(self.norm(h)))
        if kv is None:
            k, v = self.kv(ctx.to(h.dtype)).chunk(2, dim=-1)
            kv = (heads(k), heads(v))
        o = F.scaled_dot_product_attention(q, *kv)
        return self.out(o.transpose(1, 2).reshape(b, l, d)), kv


class GatedCrossAttention(nn.Module):
    """One gated cross-attention per decoder layer, applied before that layer.

    Args:
        layers: OPT's `decoder.layers` ModuleList (hooked, not modified).
        dim, n_head: OPT hidden size and head count.
    """

    def __init__(self, layers, dim, n_head):
        super().__init__()
        self.blocks = nn.ModuleList(CrossAttention(dim, n_head) for _ in layers)
        self.gates = nn.Parameter(torch.zeros(len(layers)))
        self.active = None
        for i, layer in enumerate(layers):
            layer.register_forward_pre_hook(self._hook(i), with_kwargs=True)

    @contextlib.contextmanager
    def attending(self, ctx):
        self.active = ctx
        try:
            yield
        finally:
            self.active = None

    def _hook(self, i):
        def hook(module, args, kwargs):
            ctx = self.active
            if ctx is None:
                return None
            h = args[0] if args else kwargs["hidden_states"]
            cached = None if ctx.kv is None else ctx.kv.get(i)
            out, kv = self.blocks[i](h, ctx.tokens, cached)
            if ctx.kv is not None:
                ctx.kv[i] = kv
            h = h + torch.tanh(self.gates[i]).to(h.dtype) * out
            if args:
                return (h, *args[1:]), kwargs
            return args, {**kwargs, "hidden_states": h}
        return hook
```

In `src/models/mesh_transformer.py`, `MeshOPTTransformer`:
- add `image=None` to `__init__`;
- at the end of `__init__`:

```python
        self.image_fusion = None if image is None else image["fusion"]
        if image is not None:
            from src.models.ortho_encoder import OrthoImageEncoder
            from src.models.ortho_fusion import GatedCrossAttention

            if self.image_fusion not in ("xattn", "prefix"):
                raise ValueError(f"Unknown image fusion {self.image_fusion!r}; "
                                 "expected 'xattn' or 'prefix'.")
            width = cfg.hidden_size if self.image_fusion == "xattn" else cfg.word_embed_proj_dim
            self.image_encoder = OrthoImageEncoder(
                width, name=image.get("name"), config=image.get("config"),
                image_size=image["image_size"], beta=image["beta"])
            if self.image_fusion == "xattn":
                self.xattn = GatedCrossAttention(self.opt.model.decoder.layers,
                                                 cfg.hidden_size, cfg.num_attention_heads)
            else:
                # Its own vector rather than a third `segment_embed` row, so the
                # v5 table keeps its shape and loads unchanged.
                self.image_segment = nn.Parameter(torch.zeros(cfg.word_embed_proj_dim))
```

Add these methods:

```python
    def encode_image(self, image, valid=None):
        from src.models.ortho_fusion import ImageContext
        return ImageContext(self.image_encoder(image, valid))

    def _image_prefix(self, ctx):
        return ctx.tokens + self.image_segment

    def _attending(self, ctx):
        import contextlib
        if self.image_fusion == "xattn" and ctx is not None:
            return self.xattn.attending(ctx)
        return contextlib.nullcontext()
```

Changes to `forward`:
- signature `forward(self, cond, tgt, cond_pad_mask=None, tgt_pad_mask=None, image_ctx=None)`;
- let `n_img = image_ctx.tokens.shape[1] if (self.image_fusion == "prefix" and image_ctx is not None) else 0`
  and add `n_img` to `length` in the bounded check;
- after building `x` and `attn`, for prefix:
  ```python
  if n_img:
      x = torch.cat([self._image_prefix(image_ctx), x], dim=1)
      attn = torch.cat([attn.new_ones(attn.shape[0], n_img), attn], dim=1)
  ```
- wrap the OPT call:
  ```python
  with self._attending(image_ctx):
      out = self.opt(inputs_embeds=x, attention_mask=attn)
  return out.logits[:, n_img + n_cond:]
  ```

`new_cache` becomes:

```python
    def new_cache(self, cond, cond_pad_mask=None, image_ctx=None):
        from transformers import DynamicCache

        attn = (torch.ones(cond.shape[:2], dtype=torch.long, device=cond.device)
                if cond_pad_mask is None else (~cond_pad_mask).long())
        n_img = 0
        if image_ctx is not None and self.image_fusion == "prefix":
            n_img = image_ctx.tokens.shape[1]
            attn = torch.cat([attn.new_ones(attn.shape[0], n_img), attn], dim=1)
        return {"past": DynamicCache(), "attn": attn, "n_cond": cond.shape[1],
                "n_tgt": 0, "cond": cond, "n_img": n_img,
                "image": None if image_ctx is None else image_ctx.caching()}
```

`cached_logits` replaces its body with this (same logic as before, plus the prefix rows and the
attending context):

```python
    def cached_logits(self, cache, tgt):
        device = tgt.device
        embedded = self.embed_tokens(tgt)
        new = embedded[:, cache["n_tgt"]:]
        if cache["n_tgt"] == 0:
            seg = torch.cat([torch.zeros(cache["n_cond"], dtype=torch.long, device=device),
                             torch.ones(tgt.shape[1], dtype=torch.long, device=device)])
            x = torch.cat([self.embed_cond(cache["cond"]), new], dim=1) + self.segment_embed(seg)
            if cache["n_img"]:
                x = torch.cat([self._image_prefix(cache["image"]), x], dim=1)
            cache["attn"] = torch.cat(
                [cache["attn"], torch.ones(tgt.shape[0], tgt.shape[1], dtype=torch.long,
                                           device=device)], dim=1)
        else:
            x = new + self.segment_embed(torch.ones(new.shape[1], dtype=torch.long,
                                                    device=device))
            cache["attn"] = torch.cat(
                [cache["attn"], torch.ones(tgt.shape[0], new.shape[1], dtype=torch.long,
                                           device=device)], dim=1)
        with self._attending(cache["image"]):
            out = self.opt(inputs_embeds=x, attention_mask=cache["attn"],
                           past_key_values=cache["past"], use_cache=True)
        cache["past"] = out.past_key_values
        cache["n_tgt"] = tgt.shape[1]
        return out.logits[:, -1]
```

(`self._attending(None)` is a null context, so image-free caches behave exactly as before. The
existing `tests/test_kv_cache.py` pins that.)

`reorder_cache` adds
`"image": None if cache.get("image") is None else cache["image"].select(order)`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ortho_fusion.py tests/test_kv_cache.py tests/test_mesh_transformer.py -q -p no:warnings`
Expected: all pass (the existing cache and transformer tests prove `image=None` is unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/models/ortho_fusion.py src/models/mesh_transformer.py tests/test_ortho_fusion.py
git commit -m "feat(model): gated cross-attention and prefix image fusion for the OPT backbone"
```

---

### Task 5: `MeshTransformerModule` wiring

This covers images in training and generation, the training scope, lr groups, `init_from` and gate
logging.

**Files:**
- Modify: `src/models/mesh_transformer.py` (`MeshTransformerModule.__init__`, `_shared_step`,
  `_step_logits`, `_beam_search`, `generate`, `training_step`, `_param_groups`, new `_image_ctx`,
  `load_weights_from`)
- Test: `tests/test_ortho_module.py`

**Interfaces:**
- Consumes: the Task 4 network API.
- Produces: new `MeshTransformerModule.__init__` kwargs, all default-off:
  - `image_encoder=None` (HF id);
  - `image_encoder_config=None` (test hook, excluded from `save_hyperparameters`);
  - `image_fusion="xattn"`, `image_size=256`, `image_margin=0.1`, `image_dropout=0.1`;
  - `train_scope="all"` or `"image_only"`;
  - `backbone_lr_scale=1.0`.
- Produces: `generate(..., image=None, image_valid=None)`. `image` is required when `self.uses_image`
  and ignored otherwise.
- Produces: `load_weights_from(path) -> list[str]` (the missing keys, all image keys).

- [ ] **Step 1: Run impact** on `MeshTransformerModule`, `generate` and `_param_groups`. Stop at
  HIGH for `generate` and ask the user.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_ortho_module.py
"""The Lightning module with an image: dropout, scopes, lr groups, init_from, generate."""
import pytest
import torch
from transformers import DINOv3ViTConfig, OPTConfig

from src.dataset.mesh_dataset import specials, vocab_size
from src.models.mesh_transformer import MeshTransformerModule

NUM_BINS, S = 16, 64
BOS, EOS, PAD = specials(NUM_BINS)


def module(**kw):
    torch.manual_seed(0)
    opt = OPTConfig(vocab_size=vocab_size(NUM_BINS), hidden_size=32, word_embed_proj_dim=32,
                    num_hidden_layers=2, num_attention_heads=4, ffn_dim=64,
                    max_position_embeddings=512, dropout=0.0)
    dino = DINOv3ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=2, patch_size=16, image_size=S)
    common = dict(num_bins=NUM_BINS, backbone="opt", opt_config=opt, dropout=0.0,
                  pos_embed="sinusoidal", max_seq_len=256)
    if kw.pop("image", True):
        common.update(image_encoder_config=dino, image_size=S)
    return MeshTransformerModule(**common, **kw)


def batch(valid=(True, True)):
    g = torch.Generator().manual_seed(1)
    tgt = torch.cat([torch.full((2, 1), BOS), torch.randint(0, NUM_BINS, (2, 18), generator=g),
                     torch.full((2, 1), EOS)], dim=1)
    return {"cond": torch.randint(0, NUM_BINS, (2, 18), generator=g), "tgt": tgt,
            "cond_pad_mask": torch.zeros(2, 18, dtype=torch.bool),
            "tgt_pad_mask": torch.zeros(2, 20, dtype=torch.bool),
            "image": torch.randint(0, 255, (2, 3, S, S), generator=g, dtype=torch.uint8),
            "image_valid": torch.tensor(valid), "ids": ["a", "b"],
            "center": torch.zeros(2, 3), "scale": torch.ones(2, 3)}


def test_image_model_requires_an_image_in_the_batch():
    m = module().eval()
    b = batch()
    del b["image"]
    with pytest.raises(ValueError, match="ortho_dir"):
        m._shared_step(b)


def test_image_only_scope_trains_only_the_image_parameters():
    m = module(train_scope="image_only")
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith(("network.image_encoder.", "network.xattn."))
                             for n in trainable)
    assert not any(n.startswith("network.image_encoder.backbone.") for n in trainable)
    groups = m._param_groups()
    assert sum(len(g["params"]) for g in groups) == len(trainable)


def test_backbone_lr_scale_gives_opt_its_own_learning_rate():
    m = module(lr=1e-3, backbone_lr_scale=0.1)
    lrs = {round(g["lr"], 6) for g in m._param_groups()}
    assert lrs == {1e-3, 1e-4}


def test_image_dropout_only_acts_in_training():
    m = module(image_dropout=1.0)
    seen = []
    m.network.encode_image = lambda img, valid: seen.append(valid.clone()) or \
        type(m.network).encode_image(m.network, img, valid)
    m.train()
    m._shared_step(batch())
    m.eval()
    m._shared_step(batch())
    assert seen[0].tolist() == [False, False] and seen[1].tolist() == [True, True]


def test_init_from_an_image_free_checkpoint_only_misses_image_keys(tmp_path):
    import lightning as L
    base = module(image=False)
    path = tmp_path / "v5.ckpt"
    torch.save({"state_dict": base.state_dict(), "hyper_parameters": dict(base.hparams),
                "pytorch-lightning_version": L.__version__}, path)
    m = module()
    missing = m.load_weights_from(path)
    assert missing and all(k.startswith(("network.image_encoder.", "network.xattn."))
                           for k in missing)
    torch.testing.assert_close(m.network.opt.lm_head.weight, base.network.opt.lm_head.weight)


def test_init_from_refuses_a_checkpoint_from_another_split_scheme(tmp_path):
    import lightning as L
    base = module(image=False)                       # data_split defaults to "random"
    path = tmp_path / "v5.ckpt"
    hp = {k: v for k, v in dict(base.hparams).items() if k != "data_split"}   # pre-2a checkpoint
    torch.save({"state_dict": base.state_dict(), "hyper_parameters": hp,
                "pytorch-lightning_version": L.__version__}, path)
    with pytest.raises(ValueError, match="split"):
        module(data_split="tile").load_weights_from(path)


def test_init_from_refuses_a_different_tokenization(tmp_path):
    import lightning as L
    base = module(image=False, tokenization="amt")
    path = tmp_path / "amt.ckpt"
    torch.save({"state_dict": base.state_dict(), "hyper_parameters": dict(base.hparams),
                "pytorch-lightning_version": L.__version__}, path)
    with pytest.raises(ValueError, match="tokenization"):
        module().load_weights_from(path)


def test_generate_with_images_cached_matches_uncached_including_beams():
    m = module().eval()
    with torch.no_grad():
        m.network.xattn.gates.fill_(0.7)
    b = batch(valid=(True, False))
    for beam in (1, 3):
        kw = dict(image=b["image"], image_valid=b["image_valid"], temperature=0.0,
                  beam_size=beam, max_new_tokens=12)
        off = m.generate(b["cond"], b["cond_pad_mask"], use_cache=False, **kw)
        on = m.generate(b["cond"], b["cond_pad_mask"], use_cache=True, **kw)
        assert torch.equal(off, on), beam


def test_scratch_backbone_with_an_image_is_refused():
    with pytest.raises(ValueError, match="opt"):
        MeshTransformerModule(num_bins=NUM_BINS, backbone="scratch", d_model=16, n_head=2,
                              num_layers=1, max_seq_len=64,
                              image_encoder_config=DINOv3ViTConfig(
                                  hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                  num_attention_heads=2, patch_size=16, image_size=S),
                              image_size=S)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ortho_module.py -q -p no:warnings`
Expected: `TypeError: ... unexpected keyword argument 'image_encoder_config'`.

- [ ] **Step 4: Implement** in `MeshTransformerModule`:

`__init__`:
1. Add the kwargs listed under Interfaces.
2. Change `save_hyperparameters(ignore=["vqvae", "opt_config", "image_encoder_config"])`.
3. Then:

```python
        self.uses_image = image_encoder is not None or image_encoder_config is not None
        self.image_dropout = float(image_dropout)
        self.backbone_lr_scale = float(backbone_lr_scale)
        if self.uses_image and backbone != "opt":
            raise ValueError("Image conditioning needs backbone='opt' (cross-attention hooks "
                             "and prefix segments are implemented for OPT only).")
        if train_scope not in ("all", "image_only"):
            raise ValueError(f"Unknown train_scope {train_scope!r}; expected 'all' or 'image_only'.")
        if train_scope == "image_only" and not self.uses_image:
            raise ValueError("train_scope='image_only' needs an image encoder.")
        image = None if not self.uses_image else {
            "fusion": image_fusion, "name": image_encoder, "config": image_encoder_config,
            "image_size": image_size, "beta": image_margin}
```

4. Pass `image=image` to `MeshOPTTransformer(...)`. After the network is built:

```python
        if train_scope == "image_only":
            # Stage A (Flamingo arXiv:2204.14198, LLaVA arXiv:2304.08485): the
            # image-free model stays frozen and only the new image path learns.
            for name, p in self.network.named_parameters():
                p.requires_grad_(name.startswith(IMAGE_PARAM_PREFIXES))
```

Module-level constant:
`IMAGE_PARAM_PREFIXES = ("image_encoder.", "xattn.", "image_segment")`. The DINOv3 backbone has
`requires_grad=False` from its own constructor, and this loop must not re-enable it. Guard with
`and not name.startswith("image_encoder.backbone.")`.

`_image_ctx`:

```python
    def _image_ctx(self, image, valid):
        """Encode the batch's crops; None for an image-free model."""
        if not self.uses_image:
            return None
        if image is None:
            raise ValueError("This model is image-conditioned but the batch has no 'image'; "
                             "set mesh_data.ortho_dir.")
        if valid is None:
            valid = torch.ones(image.shape[0], dtype=torch.bool, device=image.device)
        if self.training and self.image_dropout > 0:
            keep = torch.rand(valid.shape, device=valid.device) >= self.image_dropout
            valid = valid & keep
        return self.network.encode_image(image, valid)
```

`_shared_step`:
`ctx = self._image_ctx(batch.get("image"), batch.get("image_valid"))`, then
`logits = self.network(cond, tgt, cond_pad_mask, tgt_pad_mask, image_ctx=ctx)` (the image-free call
stays valid because `image_ctx=None`). For the scratch backbone keep the old 4-argument call: use
`kwargs = {"image_ctx": ctx} if ctx is not None else {}`.

`_step_logits(self, cond, tgt, cond_pad_mask, cache=None, image_ctx=None)`: the uncached branch passes
`image_ctx=image_ctx` when it is not None.

`_beam_search(..., image_ctx=None)`:
- `image_ctx = None if image_ctx is None else image_ctx.repeat(beam_size)` next to the `cond` repeat;
- `new_cache(cond, cond_pad_mask, image_ctx=image_ctx)` when it is not None;
- pass `image_ctx` to `_step_logits`;
- in the uncached path the beams are reordered through `tgt[order]` only, so also do
  `image_ctx = image_ctx.select(order)` after each reorder when `cache is None`. With the cache,
  `reorder_cache` does it.

`generate(..., image=None, image_valid=None)`: after the existing argument validation, add

```python
        ctx = self._image_ctx(image, image_valid) if self.uses_image else None
        img = {} if ctx is None else {"image_ctx": ctx}
        n_prefix = ctx.tokens.shape[1] if (ctx is not None and
                                           self.network.image_fusion == "prefix") else 0
        budget = self.network.generation_budget(cond.shape[1] + n_prefix)
```

(the last line replaces the existing `budget = ...` line). Then:
- `return self._beam_search(cond, cond_pad_mask, max_new_tokens, beam_size, length_penalty,
  use_cache, **img)`;
- in the greedy loop,
  `cache = self.network.new_cache(cond, cond_pad_mask, **img) if use_cache else None`;
- `logits = self._step_logits(cond, tgt, cond_pad_mask, cache, **img)`.

`_beam_search(..., use_cache=True, image_ctx=None)` gains the same `img` pattern:

```python
        if image_ctx is not None:
            image_ctx = image_ctx.repeat(beam_size)
        img = {} if image_ctx is None else {"image_ctx": image_ctx}
        cache = self.network.new_cache(cond, cond_pad_mask, **img) if use_cache else None
        for _ in range(max_new_tokens):
            logits = self._step_logits(cond, tgt, cond_pad_mask, cache,
                                       **({} if cache is not None else img))
            ...unchanged scoring...
            if cache is not None:
                cache = self.network.reorder_cache(cache, order)
            elif image_ctx is not None:
                image_ctx = image_ctx.select(order)
                img = {"image_ctx": image_ctx}
```

`_step_logits(self, cond, tgt, cond_pad_mask, cache=None, image_ctx=None)`, uncached branch:
`logits = self.network(cond, step, cond_pad_mask, torch.zeros_like(step, dtype=torch.bool),
**({} if image_ctx is None else {"image_ctx": image_ctx}))[:, -1]`.

`_param_groups`: keep the decay/no-decay classification and change only how groups are assembled:

```python
        image_prefixes = tuple("network." + p for p in IMAGE_PARAM_PREFIXES)
        buckets = {}                                  # (is_image, decays) -> [params]
        seen = set()
        for name, p in self.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            decays = not (id(p) in no_decay_ids or p.dim() < 2)
            is_image = self.uses_image and name.startswith(image_prefixes)
            buckets.setdefault((is_image, decays), []).append(p)
        groups = []
        for (is_image, decays), params in sorted(buckets.items(), key=lambda kv: (not kv[0][0], not kv[0][1])):
            if decays and not self.weight_decay:
                continue
            lr = self.lr if (is_image or not self.uses_image) else self.lr * self.backbone_lr_scale
            groups.append({"params": params, "lr": lr,
                           "weight_decay": float(self.weight_decay) if decays else 0.0})
        return groups
```

For an image-free model this yields the same two groups as today (decayed first, then undecayed),
each with `"lr": self.lr`. `tests/test_mesh_transformer.py::test_embeddings_norms_and_biases_are_excluded_from_decay`
must keep passing unchanged. The current code drops the decayed group when `weight_decay` is 0;
the `continue` keeps that behaviour.

`configure_optimizers` passes `lr=self.lr` as before; the per-group `lr` overrides it.

`training_step`, after the existing logging:

```python
        if self.uses_image and self.network.image_fusion == "xattn":
            gates = torch.tanh(self.network.xattn.gates.detach())
            self.log("train_gate_abs_mean", gates.abs().mean(), on_step=True, on_epoch=False)
            for i, g in enumerate(gates):
                self.log(f"train_gate_{i}", g, on_step=True, on_epoch=False)
```

`load_weights_from`:

```python
    def load_weights_from(self, path):
        """Initialise from another run's weights (not a Lightning resume).

        Architecture hyperparameters must match; the only keys allowed to be
        missing are the image path's, which an image-free checkpoint lacks.

        Returns:
            list[str]: the missing (newly initialised) keys.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        theirs = ckpt.get("hyper_parameters", {})
        for key in ("backbone", "tokenization", "num_bins", "pos_embed", "opt_name"):
            if key in theirs and theirs[key] != self.hparams.get(key):
                raise ValueError(f"{path}: {key}={theirs[key]!r} but this model has "
                                 f"{self.hparams.get(key)!r}.")
        # Task 2a: a checkpoint from another split scheme has trained on this
        # run's val/test buildings. Checkpoints older than the key were random.
        if theirs.get("data_split", "random") != self.hparams.get("data_split", "random"):
            raise ValueError(f"{path} was trained on the {theirs.get('data_split', 'random')!r} "
                             f"split; this run uses {self.hparams.get('data_split')!r}. Its "
                             "weights have seen this run's val/test buildings.")
        missing, unexpected = self.load_state_dict(ckpt["state_dict"], strict=False)
        stray = [k for k in missing if not k.startswith(
            tuple("network." + p for p in IMAGE_PARAM_PREFIXES))]
        if unexpected or stray:
            raise ValueError(f"{path} does not fit this model: unexpected {unexpected[:5]}, "
                             f"missing non-image {stray[:5]}.")
        logger.info("Initialised from %s; %d image keys are new.", path, len(missing))
        return missing
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ortho_module.py tests/test_kv_cache.py tests/test_mesh_transformer.py tests/test_beam_search.py tests/test_sampling_filters.py -q -p no:warnings`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/models/mesh_transformer.py tests/test_ortho_module.py
git commit -m "feat(model): image-conditioned MeshTransformerModule (dropout, scopes, lr groups, init_from)"
```

---

### Task 6: Config, training, evaluation and inference wiring

**Files:**
- Modify: `src/utils/config.py` (`MeshModelConfig`), `src/train_mesh.py`, `src/eval/mesh_eval.py`
  (`run_mesh_eval`), `src/eval/posthoc_eval.py` (`run_posthoc_eval`), `src/inference.py`
  (`_check_config`)
- Test: `tests/test_ortho_wiring.py`

**Interfaces:**
- Produces: `MeshModelConfig` fields with defaults:
  - `image_encoder: Optional[str] = None`
  - `image_fusion: str = "xattn"`
  - `image_dropout: float = 0.1`
  - `train_scope: str = "all"`
  - `backbone_lr_scale: float = 1.0`
  - `init_from: Optional[str] = None`
- Produces: `train_mesh` validation. `image_encoder` requires `mesh_data.ortho_dir`, and `init_from`
  must exist.

- [ ] **Step 1: Run impact** on `run_mesh_eval` and `run_posthoc_eval`. Stop at HIGH and ask the user.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_ortho_wiring.py
from pathlib import Path

import pytest
import torch

from src.utils.config import Config
from tests.test_ortho_module import S, batch, module


def test_config_defaults_keep_v5_image_free():
    m = Config().mesh_model
    assert m.image_encoder is None and m.init_from is None
    assert (m.image_fusion, m.train_scope, m.backbone_lr_scale) == ("xattn", "all", 1.0)


def test_train_mesh_refuses_an_image_model_without_ortho_dir():
    from src.train_mesh import train_mesh
    cfg = Config()
    cfg.mesh_data.dataset_dir = "unused"
    cfg.mesh_model.image_encoder = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    with pytest.raises(ValueError, match="ortho_dir"):
        train_mesh(cfg)


class _Seen(Exception):
    """Raised by the generate spy: the test is about the call, not the metrics after it."""


def test_run_mesh_eval_passes_the_image_to_generate(monkeypatch):
    from src.eval import mesh_eval
    m = module().eval()
    seen = {}

    def spy(self, cond, cond_pad_mask=None, **kw):
        seen.update(kw)
        raise _Seen
    monkeypatch.setattr(type(m), "generate", spy)
    b = batch()
    items = [{"cond": b["cond"][i], "tgt": b["tgt"][i], "id": b["ids"][i],
              "center": b["center"][i], "scale": b["scale"][i],
              "image": b["image"][i]} for i in range(2)]
    cfg = type("Cfg", (), dict(batch_size=2, beam_size=1, length_penalty=1.0, taus=[0.25],
                               n_points=64, voxel_m=0.5, save_samples=0))()
    with pytest.raises(_Seen):
        mesh_eval.run_mesh_eval(m, items, [0, 1], cfg, max_new_tokens=4)
    assert seen["image"].shape == (2, 3, S, S) and seen.get("image_valid") is None


def test_inference_refuses_image_models():
    from src.inference import _check_config
    cfg = Config()
    cfg.inference.checkpoint_path, cfg.inference.input_path = "x", "y"
    cfg.mesh_model.image_encoder = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    with pytest.raises(ValueError, match="orthophoto"):
        _check_config(cfg)
```

(`items` in the eval test are plain dicts shaped like `MeshDataset` items. `run_mesh_eval` only
indexes them, so a list works as the dataset.)

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ortho_wiring.py -q -p no:warnings`
Expected: `AttributeError: 'MeshModelConfig' object has no attribute 'image_encoder'`.

- [ ] **Step 4: Implement**

`MeshModelConfig`, after `pos_embed`:

```python
    # Orthophoto conditioning (docs/research/2026-10-14-orthophoto-conditioning.md).
    # image_encoder: Hugging Face id of a DINOv3 ViT, e.g.
    #   facebook/dinov3-vitl16-pretrain-lvd1689m (web) or ...-sat493m (satellite);
    #   gated on the hub -- `huggingface-cli login` and accept the licence first.
    #   None keeps the model image-free (every run up to v5).
    image_encoder: Optional[str] = None
    image_fusion: str = "xattn"        # "xattn" (gated cross-attention) | "prefix"
    image_dropout: float = 0.1         # training-time rate of replacing the image by the null token
    # "image_only" = stage A: OPT frozen, only the image path trains.
    train_scope: str = "all"
    # Multiplies training.lr for every non-image parameter (stage B: 0.1).
    backbone_lr_scale: float = 1.0
    # Weights to start from (a .ckpt), e.g. the v5 best for stage A. Not a resume:
    # optimizer state and epoch counters start fresh.
    init_from: Optional[str] = None
```

`train_mesh.py`:
- after the `dataset_dir` check:

```python
    if cfg.mesh_model.image_encoder and not cfg.mesh_data.ortho_dir:
        raise ValueError("mesh_model.image_encoder needs mesh_data.ortho_dir (the crops it encodes).")
    if cfg.mesh_model.init_from and not Path(cfg.mesh_model.init_from).is_file():
        raise ValueError(f"mesh_model.init_from not found: {cfg.mesh_model.init_from}")
```

- add to the `MeshTransformerModule(...)` call:

```python
        image_encoder=cfg.mesh_model.image_encoder,
        image_fusion=cfg.mesh_model.image_fusion,
        image_size=cfg.mesh_data.image_size,
        image_margin=cfg.mesh_data.image_margin,
        image_dropout=cfg.mesh_model.image_dropout,
        train_scope=cfg.mesh_model.train_scope,
        backbone_lr_scale=cfg.mesh_model.backbone_lr_scale,
```

- then `if cfg.mesh_model.init_from: model.load_weights_from(cfg.mesh_model.init_from)`.

`run_mesh_eval`: add `image=batch.get("image")` to the `model.generate(...)` call.
`run_posthoc_eval`: add `image=batch.get("image"), image_valid=image_valid` (Task 7 sets
`image_valid`; until then pass `image=batch.get("image")` only).

`inference._check_config`:

```python
    if cfg.mesh_model.image_encoder:
        raise ValueError("Image-conditioned checkpoints are not supported by inference yet: "
                         "it would need an orthophoto crop for every input building.")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ortho_wiring.py tests/test_mesh_eval.py tests/test_posthoc_eval.py tests/test_inference.py -q -p no:warnings`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/utils/config.py src/train_mesh.py src/eval/mesh_eval.py src/eval/posthoc_eval.py src/inference.py tests/test_ortho_wiring.py
git commit -m "feat(train): wire image conditioning through config, training, eval; refuse in inference"
```

---

### Task 7: Post-hoc image controls (`image_mode`)

**Files:**
- Modify: `src/utils/config.py` (`PosthocEvalConfig.image_mode`), `src/eval/posthoc_eval.py`
  (`validate_posthoc_config`, `run_posthoc_eval`, `main`)
- Test: `tests/test_posthoc_eval.py`

**Interfaces:**
- Produces: `image_mode: "true" | "shuffled" | "none"`, default `"true"`.
  - `shuffled`: split position k is paired with the image of split position `(k + n//2) % n`. That is
    deterministic, never the same building for n ≥ 2, and spatially unrelated, because the split is a
    random permutation.
  - `none`: passes `image_valid=False` for every row to `generate`, which selects the null token.
    Dataset items never carry `image_valid`.
  - Outputs go to `<…>/<ckpt>/<image_mode>-<timestamp>` when the mode is not `"true"`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_posthoc_eval.py`):

```python
def test_image_mode_is_validated():
    with pytest.raises(ValueError, match="image_mode"):
        validate_posthoc_config(_cfg(image_mode="random"))


def test_shuffled_pairs_each_object_with_another_buildings_image():
    from src.eval.posthoc_eval import image_source
    n = 5
    src = [image_source(k, n, "shuffled") for k in range(n)]
    assert sorted(src) == list(range(n)) and all(s != k for k, s in enumerate(src))
    assert [image_source(k, n, "true") for k in range(n)] == list(range(n))


def test_shuffled_needs_two_objects():
    from src.eval.posthoc_eval import image_source
    with pytest.raises(ValueError, match="two"):
        image_source(0, 1, "shuffled")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_posthoc_eval.py -q -p no:warnings -k "image_mode or shuffled"`
Expected: `ImportError: cannot import name 'image_source'`, and an `OmegaConf` error for the unknown
key.

- [ ] **Step 3: Implement**

`PosthocEvalConfig`:
`image_mode: str = "true"   # "true" | "shuffled" (another building's image) | "none" (null token)`.

`posthoc_eval.py`:

```python
IMAGE_MODES = ("true", "shuffled", "none")


def image_source(k, n, mode):
    """Split position whose image object k is scored with."""
    if mode != "shuffled":
        return k
    if n < 2:
        raise ValueError("image_mode='shuffled' needs at least two objects in the split.")
    return (k + n // 2) % n
```

In `validate_posthoc_config`:

```python
    if cfg.image_mode not in IMAGE_MODES:
        raise ValueError(f"Unknown image_mode {cfg.image_mode!r}; expected one of {list(IMAGE_MODES)}.")
```

In `run_posthoc_eval`, replace the batch construction:

Before the `for run in runs:` loop:

```python
    indices = [int(i) for i in indices]
    position = {k: p for p, k in enumerate(indices)}
```

Then replace `batch = mesh_collate_fn([dataset[int(i)] for i in chunk], pad=pad)` with:

```python
            items = []
            for k in chunk:
                item = dataset[k]
                if "image" in item and cfg.image_mode == "shuffled":
                    source = indices[image_source(position[k], len(indices), "shuffled")]
                    full, orig = _resolve(dataset, source)
                    item["image"] = full.image(orig)
                items.append(item)
            batch = mesh_collate_fn(items, pad=pad)
            image_valid = (torch.zeros(len(chunk), dtype=torch.bool)
                           if cfg.image_mode == "none" else None)
```

(`_resolve` is already imported from `mesh_eval`. Move `image_valid` to `device` with the batch, and
pass `image=batch.get("image"), image_valid=image_valid` to `model.generate`.)

In `main`, use `stamp = datetime.now().strftime("%Y%m%d-%H%M%S")` and
`out_dir = base / ckpt.stem / (stamp if cfg.image_mode == "true" else f"{cfg.image_mode}-{stamp}")`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_posthoc_eval.py -q -p no:warnings`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/utils/config.py src/eval/posthoc_eval.py tests/test_posthoc_eval.py
git commit -m "feat(eval): post-hoc image_mode true/shuffled/none controls"
```

---

### Task 8: Configs, stratified analysis, runbook

**Files:**
- Create: `configs/mesh-v6-ortho-xattn-a.yaml`, `configs/mesh-v6-ortho-xattn-b.yaml`,
  `configs/mesh-v6-ortho-prefix.yaml`, `configs/mesh-v6-ortho-smoke.yaml`
- Create: `src/analysis/ortho_strata.py`
- Create: `docs/ortho-conditioning-runbook.md`
- Test: `tests/test_ortho_strata.py`, plus a config-load test in `tests/test_ortho_wiring.py`

**Interfaces:**
- Produces: `stratify(subset_root, runs, metric, assignment=None)`. It scores every run on the
  **same buildings**: the ids present in all runs. That matters because image runs skip buildings
  without an orthophoto, and v5 does not. With `assignment` ({id: imagery year}, from
  `ortho/year_assignment.json`) it adds an `imagery_gap` stratum: |`b3_pw_datum` − imagery year| as
  "0", "1", "2" or ">2".
- Produces: `python -m src.analysis.ortho_strata --dataset-dir <subset> --run true=<dir>
  --run shuffled=<dir> --run none=<dir> --metric chamfer_m_vs_lod2 --out <dir>`. It writes
  `strata.csv` with columns `mode, stratum, value, n, mean, median` for strata `pw_datum`,
  `height_bin` (<10, 10–20, 20–40, ≥40 m from the LOD2 z extent) and `all`, plus `strata.md`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ortho_strata.py
import pandas as pd

from src.analysis.ortho_strata import stratify
from tests.test_ortho_tiles import write_tile


def test_stratify_groups_by_point_cloud_year_and_height(tmp_path):
    root = tmp_path / "The Hague" / "mini_cleaner"
    write_tile(root / "LOD2" / "Source B" / "t.city.json",
               {"a": (0, 0, 10, 10, 6), "b": (20, 0, 30, 10, 25), "c": (40, 0, 50, 10, 8)},
               datum={"a": 2023, "b": 2023, "c": 2020})
    runs = {"true": pd.DataFrame({"object_id": ["a", "b", "c"], "m": [1.0, 3.0, 5.0]}),
            "shuffled": pd.DataFrame({"object_id": ["a", "b", "c"], "m": [2.0, 4.0, 6.0]})}
    out = stratify(root, runs, "m")
    row = out[(out["mode"] == "true") & (out["stratum"] == "pw_datum") & (out["value"] == "2023")]
    assert row["n"].item() == 2 and row["mean"].item() == 2.0
    tall = out[(out["mode"] == "shuffled") & (out["stratum"] == "height_bin") & (out["value"] == "20-40")]
    assert tall["mean"].item() == 4.0


def test_stratify_compares_on_common_ids_and_reports_the_imagery_gap(tmp_path):
    root = tmp_path / "The Hague" / "mini_cleaner"
    write_tile(root / "LOD2" / "Source B" / "t.city.json",
               {"a": (0, 0, 10, 10, 6), "b": (20, 0, 30, 10, 6), "c": (40, 0, 50, 10, 6)},
               datum={"a": 2023, "b": 2020, "c": 2014})
    runs = {"v5": pd.DataFrame({"object_id": ["a", "b", "c"], "m": [1.0, 2.0, 9.0]}),
            "true": pd.DataFrame({"object_id": ["a", "b"], "m": [0.5, 1.5]})}   # "c" skipped
    out = stratify(root, runs, "m", assignment={"a": 2023, "b": 2022, "c": 2022})
    v5_all = out[(out["mode"] == "v5") & (out["stratum"] == "all")]
    assert v5_all["n"].item() == 2 and v5_all["mean"].item() == 1.5        # "c" excluded
    gaps = out[(out["mode"] == "true") & (out["stratum"] == "imagery_gap")]
    assert dict(zip(gaps["value"], gaps["n"])) == {"0": 1, "2": 1}
```

```python
# tests/test_ortho_wiring.py (append)
@pytest.mark.parametrize("name", ["xattn-a", "xattn-b", "prefix", "smoke"])
def test_shipped_v6_configs_load(name):
    from src.utils.initialization import load_config
    cfg = load_config(Path(f"configs/mesh-v6-ortho-{name}.yaml"), [])
    assert cfg.mesh_model.image_encoder and cfg.mesh_data.ortho_dir == "ortho"
    assert cfg.training.split == "tile"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ortho_strata.py tests/test_ortho_wiring.py -q -p no:warnings`
Expected: `ModuleNotFoundError: src.analysis.ortho_strata`, and `FileNotFoundError` for the configs.

- [ ] **Step 3: Implement** `src/analysis/ortho_strata.py`:

```python
"""Post-hoc metrics of image-conditioned runs, split by what the image can and cannot show.

Strata: `b3_pw_datum` (point-cloud year vs the 2022 imagery -- 64.9% of `cleaner`
is 2023), LOD2 height (relief displacement grows with height), and all.
Reads the per-object `objects.csv` of one post-hoc run per image mode.

Usage:
    python -m src.analysis.ortho_strata --dataset-dir "data/The Hague/cleaner" \\
        --run true=<posthoc dir>/<run> --run shuffled=<...> --run none=<...> \\
        --metric chamfer_m_vs_lod2 --out outputs/ortho_strata
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.ortho_tiles import pw_datum_by_id
from src.visualize_cityjson import load_cityjson

logger = logging.getLogger(__name__)
HEIGHT_BINS = [(0, 10, "<10"), (10, 20, "10-20"), (20, 40, "20-40"), (40, np.inf, ">=40")]


def building_attributes(subset_root):
    """``{id: (pw_datum, height_m)}`` over the subset's LOD2 files."""
    out = {}
    for path in sorted((Path(subset_root) / "LOD2").rglob("*.json")):
        cj, verts = load_cityjson(path)
        datum = pw_datum_by_id(path)
        for oid, obj in cj["CityObjects"].items():
            idx = {i for g in obj.get("geometry", []) for i in _flat(g["boundaries"])}
            z = np.asarray(verts)[sorted(idx), 2]
            out[oid] = (datum[oid], float(z.max() - z.min()))
    return out


def to_markdown(table):
    """Plain markdown table (no `tabulate` dependency)."""
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in table.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v)
                                        for v in row) + " |")
    return "\n".join(lines) + "\n"


def _flat(nested):
    if isinstance(nested, int):
        return [nested]
    return [i for item in nested for i in _flat(item)]


def _height_bin(h):
    return next(label for lo, hi, label in HEIGHT_BINS if lo <= h < hi)


def _gap(datum, year):
    if datum == "unknown":
        return "unknown"
    gap = abs(int(datum) - int(year))
    return str(gap) if gap <= 2 else ">2"


def stratify(subset_root, runs, metric, assignment=None):
    """Per-stratum mean/median of `metric` for each run ({mode: objects DataFrame}).

    Every run is restricted to the ids present in all of them, so modes and
    models are compared on identical buildings.
    """
    attrs = building_attributes(subset_root)
    common = set(attrs).intersection(*(set(df["object_id"]) for df in runs.values()))
    strata = ["all", "pw_datum", "height_bin"] + (["imagery_gap"] if assignment else [])
    rows = []
    for mode, df in runs.items():
        df = df[df["object_id"].isin(common)].copy()
        df["pw_datum"] = df["object_id"].map(lambda o: attrs[o][0])
        df["height_bin"] = df["object_id"].map(lambda o: _height_bin(attrs[o][1]))
        if assignment:
            df["imagery_gap"] = df["object_id"].map(lambda o: _gap(attrs[o][0], assignment[o]))
        df["all"] = "all"
        for stratum in strata:
            for value, g in df.groupby(stratum):
                vals = g[metric].dropna()
                rows.append({"mode": mode, "stratum": stratum, "value": str(value),
                             "n": len(vals), "mean": float(vals.mean()),
                             "median": float(vals.median())})
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="mode=<run dir>")
    parser.add_argument("--metric", default="chamfer_m_vs_lod2")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    runs = {}
    for spec in args.run:
        mode, path = spec.split("=", 1)
        runs[mode] = pd.read_csv(Path(path) / "objects.csv")
    assignment_path = args.dataset_dir / "ortho" / "year_assignment.json"
    assignment = (json.loads(assignment_path.read_text(encoding="utf-8"))
                  if assignment_path.is_file() else None)
    table = stratify(args.dataset_dir, runs, args.metric, assignment)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "strata.csv", index=False)
    (args.out / "strata.md").write_text(to_markdown(table), encoding="utf-8")
    (args.out / "strata_args.json").write_text(json.dumps(vars(args), default=str, indent=2),
                                               encoding="utf-8")
    logger.info("wrote %s", args.out / "strata.csv")


if __name__ == "__main__":
    main()
```

**Configs.** First the baseline: copy `configs/mesh-v5-full-opt-coord.yaml` to
`configs/mesh-v5-full-opt-coord-tile.yaml`, change only `training.split: tile` and
`logging.run_name: mesh-v5-full-opt-coord-tile`, and say in the header comment that it is v5 retrained
on the geographic split (Task 2a), so image runs have a leak-free init and a fair baseline.

Then copy `configs/mesh-v5-full-opt-coord.yaml` to `configs/mesh-v6-ortho-xattn-a.yaml` and change
exactly these keys:

```yaml
mesh_data:
  ortho_dir: ortho
  image_size: 256
  image_margin: 0.1
  image_min_coverage: 1.0
mesh_model:
  image_encoder: facebook/dinov3-vitl16-pretrain-lvd1689m
  image_fusion: xattn
  image_dropout: 0.1
  train_scope: image_only
  backbone_lr_scale: 1.0
  # v5 retrained on the tile split (Task 9). The random-split v5 checkpoint is
  # refused by load_weights_from: it has seen this run's val/test tiles.
  init_from: outputs/initial-runs/mesh-v5-full-opt-coord-tile/checkpoints/best.ckpt   # replace with the real best path
training:
  split: tile
  lr: 1.0e-4            # new parameters only; Flamingo-scale, not v5's from-scratch 1e-3
  warmup_steps: 1000
  max_epochs: 5         # ~65k optimizer steps on cleaner; the gates show within an epoch
  early_stopping: {enabled: true, monitor: val_gen_chamfer_m, patience: 2, mode: min}
logging:
  run_name: mesh-v6-ortho-xattn-a
```

In the copied file's header comment, state:
- this is stage A;
- step 0 equals v5;
- the gate to stage B, quoted from the research doc §4.3.

`mesh-v6-ortho-xattn-b.yaml` is the same as A except:
- `train_scope: all`
- `backbone_lr_scale: 0.1`
- `init_from: <stage A best ckpt>`: leave the path as the literal string
  `outputs/initial-runs/mesh-v6-ortho-xattn-a/checkpoints/best.ckpt` with a comment "replace with
  stage A's best checkpoint path"
- `max_epochs: 10`
- `patience: 3`
- `run_name: mesh-v6-ortho-xattn-b`

`mesh-v6-ortho-prefix.yaml` is the same as A except:
- `image_fusion: prefix`
- `train_scope: all`
- `backbone_lr_scale: 0.1`
- `max_epochs: 10`
- `run_name: mesh-v6-ortho-prefix`

`mesh-v6-ortho-smoke.yaml` is the same as A except:
- `mesh_data.dataset_dir: data/The Hague/mini_cleaner`
- `max_epochs: 1`
- `mesh_eval.n_val: 4`
- `logging.loggers: []`
- `run_name: mesh-v6-ortho-smoke`

**Runbook** `docs/ortho-conditioning-runbook.md`:
- prerequisites: HF login and DINOv3 licence acceptance;
- the order smoke → A → gate check → B → prefix control;
- the post-hoc commands for `image_mode` true/shuffled/none;
- the `ortho_strata` command;
- how to read the gates and the shuffled gap;
- a "decision points" section covering two things:
  - 2021 imagery for the 2020/2014 point-cloud buildings, worth ~14 GB only if the `imagery_gap = 2`
    stratum clearly underperforms the `0` stratum;
  - the web vs sat encoder ablation.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ortho_strata.py tests/test_ortho_wiring.py tests/test_requirements_complete.py -q -p no:warnings`
Expected: all pass.

- [ ] **Step 5: Full suite, `detect_changes`, commit**

Run: `python -m pytest -q -p no:warnings`. Expected: all pass (previous 572 + new).
Run gitnexus `detect_changes({scope: "all"})` and review it.

```bash
git add configs/mesh-v6-ortho-*.yaml configs/mesh-v5-full-opt-coord-tile.yaml src/analysis/ortho_strata.py docs/ortho-conditioning-runbook.md tests/test_ortho_strata.py tests/test_ortho_wiring.py
git commit -m "feat(ortho): v6 stage configs, stratified analysis, runbook"
```

---

### Task 9: Verification runs (each step only with the user's go-ahead)

There is no code here; these are runs to execute and record.

- [ ] **Step 1: DINOv3 access.** The user runs `huggingface-cli login` and accepts the licence on
  `facebook/dinov3-vitl16-pretrain-lvd1689m`. Check with
  `python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m')"`.
- [x] **Step 2: Data (done 2026-10-14).** 2022 + 2023 imagery downloaded with `--years 2022 2023`.
  All 437 2023 tiles came down, and the three 2022 tiles that returned 403 were recovered by the block
  fallback (0 failed). `cleaner` and `mini_cleaner` were re-parsed into per-year mosaics. Read their
  `ortho_manifest.json` for the residual partial coverage; those buildings are the ones Task 2 skips.
- [ ] **Step 2b: Spatial-correlation gate.** Run
  `python -m src.analysis.spatial_correlation "data/The Hague/cleaner" --out outputs/spatial_correlation`.
  - **Expected:** ICC(1) well above its shuffled-label 99th percentile for all three metrics, a correlogram
    that is positive at short range and decays, and random-split nearest-train neighbours that are
    closer and more similar than under the tile split.
  - **If confirmed:** adopt `training.split: tile`. **Train `configs/mesh-v5-full-opt-coord-tile.yaml`
    first.** It is a multi-day run, as v5 was, and it is both the fair baseline and stage A's init.
  - **If not confirmed:** set the v6 configs back to `split: random` with `init_from` the existing v5
    checkpoint, and record why in the runbook.
- [ ] **Step 3: Loader timing.** Time 2,000 random `MeshDataset(..., ortho_dir="ortho")[i]` reads on
  `cleaner` with 4 workers. Record items per second. The plan assumed 300–800/s.
- [ ] **Step 4: Smoke.** `python main.py --train --config configs/mesh-v6-ortho-smoke.yaml`. It must
  finish an epoch. `train_gate_abs_mean` must move off 0, and `val_gen_chamfer_m` at epoch 0 must be
  within noise of v5 on the same buildings (step 0 is v5).
- [ ] **Step 5: Stage A full run.** Afterwards, run post-hoc eval with `image_mode` true, shuffled and
  none on `val`, then `ortho_strata`. Apply the research doc §4.3 gate: gates open, and a shuffled gap
  beyond metric noise.
- [ ] **Step 6: Stage B and prefix control**, only if the gate in Step 5 passes. Compare all three
  against v5 with the same post-hoc protocol.

---

## Self-review notes

- **Spec coverage:**
  - §2.1 imagery: done; 2023 is a decision point in the Task 8 runbook.
  - §2.2 crop: done (dataset).
  - §2.3 coordinate embedding: Task 3.
  - §2.4 frozen DINOv3: Task 3.
  - §2.5 image dropout: Task 5.
  - §3 prefix: Tasks 4–5.
  - §4 staged cross-attention: Tasks 4–6, 8.
  - §6 evaluation protocol: Tasks 7–8.
  - §7 decisions: runbook (Task 8) and Task 9.
- **Out of scope:**
  - Image-conditioned inference on unseen data (needs per-building crops outside the dataset):
    refused explicitly in Task 6.
  - The scratch backbone: refused in Task 5.
- **Known limitation:** the forward pre-hooks are bound to the module instance. `copy.deepcopy` of
  the network would keep hooks pointing at the original `GatedCrossAttention`. Nothing in the
  training path deep-copies the model (no EMA here), so the plan accepts this and documents it in
  `ortho_fusion.py`'s docstring.
