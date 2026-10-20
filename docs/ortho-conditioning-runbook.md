# Orthophoto conditioning runbook (v6)

How to train and read the image-conditioned mesh transformer. Design:
`docs/research/2026-10-14-orthophoto-conditioning.md`. Implementation plan:
`docs/superpowers/plans/2026-10-14-orthophoto-conditioning.md`.

The model is v5 (OPT, coordinate tokens, sinusoidal positions) plus a frozen
DINOv3 encoding of the building's top-down crop. The crop enters through one
zero-gated cross-attention block per OPT layer (Flamingo, arXiv:2204.14198),
or as prefix tokens in the control arm (LLaVA, arXiv:2304.08485).

## Prerequisites

1. **DINOv3 access.** The weights are gated on Hugging Face under the DINOv3
   licence (not Apache):
   ```bash
   huggingface-cli login
   # then accept the licence on facebook/dinov3-vitl16-pretrain-lvd1689m in the browser
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m')"
   ```
2. **Imagery.** `data/The Hague/{cleaner,mini_cleaner}/ortho/` with
   `ortho_manifest.json` and `year_assignment.json` (2022 + 2023 flights,
   already downloaded and parsed). Buildings without full coverage are skipped
   after the split; the count is in each run's `split.json`.
3. **Split decision.** The v6 configs use `training.split: tile`. That is
   provisional: if the block-size sweep in `outputs/split_sweep` points to a
   grid split, change `split` (plus `split_block_m` / `split_buffer_m`) in the
   baseline and in every v6 config together. `load_weights_from` refuses an
   init trained on a different split, so a mismatch fails at startup rather
   than leaking val/test buildings.
4. **Baseline.** Train `configs/mesh-v5-full-opt-coord-tile.yaml` (multi-day).
   It is both stage A's init and the image-free baseline. The original v5 was
   trained on the random split, so it can be neither.

## Order of runs

| Step | Config | Starts from | Trains |
|---|---|---|---|
| 0 | `mesh-v5-full-opt-coord-tile.yaml` | scratch | everything (image-free) |
| 1 | `mesh-v6-ortho-smoke.yaml` | v5-tile best | image path, mini_cleaner, 1 epoch |
| 2 | `mesh-v6-ortho-xattn-a.yaml` (stage A) | v5-tile best | image path only |
| 3 | gate check (below) | | |
| 4 | `mesh-v6-ortho-xattn-b.yaml` (stage B) | stage A best | everything but DINOv3; OPT at 0.1x lr |
| 5 | `mesh-v6-ortho-prefix.yaml` (control) | v5-tile best | everything but DINOv3; OPT at 0.1x lr |

Before each run, replace the `mesh_model.init_from` placeholder with the real
best checkpoint path (`{epoch:02d}-{val_gen_chamfer_m:.4f}.ckpt` in the
source run's `checkpoints/`).

```bash
python main.py --train --config configs/mesh-v6-ortho-smoke.yaml
python main.py --train --config configs/mesh-v6-ortho-xattn-a.yaml
```

**Smoke pass criteria:** it finishes an epoch, `train_gate_abs_mean` moves off
0, and epoch-0 `val_gen_chamfer_m` is within noise of the v5-tile model on the
same buildings. With zero gates, step 0 *is* that model, so a large epoch-0 gap
means a wiring bug, not a learning effect. (The smoke runs on mini_cleaner,
whose tile split is not cleaner's; it is a plumbing check, not a measurement.)

## Post-hoc evaluation with image controls

`image_mode` swaps what each building sees at generation time:

- `true`: its own crop.
- `shuffled`: the crop of the building half the split away, which is never
  itself and spatially unrelated.
- `none`: the learned null token, the same thing image dropout trained the
  model on.

Make a post-hoc config for the run, with `reference_config` set to the v6
training config and `checkpoint` set to its best checkpoint. Then:

```bash
python -m src.eval.posthoc_eval --config configs/posthoc-eval-<run>.yaml image_mode=true
python -m src.eval.posthoc_eval --config configs/posthoc-eval-<run>.yaml image_mode=shuffled
python -m src.eval.posthoc_eval --config configs/posthoc-eval-<run>.yaml image_mode=none
```

Outputs go to `<run>/posthoc_eval/<ckpt>/<stamp>` for `true` and
`<run>/posthoc_eval/<ckpt>/<mode>-<stamp>` for the others. Score the v5-tile
baseline with its own post-hoc config. `image_mode` has no effect on an
image-free model.

## Stratified comparison

```bash
python -m src.analysis.ortho_strata --dataset-dir "data/The Hague/cleaner" \
    --run v5=<v5-tile posthoc dir>/greedy \
    --run true=<true dir>/greedy --run shuffled=<shuffled dir>/greedy --run none=<none dir>/greedy \
    --metric chamfer_m_vs_lod2 --out outputs/ortho_strata/<run>
```

This writes `strata.csv` and `strata.md` with the columns `mode, stratum,
value, n, mean, median`. Every run is restricted to the **ids present in all
runs**, so image runs, which skip buildings without imagery, are compared
with v5 on the same buildings. The strata are:

- `all`;
- `pw_datum`: the 3DBAG point-cloud year;
- `height_bin`: LOD2 z extent, `<10`, `10-20`, `20-40` or `>=40` m. Tall
  buildings lean in the mosaic.
- `imagery_gap`: |point-cloud year - assigned imagery year|. This needs
  `ortho/year_assignment.json`.

## Reading the results

**Gates (stage A).** `train_gate_{i}` is tanh(alpha_i) for layer i, and
`train_gate_abs_mean` is their mean magnitude. They start at exactly 0.

- Gates that open in some layers mean the model found something usable in
  the image.
- Gates that all stay near 0 mean it did not. Unfreezing OPT will not fix
  that. Look upstream instead: crop alignment, building lean, or the encoder
  variant.

**Shuffled gap (the headline).** This is the `true` versus `shuffled` mean
chamfer on the same buildings. It measures what the image contributes,
independent of how good the absolute chamfer is. It counts only if it exceeds
the per-object metric noise measured in `docs/posthoc-eval-runbook.md`.

**`none` against v5.** This should land near v5. If `none` is much worse than
v5, image dropout was too low for the model to learn to work without the
image.

**Gate to stage B** (research doc section 4.3). Move on only when all three
hold:

1. Stage A plateaued: early stopping fired.
2. The gates opened.
3. The shuffled gap is real.

## Decision points

- **2021 imagery for the 2020/2014 point-cloud buildings.** Those buildings
  currently use the 2022 flight, a 2-year gap. Adding 2021 costs about 14 GB
  and moves only ~10% of buildings to a 1-year gap. Download it only if the
  `imagery_gap = 2` stratum clearly underperforms the `0` stratum in the
  `true` mode, beyond noise.
- **Web vs sat encoder.** Start with `facebook/dinov3-vitl16-pretrain-lvd1689m`
  (web). The sat variant (`...-sat493m`) was trained on 0.6 m Maxar imagery
  and is stronger on metric tasks. Ablate it by changing only
  `mesh_model.image_encoder` and rerunning stage A. Compare the shuffled gaps,
  not the absolute chamfer.
