# VQ-VAE mesh tokenizer for the LOD1-conditioned mesh transformer

Date: 2026-08-08
Status: approved, stage 1 of 2

## Why

`mesh-1` (wandb `rp7ssm9y`, 5.27M params, 46 epochs) converged with the copyable
half of the task solved and the added half lagging:

| signal | value | reading |
| --- | --- | --- |
| `footprint_iou` | 0.979 | the LOD1-given footprint is essentially exact |
| `bin_mae` y / z / x | 0.52 / 1.33 / 6.15 | z is 2.5x worse than y despite being the *primary* sort key in `canonicalize` |
| `roof_mean_z_err_m` | 0.305 | at `coord_mae_m` 0.292, i.e. the roof is no better predicted than an average coordinate |
| `merged_face_ratio` | 0.83 | 17% fewer polygons than ground truth: systematically simplified |
| `watertight_rate` | 0.53 | against a 0.78 ground-truth ceiling |
| `tf_chamfer_m` vs `gen_chamfer_m` | 0.260 vs 0.179 | free-running beats teacher-forced, so **not** exposure bias |

val_loss moved 0.397 -> 0.389 across epochs 26-42 and early stopping fired at
46, so it is not undertrained either. The failure is representational: the model
emits a coherent but simplified mesh. That is the case a learned mesh vocabulary
addresses, per MeshAnything (Chen et al., arXiv:2406.10163).

## Scope

Stage 1, this spec: the VQ-VAE as a **standalone tokenizer/detokenizer** with
its own training entry point, selectable alongside the existing coordinate
tokenizer. Stage 2, a later spec: rewiring `MeshTransformer` to generate codes.

The coordinate tokenizer in `src/dataset/mesh_dataset.py` is not removed or
changed. It stays the default and the measurement baseline.

## Success gate

The VQ-VAE decoder predicts logits over discretized vertex coordinates
(paper §4.2, eq. 8), so its output lands on the same 128-bin grid as the
coordinate tokenizer. Its reconstruction ceiling is therefore that tokenizer's
measured 0.107 m chamfer **plus** whatever the codebook loses -- it can never
beat it, and a spec that asks it to would be asking for the impossible.

The trade being bought is 3x sequence compression (9 tokens/face -> 3) and a
decoder that tolerates imperfect codes. So the gate is:

1. Reconstruction chamfer **<= 0.15 m** on the val split (vs the 0.107 m floor).
2. Codebook usage healthy: no more than ~20% dead entries at the configured
   size, judged from the logged perplexity and distinct-code count.

Failing (1) means the codebook is lossier than the compression is worth.
Failing (2) means the codebook is mis-sized, and the size is a config knob.

## Architecture

```
verts, faces
   |  quantize()                     (reused, src/dataset/mesh_dataset.py)
   v
coords [F, 9]  int64
   |  per-coordinate embedding, 9 -> d_model projection
   v
face embeddings [F, d]
   |  encoder: nn.TransformerEncoder over the face sequence
   v
z [F, d]  ------------------------------> CONDITION PATH STOPS HERE
   |  residual vector quantization, depth D
   v
codes [F, D]  ---------------------------> what stage 2 will generate
   |  decoder: nn.TransformerEncoder
   v
logits [F, 9, num_bins]
   |  argmax, dequantize(), canonicalize()  (reused)
   v
verts, faces
```

Encoder and decoder are both `nn.TransformerEncoder` stacks, following the paper
("we employ transformers with identical structures for both", §4.2), not
MeshGPT's graph-convolution encoder.

### Public surface of `MeshVQVAE`

| method | returns | used by |
| --- | --- | --- |
| `encode(verts, faces)` | `z [F, d]` continuous | stage 2 condition path |
| `tokenize(verts, faces)` | `codes [F, D]` | stage 2 target path |
| `detokenize(codes)` | `(verts, faces)` | stage 2 decode |

`tokenize`/`detokenize` mirror the module-level functions of the same name in
`mesh_dataset.py`, so stage 2 selects between them rather than restructuring.

### Why the condition is encoded but not quantized

Decided in design discussion; recorded here because it is the one place this
design departs from a literal reading of the paper.

MeshAnything conditions on a point cloud through a frozen pretrained encoder
into 257 continuous projected tokens (§4.3) -- the VQ-VAE is target-side only.
For them condition and target are different modalities, so separate encoders
were forced and the question never arises. Ours are the same modality in the
same coordinate frame, so it does.

Quantization exists to make a sequence generatable by cross-entropy. The
condition is never sampled and never decoded, so quantizing it buys nothing
and costs precision on the one input known exactly. It also fails ungracefully:
LOD1 faces are large wall quads and a ground plane, structurally unlike the
small LOD2 roof triangles the codebook mostly sees, and a lookup snaps to the
nearest entry however far that is, where a continuous embedding degrades
smoothly.

Encoding it (rather than leaving it as raw coordinate tokens) protects what
already works. `LOD1_synth` is extruded from the LOD2 footprint, so x and y
coincide by construction and the footprint is literally copyable token for
token; `footprint_iou` 0.979 is the model exploiting that. Moving the target to
a codebook while leaving the condition in coordinate bins would break the
correspondence and force the model to learn a coordinate-to-code translation.
Running the same encoder on both sides restores it at face granularity.

This is a bet, not a certainty, so stage 2 exposes the condition path as a
config switch and the question stays empirical.

## Sizing

The paper uses codebook 8192 x RVQ depth 3 over 56k diverse Objaverse/ShapeNet
meshes. Our faces are walls, ground planes and roof slopes -- far less varied --
over roughly 800k faces (~20k buildings x ~40 faces on `mini`).

Starting point: **codebook 1024, depth 3**, both configurable. Depth 3 keeps the
paper's compression. 1024 rather than 8192 because a codebook far larger than
the diversity of the data leaves most entries dead and the live ones
undertrained. Codebook perplexity and distinct-code count are logged every
epoch, so the size is calibrated from the run rather than guessed twice.

## Data

No change to `MeshDataset`. The VQ-VAE trains on faces reshaped from the
existing `cond` and `tgt` token tensors (`[9F] -> [F, 9]`), drawn from **both**
LODs so the codebook covers LOD1's large quads as well as LOD2's roof
triangles. A dedicated datamodule wraps the existing one and flattens
per-building items into per-face batches.

## Files

Purely additive except two one-line edits.

| file | change |
| --- | --- |
| `src/models/mesh_vqvae.py` | new: `MeshVQVAE` (nn.Module), `MeshVQVAEModule` (Lightning) |
| `src/dataset/mesh_vqvae_datamodule.py` | new: face batches from `MeshDataset` |
| `src/train_mesh_vqvae.py` | new: stage-1 training, mirrors `train_mesh.py` |
| `configs/mesh-vqvae.yaml` | new: `config_set: mesh_vqvae` |
| `tests/test_mesh_vqvae.py` | new: written first and failing, per the TDD gate |
| `src/utils/config.py` | add `MeshVQVAEConfig` + one field on `Config` |
| `src/train.py` | one dispatch branch |
| `src/utils/initialization.py` | add `mesh_vqvae` to `--config-set` choices |

Impact analysis run before editing: `train` LOW (1 caller, `main.main`;
additive branch). `Config` MEDIUM (5 direct importers) but the change is a new
dataclass field with a default, so every existing config still resolves.

## Error handling

- `detokenize` on empty or degenerate codes returns `(zeros((0,3)), zeros((0,3)))`,
  matching the coordinate `detokenize` contract, so a caller can `nanmean` over
  a batch where some items decoded to nothing.
- Codebook indices are clamped to the codebook size on decode, so a stage-2
  transformer emitting an out-of-range code cannot raise inside the decoder.
- Dead-codebook-entry reinitialization is **out of scope** for stage 1; the
  usage metric reports the problem, and a fix is only worth writing once the
  metric shows one.

## Testing

An untrained codebook cannot reconstruct, so unit tests assert the contract and
leave quality to the training gate above.

- shapes at each stage: `encode -> [F, d]`, `tokenize -> [F, D]`,
  decoder logits `-> [F, 9, num_bins]`
- codes lie in `[0, codebook_size)`
- `detokenize` returns a mesh with the expected face count, and empty input
  returns the empty-mesh pair rather than raising
- straight-through estimator: gradients reach encoder parameters through the
  quantizer
- commitment loss is finite and strictly positive on non-degenerate input
- usage metric counts distinct codes correctly on a known assignment
- `tokenize` is deterministic in eval mode

## Out of scope

- The noise-resistant decoder (paper §4.2). It is a fine-tuning stage on top of
  a trained VQ-VAE and conditioned on the shape input, so it presupposes both
  stages. Worth revisiting once stage 2 has a number.
- Any change to `MeshTransformer`, `MeshDataset`, or the coordinate tokenizer.
- Dead-entry reinitialization, as above.
