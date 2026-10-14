import logging
from dataclasses import dataclass, field
from typing import Optional, List, Union
from omegaconf import MISSING

logger = logging.getLogger(__name__)

@dataclass
class MeshDataConfig:
    """Dataset/datamodule for the LOD1-conditioned mesh transformer.

    Reads two LOD directories by name rather than by LOD number.
    """
    # Not MISSING, so a config that never reads mesh data (posthoc eval's own
    # file, the schema round-trip tests) still resolves. Checked in
    # `train_mesh` / `train_mesh_vqvae` instead.
    dataset_dir: Optional[str] = None
    # Directory names under dataset_dir, not LOD numbers. LOD1_synth is the
    # default input because the real LOD1 folder only covers Source B, while
    # every LOD2 building has a synthesized LOD1 counterpart.
    lod_in: str = "LOD1_synth"
    lod_out: str = "LOD2"
    # Coordinate discretization. 128 is the MeshAnything/MeshGPT default; it
    # bounds how exactly the tokenizer can reproduce a mesh.
    num_bins: int = 128
    # How the face list is spelled out as a sequence. Orthogonal to
    # `mesh_model.tokenizer`, which chooses what a token *is*:
    #   "coord" -- three vertices per face, 9 tokens. The exact inverse, and the
    #       default, so every existing run and checkpoint is unaffected.
    #   "amt"   -- Adjacent Mesh Tokenization (MeshAnything V2,
    #       arXiv:2408.02555, Algorithm 1): one new vertex per face wherever it
    #       is adjacent to the previous one, with a break token otherwise.
    #       Measured 0.55x the sequence length on The Hague/mini LOD2, against
    #       ~0.49x reported on Objaverse. Adds one id to the vocabulary.
    # AMT requires `mesh_model.tokenizer: coord` -- V2 drops the VQ-VAE for
    # exactly this reason, and the combination raises rather than silently
    # ignoring one of them.
    tokenization: str = "coord"
    # Per-axis (x, y, z) headroom below / above the LOD1 normalization box, as a
    # fraction of its scale, so LOD2 geometry outside that box is not quantized
    # flat onto a box face. Split by side, and settable on all six, because the
    # overflow is one-directional. p99 over the 8016 mini (LOD1_synth, LOD2)
    # pairs under max_faces=200: every side is 0.000 except +z, which needs
    # 0.087 for LOD1_synth and 0.102 for real LOD1 -- ridges above the median
    # roof height, which is all convert_to_lod1 keeps. Rounded to 0.1 to cover
    # both inputs. x and y are 0 because LOD1_synth is extruded from the LOD2
    # footprint, so they coincide by construction; -z is 0 because the two share
    # a ground plane. The tail past p99 (+z p99.9 = 0.58, max 2.11) is source
    # defects: covering it would cost every building most of its z resolution.
    # MeshDataset logs the measured requirement for all six sides at load;
    # re-read it on `clean`, which mini under-represents at the large end.
    margin_lo: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    margin_hi: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.1])
    # Buildings above this triangle count are dropped. 9 tokens per triangle
    # and quadratic attention, so this is the real sequence-length knob.
    max_faces: Optional[int] = 200
    # Read only the first N files per LOD. Smoke tests and sample dumps only.
    max_files: Optional[int] = None
    num_workers: int = 4
    persistent_workers: bool = False

@dataclass
class MeshEvalConfig:
    """Free-running generation scored against the paired ground-truth LOD2.

    Per-pair rather than distributional: every LOD1 condition has exactly one
    correct answer, so unconditional distribution metrics are meaningless here.
    """
    enabled: bool = True
    # generate() has no KV cache, so a 200-face building is ~1800 sequential
    # forward passes. This is the knob that decides how much a run costs.
    every_n_epochs: int = 10
    n_val: int = 8
    n_test: int = 32
    batch_size: int = 8
    n_points: int = 4096          # surface samples per mesh for chamfer/F-score
    taus: List[float] = field(default_factory=lambda: [0.25, 0.5])  # F-score, metres
    voxel_m: float = 0.25         # IoU grid resolution, metres
    save_samples: int = 8         # .obj + .city.json written per test run
    # 1 is greedy argmax, the reference's setting (MeshAnything V2 passes
    # `num_beams=1`). >1 searches for the best *sequence* rather than the best
    # next token, at beam_size x the compute -- and generation has no KV cache,
    # so that multiplies an already O(L^2) loop. Deterministic, so it forces
    # temperature 0.
    beam_size: int = 1
    # Beam scores are divided by ``length ** penalty``. 0.0 is raw
    # sum-of-log-probs, which prefers short sequences -- and short here means a
    # mesh with fewer triangles, so the default normalises per token.
    length_penalty: float = 1.0

@dataclass
class MeshModelConfig:
    """Architecture of the autoregressive mesh transformer (arXiv:2406.10163)."""
    d_model: int = 256        # must be divisible by n_head
    n_head: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    # Model capacity in triangles -- how large a mesh the model has room for.
    # Deliberately NOT `mesh_data.max_faces`, which decides which buildings
    # are in the corpus. They were one knob until now, so capacity was a
    # side effect of a data filter and the mesh-v3 arms had to run at 200
    # (scratch) and 113 (OPT) -- which confounded scratch-vs-OPT with
    # capacity. MeshAnything V2 keeps them apart (`data_process.py
    # --max_face_num` filters, `n_max_triangles` sizes) and so does this.
    #
    # None = fall back to the loaded corpus's longest segment, which is the
    # old coupled behaviour and stays the default so existing configs are
    # unchanged. Set it to compare backbones at one capacity.
    n_max_triangles: Optional[int] = None
    # Capacity for the LOD1 *condition*, in triangles. Only OPT pays for this
    # (its positions run through both halves); the scratch stack restarts per
    # segment. Defaults to `n_max_triangles`, the worst case.
    #
    # Sizing it down is the lever that makes `opt_pretrained: true` reachable:
    # LOD1 is a prism, and at cap 200 it drops zero buildings on its own, so
    # budgeting it like a full LOD2 doubles the table for nothing. AMT at 200
    # target faces needs 2523 shared positions at worst case but 1578 with a
    # 50-triangle condition -- inside opt-350m's 2048 trained rows.
    cond_max_triangles: Optional[int] = None
    # Explicit override for the positional capacity, in *tokens*. Highest
    # precedence; leave it None and let `n_max_triangles` do the arithmetic.
    max_seq_len: Optional[int] = None
    # "coord" -- 9 discretized coordinates per face, an exact inverse, the
    # measurement baseline and the default. "vqvae" -- codes from a trained
    # stage-1 checkpoint, `3 * depth` per face, and the LOD1 condition goes
    # through the same encoder unquantized (see the design spec for why not).
    tokenizer: str = "coord"
    # Stage-1 checkpoint. Required when tokenizer == "vqvae"; the codebook is
    # loaded frozen, so nothing here can move the vocabulary mid-run.
    vqvae_ckpt: Optional[str] = None
    # Masking Invalid Predictions (MeshAnything V2 section 3.2, from PolyGen):
    # at sampling time, zero out logits for tokens that cannot legally follow --
    # EOS in the middle of a face, a break straight after a break, a break
    # before a strip has three vertices. Affects generation only, never
    # training, so it can be turned on for an existing checkpoint. Off by
    # default so an existing run's samples stay reproducible.
    mask_invalid: bool = False
    # "scratch" -- the transformer defined here, sized by d_model / n_head /
    # num_layers / max_seq_len above. "opt" -- an OPT backbone, which is what
    # the paper adopts for stage 2; those four fields are then read from the
    # OPT config instead and only `dropout` still applies.
    #
    # OPT is ~331M parameters against ~20M for the scratch model at d_model 512.
    # On a 16k-building corpus that is a large bet on the pretrained
    # initialization doing the regularizing, so treat it as an experiment
    # against a measured scratch baseline, not as a default.
    backbone: str = "scratch"
    # Any OPT id; the two sizes worth trying here are `facebook/opt-125m`
    # (~125M, hidden 768, no embed projection) and `facebook/opt-350m` (~331M,
    # hidden 1024 projected from 512), both 2048 positions. Every dimension is
    # read from the loaded config, so switching is this line and nothing else.
    opt_name: str = "facebook/opt-350m"
    # Load OPT's weights, or only its architecture. The reference builds it with
    # `from_config` (random) and loads its own checkpoint, so a warm start from
    # the language model is the paper's text rather than its released code.
    opt_pretrained: bool = True
    # How sequence position reaches the model, on *either* backbone.
    #
    # "learned" -- an embedding table, the default and what every run so far
    # used. Its row count is a hard ceiling: `max_seq_len` on the scratch stack,
    # and on OPT the `max_position_embeddings` that rides in from a hub config
    # whose weights are not even loaded when opt_pretrained is false.
    #
    # "sinusoidal" -- fixed Fourier positions (Vaswani et al., arXiv:1706.03762
    # section 3.5). No parameters, defined at every position, so the ceiling
    # disappears rather than moving and long LOD2 roofs stop being a capacity
    # question. Positions still restart per segment on the scratch backbone and
    # still run straight through on OPT; only the lookup changes.
    #
    # RoPE and ALiBi are deliberately absent: both act inside attention, and
    # `OPTAttention.forward` takes no position argument, so they would need a
    # fork of every layer rather than a config switch.
    pos_embed: str = "learned"

@dataclass
class MeshVQVAEConfig:
    """Stage-1 learned mesh vocabulary (arXiv:2406.10163 §4.2).

    Read only under `config_set: mesh_vqvae`. The mesh data itself still comes
    from `mesh_data`, so the two stages cannot disagree about bins, margins or
    the train/val/test split.
    """
    codebook_size: int = 1024
    # Residual stages, i.e. codes per *vertex*. 3 is the paper's setting; with 3
    # vertices to a face that is 9 tokens per face -- the same count as raw
    # coordinates, because the codebook buys a learned vocabulary rather than
    # compression (`face_per_token = num_quantizers * 3` in the reference).
    # Quantizing one feature per face instead gave 3 tokens and a bottleneck 2x
    # narrower than the data, which capped reconstruction at 0.246 m against the
    # coordinate tokenizer's 0.107 m.
    depth: int = 3
    d_model: int = 256        # must be divisible by n_head
    # Width of the code vectors, independent of d_model. MeshGPT's
    # `project_dim_codebook = Linear(curr_dim, dim_codebook * nvf)` uses 192
    # against a model dim of 512, so codes live in a narrower space than the
    # transformer that reads them. None ties it to d_model, which is a
    # coincidence rather than a design. Stage 2 embeds code ids by projecting
    # these vectors, so this is also the input width of that projection.
    codebook_dim: Optional[int] = None
    n_head: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    # Capacity of the per-face positional embedding. None = size it from the
    # loaded corpus's longest mesh, logged at startup.
    max_faces: Optional[int] = None
    # Weight on the term pulling the encoder toward the codebook
    # (van den Oord et al., 2017). 0.1 is MeshGPT's `commit_loss_weight`. With
    # EMA codebook updates this is the *only* quantizer term in the objective,
    # so it is also the whole knob for balancing against the reconstruction
    # cross-entropy.
    commitment: float = 0.1
    # EMA decay for the codebook, matching vector-quantize-pytorch's default.
    decay: float = 0.8
    # Replace the straight-through estimator with the rotation trick (Fifty et
    # al., ICLR 2025, arXiv:2410.06424): the gradient at the code is rotated
    # onto the encoder output rather than copy-pasted, so points in one Voronoi
    # cell get different updates by angle. Reported to cut quantization error by
    # an order of magnitude and raise codebook usage. Off by default: it
    # postdates MeshAnything, so turning it on is a departure from the method,
    # not a reproduction of it. A/B it against a settled baseline -- and note
    # its stated failure mode (paper §6) is codewords near zero norm, which is
    # what `kmeans_init` exists to prevent.
    rotation_trick: bool = False
    # Noise-resistant decoder fine-tune (arXiv:2406.10163 §4.2). A *second*
    # stage-1 run: load `init_from`, freeze everything the codes depend on, and
    # retrain the decoder alone with the LOD1 condition injected and codes drawn
    # with Gumbel noise. This is what makes the decoder tolerate the imperfect
    # tokens a transformer emits -- without it the decoder has only ever seen
    # its own encoder's exact codes, which is the reason to prefer a learned
    # vocabulary over the coordinate tokenizer in the first place.
    noise_resistant: bool = False
    # Gumbel temperature for that fine-tune. Higher = codes further from the
    # ones the encoder would have picked. Ignored unless noise_resistant.
    noise_temp: float = 1.0
    # Stage-1 checkpoint to initialize from. Required when noise_resistant:
    # starting the fine-tune from scratch trains a decoder against a codebook
    # that means nothing yet.
    init_from: Optional[str] = None

@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping callback."""
    enabled: bool = True
    monitor: str = "val_coord_mse"
    patience: int = 10
    mode: str = "min"

@dataclass
class CheckpointConfig:
    """Configuration for checkpoint callback."""
    enabled: bool = True
    monitor: str = "val_coord_mse"
    mode: str = "min"
    save_top_k: int = 3
    filename: str = "{epoch:02d}-{val_coord_mse:.4f}"

@dataclass
class TrainingConfig:
    """Configuration for training loop."""
    batch_size: int = 32
    train_val_test_split: List[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    lr: float = 1e-3
    max_epochs: int = 100
    gradient_clip_val: Optional[float] = None
    # Optimizer steps are taken every N batches, so the *effective* batch is
    # batch_size * this. Lightning accumulates the gradient itself and clips
    # after accumulating, so gradient_clip_val keeps its meaning. 1 = off.
    #
    # Two things it does not do on its own. `lr` is not rescaled -- a larger
    # effective batch usually wants a larger or better-warmed lr, and that is a
    # separate decision. And `trainer/global_step` counts optimizer steps, so
    # steps-per-epoch drops by this factor; a step-interval scheduler would
    # change meaning, though this project's schedulers are all epoch-interval.
    accumulate_grad_batches: int = 1
    lr_scheduler: str = "none"  # "none", "cosine", "step"
    # Linear ramp from ~0 to `lr` over this many OPTIMIZER STEPS (not epochs),
    # composable with any lr_scheduler value. 0 disables.
    #
    # Warmup is measured in steps because that is what it protects: the first
    # few hundred updates, when Adam's second-moment estimate is still cold and
    # a large lr can take a step it cannot walk back. At 127 steps/epoch an
    # epoch-scaled warmup would ramp for thousands of steps, which is a
    # different intervention entirely.
    #
    # NOTE: honoured by `config_set: mesh` only. mesh_vqvae ignores it, so
    # setting it there is a silent no-op rather than an error.
    warmup_steps: int = 0
    # AdamW's own default, so leaving this unset reproduces every run trained
    # before it was exposed. It applies ONLY to weight matrices: embeddings,
    # norm gains and biases are put in a second, undecayed param group.
    # Decay shrinks a parameter toward zero, which is a sensible prior for a
    # weight matrix and a wrong one for a LayerNorm gain (whose meaningful
    # default is 1) or a bias (one parameter per unit, no capacity to
    # regularize). Decoupled decay is applied every step independently of the
    # gradient, so its strength scales with lr x steps -- at the v3 arms' flat
    # 5e-5 it was a 5% nudge, at 1e-3 with cosine over 30 full-corpus epochs it
    # is a ~7x pull, which is what made the exclusion worth doing.
    #
    # Read by the mesh transformer only.
    weight_decay: float = 0.01
    lr_decay_steps: int = 50
    lr_decay_rate: float = 0.5
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

@dataclass
class TrainerConfig:
    """Configuration for Lightning Trainer."""
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "32"
    log_every_n_steps: int = 50

@dataclass
class LoggingConfig:
    """Configuration for loggers."""
    loggers: List[str] = field(default_factory=lambda: ["tensorboard"])  # combo of "tensorboard", "wandb", "mlflow"
    project_name: str = "lod-generation"
    experiment_name: Optional[str] = "regnn_diffusion"
    run_name: Optional[str] = "default"
    save_dir: str = "outputs"
    wandb_entity: Optional[str] = None
    wandb_offline: bool = False
    log_model: bool = True

@dataclass
class StrategyConfig:
    """One decoding strategy for `src.eval.posthoc_eval`.

    Every param is a list, and lists expand as a cartesian product within the
    strategy. Params that do not belong to `kind` must stay null:
      greedy -- none
      beam   -- beam_size, length_penalty (defaults to [1.0])
      top_p  -- top_p, temperature
      min_p  -- min_p, temperature
    """
    kind: str = MISSING
    beam_size: Optional[List[int]] = None
    length_penalty: Optional[List[float]] = None
    top_p: Optional[List[float]] = None
    min_p: Optional[List[float]] = None
    temperature: Optional[List[float]] = None


@dataclass
class SinkhornConfig:
    """Debiased Sinkhorn divergence settings (`src.eval.sinkhorn`)."""
    n_points: int = 2048      # surface samples per mesh, same seed both sides
    blur_m: float = 0.1       # entropic length scale; eps = blur_m ** 2
    max_iters: int = 500      # cap at the final epsilon
    tol: float = 1e-6         # dual-potential change, m^2


def _default_strategies():
    return [StrategyConfig(kind="greedy"),
            StrategyConfig(kind="beam", beam_size=[3, 5], length_penalty=[1.0]),
            StrategyConfig(kind="top_p", top_p=[0.95], temperature=[0.6, 1.2]),
            StrategyConfig(kind="min_p", min_p=[0.05], temperature=[0.6, 1.2])]


@dataclass
class PosthocEvalConfig:
    """Post-hoc scoring of one mesh-transformer checkpoint under many decoders.

    Holds only eval settings. Data, split seed, tokenizer and model shape all
    come from `reference_config`, the run's own training config, so the split
    and the vocabulary cannot drift from what the checkpoint was trained on.
    """
    reference_config: str = MISSING
    checkpoint: str = MISSING
    # Dot-list applied to the reference, e.g. a relocated dataset_dir.
    reference_overrides: List[str] = field(default_factory=list)
    split: str = "val"
    # None scores the whole split; else a seeded subsample via eval_indices.
    max_objects: Optional[int] = None
    batch_size: int = 8
    seed: int = 1234
    device: str = "auto"
    # None -> <save_dir>/<experiment_name>/<run_name>/posthoc_eval
    output_dir: Optional[str] = None
    strategies: List[StrategyConfig] = field(default_factory=_default_strategies)
    metrics: List[str] = field(default_factory=lambda: [
        "chamfer", "hausdorff", "vol_symdiff", "sinkhorn", "roof_chamfer"])
    references: List[str] = field(default_factory=lambda: ["lod1", "lod2"])
    n_points: int = 4096      # surface samples for the chamfer family
    sinkhorn: SinkhornConfig = field(default_factory=SinkhornConfig)


@dataclass
class InferenceConfig:
    """LOD1 -> LOD2 generation with a trained mesh transformer (`--inference`).

    Run it against the checkpoint's *own* training config, so that
    `mesh_data` (bins, margins, tokenization) and `mesh_eval` (beam settings)
    are the ones the model was trained and selected under.
    """
    # Weights to load. Resolved by `src.inference._resolve_checkpoint`, which
    # dispatches on the prefix:
    #   local path      "outputs/initial-runs/mesh-v5-full-opt-coord/checkpoints/last.ckpt"
    #                   Relative to the cwd. Must exist, or inference exits.
    #   WandB artifact  "wandb://[entity/]project/name:alias"
    #                   e.g. "wandb://me/lod-generation/model-abc123:best". A bare
    #                   "wandb://name:alias" is qualified from logging.wandb_entity
    #                   and logging.project_name. Downloaded to
    #                   <logging.save_dir>/wandb_artifacts/ and the .ckpt inside is
    #                   loaded. Artifacts written by logging.log_model hold one
    #                   model.ckpt; aliases "best"/"latest"/"v<N>" all work.
    #   fsspec URL      "https://...", "s3://...", "gs://..."
    #                   Anything with a "://" that isn't wandb:// is handed to
    #                   Lightning unchanged; it opens the URL itself. Needs the
    #                   matching fsspec backend installed (s3fs, gcsfs, ...).
    checkpoint_path: Optional[str] = None
    # LOD1 to condition on: one CityJSON file, or a directory searched
    # recursively for *.json / *.city.json. Every CityObject with geometry gets
    # one generated LOD2. No LOD2 is read, so this works on unseen data.
    input_path: Optional[str] = None
    batch_size: int = 8
    # Each run writes to a fresh <output_dir>/<timestamp>/ so outputs of two
    # runs are never mixed into one all_buildings.city.json.
    output_dir: str = "outputs/generated"


@dataclass
class Config:
    """Root configuration class."""
    mode: str = "train"  # "train" or "inference"
    # Which set of config objects is built and run:
    #   "mesh"      — mesh_data + mesh_model, the LOD1-conditioned transformer
    #   "mesh_vqvae"— mesh_data + mesh_vqvae, stage-1 learned mesh vocabulary
    # Resolved in `src.train.train`. Deliberately a plain string switch rather
    # than a registry; there are two of these and they are all in one repo.
    config_set: str = "mesh"
    seed: int = 42
    mesh_data: MeshDataConfig = field(default_factory=MeshDataConfig)
    mesh_model: MeshModelConfig = field(default_factory=MeshModelConfig)
    mesh_eval: MeshEvalConfig = field(default_factory=MeshEvalConfig)
    mesh_vqvae: MeshVQVAEConfig = field(default_factory=MeshVQVAEConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    resume_from: Optional[str] = None

