from dataclasses import dataclass, field
from typing import Optional, List, Union
from omegaconf import MISSING

@dataclass
class DataConfig:
    """Configuration for dataset and datamodule."""
    dataset_dir: str = MISSING
    lods: List[int] = field(default_factory=lambda: [1, 2])
    normalize_coords: bool = False
    num_workers: int = 4
    # Keep DataLoader workers alive across epochs (ignored when num_workers=0).
    persistent_workers: bool = False
    n_max: Optional[int] = None
    upper_limit_nodes: Optional[int] = None
    # Metres per unit of the model's coordinate space. None = compute the pooled
    # std of the train split. Set explicitly to reuse a scale across runs.
    coord_scale: Optional[float] = None
    # se2 only: metres subtracted from z before scaling, so the absolute-height
    # channel is zero-mean under the N(0,1) prior. None = train-split mean
    # vertex z (ignored unless model.equivariance == "se2").
    z_shift: Optional[float] = None
    # Bessel distance basis (model.dist_embed == "bessel") only: cutoff radius in
    # normalised coord units. None = compute a high quantile of train-split
    # pairwise distance on the fly. Ignored for other dist_embed modes.
    dist_r_max: Optional[float] = None

@dataclass
class ModelConfig:
    """Configuration for model architecture and the diffusion noise schedule."""
    num_node_classes: int = 5  # vertex, ground, roof, wall, off (Levi graph)
    num_edge_classes: int = 3  # off, vertex-vertex, vertex-face
    equivariance: str = "so2"  # "so2" | "se2" | "o3"
    time_embed: str = "scalar"  # "scalar" (raw t/T) | "sinusoidal" (Fourier lift)
    # Pairwise-distance featurization into lin_dist1. "raw" = MiDi's single
    # linear channel; the others lift distance for multi-scale resolution.
    dist_embed: str = "raw"  # "raw" | "sinusoidal" | "mlp" | "bessel"
    dist_embed_dim: int = 16  # feature width of the lift (even for sinusoidal); ignored for "raw"
    hidden_dim: int = 64      # dx: node channel width, must be divisible by n_head
    edge_dim: int = 32        # de: edge channel width; drives [B,N,N,de] memory
    global_dim: int = 32      # dy: global feature width
    n_head: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    T: int = 500
    # "marginal" = limit distribution is the training class marginals (MiDi default)
    discrete_noise_type: str = "marginal"  # "marginal" | "uniform"

    # Cosine schedule exponent per feature (positions, node classes, edges).
    # nu_pos > nu_e destroys coordinates faster than graph structure.
    nu_pos: float = 2.5
    nu_x: float = 1.0
    nu_e: float = 1.5

    # Loss weights, MiDi's lambda_train restricted to (pos, X, E).
    lambda_pos: float = 3.0
    lambda_x: float = 0.4
    lambda_e: float = 2.0

    # Weight of the Off-slot coordinate term. The coord MSE is normalised over
    # real slots only; Off keeps a smaller anchor because the zero-CoM projection
    # spans every slot, so unheld Off positions rigidly translate the building.
    off_anchor_weight: float = 0.1

    # Cross-entropy class balancing, the categorical counterpart of the above.
    # Off is 73.7% of node slots and 99.0% of node pairs; GroundSurface is 0.67%
    # of node slots. class_balance is the exponent on inverse frequency: 0 leaves
    # the loss proper, 1 equalises every class's contribution. off_ce_weight
    # scales the Off class on top, since how often the model emits Off sets the
    # generated graph's size. Any non-default value biases the network's x0
    # posterior, which `_debias` divides back out before sampling.
    class_balance: float = 0.5
    off_ce_weight: float = 1.0

@dataclass
class MeshDataConfig:
    """Dataset/datamodule for the LOD1-conditioned mesh transformer.

    Separate from `DataConfig`: the mesh pipeline shares none of its fields
    (no n_max, no coord_scale, no marginals) and reads two LOD directories by
    name rather than by LOD number.
    """
    # Not MISSING: every existing diffusion config would then fail to resolve
    # just for leaving this whole block out. Checked in `train_mesh` instead.
    dataset_dir: Optional[str] = None
    # Directory names under dataset_dir, not LOD numbers. LOD1_synth is the
    # default input because the real LOD1 folder only covers Source B, while
    # every LOD2 building has a synthesized LOD1 counterpart.
    lod_in: str = "LOD1_synth"
    lod_out: str = "LOD2"
    # Coordinate discretization. 128 is the MeshAnything/MeshGPT default; it
    # bounds how exactly the tokenizer can reproduce a mesh.
    num_bins: int = 128
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

    Replaces `GenerativeEvalConfig` on the mesh path: that one measures an
    unconditional model against a distribution, which is meaningless here since
    every LOD1 condition has exactly one correct answer.
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

@dataclass
class MeshModelConfig:
    """Architecture of the autoregressive mesh transformer (arXiv:2406.10163)."""
    d_model: int = 256        # must be divisible by n_head
    n_head: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    # Positional-embedding capacity. Positions restart per segment, so this
    # covers max(len(cond), len(tgt) - 1). None = size it from the loaded
    # dataset's longest segment, logged at startup.
    max_seq_len: Optional[int] = None
    # "coord" -- 9 discretized coordinates per face, an exact inverse, the
    # measurement baseline and the default. "vqvae" -- codes from a trained
    # stage-1 checkpoint, `3 * depth` per face, and the LOD1 condition goes
    # through the same encoder unquantized (see the design spec for why not).
    tokenizer: str = "coord"
    # Stage-1 checkpoint. Required when tokenizer == "vqvae"; the codebook is
    # loaded frozen, so nothing here can move the vocabulary mid-run.
    vqvae_ckpt: Optional[str] = None
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
    opt_name: str = "facebook/opt-350m"
    # Load OPT's weights, or only its architecture. The reference builds it with
    # `from_config` (random) and loads its own checkpoint, so a warm start from
    # the language model is the paper's text rather than its released code.
    opt_pretrained: bool = True

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
    lr_scheduler: str = "none"  # "none", "cosine", "step"
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
class InferenceConfig:
    """Configuration for inference."""
    # Weights to sample from. Resolved by `src.inference._resolve_checkpoint`, which
    # dispatches on the prefix:
    #   local path      "outputs/regnn_diffusion/default/checkpoints/last.ckpt"
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
    batch_size: int = 10
    output_dir: str = "outputs/generated"
    # Attach the generative-eval metrics/samples to an existing WandB run instead of
    # logging locally only. Use when training ran without generative_eval.enabled.
    wandb_run_id: Optional[str] = None

@dataclass
class GenerativeEvalConfig:
    """End-of-pipeline full-generation evaluation (fires on test end)."""
    enabled: bool = True
    num_batches: int = 4          # batches sampled through the full reverse chain
    batch_size: int = 16
    log_n_samples: int = 8        # graphs+geometries persisted locally and to WandB
    save_dir: Optional[str] = None  # None -> <run_save_dir>/generative_eval
    feature_set: str = "full"     # "full" | "welldefined"
    val3dity_path: Optional[str] = None  # None -> shutil.which("val3dity")
    novelty_tol: float = 0.1      # feature-space distance for a "novel" sample
    # Reference buildings drawn per split for the Wasserstein/MMD/novelty arms.
    # The splits hold O(1e5) buildings and kernel_mmd is O(n^2) in the reference
    # set, so the full test split is both minutes of conversion and a ~80GB
    # gram matrix. A few thousand samples estimate these statistics fine.
    ref_max_samples: int = 2000

@dataclass
class Config:
    """Root configuration class."""
    mode: str = "train"  # "train" or "inference"
    # Which set of config objects is built and run:
    #   "diffusion" — data + model, the Levi-graph discrete diffusion (default)
    #   "mini"      — the same objects pointed at data/The Hague/mini; it needs
    #                 no fields of its own, so it gets no dataclass of its own
    #   "mesh"      — mesh_data + mesh_model, the LOD1-conditioned transformer
    #   "mesh_vqvae"— mesh_data + mesh_vqvae, stage-1 learned mesh vocabulary
    # Resolved in `src.train.train`. Deliberately a plain string switch rather
    # than a registry; there are four of these and they are all in one repo.
    config_set: str = "diffusion"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mesh_data: MeshDataConfig = field(default_factory=MeshDataConfig)
    mesh_model: MeshModelConfig = field(default_factory=MeshModelConfig)
    mesh_eval: MeshEvalConfig = field(default_factory=MeshEvalConfig)
    mesh_vqvae: MeshVQVAEConfig = field(default_factory=MeshVQVAEConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    generative_eval: GenerativeEvalConfig = field(default_factory=GenerativeEvalConfig)

    resume_from: Optional[str] = None
