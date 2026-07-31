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
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    generative_eval: GenerativeEvalConfig = field(default_factory=GenerativeEvalConfig)

    resume_from: Optional[str] = None
