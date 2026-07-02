from dataclasses import dataclass, field
from typing import Optional, List, Union
from omegaconf import MISSING

@dataclass
class DataConfig:
    """Configuration for dataset and datamodule."""
    dataset_dir: str = MISSING
    lods: List[int] = field(default_factory=lambda: [1, 2])
    normalize_coords: bool = True
    num_workers: int = 4
    n_max: Optional[int] = None

@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    num_node_classes: int = 2
    hidden_dim: int = 64
    num_layers: int = 4
    T: int = 500
    discrete_noise_type: str = "uniform"  # "uniform", "absorbing", "discretized_gaussian"

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
    log_model: bool = False

@dataclass
class InferenceConfig:
    """Configuration for inference."""
    checkpoint_path: Optional[str] = None
    batch_size: int = 10
    edge_threshold: float = 0.5
    output_dir: str = "outputs/generated"

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
    
    resume_from: Optional[str] = None
