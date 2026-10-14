import logging
from pathlib import Path
from typing import List

from lightning.pytorch.loggers import Logger, TensorBoardLogger, WandbLogger, MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar

from src.utils.config import Config

logger = logging.getLogger(__name__)

def create_loggers(cfg: Config, save_dir: Path) -> List[Logger]:
    """
    Create experiment loggers based on configuration.
    Supports TensorBoard (local), MLflow (local/remote), and Wandb (remote).
    """
    loggers: List[Logger] = []
    
    for name in cfg.logging.loggers:
        if name == "tensorboard":
            loggers.append(TensorBoardLogger(
                save_dir=str(save_dir),
                name=cfg.logging.project_name,
                version=cfg.logging.run_name,
            ))
            logger.info("Initialized TensorBoard logger (local)")
            
        elif name == "wandb":
            loggers.append(WandbLogger(
                project=cfg.logging.project_name,
                entity=cfg.logging.wandb_entity,
                name=cfg.logging.run_name,
                offline=cfg.logging.wandb_offline,
                log_model=cfg.logging.log_model,
                save_dir=str(save_dir),
                group=cfg.logging.experiment_name,
            ))
            logger.info("Initialized WandB logger (remote)")
            
        elif name == "mlflow":
            loggers.append(MLFlowLogger(
                experiment_name=cfg.logging.experiment_name or cfg.logging.project_name,
                run_name=cfg.logging.run_name,
                save_dir=str(save_dir),
            ))
            logger.info("Initialized MLflow logger (local/remote)")
            
        else:
            raise ValueError(f"Unknown logger: {name}")
            
    return loggers


def create_callbacks(cfg: Config, save_dir: Path) -> list:
    """Create Lightning callbacks based on configuration."""
    callbacks = []
    
    # Early Stopping
    es_cfg = cfg.training.early_stopping
    if es_cfg.enabled:
        callbacks.append(EarlyStopping(
            monitor=es_cfg.monitor,
            patience=es_cfg.patience,
            mode=es_cfg.mode,
            verbose=True,
        ))
        
    # Model Checkpoint
    ckpt_cfg = cfg.training.checkpoint
    if ckpt_cfg.enabled:
        checkpoint_dir = save_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor=ckpt_cfg.monitor,
            mode=ckpt_cfg.mode,
            save_top_k=ckpt_cfg.save_top_k,
            filename=ckpt_cfg.filename,
            save_last=True,
            verbose=True,
        ))
        
    # LR Monitor -- only when there is somewhere to write it. Lightning does not
    # ignore a logger-less LearningRateMonitor, it refuses the run:
    #   MisconfigurationException: Cannot use `LearningRateMonitor` callback
    #   with `Trainer` that has no logger.
    # `logging.loggers: []` is a legitimate configuration (the pipeline's smoke
    # mode uses it so a throwaway run creates no wandb run), and create_loggers
    # returns [] for it, which train() passes to the Trainer as logger=False.
    if cfg.logging.loggers:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    
    # Rich Progress Bar
    try:
        callbacks.append(RichProgressBar())
    except Exception:
        pass

    return callbacks
