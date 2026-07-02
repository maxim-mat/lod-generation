import logging
from pathlib import Path
from datetime import datetime

import lightning as L

from src.utils.config import Config
from src.utils.setup_utils import (
    create_datamodule,
    create_model,
    create_loggers,
    create_callbacks
)

logger = logging.getLogger(__name__)

def train(cfg: Config):
    """
    Main training pipeline.
    Sets up Datamodule, Model, Logger, Callbacks and runs the Lightning Trainer.
    """
    # Seed everything for reproducibility
    L.seed_everything(cfg.seed, workers=True)
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup save directory
    save_dir = Path(cfg.logging.save_dir, cfg.logging.experiment_name, cfg.logging.run_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Initializing Datamodule...")
    datamodule = create_datamodule(cfg)
    datamodule.setup()
    
    resolved_n_max = datamodule.n_max
    logger.info(f"Resolved N_max = {resolved_n_max}")
    
    logger.info(
        "Dataset loaded: train=%d, val=%d, test=%d",
        len(datamodule.train_dataset) if datamodule.train_dataset else 0,
        len(datamodule.val_dataset) if datamodule.val_dataset else 0,
        len(datamodule.test_dataset) if datamodule.test_dataset else 0,
    )
    
    logger.info("Creating Model...")
    model = create_model(cfg, resolved_n_max)
    
    logger.info("Creating loggers and callbacks...")
    exp_loggers = create_loggers(cfg, save_dir)
    callbacks = create_callbacks(cfg, save_dir)
    
    # Configure Lightning Trainer
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        callbacks=callbacks,
        logger=exp_loggers if exp_loggers else False,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        deterministic=False,
        default_root_dir=str(save_dir),
    )
    
    # Log hyperparameters to loggers
    if exp_loggers:
        from omegaconf import OmegaConf
        hparams = OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)
        for lg in trainer.loggers:
            lg.log_hyperparams(hparams)
            
    # Run training
    logger.info("Starting training...")
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.resume_from)
    
    # Test best model if test split exists
    if datamodule.test_dataset and len(datamodule.test_dataset) > 0:
        best_model_path = trainer.checkpoint_callback.best_model_path
        if best_model_path:
            logger.info(f"Running test evaluation using best checkpoint: {best_model_path}")
            trainer.test(model, datamodule=datamodule, ckpt_path=best_model_path)
        else:
            logger.warning("No best checkpoint found! Running test evaluation with current model weights.")
            trainer.test(model, datamodule=datamodule)
            
    logger.info("Training complete.")
    if cfg.training.checkpoint.enabled and trainer.checkpoint_callback.best_model_path:
        logger.info(f"Best checkpoint saved to: {trainer.checkpoint_callback.best_model_path}")
