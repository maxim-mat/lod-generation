"""Training pipeline for the LOD1-conditioned mesh transformer (config_set: mesh).

Reached from `src.train.train` when `cfg.config_set == "mesh"`. Kept separate
from the diffusion path because that one resolves marginals, a coordinate scale
and a z-shift from the datamodule, none of which exist here.
"""
import logging
from pathlib import Path

import lightning as L
from omegaconf import OmegaConf

from src.dataset.mesh_datamodule import MeshDataModule
from src.models.mesh_transformer import MeshTransformerModule
from src.utils.config import Config
from src.utils.setup_utils import create_callbacks, create_loggers

logger = logging.getLogger(__name__)


def train_mesh(cfg: Config):
    """Train `MeshTransformerModule` on (LOD1, LOD2) mesh-token pairs."""
    L.seed_everything(cfg.seed, workers=True)

    if not cfg.mesh_data.dataset_dir:
        raise ValueError("config_set='mesh' requires mesh_data.dataset_dir to be set.")

    save_dir = Path(cfg.logging.save_dir, cfg.logging.experiment_name, cfg.logging.run_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing mesh datamodule...")
    datamodule = MeshDataModule(
        dataset_dir=cfg.mesh_data.dataset_dir,
        lod_in=cfg.mesh_data.lod_in,
        lod_out=cfg.mesh_data.lod_out,
        num_bins=cfg.mesh_data.num_bins,
        max_faces=cfg.mesh_data.max_faces,
        max_files=cfg.mesh_data.max_files,
        batch_size=cfg.training.batch_size,
        train_val_test_split=tuple(cfg.training.train_val_test_split),
        num_workers=cfg.mesh_data.num_workers,
        persistent_workers=cfg.mesh_data.persistent_workers,
        seed=cfg.seed,
    )
    datamodule.setup()
    logger.info("Mesh pairs: train=%d, val=%d, test=%d",
                len(datamodule.train_dataset), len(datamodule.val_dataset),
                len(datamodule.test_dataset))

    # Sized from the data unless pinned, so a longer building cannot silently
    # index past the positional embedding mid-run.
    max_seq_len = cfg.mesh_model.max_seq_len or datamodule.max_seq_len
    logger.info("max_seq_len = %d (dataset longest: %d)", max_seq_len, datamodule.max_seq_len)

    model = MeshTransformerModule(
        num_bins=cfg.mesh_data.num_bins,
        d_model=cfg.mesh_model.d_model,
        n_head=cfg.mesh_model.n_head,
        num_layers=cfg.mesh_model.num_layers,
        dropout=cfg.mesh_model.dropout,
        max_seq_len=max_seq_len,
        lr=cfg.training.lr,
        lr_scheduler=cfg.training.lr_scheduler,
        lr_decay_steps=cfg.training.lr_decay_steps,
        lr_decay_rate=cfg.training.lr_decay_rate,
    )

    exp_loggers = create_loggers(cfg, save_dir)
    callbacks = create_callbacks(cfg, save_dir)

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

    if exp_loggers:
        hparams = OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)
        hparams["mesh_model"]["max_seq_len"] = max_seq_len   # log the resolved value
        for lg in trainer.loggers:
            lg.log_hyperparams(hparams)

    logger.info("Starting training...")
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.resume_from)

    if datamodule.test_dataset and len(datamodule.test_dataset) > 0:
        best = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else None
        trainer.test(model, datamodule=datamodule, ckpt_path=best or None)

    logger.info("Training complete.")
