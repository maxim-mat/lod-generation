"""Stage-1 training for the learned mesh vocabulary (config_set: mesh_vqvae).

Reached from `src.train.train`. Kept separate from `train_mesh` for the same
reason that one is separate from the diffusion path: it resolves a different
model from a different datamodule, and sharing an entry point would mean a
function that is mostly branches.

The output is a checkpoint whose `MeshVQVAE` stage 2 loads frozen. Success is
defined in docs/superpowers/specs/2026-08-08-mesh-vqvae-design.md: reconstruction
chamfer <= 0.15 m against the coordinate tokenizer's measured 0.107 m floor,
with a codebook that is actually being used.
"""
import logging
from pathlib import Path

import lightning as L
from omegaconf import OmegaConf

from src.dataset.mesh_vqvae_datamodule import MeshVQVAEDataModule
from src.models.mesh_vqvae import MeshVQVAEModule
from src.utils.config import Config
from src.utils.setup_utils import create_callbacks, create_loggers

logger = logging.getLogger(__name__)


def train_mesh_vqvae(cfg: Config):
    """Train `MeshVQVAEModule` to reconstruct LOD1 and LOD2 faces."""
    L.seed_everything(cfg.seed, workers=True)

    if not cfg.mesh_data.dataset_dir:
        raise ValueError("config_set='mesh_vqvae' requires mesh_data.dataset_dir.")

    save_dir = Path(cfg.logging.save_dir, cfg.logging.experiment_name, cfg.logging.run_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing mesh VQ-VAE datamodule...")
    datamodule = MeshVQVAEDataModule(
        batch_size=cfg.training.batch_size,
        dataset_dir=cfg.mesh_data.dataset_dir,
        lod_in=cfg.mesh_data.lod_in,
        lod_out=cfg.mesh_data.lod_out,
        num_bins=cfg.mesh_data.num_bins,
        # OmegaConf ListConfig -> plain list, so numpy can broadcast it
        margin_lo=list(cfg.mesh_data.margin_lo),
        margin_hi=list(cfg.mesh_data.margin_hi),
        max_faces=cfg.mesh_data.max_faces,
        max_files=cfg.mesh_data.max_files,
        train_val_test_split=tuple(cfg.training.train_val_test_split),
        num_workers=cfg.mesh_data.num_workers,
        persistent_workers=cfg.mesh_data.persistent_workers,
        seed=cfg.seed,
    )
    datamodule.setup()
    inner = datamodule.inner
    logger.info("Mesh pairs: train=%d, val=%d, test=%d -> %d training meshes "
                "(both LODs)", len(inner.train_dataset), len(inner.val_dataset),
                len(inner.test_dataset), 2 * len(inner.train_dataset))

    # Sized from the data unless pinned, so the longest building cannot index
    # past the per-face positional embedding mid-run.
    max_faces = cfg.mesh_vqvae.max_faces or datamodule.max_faces_seen
    logger.info("max_faces = %d (corpus longest mesh: %d faces)",
                max_faces, datamodule.max_faces_seen)

    model = MeshVQVAEModule(
        num_bins=cfg.mesh_data.num_bins,
        codebook_size=cfg.mesh_vqvae.codebook_size,
        depth=cfg.mesh_vqvae.depth,
        d_model=cfg.mesh_vqvae.d_model,
        n_head=cfg.mesh_vqvae.n_head,
        num_layers=cfg.mesh_vqvae.num_layers,
        dropout=cfg.mesh_vqvae.dropout,
        max_faces=max_faces,
        commitment=cfg.mesh_vqvae.commitment,
        lr=cfg.training.lr,
        lr_scheduler=cfg.training.lr_scheduler,
        lr_decay_steps=cfg.training.lr_decay_steps,
        lr_decay_rate=cfg.training.lr_decay_rate,
    )
    logger.info("Codebook: %d entries x depth %d -> %d tokens per face "
                "(coordinate tokenizer uses 9)", cfg.mesh_vqvae.codebook_size,
                cfg.mesh_vqvae.depth, cfg.mesh_vqvae.depth)

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
        hparams["mesh_vqvae"]["max_faces"] = max_faces   # log the resolved value
        for lg in trainer.loggers:
            lg.log_hyperparams(hparams)

    logger.info("Starting VQ-VAE training...")
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.resume_from)

    if inner.test_dataset and len(inner.test_dataset) > 0:
        best = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else None
        trainer.test(model, datamodule=datamodule, ckpt_path=best or None)

    logger.info("VQ-VAE training complete.")
