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


def _load_vqvae(cfg: Config):
    """The frozen stage-1 tokenizer, or None on the coordinate path.

    Loaded from the checkpoint's own hyperparameters rather than from
    `cfg.mesh_vqvae`, so a stage-2 config cannot silently describe a different
    codebook than the one whose weights it is loading.
    """
    if cfg.mesh_model.tokenizer == "coord":
        return None
    if cfg.mesh_model.tokenizer != "vqvae":
        raise ValueError(f"Unknown mesh_model.tokenizer: {cfg.mesh_model.tokenizer!r}. "
                         "Expected 'coord' or 'vqvae'.")
    if not cfg.mesh_model.vqvae_ckpt:
        raise ValueError("mesh_model.tokenizer='vqvae' requires "
                         "mesh_model.vqvae_ckpt (a stage-1 checkpoint).")

    from src.models.mesh_vqvae import MeshVQVAEModule

    module = MeshVQVAEModule.load_from_checkpoint(cfg.mesh_model.vqvae_ckpt,
                                                  map_location="cpu")
    vqvae = module.network
    if vqvae.num_bins != cfg.mesh_data.num_bins:
        raise ValueError(
            f"VQ-VAE was trained at num_bins={vqvae.num_bins} but mesh_data "
            f"says {cfg.mesh_data.num_bins}; the codebook would be meaningless.")
    logger.info("Loaded frozen VQ-VAE from %s: %d codes x depth %d -> %d tokens "
                "per face", cfg.mesh_model.vqvae_ckpt, vqvae.codebook_size,
                vqvae.depth, vqvae.depth)
    return vqvae


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
        # OmegaConf ListConfig -> plain list, so numpy can broadcast it
        margin_lo=list(cfg.mesh_data.margin_lo),
        margin_hi=list(cfg.mesh_data.margin_hi),
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

    vqvae = _load_vqvae(cfg)

    # Sized from the data unless pinned, so a longer building cannot silently
    # index past the positional embedding mid-run.
    max_seq_len = cfg.mesh_model.max_seq_len or datamodule.max_seq_len
    if vqvae is not None and cfg.mesh_model.max_seq_len is None:
        # Codes, not coordinates: `depth` per face instead of 9, plus BOS. The
        # dataset's figure counts coordinate tokens and would oversize the
        # positional embedding by 3x.
        faces = (datamodule.max_seq_len + 8) // 9
        max_seq_len = faces * vqvae.depth + 1
    logger.info("max_seq_len = %d (dataset longest segment: %d coordinate tokens)",
                max_seq_len, datamodule.max_seq_len)

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
        vqvae=vqvae,
    )

    exp_loggers = create_loggers(cfg, save_dir)
    callbacks = create_callbacks(cfg, save_dir)

    # Mesh-only, so it is built here rather than inside the shared helper.
    if cfg.mesh_eval.enabled:
        from src.eval.mesh_eval import MeshEvalCallback
        callbacks.append(MeshEvalCallback(cfg.mesh_eval, save_dir,
                                          max_faces=cfg.mesh_data.max_faces or 200,
                                          seed=cfg.seed))

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
