"""Entry point for `config_set: mesh_diffusion`.

Deliberately parallel to `src/train_mesh.py` -- same save_dir layout, same
`create_loggers` / `create_callbacks` helpers, same split semantics -- so a
diffusion run and an autoregressive run land side by side in wandb and can be
read against each other without a translation step.
"""
import logging
from functools import partial
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, random_split

from src.dataset.mesh_set_dataset import MeshSetDataset, mesh_set_collate_fn
from src.models.mesh_diffusion_module import MeshDiffusionModule
from src.utils.config import Config, validate_combination
from src.utils.setup_utils import create_callbacks, create_loggers

logger = logging.getLogger(__name__)


class MeshSetDataModule(L.LightningDataModule):
    """Train/val/test split over one `MeshSetDataset`.

    Split is by `training.train_val_test_split` under a generator seeded with
    `cfg.seed`, matching `MeshDataModule`, so the same building lands in the
    same split on both branches and a cross-branch comparison is honest.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size
        self.collate = partial(mesh_set_collate_fn, multiple_of=8)
        self.dataset = None
        self.train_dataset = self.val_dataset = self.test_dataset = None

    def setup(self, stage=None):
        d, md = self.cfg.mesh_diffusion, self.cfg.mesh_data
        self.dataset = MeshSetDataset(
            dataset_dir=md.dataset_dir, lod_in=md.lod_in, lod_out=md.lod_out,
            num_bins=md.num_bins, margin_lo=list(md.margin_lo),
            margin_hi=list(md.margin_hi), max_faces=md.max_faces,
            max_files=md.max_files, order=d.order, state=d.state)
        fracs = list(self.cfg.training.train_val_test_split)
        gen = torch.Generator().manual_seed(self.cfg.seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, fracs, generator=gen)

    def _loader(self, ds, shuffle):
        md = self.cfg.mesh_data
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          collate_fn=self.collate, num_workers=md.num_workers,
                          persistent_workers=md.persistent_workers and md.num_workers > 0)

    def train_dataloader(self):
        return self._loader(self.train_dataset, True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, False)


def train_mesh_diffusion(cfg: Config):
    """Train the non-autoregressive mesh diffusion branch."""
    validate_combination(cfg)
    L.seed_everything(cfg.seed, workers=True)
    if not cfg.mesh_data.dataset_dir:
        raise ValueError(
            "config_set: mesh_diffusion needs mesh_data.dataset_dir. It shares "
            "the autoregressive branch's data block on purpose -- the two must "
            "read the same corpus to be comparable.")

    save_dir = Path(cfg.logging.save_dir, cfg.logging.experiment_name,
                    cfg.logging.run_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    datamodule = MeshSetDataModule(cfg)
    datamodule.setup()
    logger.info("Mesh pairs: train=%d, val=%d, test=%d",
                len(datamodule.train_dataset), len(datamodule.val_dataset),
                len(datamodule.test_dataset))
    logger.info("Arm: order=%s pos_embed=%s loss=%s denoiser=%s process=%s "
                "target=%s scaffold=%s",
                cfg.mesh_diffusion.order, cfg.mesh_diffusion.pos_embed,
                cfg.mesh_diffusion.loss, cfg.mesh_diffusion.denoiser,
                cfg.mesh_diffusion.process, cfg.mesh_diffusion.target,
                cfg.mesh_diffusion.scaffold.enabled)

    model = MeshDiffusionModule(cfg)
    logger.info("Denoiser: %s, %.1fM parameters", cfg.mesh_diffusion.denoiser,
                sum(p.numel() for p in model.denoiser.parameters()) / 1e6)

    callbacks = create_callbacks(cfg, save_dir)
    if cfg.mesh_diffusion.every_n_epochs > 0:
        from src.eval.mesh_set_eval import MeshSetEvalCallback
        callbacks.append(MeshSetEvalCallback(cfg, save_dir, seed=cfg.seed))

    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        callbacks=callbacks,
        logger=create_loggers(cfg, save_dir) or False,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.resume_from)
    trainer.test(model, datamodule=datamodule)
    return model
