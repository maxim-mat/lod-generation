#!/usr/bin/env python3
"""
CityJSON LOD Generation — Main Entry Point

Training:
    python src/main.py
    python src/main.py mode=train model.hidden_dim=128 training.max_epochs=100

Inference:
    python src/main.py mode=inference inference.checkpoint_path=path/to/ckpt

Override any config parameter using OmegaConf dot notation:
    python src/main.py training.batch_size=16 model.T=1000 wandb.enabled=false
"""
import sys
import os
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Ensure src/ is importable regardless of working directory
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from model import CityJSONDiffusionModule
from datamodule import CityJSONDataModule
from post_process import graph_to_cityjson, save_to_file

# Conditionally import PyTorch Lightning
try:
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
        RichProgressBar,
    )
except ImportError:
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
        RichProgressBar,
    )


# ==============================================================================
# Configuration
# ==============================================================================

def load_config():
    """
    Load configuration from YAML file with CLI overrides via OmegaConf.
    
    Resolution order:
    1. configs/default.yaml (base)
    2. CLI overrides (e.g. model.hidden_dim=128)
    """
    # Locate default config relative to the project root
    project_root = _script_dir.parent
    default_cfg_path = project_root / "configs" / "default.yaml"
    
    if default_cfg_path.exists():
        base_cfg = OmegaConf.load(str(default_cfg_path))
    else:
        print(f"Warning: Default config not found at {default_cfg_path}, using empty config.")
        base_cfg = OmegaConf.create()
    
    # Parse CLI overrides (skip script name)
    cli_cfg = OmegaConf.from_cli(sys.argv[1:])
    
    # Merge: CLI overrides take precedence
    cfg = OmegaConf.merge(base_cfg, cli_cfg)
    
    return cfg


# ==============================================================================
# Training
# ==============================================================================

def train(cfg):
    """Run training loop with PyTorch Lightning."""
    # Seed everything for reproducibility
    pl.seed_everything(cfg.seed, workers=True)
    
    # --- DataModule ---
    datamodule = CityJSONDataModule(
        dataset_dir=cfg.data.dataset_dir,
        lods=cfg.data.lods,
        batch_size=cfg.training.batch_size,
        train_val_test_split=tuple(cfg.training.train_val_test_split),
        normalize_coords=cfg.data.normalize_coords,
        num_workers=cfg.data.num_workers,
        n_max=cfg.data.n_max,
    )
    
    # Setup to resolve n_max before model creation
    datamodule.setup()
    resolved_n_max = datamodule.n_max
    print(f"Resolved N_max = {resolved_n_max}")
    
    # --- Model ---
    model = CityJSONDiffusionModule(
        num_node_classes=cfg.model.num_node_classes,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        T=cfg.model.T,
        lr=cfg.training.lr,
        discrete_noise_type=cfg.model.discrete_noise_type,
        n_max=resolved_n_max,
    )
    
    # --- Callbacks ---
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
    
    # Model Checkpointing
    ckpt_cfg = cfg.training.checkpoint
    if ckpt_cfg.enabled:
        callbacks.append(ModelCheckpoint(
            monitor=ckpt_cfg.monitor,
            mode=ckpt_cfg.mode,
            save_top_k=ckpt_cfg.save_top_k,
            filename=ckpt_cfg.filename,
            save_last=True,
            verbose=True,
        ))
    
    # Learning Rate Monitor (logged to wandb)
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    
    # Rich progress bar (optional, silently skip if not installed)
    try:
        callbacks.append(RichProgressBar())
    except Exception:
        pass
    
    # --- Logger ---
    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            tags=list(cfg.wandb.tags) if cfg.wandb.tags else None,
            log_model=cfg.wandb.log_model,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    
    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        deterministic=False,
    )
    
    # --- Train ---
    print("=" * 60)
    print("Starting training")
    print(f"  Model:      rEGNN ({cfg.model.num_layers} layers, hidden_dim={cfg.model.hidden_dim})")
    print(f"  Diffusion:  T={cfg.model.T}, noise={cfg.model.discrete_noise_type}")
    print(f"  Data:       {cfg.data.dataset_dir}, LODs={cfg.data.lods}, N_max={resolved_n_max}")
    print(f"  Training:   batch_size={cfg.training.batch_size}, lr={cfg.training.lr}")
    print(f"  WandB:      {'enabled' if cfg.wandb.enabled else 'disabled'}")
    print("=" * 60)
    
    trainer.fit(model, datamodule=datamodule)
    
    # --- Test (if test split exists) ---
    if datamodule.test_dataset and len(datamodule.test_dataset) > 0:
        trainer.test(model, datamodule=datamodule, ckpt_path="best")
    
    print("Training complete.")
    if ckpt_cfg.enabled:
        print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    
    return model, trainer


# ==============================================================================
# Inference
# ==============================================================================

def inference(cfg):
    """Generate buildings from a trained checkpoint and export to CityJSON."""
    ckpt_path = cfg.inference.checkpoint_path
    if ckpt_path is None:
        print("Error: inference.checkpoint_path must be specified for inference mode.")
        print("Usage: python src/main.py mode=inference inference.checkpoint_path=path/to/ckpt")
        sys.exit(1)
    
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    
    print(f"Loading model from: {ckpt_path}")
    model = CityJSONDiffusionModule.load_from_checkpoint(str(ckpt_path))
    
    # Move to best available device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    batch_size = cfg.inference.batch_size
    threshold = cfg.inference.edge_threshold
    output_dir = Path(cfg.inference.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Starting inference")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  N_max:      {model.n_max}")
    print(f"  Batch size: {batch_size}")
    print(f"  Threshold:  {threshold}")
    print(f"  Output dir: {output_dir}")
    print("=" * 60)
    
    # Generate
    print(f"Generating {batch_size} building(s)...")
    results = model.generate_cityjson(batch_size=batch_size, threshold=threshold)
    
    print(f"Successfully generated {len(results)} building(s).")
    
    # Save each building to a separate CityJSON file
    for i, cj_dict in enumerate(results):
        out_path = output_dir / f"building_{i:04d}.city.json"
        save_to_file(cj_dict, out_path)
    
    # Also save a combined file with all buildings
    if len(results) > 1:
        combined = {
            "type": "CityJSON",
            "version": "1.1",
            "CityObjects": {},
            "vertices": [],
            "metadata": {"datasetLod": "1"},
        }
        vertex_offset = 0
        for i, cj_dict in enumerate(results):
            # Offset vertex indices in boundaries
            for obj_id, city_obj in cj_dict.get("CityObjects", {}).items():
                new_obj = dict(city_obj)
                new_geoms = []
                for geom in city_obj.get("geometry", []):
                    new_geom = dict(geom)
                    new_boundaries = _offset_boundaries(geom["boundaries"], vertex_offset)
                    new_geom["boundaries"] = new_boundaries
                    new_geoms.append(new_geom)
                new_obj["geometry"] = new_geoms
                combined["CityObjects"][f"{obj_id}"] = new_obj
            
            combined["vertices"].extend(cj_dict.get("vertices", []))
            vertex_offset += len(cj_dict.get("vertices", []))
        
        combined_path = output_dir / "all_buildings.city.json"
        save_to_file(combined, combined_path)
    
    print("Inference complete.")


def _offset_boundaries(boundaries, offset):
    """Recursively offset vertex indices in CityJSON boundary structures."""
    if isinstance(boundaries, int):
        return boundaries + offset
    return [_offset_boundaries(item, offset) for item in boundaries]


# ==============================================================================
# Entry Point
# ==============================================================================

def main():
    cfg = load_config()
    
    # Print resolved config
    print("\n--- Resolved Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("------------------------------\n")
    
    mode = cfg.get("mode", "train")
    
    if mode == "train":
        train(cfg)
    elif mode == "inference":
        inference(cfg)
    else:
        print(f"Error: Unknown mode '{mode}'. Must be 'train' or 'inference'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
