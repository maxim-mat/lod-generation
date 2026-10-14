#!/usr/bin/env python3
"""
CityJSON LOD Generation — Main Entry Point

Training:
    python main.py --train --config configs/mesh-v5-full-opt-coord.yaml
    python main.py --train --config configs/mesh-vqvae.yaml --config-set mesh_vqvae

Inference (LOD1 CityJSON in, LOD2 CityJSON out; pass the run's own training config):
    python main.py --inference --config configs/mesh-v5-full-opt-coord.yaml \
        inference.checkpoint_path=wandb://maxim-lod/lod-generation/model-<run_id>:best \
        inference.input_path="data/The Hague/cleaner/LOD1_synth/Source A"

Override any config parameter using OmegaConf dot notation:
    python main.py --train --config configs/mesh-v5-full-opt-coord.yaml training.batch_size=16 logging.loggers="['tensorboard', 'wandb']"
"""
import logging
import sys
from omegaconf import OmegaConf

from src.train import train
from src.inference import run_inference
from src.utils.initialization import load_config, parse_args

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    args = parse_args()
    
    logger.info("Loading config from: %s", args.config)
    cfg = load_config(args.config, args.overrides)
    if args.config_set:
        cfg.config_set = args.config_set

    logger.info("Config set: %s", cfg.config_set)
    logger.info("Config:\n%s", OmegaConf.to_yaml(OmegaConf.structured(cfg)))
    
    if args.train:
        # Override mode in config for consistency
        cfg.mode = "train"
        train(cfg)
    elif args.inference:
        cfg.mode = "inference"
        run_inference(cfg)
    else:
        raise ValueError("Invalid mode. Use --train or --inference.")


if __name__ == "__main__":
    main()
