#!/usr/bin/env python3
"""
CityJSON LOD Generation — Main Entry Point

Training:
    python main.py --train --config configs/train.yaml
    python main.py --train --config configs/train.yaml model.hidden_dim=128 training.max_epochs=100

Inference:
    python main.py --inference --config configs/inference.yaml
    python main.py --inference --config configs/train.yaml inference.checkpoint_path=outputs/regnn_diffusion/default/checkpoints/last.ckpt

Override any config parameter using OmegaConf dot notation:
    python main.py --train --config configs/train.yaml training.batch_size=16 model.T=1000 logging.loggers="['tensorboard', 'wandb']"
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
    
    logger.info("Config:\n%s", OmegaConf.to_yaml(OmegaConf.structured(cfg)))
    
    if args.train:
        # Override mode in config for consistency
        cfg.mode = "train"
        train(cfg)
    elif args.inference:
        # Override mode in config for consistency
        cfg.mode = "inference"
        run_inference(cfg)
    else:
        raise ValueError("Invalid mode. Use --train or --inference.")


if __name__ == "__main__":
    main()
