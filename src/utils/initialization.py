import argparse
import logging
from pathlib import Path
from omegaconf import OmegaConf
from src.utils.config import Config

logger = logging.getLogger(__name__)

def load_config(config_path: Path, overrides: list[str]) -> Config:
    """
    Load configuration from YAML file with CLI overrides.
    
    Args:
        config_path: Path to YAML config file
        overrides: List of dotlist overrides (e.g., ["training.max_epochs=50"])
        
    Returns:
        Structured Config object
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load YAML config
    file_cfg = OmegaConf.load(config_path)
    
    # Create structured config with defaults
    schema = OmegaConf.structured(Config)
    
    # Merge: schema (defaults) <- file config <- CLI overrides
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(schema, file_cfg, cli_cfg)
    else:
        cfg = OmegaConf.merge(schema, file_cfg)
    
    # Convert to structured Config object for type safety
    return OmegaConf.to_object(cfg)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CityJSON LOD Generation with Graph Diffusion Models",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Train the model")
    group.add_argument("--inference", action="store_true", help="Run inference")
    parser.add_argument(
        "--config", "-c",
        required=True,
        type=Path,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in dotlist format (e.g., training.max_epochs=50)",
    )
    return parser.parse_args()
