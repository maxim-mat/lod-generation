import logging

from src.utils.config import Config

logger = logging.getLogger(__name__)

def train(cfg: Config):
    """
    Main training pipeline.

    `cfg.config_set` selects which set of config objects is built: "mesh" runs
    the LOD1-conditioned mesh transformer, "mesh_vqvae" its stage-1 learned
    vocabulary.
    """
    if cfg.config_set == "mesh":
        from src.train_mesh import train_mesh
        return train_mesh(cfg)
    if cfg.config_set == "mesh_vqvae":
        from src.train_mesh_vqvae import train_mesh_vqvae
        return train_mesh_vqvae(cfg)
    raise ValueError(f"Unknown config_set: {cfg.config_set!r}. Expected "
                     "'mesh' or 'mesh_vqvae'.")
