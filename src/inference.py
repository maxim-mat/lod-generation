import logging
import sys
from dataclasses import replace
from pathlib import Path

import torch

from src.eval.callback import run_generative_eval
from src.models.diffusion import CityJSONDiffusionModule
from src.post_process.post_process import save_to_file
from src.utils.config import Config

logger = logging.getLogger(__name__)


def run_inference(cfg: Config):
    """
    Main inference pipeline.

    Samples buildings from a trained checkpoint through `run_generative_eval`, which
    converts every graph with `graph_to_cityjson`, writes one .city.json/.obj/.npz per
    generated graph and scores the batch with the src/eval metrics. Optionally attaches
    the results to an existing WandB run (for runs trained without the eval callback).
    """
    ckpt_path = cfg.inference.checkpoint_path
    if ckpt_path is None:
        logger.error("inference.checkpoint_path must be specified for inference mode.")
        sys.exit(1)

    ckpt_path_obj = Path(ckpt_path)
    if not ckpt_path_obj.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    logger.info(f"Loading model from: {ckpt_path}")
    model = CityJSONDiffusionModule.load_from_checkpoint(str(ckpt_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    output_dir = Path(cfg.inference.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # One batch of inference.batch_size, every survivor persisted (the callback caps
    # at log_n_samples; inference wants the whole batch on disk).
    eval_cfg = replace(
        cfg.generative_eval,
        enabled=True,
        num_batches=1,
        batch_size=cfg.inference.batch_size,
        save_dir=str(output_dir),
        log_n_samples=cfg.inference.batch_size,
    )

    logger.info("Starting inference:")
    logger.info(f"  Checkpoint: {ckpt_path}")
    logger.info(f"  N_max:      {model.n_max}")
    logger.info(f"  Batch size: {eval_cfg.batch_size}")
    logger.info(f"  Output dir: {output_dir}")

    datamodule = _load_datamodule(cfg)
    loggers = _resume_wandb_logger(cfg, output_dir)

    try:
        metrics = run_generative_eval(model, datamodule, eval_cfg, loggers, output_dir)
    finally:
        if loggers:
            import wandb
            wandb.finish()

    for key in sorted(metrics):
        logger.info("  %s = %s", key, metrics[key])

    _write_combined(output_dir)
    logger.info("Inference complete.")
    return metrics


def _load_datamodule(cfg: Config):
    """Datamodule for the reference-distribution metrics, or None if unavailable.

    Wasserstein/MMD/novelty need the train+test splits; validity, rejection rate and
    face coherence do not. A missing dataset degrades the run instead of killing it.
    """
    from src.utils.setup_utils import create_datamodule
    try:
        datamodule = create_datamodule(cfg)
        datamodule.setup()
        return datamodule
    except Exception as exc:
        logger.warning(
            "Could not load dataset from %s (%s); skipping distribution/novelty metrics.",
            cfg.data.dataset_dir, exc,
        )
        return None


def _resume_wandb_logger(cfg: Config, save_dir: Path):
    """[WandbLogger] bound to an existing run when inference.wandb_run_id is set, else [].

    `run_generative_eval` picks a WandbLogger out of this list and logs the scalars,
    sample table and artifact to it, so results from a training run that had the
    generative-eval callback disabled can still be attached after the fact.
    """
    run_id = cfg.inference.wandb_run_id
    if not run_id:
        return []
    from lightning.pytorch.loggers import WandbLogger
    logger.info("Attaching results to existing WandB run: %s", run_id)
    return [WandbLogger(
        id=run_id,
        resume="must",  # fail loudly rather than silently opening a fresh run
        project=cfg.logging.project_name,
        entity=cfg.logging.wandb_entity,
        save_dir=str(save_dir),
    )]


def _write_combined(output_dir: Path):
    """Merge the per-building CityJSON files into one all_buildings.city.json."""
    import json

    paths = sorted(output_dir.glob("gen_*.city.json"))
    if len(paths) < 2:
        return

    combined = {
        "type": "CityJSON",
        "version": "1.1",
        "CityObjects": {},
        "vertices": [],
        "metadata": {"datasetLod": "1"},
    }
    vertex_offset = 0
    for path in paths:
        cj_dict = json.loads(path.read_text(encoding="utf-8"))
        for obj_id, city_obj in cj_dict.get("CityObjects", {}).items():
            new_obj = dict(city_obj)
            new_obj["geometry"] = [
                {**geom, "boundaries": _offset_boundaries(geom["boundaries"], vertex_offset)}
                for geom in city_obj.get("geometry", [])
            ]
            combined["CityObjects"][obj_id] = new_obj

        combined["vertices"].extend(cj_dict.get("vertices", []))
        vertex_offset += len(cj_dict.get("vertices", []))

    save_to_file(combined, output_dir / "all_buildings.city.json")


def _offset_boundaries(boundaries, offset):
    """Recursively offset vertex indices in CityJSON boundary structures."""
    if isinstance(boundaries, int):
        return boundaries + offset
    return [_offset_boundaries(item, offset) for item in boundaries]
