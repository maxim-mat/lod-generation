import logging
import sys
from pathlib import Path
import torch

from src.utils.config import Config
from src.models.diffusion import CityJSONDiffusionModule
from src.post_process.post_process import save_to_file

logger = logging.getLogger(__name__)

def run_inference(cfg: Config):
    """
    Main inference pipeline.
    Generates buildings from a trained checkpoint and exports them to CityJSON.
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
    
    # Device placement
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    batch_size = cfg.inference.batch_size
    threshold = cfg.inference.edge_threshold
    output_dir = Path(cfg.inference.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting inference:")
    logger.info(f"  Checkpoint: {ckpt_path}")
    logger.info(f"  N_max:      {model.n_max}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Threshold:  {threshold}")
    logger.info(f"  Output dir: {output_dir}")
    
    # Generate
    logger.info(f"Generating {batch_size} building(s)...")
    results = model.generate_cityjson(batch_size=batch_size, threshold=threshold)
    
    logger.info(f"Successfully generated {len(results)} building(s).")
    
    # Save individual buildings
    for i, cj_dict in enumerate(results):
        out_path = output_dir / f"building_{i:04d}.city.json"
        save_to_file(cj_dict, out_path)
        
    # Save combined file if multiple buildings generated
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
        
    logger.info("Inference complete.")


def _offset_boundaries(boundaries, offset):
    """Recursively offset vertex indices in CityJSON boundary structures."""
    if isinstance(boundaries, int):
        return boundaries + offset
    return [_offset_boundaries(item, offset) for item in boundaries]
