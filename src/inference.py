"""LOD1 -> LOD2 inference with a trained mesh transformer (`main.py --inference`).

Reads LOD1 CityJSON (one file or a directory of tiles), generates one LOD2 per
CityObject, and writes it back as CityJSON in the input's own coordinates:

    <output_dir>/<timestamp>/<object_id>.city.json   one per generated building
    <output_dir>/<timestamp>/<object_id>.obj         the same triangle mesh
    <output_dir>/<timestamp>/all_buildings.city.json the above merged
    <output_dir>/<timestamp>/manifest.json           what produced them

Run it with the checkpoint's own training config: normalization (bins,
margins), tokenization and the beam settings it was selected under all come
from there, so the LOD1 frame cannot drift from the one it was trained in.
"""
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import lightning as L
import torch
from omegaconf import OmegaConf

from src.dataset.mesh_dataset import (amt_tokenize, normalize_to_unit_box,
                                      parse_cityjson_file_to_meshes, tokenize, write_obj)
from src.eval.posthoc_eval import (_git_state, _versions, _write_json, check_tokenization,
                                   generation_budget, load_model, resolve_device)
from src.post_process.post_process import mesh_to_cityjson, save_to_file
from src.utils.config import Config

logger = logging.getLogger(__name__)

COMBINED = "all_buildings.city.json"


def run_inference(cfg: Config) -> Path:
    """Generate LOD2 for every building in `cfg.inference.input_path`.

    Returns:
        Path: the run's output directory, holding the CityJSON/OBJ files and
        `manifest.json` (checkpoint, decoding, git state, per-object status).

    Raises:
        ValueError: the config cannot describe a mesh-transformer inference
            run, or the checkpoint's tokenization differs from the config's.
    """
    _check_config(cfg)
    L.seed_everything(cfg.seed, workers=True)

    ckpt_path = _resolve_checkpoint(cfg)
    device = resolve_device("auto")
    logger.info("Loading model from: %s", ckpt_path)
    model = load_model(cfg, ckpt_path, device)
    check_tokenization(dict(model.hparams), cfg)

    meshes = load_lod1(cfg.inference.input_path)
    out_dir = Path(cfg.inference.output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.info("Generating LOD2 for %d buildings from %s -> %s",
                len(meshes), cfg.inference.input_path, out_dir)

    status = infer_buildings(model, cfg, meshes, out_dir)
    _write_combined(out_dir)

    git, diff = _git_state()
    _write_json(out_dir / "manifest.json", {
        "checkpoint": ckpt_path,
        "input_path": cfg.inference.input_path,
        "device": str(device),
        "decoding": {"temperature": 0.0, "beam_size": cfg.mesh_eval.beam_size,
                     "length_penalty": cfg.mesh_eval.length_penalty,
                     "max_new_tokens": generation_budget(cfg)},
        "git": git,
        "versions": _versions(),
        "argv": sys.argv,
        "config": OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True),
        "status": status,
    })
    if diff:
        (out_dir / "git_diff.patch").write_text(diff, encoding="utf-8")

    counts = {s: sum(v == s for v in status.values()) for s in ("ok", "failed", "skipped")}
    logger.info("Inference complete: %s -> %s", counts, out_dir)
    return out_dir


def _check_config(cfg: Config):
    """Everything that can be rejected before a checkpoint or tile is read."""
    if cfg.config_set != "mesh":
        raise ValueError(f"Inference runs a mesh-transformer checkpoint; config_set "
                         f"must be 'mesh', got {cfg.config_set!r}.")
    if not cfg.inference.checkpoint_path:
        raise ValueError("inference.checkpoint_path must be set.")
    if not cfg.inference.input_path:
        raise ValueError("inference.input_path must be set to a LOD1 CityJSON file "
                         "or a directory of them.")
    if cfg.mesh_model.image_encoder:
        raise ValueError("Image-conditioned checkpoints are not supported by inference yet: "
                         "it would need an orthophoto crop for every input building.")
    if cfg.mesh_model.tokenizer != "coord":
        # The VQ-VAE path encodes the condition inside `_prepare`, which also
        # tokenizes a ground-truth target that inference does not have.
        raise ValueError(f"Inference supports mesh_model.tokenizer='coord' only, got "
                         f"{cfg.mesh_model.tokenizer!r} (vqvae checkpoints are not "
                         "supported).")


def load_lod1(input_path):
    """``{object_id: (verts [V, 3] metres, faces [F, 3])}`` from a file or directory.

    A directory is searched recursively for ``*.json`` / ``*.city.json``. An
    unreadable tile is logged and skipped rather than ending the run.

    Raises:
        FileNotFoundError: ``input_path`` does not exist.
    """
    path = Path(input_path)
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(p for p in path.rglob("*")
                       if p.is_file() and p.name.lower().endswith(".json"))
    else:
        raise FileNotFoundError(f"LOD1 input not found: {path}")

    meshes = {}
    for f in files:
        try:
            tile = parse_cityjson_file_to_meshes(f)
        except (OSError, ValueError) as exc:
            logger.warning("skipping %s: %s", f.name, exc)
            continue
        repeated = meshes.keys() & tile.keys()
        if repeated:
            logger.warning("%s repeats %d object id(s) already read; the later "
                           "geometry wins", f.name, len(repeated))
        meshes.update(tile)
    if not meshes:
        logger.warning("No buildings with geometry under %s", path)
    return meshes


def condition_tokens(verts, faces, mesh_data, tokenization):
    """One LOD1 mesh as condition tokens, in the frame `MeshDataset` trains in.

    Returns:
        tuple: (tokens [L] int64, center [3], scale [3]); a decoded mesh goes
        back to metres as ``verts * scale + center``.
    """
    verts_n, center, scale = normalize_to_unit_box(
        verts, margin_lo=list(mesh_data.margin_lo), margin_hi=list(mesh_data.margin_hi))
    tok = amt_tokenize if tokenization == "amt" else tokenize
    return tok(verts_n, faces, mesh_data.num_bins), center, scale


def infer_buildings(model, cfg: Config, meshes, out_dir):
    """Generate, decode and write back each building.

    Returns:
        dict: object_id -> "ok" (written), "failed" (no usable mesh came out),
        or "skipped" (its LOD1 does not fit the model's position table).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = generation_budget(cfg)
    device = next(model.parameters()).device

    status, queue = {}, []
    for obj_id, (verts, faces) in meshes.items():
        tokens, center, scale = condition_tokens(verts, faces, cfg.mesh_data,
                                                 model.tokenization)
        if not _fits(model.network, len(tokens)):
            logger.warning("%s: %d condition tokens do not fit the model's %d "
                           "positions; skipped", obj_id, len(tokens),
                           model.network.max_seq_len)
            status[obj_id] = "skipped"
            continue
        queue.append((obj_id, tokens, center, scale))

    batch_size = cfg.inference.batch_size
    for start in range(0, len(queue), batch_size):
        chunk = queue[start:start + batch_size]
        width = max(len(tokens) for _, tokens, _, _ in chunk)
        cond = torch.full((len(chunk), width), model.pad, dtype=torch.long)
        cond_pad = torch.ones((len(chunk), width), dtype=torch.bool)
        for k, (_, tokens, _, _) in enumerate(chunk):
            cond[k, :len(tokens)] = torch.from_numpy(tokens)
            cond_pad[k, :len(tokens)] = False

        with torch.no_grad():
            out = model.generate(cond.to(device), cond_pad.to(device),
                                 max_new_tokens=budget, temperature=0.0,
                                 beam_size=cfg.mesh_eval.beam_size,
                                 length_penalty=cfg.mesh_eval.length_penalty)

        for k, (obj_id, _, center, scale) in enumerate(chunk):
            status[obj_id] = _write_building(model, out[k].cpu(), obj_id,
                                             center, scale, out_dir)
        logger.info("Generated %d/%d", min(start + batch_size, len(queue)), len(queue))
    return status


def _fits(network, n_cond):
    """Whether a condition of `n_cond` tokens fits the backbone's positions."""
    if not network.bounded_positions:
        return True
    return n_cond <= network.max_seq_len and network.generation_budget(n_cond) >= 1


def _write_building(model, tokens, obj_id, center, scale, out_dir):
    """Decode one generation to metres and write it; returns its status."""
    verts_n, faces = model.decode_tokens(tokens)
    if len(faces) == 0:
        logger.warning("%s: generation decoded to no faces", obj_id)
        return "failed"
    verts = verts_n * scale + center
    cj = mesh_to_cityjson(verts, faces, building_id=obj_id)
    if not cj:
        logger.warning("%s: CityJSON write-back failed on %d faces", obj_id, len(faces))
        return "failed"
    name = re.sub(r'[<>:"/\\|?*]', "_", str(obj_id))
    save_to_file(cj, out_dir / f"{name}.city.json")
    write_obj(out_dir / f"{name}.obj", verts, faces)
    return "ok"


def _resolve_checkpoint(cfg: Config) -> str:
    """inference.checkpoint_path -> something `load_from_checkpoint` can open.

    Three forms:
      - local path              outputs/.../last.ckpt
      - WandB model artifact    wandb://[entity/]project/model-<run_id>:best
      - any fsspec URL          https://..., s3://..., gs://...  (passed straight
                                through; Lightning opens these itself)
    """
    ref = cfg.inference.checkpoint_path

    if ref.startswith("wandb://"):
        return _download_wandb_checkpoint(ref[len("wandb://"):], cfg)

    if "://" in ref:
        return ref

    if not Path(ref).exists():
        logger.error(f"Checkpoint not found: {ref}")
        sys.exit(1)
    return ref


def _download_wandb_checkpoint(artifact_ref: str, cfg: Config) -> str:
    """Pull a WandB model artifact and return the local .ckpt path.

    `artifact_ref` is `[entity/]project/name:alias`; a bare `name:alias` is qualified
    from cfg.logging. Lightning's `log_model` writes one model.ckpt per artifact.
    """
    import wandb

    if "/" not in artifact_ref:  # bare name:alias -> qualify from the logging config
        prefix = f"{cfg.logging.wandb_entity}/" if cfg.logging.wandb_entity else ""
        artifact_ref = f"{prefix}{cfg.logging.project_name}/{artifact_ref}"

    cache_dir = Path(cfg.logging.save_dir, "wandb_artifacts", artifact_ref.replace("/", "_").replace(":", "_"))
    logger.info("Downloading WandB artifact %s -> %s", artifact_ref, cache_dir)
    try:
        artifact = wandb.Api().artifact(artifact_ref)
        root = Path(artifact.download(root=str(cache_dir)))
    except Exception as exc:
        logger.error("Could not download WandB artifact %s: %s", artifact_ref, exc)
        sys.exit(1)

    ckpts = sorted(root.rglob("*.ckpt"))
    if not ckpts:
        logger.error("No .ckpt file inside WandB artifact %s (got: %s)",
                     artifact_ref, [p.name for p in root.rglob("*")])
        sys.exit(1)
    if len(ckpts) > 1:
        logger.warning("Artifact holds %d checkpoints; using %s", len(ckpts), ckpts[0].name)
    return str(ckpts[0])


def _write_combined(output_dir: Path):
    """Merge the per-building CityJSON files into one all_buildings.city.json."""
    import json

    output_dir = Path(output_dir)
    paths = sorted(p for p in output_dir.glob("*.city.json") if p.name != COMBINED)
    if len(paths) < 2:
        return

    combined = {
        "type": "CityJSON",
        "version": "1.1",
        "CityObjects": {},
        "vertices": [],
        "metadata": {"datasetLod": "2"},
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

    save_to_file(combined, output_dir / COMBINED)


def _offset_boundaries(boundaries, offset):
    """Recursively offset vertex indices in CityJSON boundary structures."""
    if isinstance(boundaries, int):
        return boundaries + offset
    return [_offset_boundaries(item, offset) for item in boundaries]
