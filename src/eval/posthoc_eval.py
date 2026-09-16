"""Post-hoc evaluation of a mesh-transformer checkpoint across decoding strategies.

Usage:
    python -m src.eval.posthoc_eval --config configs/posthoc-eval.yaml [dot.overrides]

`src/eval/mesh_eval.py` scores greedy or beam decoding on a small subsample
during training. This scores a finished checkpoint on a whole split under a grid
of strategies, one CSV row per object, and writes everything needed to
reproduce any row: resolved configs, checkpoint hash, git state, versions,
per-object seeds.
"""
import csv
import hashlib
import itertools
import json
import logging
import platform
import subprocess
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.analysis.exact_volume import exact_pair_volumes
from src.dataset.mesh_dataset import (amt_detokenize, detokenize, mesh_collate_fn,
                                      specials)
from src.eval.mesh_eval import _resolve, _to_metres, eval_indices
from src.eval.mesh_metrics import (chamfer_distance, hausdorff_max, hausdorff_p95,
                                   is_watertight_mesh, roof_submesh,
                                   surface_distances)
from src.eval.sinkhorn import mesh_sinkhorn
from src.utils.config import PosthocEvalConfig, SinkhornConfig

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
# Surface-sampling seed for every metric, as in `run_mesh_eval`. Recorded in run.json.
SURFACE_SEED = 0

METRICS = ("chamfer", "hausdorff", "vol_symdiff", "sinkhorn", "roof_chamfer")
REFERENCES = ("lod1", "lod2")
SPLITS = ("train", "val", "test")

# Per-reference numeric columns that summaries average. Order is CSV order.
METRIC_COLUMNS = ("chamfer_m", "hausdorff_p95_m", "hausdorff_max_m", "symdiff_m3",
                  "rel_symdiff", "sinkhorn_m", "roof_chamfer_m")

# Which StrategyConfig fields each kind reads, in run-name order.
STRATEGY_PARAMS = {
    "greedy": (),
    "beam": ("beam_size", "length_penalty"),
    "top_p": ("top_p", "temperature"),
    "min_p": ("min_p", "temperature"),
}
_ALL_PARAMS = ("beam_size", "length_penalty", "top_p", "min_p", "temperature")
_PARAM_DEFAULTS = {"length_penalty": [1.0]}
_ABBREV = {"beam_size": "k", "length_penalty": "lp", "top_p": "p", "min_p": "p",
           "temperature": "t"}


@dataclass
class DecodeRun:
    """One point of the strategy grid: a name for the output folder plus params."""
    name: str
    kind: str
    params: dict

    def generate_kwargs(self):
        """Keyword arguments for `MeshTransformerModule.generate`, seeds excluded."""
        p = self.params
        if self.kind == "greedy":
            return {"temperature": 0.0}
        if self.kind == "beam":
            return {"temperature": 0.0, "beam_size": int(p["beam_size"]),
                    "length_penalty": float(p["length_penalty"])}
        return {"temperature": float(p["temperature"]), self.kind: float(p[self.kind])}


def expand_strategies(strategies):
    """Cartesian product of each strategy's list-valued params.

    Raises:
        ValueError: unknown kind, a param that does not belong to the kind, a
            missing or empty required list, a non-positive temperature, or two
            runs with the same name.
    """
    runs = []
    for s in strategies:
        if s.kind not in STRATEGY_PARAMS:
            raise ValueError(f"Unknown strategy kind {s.kind!r}; expected one of "
                             f"{sorted(STRATEGY_PARAMS)}.")
        allowed = STRATEGY_PARAMS[s.kind]
        for key in _ALL_PARAMS:
            if key not in allowed and getattr(s, key) is not None:
                raise ValueError(f"Strategy {s.kind!r} takes no {key!r}; "
                                 f"allowed: {list(allowed)}.")
        grid = []
        for key in allowed:
            values = getattr(s, key)
            if values is None:
                values = _PARAM_DEFAULTS.get(key)
            if not values:
                raise ValueError(f"Strategy {s.kind!r} needs a non-empty {key!r} list.")
            if key == "temperature" and min(values) <= 0:
                raise ValueError(f"Strategy {s.kind!r} samples, so every temperature "
                                 f"must be > 0; got {list(values)}.")
            grid.append([(key, v) for v in values])
        for combo in itertools.product(*grid):
            name = "_".join([s.kind] + [f"{_ABBREV[k]}{v}" for k, v in combo])
            runs.append(DecodeRun(name=name, kind=s.kind, params=dict(combo)))

    names = [r.name for r in runs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"Duplicate strategy runs: {dupes}.")
    return runs


def validate_posthoc_config(cfg):
    """Everything checkable without touching a model, data or checkpoint.

    Returns:
        list[DecodeRun]: the expanded strategy grid.
    """
    bad = sorted(set(cfg.metrics) - set(METRICS))
    if bad or not cfg.metrics:
        raise ValueError(f"Unknown or empty metric list {list(cfg.metrics)}; "
                         f"expected a subset of {list(METRICS)}.")
    bad = sorted(set(cfg.references) - set(REFERENCES))
    if bad or not cfg.references:
        raise ValueError(f"Unknown or empty reference list {list(cfg.references)}; "
                         f"expected a subset of {list(REFERENCES)}.")
    if cfg.split not in SPLITS:
        raise ValueError(f"Unknown split {cfg.split!r}; expected one of {list(SPLITS)}.")
    if cfg.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {cfg.batch_size}.")
    if not cfg.strategies:
        raise ValueError("At least one strategy is required.")
    return expand_strategies(cfg.strategies)


def load_posthoc_config(path, overrides):
    """YAML + dot-list overrides merged onto the `PosthocEvalConfig` schema.

    Unknown keys and missing `reference_config` / `checkpoint` raise an OmegaConf
    error here, before anything expensive is loaded.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Eval config not found: {path}")
    merged = OmegaConf.merge(OmegaConf.structured(PosthocEvalConfig),
                             OmegaConf.load(path),
                             OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_object(merged)


def score_object(gen, refs, metrics, n_points=4096, sinkhorn_cfg=None, seed=0,
                 device="cpu"):
    """Every selected metric of one generated mesh against every reference.

    Args:
        gen: ``(verts [V, 3] metres, faces [F, 3])``.
        refs: reference name -> mesh; columns get the suffix ``_vs_<name>``.
        metrics: subset of `METRICS`.
        n_points: surface samples for chamfer, Hausdorff and roof chamfer.
        sinkhorn_cfg: `SinkhornConfig`; None uses its defaults.
        seed: surface-sampling seed shared by every sampled metric.
        device: torch device for the Sinkhorn solve.

    Returns:
        dict: column -> value. Undefined metrics are ``nan``, never an
        exception: a free-running generation is often empty or open.
    """
    sink = sinkhorn_cfg or SinkhornConfig()
    row = {"n_faces_gen": int(len(np.asarray(gen[1]).reshape(-1, 3))),
           "watertight_gen": float(is_watertight_mesh(gen[1]))}
    nan = float("nan")

    for name, ref in refs.items():
        sfx = f"_vs_{name}"
        if "chamfer" in metrics or "hausdorff" in metrics:
            d_ab, d_ba = surface_distances(gen, ref, n=n_points, seed=seed)
            if "chamfer" in metrics:
                row["chamfer_m" + sfx] = chamfer_distance(d_ab, d_ba)
            if "hausdorff" in metrics:
                row["hausdorff_p95_m" + sfx] = hausdorff_p95(d_ab, d_ba)
                row["hausdorff_max_m" + sfx] = hausdorff_max(d_ab, d_ba)
        if "vol_symdiff" in metrics:
            # Exact CSG; `ok` is False unless both are closed, outward solids.
            vol = exact_pair_volumes(gen, ref)
            ok = bool(vol["ok"])
            row["symdiff_m3" + sfx] = float(vol["symdiff_m3"]) if ok else nan
            row["rel_symdiff" + sfx] = float(vol["rel_symdiff"]) if ok else nan
            row["vol_ok" + sfx] = ok
            row["vol_note" + sfx] = vol["note"]
        if "sinkhorn" in metrics:
            row["sinkhorn_m" + sfx] = mesh_sinkhorn(
                gen, ref, n_points=sink.n_points, blur=sink.blur_m,
                max_iters=sink.max_iters, tol=sink.tol, seed=seed, device=device)
        if "roof_chamfer" in metrics:
            roof_gen, roof_ref = roof_submesh(*gen), roof_submesh(*ref)
            if len(roof_gen[1]) and len(roof_ref[1]):
                row["roof_chamfer_m" + sfx] = chamfer_distance(
                    *surface_distances(roof_gen, roof_ref, n=n_points, seed=seed))
            else:
                row["roof_chamfer_m" + sfx] = nan
    return row


def object_seed(seed, object_id):
    """Sampling seed for one object: ``crc32("<seed>:<object_id>")``.

    The same for every strategy, so strategies are compared on common random
    numbers, and independent of batch position. crc32 rather than `hash()`,
    which Python salts per process.
    """
    return zlib.crc32(f"{seed}:{object_id}".encode())


def generation_budget(ref_cfg):
    """Tokens to allow per object: the `MeshEvalCallback` rule, 9 per face + EOS.

    `generate` clamps this to what the position scheme allows.
    """
    faces = ref_cfg.mesh_model.n_max_triangles or ref_cfg.mesh_data.max_faces or 200
    return 9 * int(faces) + 1


def checkpoint_info(path):
    """Path, sha256 and Lightning metadata of a checkpoint, for checkpoint.json."""
    path = Path(path)
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(),
            "epoch": ckpt.get("epoch"), "global_step": ckpt.get("global_step"),
            "hyper_parameters": dict(ckpt.get("hyper_parameters") or {})}


def _git_state():
    """``(info, diff)``: commit, dirty flag and status lines, plus ``git diff HEAD``."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout
    try:
        status = git("status", "--porcelain")
        info = {"commit": git("rev-parse", "HEAD").strip(),
                "dirty": bool(status.strip()), "status": status.splitlines()}
        return info, git("diff", "HEAD")
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning("git state unavailable: %s", exc)
        return {"commit": None, "error": str(exc)}, ""


def _versions():
    out = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
           "cuda_available": torch.cuda.is_available()}
    for pkg in ("numpy", "pandas", "trimesh", "manifold3d", "lightning", "omegaconf"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _write_artifacts(out_dir, cfg, ref_cfg, ids, n_split, checkpoint, split_sizes,
                     device):
    """Everything besides the per-run CSVs that a reproduction needs."""
    (out_dir / "eval_config.yaml").write_text(
        OmegaConf.to_yaml(OmegaConf.structured(cfg)), encoding="utf-8")
    (out_dir / "reference_config.yaml").write_text(
        OmegaConf.to_yaml(OmegaConf.structured(ref_cfg)), encoding="utf-8")

    git, diff = _git_state()
    _write_json(out_dir / "environment.json", {
        "git": git, "versions": _versions(), "device": str(device),
        "hostname": platform.node(), "platform": platform.platform(), "argv": sys.argv,
        "note": "Untracked files are listed under git.status but are not "
                "captured in git_diff.patch."})
    if diff:
        (out_dir / "git_diff.patch").write_text(diff, encoding="utf-8")
    if checkpoint is not None:
        _write_json(out_dir / "checkpoint.json", checkpoint)
    _write_json(out_dir / "split.json", {
        "split": cfg.split, "dataset_dir": ref_cfg.mesh_data.dataset_dir,
        "split_sizes": split_sizes, "n_split": n_split,
        "max_objects": cfg.max_objects, "n_evaluated": len(ids),
        "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()})


def _append_csv(path, rows):
    """Append rows, writing the header only when the file is new."""
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            writer.writeheader()
        writer.writerows(rows)


def _summarize(run, rows, cfg):
    """One summary.csv row: rates, nanmean/nanmedian of each metric, defined %."""
    df = pd.DataFrame(rows)
    out = {"run": run.name, "kind": run.kind,
           **{f"param_{k}": v for k, v in run.params.items()},
           "n_objects": len(df),
           "watertight_pct": 100.0 * df["watertight_gen"].mean(),
           "decode_rate": df["decode_ok"].mean(), "eos_rate": df["eos_emitted"].mean()}
    for ref in cfg.references:
        for col in METRIC_COLUMNS:
            name = f"{col}_vs_{ref}"
            if name in df:
                out[f"{name}_mean"] = df[name].mean()
                out[f"{name}_median"] = df[name].median()
        if "vol_symdiff" in cfg.metrics:
            out[f"vol_defined_pct_vs_{ref}"] = 100.0 * df[f"vol_ok_vs_{ref}"].mean()
        if "roof_chamfer" in cfg.metrics:
            out[f"roof_defined_pct_vs_{ref}"] = (
                100.0 * np.isfinite(df[f"roof_chamfer_m_vs_{ref}"].astype(float)).mean())
    return out


def _score_batch(model, batch, out, cond, tgt, cond_pad, cfg, seeds, seconds, device):
    """Decode each generated sequence and score it against its references."""
    inverse = amt_detokenize if model.tokenization == "amt" else detokenize
    rows = []
    for k, oid in enumerate(batch["ids"]):
        centre = batch["center"][k].cpu().numpy()
        scale = batch["scale"][k].cpu().numpy()
        item_cond = model._item_cond((cond, cond_pad), k)
        gen = _to_metres(model, out[k].cpu(), centre, scale, item_cond)
        # Both references go through the tokenizer round trip, as in run_mesh_eval.
        lod2 = _to_metres(model, tgt[k].cpu(), centre, scale, item_cond)
        v1, f1 = inverse(batch["cond"][k].cpu().numpy(), model.num_bins)
        lod1 = (v1 * scale + centre, f1)
        refs = {"lod1": lod1, "lod2": lod2}

        row = {"object_id": str(oid), "seed": seeds[k],
               "n_faces_lod1": len(lod1[1]), "n_faces_lod2": len(lod2[1]),
               "eos_emitted": bool((out[k] == model.eos).any()),
               "decode_ok": bool(len(gen[1]) > 0), "gen_seconds": seconds}
        row.update(score_object(gen, {name: refs[name] for name in cfg.references},
                                cfg.metrics, n_points=cfg.n_points,
                                sinkhorn_cfg=cfg.sinkhorn, seed=SURFACE_SEED,
                                device=device))
        rows.append(row)
    return rows


def run_posthoc_eval(cfg, ref_cfg, model, dataset, out_dir, checkpoint=None,
                     split_sizes=None, device="cpu"):
    """Generate and score every selected object under every expanded strategy.

    Args:
        cfg: `PosthocEvalConfig`.
        ref_cfg: the resolved training `Config` the checkpoint belongs to.
        model: a `MeshTransformerModule`, already on ``device``.
        dataset: `MeshDataset`, a `Subset` of one, or anything with the same
            item contract.
        out_dir: this evaluation's directory; created if missing.
        checkpoint: `checkpoint_info` dict, written to checkpoint.json.
        split_sizes: split name -> size, recorded in split.json.
        device: torch device for generation and the Sinkhorn solve.

    Returns:
        list[dict]: one summary row per run, as written to summary.csv.
    """
    runs = validate_posthoc_config(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = (np.arange(len(dataset)) if cfg.max_objects is None
               else eval_indices(len(dataset), cfg.max_objects, cfg.seed))
    if len(indices) == 0:
        raise ValueError(f"The {cfg.split} split selected no objects to evaluate.")
    ids = []
    for i in indices:
        full, orig = _resolve(dataset, int(i))
        ids.append(str(full.ids[orig]))
    _write_artifacts(out_dir, cfg, ref_cfg, ids, len(dataset), checkpoint,
                     split_sizes, device)

    max_new_tokens = generation_budget(ref_cfg)
    pad = specials(model.num_bins)[2]
    model.eval()
    summary = []
    for run in runs:
        run_dir = out_dir / run.name
        run_dir.mkdir(exist_ok=True)
        kwargs = run.generate_kwargs()
        _write_json(run_dir / "run.json", {
            **asdict(run), "generate_kwargs": kwargs, "max_new_tokens": max_new_tokens,
            "batch_size": cfg.batch_size, "seed": cfg.seed,
            "seed_rule": "zlib.crc32(f'{seed}:{object_id}'), same for every strategy",
            "surface_seed": SURFACE_SEED})
        logger.info("[posthoc] %s: %d objects", run.name, len(indices))

        rows = []
        for start in range(0, len(indices), cfg.batch_size):
            chunk = indices[start:start + cfg.batch_size]
            batch = mesh_collate_fn([dataset[int(i)] for i in chunk], pad=pad)
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            seeds = [object_seed(cfg.seed, oid) for oid in batch["ids"]]

            started = time.perf_counter()
            with torch.no_grad():
                cond, tgt, cond_pad, _ = model._prepare(batch)
                out = model.generate(cond, cond_pad, max_new_tokens=max_new_tokens,
                                     seeds=seeds, **kwargs)
            seconds = (time.perf_counter() - started) / len(chunk)

            batch_rows = _score_batch(model, batch, out, cond, tgt, cond_pad, cfg,
                                      seeds, seconds, device)
            # Flushed per batch: a crash keeps every finished object.
            _append_csv(run_dir / "objects.csv", batch_rows)
            rows.extend(batch_rows)

        summary.append(_summarize(run, rows, cfg))
        pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)
    return summary
