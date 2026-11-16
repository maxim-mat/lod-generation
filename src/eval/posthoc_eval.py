"""Post-hoc evaluation of a mesh-transformer checkpoint across decoding strategies.

Usage:
    python -m src.eval.posthoc_eval --config configs/posthoc-eval.yaml [dot.overrides]

`src/eval/mesh_eval.py` scores greedy or beam decoding on a small subsample
during training. This scores a finished checkpoint on a whole split under a grid
of strategies, one CSV row per object, and writes everything needed to
reproduce any row: resolved configs, checkpoint hash, git state, versions,
per-object seeds.
"""
import argparse
import csv
import hashlib
import itertools
import json
import logging
import platform
import re
import subprocess
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.analysis.exact_volume import exact_pair_volumes
from src.dataset.mesh_dataset import (amt_detokenize, detokenize, mesh_collate_fn,
                                      specials)
from src.eval.face_match import (COUNT_COLUMNS, FACE_SCORE_COLUMNS, face_scores,
                                 match_rates)
from src.eval.mesh_eval import _resolve, eval_indices
from src.eval.mesh_metrics import (chamfer_distance, chamfer_floor, chamfer_ratio,
                                   hausdorff_max, hausdorff_p95, is_watertight_mesh,
                                   roof_submesh, surface_distances)
from src.eval.sinkhorn import mesh_sinkhorn
from src.utils.config import PosthocEvalConfig, SinkhornConfig
from src.utils.initialization import load_config

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
# Reference-free columns that summaries also average (the raw counts are not).
EXTRA_COLUMNS = (tuple(c for c in FACE_SCORE_COLUMNS if c not in COUNT_COLUMNS)
                 + ("lod1_chamfer_norm",))

# The do-nothing baseline: the decoded LOD1 condition scored as the generation.
LOD1_COPY = "lod1_copy"

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
        if self.kind == LOD1_COPY:
            return {}
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


IMAGE_MODES = ("true", "shuffled", "none")


def image_source(k, n, mode):
    """Split position whose image object ``k`` (of ``n``) is scored with.

    ``shuffled`` pairs each object with the one half the split away: never
    itself for n >= 2, deterministic, and spatially unrelated because the split
    order is a random permutation. Every other mode keeps the object's own.
    """
    if mode != "shuffled":
        return k
    if n < 2:
        raise ValueError("image_mode='shuffled' needs at least two objects in the split.")
    return (k + n // 2) % n


def validate_posthoc_config(cfg):
    """Everything checkable without touching a model, data or checkpoint.

    Returns:
        list[DecodeRun]: the expanded strategy grid.
    """
    if cfg.image_mode not in IMAGE_MODES:
        raise ValueError(f"Unknown image_mode {cfg.image_mode!r}; expected one of "
                         f"{list(IMAGE_MODES)}.")
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
    if not cfg.reference_configs:
        raise ValueError("reference_configs is empty: name at least one training config.")
    if len(cfg.checkpoints) != len(cfg.reference_configs):
        raise ValueError(f"checkpoints has {len(cfg.checkpoints)} entries but "
                         f"reference_configs has {len(cfg.reference_configs)}; they "
                         "pair by position (null = the run's best checkpoint).")
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
                              text=True, encoding="utf-8", errors="replace", check=True).stdout
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
    for col in EXTRA_COLUMNS:
        if col in df:
            out[f"{col}_mean"] = df[col].mean()
            out[f"{col}_median"] = df[col].median()
    if set(COUNT_COLUMNS) <= set(df):
        # face_match's micro average: rates of the counts pooled over the split.
        micro = match_rates(df[list(COUNT_COLUMNS)].sum().to_dict())
        out.update({f"{k}_micro": v for k, v in micro.items()})
    return out


SAMPLE_COLUMN = "roof_chamfer_m_vs_lod2"


def select_samples(df, column=SAMPLE_COLUMN, k=5):
    """Object ids of the ``k`` best, most typical and worst rows by ``column``.

    Rows where ``column`` is NaN are not ranked. ``good`` runs best first,
    ``bad`` worst first, ``avg`` closest-to-the-median first. With fewer than
    ``3 * k`` rows the buckets overlap rather than come up short.

    Returns:
        dict: ``{"good": [...], "avg": [...], "bad": [...]}``.
    """
    ranked = df[np.isfinite(df[column].astype(float))].sort_values(column, kind="stable")
    ids = ranked["object_id"].astype(str)
    dist = (ranked[column] - ranked[column].median()).abs()
    return {"good": ids.head(k).tolist(),
            "avg": ids[dist.sort_values(kind="stable").index[:k]].tolist(),
            "bad": ids.iloc[::-1].head(k).tolist()}


def write_samples(run_dir, picks, meshes, rows):
    """One LOD1 | LOD2 | generated plotly page per picked object.

    Args:
        picks: `select_samples` output.
        meshes: object id -> ``{"lod1", "lod2", "gen"}`` metre meshes.
        rows: this run's `objects.csv` rows, for the titles.

    Returns:
        list[Path]: the files written, in bucket then rank order.
    """
    from src.visualize_mesh import mesh_figure, side_by_side

    by_id = {str(r["object_id"]): r for r in rows}
    out_dir = Path(run_dir) / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for bucket in ("good", "avg", "bad"):
        for rank, oid in enumerate(picks.get(bucket, []), start=1):
            m, row = meshes[oid], by_id[oid]
            panels = [("LOD1 input", m["lod1"], "#898781"),
                      ("LOD2 truth", m["lod2"], "#2a78d6"),
                      (f"generated ({Path(run_dir).name})", m["gen"], "#eda100")]
            fig = side_by_side([mesh_figure(*mesh, color=c) for _, mesh, c in panels],
                               [f"{name}, {len(mesh[1])} tris" for name, mesh, _ in panels])
            fig.update_layout(title=(
                f"[{bucket} #{rank}] {oid} | roof chamfer {row.get(SAMPLE_COLUMN, np.nan):.3f} m"
                f" | chamfer {row.get('chamfer_m_vs_lod2', np.nan):.3f} m"
                f" | face F1 {row.get('face_f1', np.nan):.3f}"))
            # sanitised outside the f-string: a backslash inside an f-string
            # expression is a SyntaxError before Python 3.12
            safe_oid = re.sub(r"[^\w.-]", "_", oid)
            path = out_dir / f"{bucket}_{rank}_{safe_oid}.html"
            fig.write_html(path, include_plotlyjs="cdn")
            paths.append(path)
    return paths


def _lod1_unit(model, batch, k):
    """The LOD1 condition of item ``k``, decoded to the unit box."""
    inverse = amt_detokenize if model.tokenization == "amt" else detokenize
    return inverse(batch["cond"][k].cpu().numpy(), model.num_bins)


def _score_batch(model, batch, gen_unit, eos_emitted, cfg, seeds, seconds, device):
    """Score each item's unit-box generation against its references.

    Args:
        gen_unit: one ``(verts, faces)`` in the unit box per item: a decoded
            generation, or the decoded condition for `LOD1_COPY`.
        eos_emitted: one bool per item, or None when nothing was generated.

    Returns:
        tuple: (rows, meshes) where ``meshes`` maps object id to the metre
        ``lod1`` / ``lod2`` / ``gen`` meshes the samples are drawn from.
    """
    with torch.no_grad():
        cond, tgt, cond_pad, _ = model._prepare(batch)
    rows, meshes = [], {}
    for k, oid in enumerate(batch["ids"]):
        centre = batch["center"][k].cpu().numpy()
        scale = batch["scale"][k].cpu().numpy()
        item_cond = model._item_cond((cond, cond_pad), k)
        # Both references go through the tokenizer round trip, as in run_mesh_eval.
        units = {"gen": gen_unit[k],
                 "lod2": model.decode_tokens(tgt[k].cpu(), cond=item_cond),
                 "lod1": _lod1_unit(model, batch, k)}
        metres = {name: (np.asarray(v) * scale + centre, f) for name, (v, f) in units.items()}
        gen, lod1, lod2 = metres["gen"], metres["lod1"], metres["lod2"]
        refs = {"lod1": lod1, "lod2": lod2}

        row = {"object_id": str(oid), "seed": seeds[k],
               "n_faces_lod1": len(lod1[1]), "n_faces_lod2": len(lod2[1]),
               "eos_emitted": float("nan") if eos_emitted is None else eos_emitted[k],
               "decode_ok": bool(len(gen[1]) > 0), "gen_seconds": seconds}
        row.update(score_object(gen, {name: refs[name] for name in cfg.references},
                                cfg.metrics, n_points=cfg.n_points,
                                sinkhorn_cfg=cfg.sinkhorn, seed=SURFACE_SEED,
                                device=device))
        row.update(face_scores(units["gen"], units["lod2"], units["lod1"], scale,
                               model.num_bins))
        if "chamfer_m_vs_lod1" in row:
            # Below 1: the generation sits closer to LOD1 than the truth does.
            row["lod1_chamfer_norm"] = chamfer_ratio(
                row["chamfer_m_vs_lod1"],
                chamfer_floor(lod1, lod2, cfg.n_points, seed=SURFACE_SEED))
        rows.append(row)
        meshes[str(oid)] = metres
    return rows, meshes


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
    indices = [int(i) for i in indices]
    position = {k: p for p, k in enumerate(indices)}
    model.eval()
    summary = []
    for run in runs + [DecodeRun(name=LOD1_COPY, kind=LOD1_COPY, params={})]:
        run_dir = out_dir / run.name
        run_dir.mkdir(exist_ok=True)
        kwargs = run.generate_kwargs()
        _write_json(run_dir / "run.json", {
            **asdict(run), "generate_kwargs": kwargs, "max_new_tokens": max_new_tokens,
            "batch_size": cfg.batch_size, "seed": cfg.seed,
            "seed_rule": "zlib.crc32(f'{seed}:{object_id}'), same for every strategy",
            "surface_seed": SURFACE_SEED})
        logger.info("[posthoc] %s: %d objects", run.name, len(indices))

        rows, meshes = [], {}
        for start in range(0, len(indices), cfg.batch_size):
            chunk = indices[start:start + cfg.batch_size]
            items = []
            for k in chunk:
                item = dataset[k]
                if "image" in item and cfg.image_mode == "shuffled":
                    source = indices[image_source(position[k], len(indices), "shuffled")]
                    full, orig = _resolve(dataset, source)
                    item = {**item, "image": full.image(orig)}
                items.append(item)
            batch = mesh_collate_fn(items, pad=pad)
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            # `none`: every row gets the null token, as image dropout trained it.
            image_valid = (torch.zeros(len(chunk), dtype=torch.bool, device=device)
                           if cfg.image_mode == "none" else None)
            seeds = [object_seed(cfg.seed, oid) for oid in batch["ids"]]

            if run.kind == LOD1_COPY:
                gen_unit = [_lod1_unit(model, batch, k) for k in range(len(chunk))]
                eos, seconds = None, 0.0
            else:
                started = time.perf_counter()
                with torch.no_grad():
                    cond, _, cond_pad, _ = model._prepare(batch)
                    out = model.generate(cond, cond_pad, max_new_tokens=max_new_tokens,
                                         seeds=seeds, image=batch.get("image"),
                                         image_valid=image_valid, **kwargs)
                seconds = (time.perf_counter() - started) / len(chunk)
                # The VQ-VAE decoder reads this item's encoded condition.
                gen_unit = [model.decode_tokens(out[k].cpu(),
                                                cond=model._item_cond((cond, cond_pad), k))
                            for k in range(len(chunk))]
                eos = [bool((out[k] == model.eos).any()) for k in range(len(chunk))]

            batch_rows, batch_meshes = _score_batch(model, batch, gen_unit, eos, cfg,
                                                    seeds, seconds, device)
            # Flushed per batch: a crash keeps every finished object.
            _append_csv(run_dir / "objects.csv", batch_rows)
            rows.extend(batch_rows)
            # ponytail: every mesh of the run stays in memory (~100 MB on a
            # 25k split) until the samples are picked; regenerate the picks
            # instead if a larger split ever runs out of RAM.
            meshes.update(batch_meshes)

        summary.append(_summarize(run, rows, cfg))
        if SAMPLE_COLUMN in rows[0]:
            write_samples(run_dir, select_samples(pd.DataFrame(rows)), meshes, rows)
        else:
            logger.warning("[posthoc] %s: no %s column, so no samples (needs the "
                           "roof_chamfer metric and the lod2 reference)",
                           run.name, SAMPLE_COLUMN)
        del meshes
        pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)
    return summary


def check_tokenization(hparams, ref_cfg):
    """Refuse a checkpoint whose token spelling differs from the reference config.

    Nothing downstream reconciles the two, and a mismatch does not raise by
    itself: a coordinate sequence decoded as AMT keeps its vertices and silently
    re-triangulates them into ~1.8x the faces (see notebooks/test-mesh-normal.ipynb).
    A checkpoint without the key predates AMT, so it is a coordinate run.
    """
    saved = hparams.get("tokenization", "coord")
    wanted = ref_cfg.mesh_data.tokenization
    if saved != wanted:
        raise ValueError(
            f"Checkpoint was trained with tokenization={saved!r} but the reference "
            f"config sets mesh_data.tokenization={wanted!r}. Point reference_config "
            "at the config this checkpoint was trained with.")


def resolve_device(name):
    """``"auto"`` -> cuda when available, else cpu; anything else unchanged."""
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def build_split(ref_cfg, split):
    """The requested split, rebuilt exactly as `train_mesh` built it.

    Same arguments and seed as training, so `random_split` reproduces the
    partition the checkpoint was trained against.

    Returns:
        tuple: (dataset, {split name: size}).
    """
    from src.dataset.mesh_datamodule import MeshDataModule

    m = ref_cfg.mesh_data
    if not m.dataset_dir:
        raise ValueError("reference config has no mesh_data.dataset_dir.")
    datamodule = MeshDataModule(
        dataset_dir=m.dataset_dir, lod_in=m.lod_in, lod_out=m.lod_out,
        num_bins=m.num_bins, margin_lo=list(m.margin_lo), margin_hi=list(m.margin_hi),
        max_faces=m.max_faces, max_files=m.max_files, tokenization=m.tokenization,
        batch_size=ref_cfg.training.batch_size,
        train_val_test_split=tuple(ref_cfg.training.train_val_test_split),
        split=ref_cfg.training.split,
        split_block_m=ref_cfg.training.split_block_m,
        split_buffer_m=ref_cfg.training.split_buffer_m,
        num_workers=0, seed=ref_cfg.seed, ortho_dir=m.ortho_dir, image_size=m.image_size,
        image_margin=m.image_margin, image_min_coverage=m.image_min_coverage)
    datamodule.setup()
    sizes = {name: len(getattr(datamodule, f"{name}_dataset")) for name in SPLITS}
    return getattr(datamodule, f"{split}_dataset"), sizes


def load_model(ref_cfg, checkpoint, device):
    """The checkpoint's module, with its frozen VQ-VAE when the run used one."""
    from src.models.mesh_transformer import MeshTransformerModule
    from src.train_mesh import _load_vqvae

    vqvae = _load_vqvae(ref_cfg)
    extra = {} if vqvae is None else {"vqvae": vqvae}
    model = MeshTransformerModule.load_from_checkpoint(str(checkpoint),
                                                      map_location="cpu", **extra)
    return model.to(device).eval()


def main(argv=None):
    """CLI entry point. Returns the evaluation directory."""
    parser = argparse.ArgumentParser(
        description="Score a mesh-transformer checkpoint under several decoding "
                    "strategies.")
    parser.add_argument("--config", "-c", type=Path, required=True,
                        help="Post-hoc eval YAML, e.g. configs/posthoc-eval.yaml")
    parser.add_argument("overrides", nargs="*",
                        help="Dot-list overrides, e.g. max_objects=256")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Cheapest checks first, for EVERY model: a bad last entry must fail in
    # seconds, not after the first models have generated for days.
    cfg = load_posthoc_config(args.config, args.overrides)
    validate_posthoc_config(cfg)
    from src.train_mesh import check_vertex_margins
    models = []
    for ref_path, ckpt_arg in zip(cfg.reference_configs, cfg.checkpoints):
        ref_cfg = load_config(Path(ref_path), list(cfg.reference_overrides))
        if ref_cfg.config_set != "mesh":
            raise ValueError(f"{ref_path} has config_set={ref_cfg.config_set!r}; post-hoc "
                             "eval scores mesh-transformer runs (config_set: mesh).")
        ckpt = resolve_checkpoint(ref_cfg, ckpt_arg)
        info = checkpoint_info(ckpt)
        check_tokenization(info["hyper_parameters"], ref_cfg)
        check_vertex_margins(info["hyper_parameters"].get("image_vertex_features", False),
                             ref_cfg.mesh_data.margin_lo, ref_cfg.mesh_data.margin_hi)
        models.append((ref_cfg, ckpt, info))
    names = [ref_cfg.logging.run_name for ref_cfg, _, _ in models]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"Two entries share logging.run_name {dupes}, so they would "
                         "write into one directory. Score them in separate calls.")

    device = resolve_device(cfg.device)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = (Path(cfg.output_dir or "outputs/posthoc_eval")
            / (stamp if cfg.image_mode == "true" else f"{cfg.image_mode}-{stamp}"))
    combined = []
    for (ref_cfg, ckpt, info), name in zip(models, names):
        # Re-seeded per model, so a model's rows do not depend on its position.
        L.seed_everything(cfg.seed, workers=True)
        dataset, sizes = build_split(ref_cfg, cfg.split)
        model = load_model(ref_cfg, ckpt, device)
        logger.info("[posthoc] %s (%s), %s split (%d objects) -> %s",
                    name, ckpt.name, cfg.split, len(dataset), root / name)
        summary = run_posthoc_eval(cfg, ref_cfg, model, dataset, root / name,
                                   checkpoint=info, split_sizes=sizes, device=device)
        combined.extend({"model": name, **row} for row in summary)
        # Rewritten after each model: a crash keeps every finished one.
        pd.DataFrame(combined).to_csv(root / "summary.csv", index=False)
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    logger.info("[posthoc] done: %s", root)
    return root


def resolve_checkpoint(ref_cfg, checkpoint):
    """The checkpoint file to score for one reference config.

    A given path must exist. ``None`` takes the run's lowest
    ``val_gen_chamfer_m=<x>.ckpt`` (what training selects on), else its
    ``last.ckpt``, from ``<save_dir>/<experiment_name>/<run_name>/checkpoints``.

    Raises:
        FileNotFoundError: the path, or every candidate, is missing.
    """
    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    lg = ref_cfg.logging
    ckpt_dir = Path(lg.save_dir, lg.experiment_name, lg.run_name, "checkpoints")
    scored = []
    for p in ckpt_dir.glob("*.ckpt"):
        m = re.search(r"val_gen_chamfer_m=([0-9.]+)\.ckpt$", p.name)
        if m:
            scored.append((float(m.group(1)), p.name, p))
    if scored:
        return min(scored)[2]
    if (ckpt_dir / "last.ckpt").is_file():
        return ckpt_dir / "last.ckpt"
    raise FileNotFoundError(f"No val_gen_chamfer_m checkpoint and no last.ckpt in "
                            f"{ckpt_dir}; set its entry in checkpoints explicitly.")


if __name__ == "__main__":
    main()
