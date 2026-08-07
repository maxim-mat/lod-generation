"""Free-running generation scored against the paired ground truth.

`src/eval/callback.py` measures an unconditional model against a *distribution*
-- MMD, Wasserstein, novelty -- because there is nothing to pair a sample with.
The mesh transformer is conditional: every LOD1 input has exactly one true LOD2,
so generations can be scored one against one, which is both far more sensitive
and far cheaper to read.

Split the same way as `run_generative_eval` / `GenerativeEvalCallback`: a plain
function that takes a model and a dataset, plus a thin Lightning adapter. The
function is what makes this testable without a trainer and re-runnable from a
checkpoint.

The expensive part is `generate()`: no KV cache, so a 200-face building is
~1800 sequential forward passes. That is why this is a gated callback and not
part of `validation_step`.
"""
import logging
from pathlib import Path

import lightning as L
import numpy as np
import torch

from src.dataset.mesh_dataset import detokenize, mesh_collate_fn, specials, write_obj
from src.eval.mesh_metrics import mesh_metrics, surface_distances, chamfer_distance
from src.post_process.post_process import mesh_to_cityjson, save_to_file

logger = logging.getLogger(__name__)


def eval_indices(n, count, seed):
    """Which dataset items to score, derived only from ``n`` and ``seed``.

    Deliberately stateless: computing the choice rather than caching it means a
    fresh `trainer.test` run scores exactly the buildings the validation curve
    was tracking, with nothing to carry across the checkpoint boundary.
    """
    if n <= 0:
        return np.zeros(0, dtype=int)
    return np.sort(np.random.default_rng(seed).choice(n, min(count, n), replace=False))


def _resolve(dataset, i):
    """``(MeshDataset, original index)`` for item ``i`` of a possible Subset."""
    if hasattr(dataset, "indices"):
        return dataset.dataset, int(dataset.indices[i])
    return dataset, int(i)


def _n_polygons(cj):
    if not cj:
        return 0
    return len(next(iter(cj["CityObjects"].values()))["geometry"][0]["boundaries"][0])


def _to_metres(tokens, num_bins, center, scale):
    verts, faces = detokenize(tokens, num_bins)
    return verts * scale + center, faces


def run_mesh_eval(model, dataset, indices, cfg, max_new_tokens, seed=1234,
                  save_dir=None, gt_ceiling=False):
    """Generate from each LOD1 condition and score against its true LOD2.

    Args:
        model: a `MeshTransformerModule`.
        dataset: `MeshDataset` or a `Subset` of one.
        indices: positions within ``dataset`` to score.
        cfg: `MeshEvalConfig`.
        max_new_tokens: generation budget, ``9 * max_faces + 1``.
        gt_ceiling: also measure the ground truth against its own tokenizer
            round trip. That is the best score the model could possibly reach at
            this bin count, and without it none of the other numbers is
            readable -- 3 cm of error means nothing until you know whether the
            floor is 1 cm or 15.

    Returns:
        dict: metric name -> float, already averaged over the sampled buildings.
    """
    if len(indices) == 0:
        return {}

    # `max_seq_len` is sized from the longest sequence actually in the corpus,
    # which is usually shorter than `max_faces` allows -- no building hits the
    # cap. Asking for more tokens than the positional embedding holds raises in
    # forward(), so the budget is whichever is smaller.
    budget = model.network.max_seq_len - 1
    if max_new_tokens > budget:
        logger.info("[mesh-eval] generation budget %d -> %d (positional limit)",
                    max_new_tokens, budget)
        max_new_tokens = budget

    _, eos, _ = specials(model.num_bins)
    was_training = model.training
    model.eval()

    per_sample, gt_rows, written = [], [], 0
    for start in range(0, len(indices), cfg.batch_size):
        chunk = indices[start:start + cfg.batch_size]
        items = [dataset[int(i)] for i in chunk]
        batch = mesh_collate_fn(items, pad=model.pad)

        device = next(model.parameters()).device
        with torch.no_grad():
            # temperature 0 -> argmax. A sampled generation would make the
            # metric a random variable and the epoch-to-epoch curve unreadable.
            out = model.generate(batch["cond"].to(device),
                                 batch["cond_pad_mask"].to(device),
                                 max_new_tokens=max_new_tokens, temperature=0.0)

        for k, i in enumerate(chunk):
            centre = batch["center"][k].numpy()
            scale = batch["scale"][k].numpy()
            tokens = out[k].cpu().numpy()

            gen = _to_metres(tokens, model.num_bins, centre, scale)
            ref = _to_metres(batch["tgt"][k].numpy(), model.num_bins, centre, scale)

            row = mesh_metrics(gen, ref, taus=tuple(cfg.taus),
                               n_points=cfg.n_points, voxel_m=cfg.voxel_m)
            row["decode_rate"] = float(len(gen[1]) > 0)
            row["eos_rate"] = float((tokens == eos).any())

            cj = mesh_to_cityjson(*gen)
            row["cityjson_rate"] = float(bool(cj))
            n_ref_poly = _n_polygons(mesh_to_cityjson(*ref))
            row["merged_face_ratio"] = (_n_polygons(cj) / n_ref_poly
                                        if n_ref_poly else float("nan"))
            per_sample.append(row)

            if save_dir is not None and written < cfg.save_samples and cj:
                name = str(batch["ids"][k]).replace("/", "_")
                save_to_file(cj, Path(save_dir) / f"{name}_gen.city.json")
                write_obj(Path(save_dir) / f"{name}_gen.obj", *gen)
                write_obj(Path(save_dir) / f"{name}_gt.obj", *ref)
                written += 1

            if gt_ceiling:
                full, orig = _resolve(dataset, i)
                raw = full.mesh_pair(orig)[1]
                d_ab, d_ba = surface_distances(ref, raw, n=cfg.n_points)
                gt_rows.append({
                    "gt_rt_chamfer_m": chamfer_distance(d_ab, d_ba),
                    "gt_watertight_rate": row["watertight_gt"],
                })

    if was_training:
        model.train()

    metrics = _aggregate(per_sample)
    # watertight_gen is the rate that makes vol_iou readable: it is averaged
    # over every sample, while vol_iou is only defined on the watertight ones.
    metrics["watertight_rate"] = metrics.pop("watertight_gen", float("nan"))
    metrics.pop("watertight_gt", None)
    metrics.update(_aggregate(gt_rows))
    return metrics


def _aggregate(rows):
    """nanmean per key, dropping keys that were undefined for every sample."""
    out = {}
    for key in {k for row in rows for k in row}:
        values = np.array([row.get(key, np.nan) for row in rows], dtype=float)
        if np.isfinite(values).any():
            out[key] = float(np.nanmean(values))
    return out


class MeshEvalCallback(L.Callback):
    """Runs `run_mesh_eval` on a fixed subsample of val, and all of test.

    Args:
        cfg: `MeshEvalConfig`.
        save_dir: run directory; samples land under ``<save_dir>/mesh_eval``.
        max_faces: from `mesh_data`, sets the generation budget.
        seed: seeds the subsample only, not the model.
    """

    def __init__(self, cfg, save_dir, max_faces=200, seed=1234):
        self.cfg = cfg
        self.save_dir = Path(save_dir)
        self.max_new_tokens = 9 * int(max_faces) + 1
        self.seed = seed

    def _run(self, trainer, pl_module, split, count, gt_ceiling, out_dir):
        dataset = getattr(trainer.datamodule, f"{split}_dataset", None)
        if dataset is None or len(dataset) == 0:
            return

        indices = eval_indices(len(dataset), count, self.seed)
        logger.info("[mesh-eval] %s: generating %d buildings (<=%d tokens each)",
                    split, len(indices), self.max_new_tokens)
        metrics = run_mesh_eval(pl_module, dataset, indices, self.cfg,
                                self.max_new_tokens, self.seed, out_dir, gt_ceiling)

        for name, value in metrics.items():
            # Through self.log, not the wandb experiment: this config logs to
            # tensorboard only, and the wandb path drops scalars silently.
            pl_module.log(f"{split}_gen_{name}" if not name.startswith("gt_")
                          else f"{split}_{name}", value)
        logger.info("[mesh-eval] %s: %s", split,
                    {k: round(v, 4) for k, v in sorted(metrics.items())})

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.cfg.enabled or trainer.sanity_checking:
            return
        every = max(int(self.cfg.every_n_epochs), 1)
        if trainer.current_epoch % every:
            return
        self._run(trainer, pl_module, "val", self.cfg.n_val, False, None)

    def on_test_epoch_end(self, trainer, pl_module):
        if not self.cfg.enabled:
            return
        out_dir = self.save_dir / "mesh_eval"
        self._run(trainer, pl_module, "test", self.cfg.n_test, True, out_dir)
