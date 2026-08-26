"""Free-running whole-mesh generation, scored against the paired LOD2.

A sibling of `src/eval/mesh_eval.py`, not a refactor of it: that module is
built around a token sequence and `decode_tokens`, and this branch produces a
face array. Everything downstream of "we have a mesh in metres" is shared --
`mesh_metrics`, the CityJSON write-back, the ground-truth ceiling.

The ceiling row is not optional. A chamfer of 0.04 m says nothing until you
know whether the tokenizer's own round-trip floor at this bin count is 0.01 or
0.15, and here there is a second floor on top of it: `faces_to_mesh` snaps to
the same grid, so the ground truth through the same snap is the best this
branch could possibly score.
"""
import logging
from pathlib import Path

import lightning as L
import numpy as np
import torch

from src.dataset.mesh_dataset import write_obj
from src.dataset.mesh_set_dataset import mesh_set_collate_fn
from src.eval.mesh_metrics import mesh_metrics
from src.models.mesh_set_postprocess import faces_to_mesh
from src.post_process.post_process import mesh_to_cityjson, save_to_file

logger = logging.getLogger(__name__)


def eval_indices(n, count, seed):
    """``count`` positions drawn without replacement, reproducibly."""
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n, size=min(count, n), replace=False).tolist())


def _aggregate(rows):
    """Mean of each key over rows, ignoring NaN, plus the sample count."""
    if not rows:
        return {}
    keys = {k for row in rows for k in row}
    out = {}
    for k in sorted(keys):
        vals = [row[k] for row in rows if k in row and np.isfinite(row[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    out["n_eval"] = float(len(rows))
    return out


def run_mesh_set_eval(model, dataset, indices, cfg, seed=1234, save_dir=None,
                      scaffold=None, gt_ceiling=True):
    """Sample a mesh per LOD1 condition and score it against its true LOD2.

    Args:
        model: a `MeshDiffusionModule`.
        dataset: `MeshSetDataset` or a `Subset` of one.
        indices: positions within `dataset` to score.
        cfg: the root `Config`; `mesh_diffusion` and `mesh_data` are read.
        seed: threaded into the index draw and torch's generator, so a rerun of
            the same checkpoint gives the same numbers.
        save_dir: where .obj and .city.json samples are written. None writes none.
        scaffold: optional ``(t, x) -> x`` projection passed to `generate`.
        gt_ceiling: also score the ground truth through `faces_to_mesh`, which
            is the floor every generated number must be read against.

    Returns:
        dict: metric name to float, averaged over the sampled buildings.
        Generated metrics are unprefixed; the ceiling is prefixed ``gt_``.
    """
    if len(indices) == 0:
        return {}
    d = cfg.mesh_diffusion
    num_bins = cfg.mesh_data.num_bins
    was_training = model.training
    model.eval()
    torch.manual_seed(seed)

    rows, gt_rows, written = [], [], 0
    for start in range(0, len(indices), d.eval_batch_size):
        chunk = indices[start:start + d.eval_batch_size]
        items = [dataset[int(i)] for i in chunk]
        batch = mesh_set_collate_fn(items, multiple_of=8)
        device = next(model.parameters()).device
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        with torch.no_grad():
            sampled = model.generate(batch, n_steps=d.eval_steps,
                                     scaffold=scaffold)

        for k, i in enumerate(chunk):
            centre = batch["center"][k].cpu().numpy()
            scale = batch["scale"][k].cpu().numpy()
            gen_v, gen_f, stats = faces_to_mesh(
                sampled[k], num_bins, snap=d.snap_before_weld,
                min_area=d.min_face_area)
            gen = (gen_v * scale + centre, gen_f)

            _, gt_raw = _resolve(dataset, int(i))
            row = {}
            if len(gen_f):
                row.update(mesh_metrics(gen, gt_raw, taus=list(d.taus),
                                        n_points=d.n_points, seed=seed,
                                        voxel_m=d.voxel_m))
            else:
                # An all-absent sample is a real failure mode of a presence
                # channel, and silently dropping it would flatter the average.
                row["chamfer_m"] = float("nan")
            row["n_faces"] = float(stats["n_faces"])
            row["n_verts"] = float(stats["n_verts"])
            row["empty"] = float(len(gen_f) == 0)
            for key in ("n_dropped_absent", "n_dropped_degenerate",
                        "n_dropped_duplicate"):
                row[key] = float(stats[key])
            rows.append(row)

            if gt_ceiling:
                ceil_v, ceil_f, _ = faces_to_mesh(
                    batch["x"][k], num_bins, snap=d.snap_before_weld,
                    min_area=d.min_face_area)
                ceil = (ceil_v * scale + centre, ceil_f)
                if len(ceil_f):
                    gt_rows.append(mesh_metrics(ceil, gt_raw, taus=list(d.taus),
                                                n_points=d.n_points, seed=seed,
                                                voxel_m=d.voxel_m))

            if save_dir is not None and written < d.save_samples and len(gen_f):
                out = Path(save_dir)
                out.mkdir(parents=True, exist_ok=True)
                name = batch["ids"][k]
                write_obj(out / f"{name}_gen.obj", *gen)
                write_obj(out / f"{name}_gt.obj", *gt_raw)
                try:
                    save_to_file(mesh_to_cityjson(*gen), out / f"{name}_gen.city.json")
                except Exception:
                    # Write-back is a convenience; a sample that cannot be
                    # expressed as CityJSON must not take the metrics down.
                    logger.warning("CityJSON write-back failed for %s", name,
                                   exc_info=True)
                written += 1

    if was_training:
        model.train()
    out = _aggregate(rows)
    out.update({f"gt_{k}": v for k, v in _aggregate(gt_rows).items()})
    return out


def _resolve(dataset, i):
    """``(index_in_base, raw_gt_mesh)`` through a `Subset` if there is one."""
    base, idx = dataset, i
    while hasattr(base, "dataset"):
        idx = base.indices[idx]
        base = base.dataset
    return idx, base.mesh_pair(idx)[1]


class MeshSetEvalCallback(L.Callback):
    """Runs `run_mesh_set_eval` every N epochs and logs the aggregate.

    Gated rather than folded into `validation_step` because the full reverse
    trajectory is `eval_steps` forward passes per building -- 50 by default,
    and 100 with classifier-free guidance on. That is affordable every fifth
    epoch on 16 buildings and unaffordable every epoch on the val split.
    """

    def __init__(self, cfg, save_dir, seed=1234):
        self.cfg = cfg
        self.d = cfg.mesh_diffusion
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.seed = seed
        self._scaffold = None      # populated in Task 14

    def _due(self, epoch):
        n = self.d.every_n_epochs
        return n > 0 and (epoch + 1) % n == 0

    def _run(self, trainer, pl_module, split, count, out_dir):
        dataset = getattr(trainer.datamodule, f"{split}_dataset")
        if dataset is None or len(dataset) == 0:
            return
        indices = eval_indices(len(dataset), count, self.seed)
        metrics = run_mesh_set_eval(
            pl_module, dataset, indices, self.cfg, seed=self.seed,
            save_dir=out_dir, scaffold=self._scaffold)
        if metrics:
            pl_module.log_dict({f"{split}_{k}": v for k, v in metrics.items()},
                               prog_bar=False, sync_dist=True)
            logger.info("%s mesh eval: %s", split,
                        {k: round(v, 4) for k, v in metrics.items()})

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not self._due(trainer.current_epoch):
            return
        self._run(trainer, pl_module, "val", self.d.n_val, None)

    def on_test_epoch_end(self, trainer, pl_module):
        out = self.save_dir / "samples" if self.save_dir else None
        self._run(trainer, pl_module, "test", self.d.n_test, out)
