import logging
import time
from contextlib import contextmanager
from pathlib import Path

import lightning as L
import numpy as np

from src.dataset.dataset import EDGE_VF, OFF, VERTEX
from src.eval.sampling import draw_samples
from src.eval.building_features import building_features, feature_matrix
from src.eval.distribution import log1p_normalize, per_feature_wasserstein, kernel_mmd
from src.eval.novelty import novelty_uniqueness
from src.eval.validity import check_validity
from src.post_process.post_process import graph_to_cityjson

logger = logging.getLogger(__name__)


@contextmanager
def _stage(label, *args):
    """Log a generative-eval stage entry/exit with its wall-clock cost.

    The arms differ by orders of magnitude (sampling is GPU-minutes, the
    reference conversion is CPU-minutes, val3dity shells out), so a stalled
    run is unreadable without knowing which one is running.
    """
    name = label % args if args else label
    logger.info("[gen-eval] %s ...", name)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.info("[gen-eval] %s done in %.1fs", name, time.perf_counter() - t0)


def face_centroid_consistency(coords, node_labels, edge_labels):
    """Mean distance between each face node and the centroid of its member vertices.

    Face position and its vertices diffuse independently and the converter ignores
    the face row, so this drift is an internal-coherence signal the converter can't see.
    """
    coords = np.asarray(coords, dtype=float)
    node_labels = np.asarray(node_labels)
    face_ids = np.flatnonzero((node_labels != VERTEX) & (node_labels != OFF))
    drifts = []
    for f in face_ids:
        members = np.flatnonzero((node_labels == VERTEX) & (edge_labels[f] == EDGE_VF))
        if len(members) >= 1:
            drifts.append(np.linalg.norm(coords[f] - coords[members].mean(axis=0)))
    return float(np.mean(drifts)) if drifts else 0.0


def reference_features(datamodule, split, feature_set, max_samples=None, seed=1234):
    """Test/train graphs -> the same converter -> feature matrix.

    Subsampled to `max_samples` because the splits hold O(1e5) buildings: the
    conversion is minutes of Python per split and `kernel_mmd` is O(n^2) in the
    reference count. The draw is seeded so the reference distribution is the
    same set across runs and the metrics stay comparable.
    """
    ds = getattr(datamodule, f"{split}_dataset")
    idx = range(len(ds))
    if max_samples is not None and len(ds) > max_samples:
        idx = np.sort(np.random.default_rng(seed).choice(
            len(ds), max_samples, replace=False))
    logger.info("Reference features: converting %d/%d %s graphs.",
                len(idx), len(ds), split)

    cjs = []
    for i in idx:
        item = ds[int(i)]
        if isinstance(item, tuple):
            item = item[0]
        coords = item["x"].numpy().astype(float)
        node_classes = item["node_categories"].argmax(-1).numpy()
        edge_classes = item["y"].squeeze(-1).numpy()
        cj = graph_to_cityjson(coords, node_classes, edge_classes)
        if cj:
            cjs.append(cj)
    logger.info("Reference features: %d/%d %s graphs converted cleanly.",
                len(cjs), len(idx), split)
    return feature_matrix(cjs, feature_set)


def run_generative_eval(model, datamodule, cfg, loggers, save_dir, seed=1234):
    """Sample buildings end-to-end and score them. Returns a dict of scalar metrics.

    Standalone (checkpoint-callable) body; the callback is a thin Lightning adapter.
    Wires the validity, distribution, novelty and face-coherence arms together and
    saves/logs the results via `_save_and_log`. Keys are prefixed `gen/`.

    `seed` is re-applied here so sampling starts from a known RNG state: training has
    consumed an arbitrary amount of randomness by the time the callback fires, so
    without it the sampled buildings are not comparable across runs.
    """
    L.seed_everything(seed)
    with _stage("sampling (%d x %d through the reverse chain)",
                cfg.num_batches, cfg.batch_size):
        records, stats = draw_samples(model, cfg.num_batches, cfg.batch_size)
    logger.info("Sampled %d/%d reconstructable buildings (%d dropped).",
                len(records), stats["attempted"], stats["dropped"])

    metrics = {}
    n_invalid = 0
    if records:
        with _stage("val3dity validity"):
            val = check_validity([r["cityjson"] for r in records], cfg.val3dity_path)
        if val is not None:
            metrics["gen/valid_fraction"] = val["valid_fraction"]
            n_invalid = sum(1 for v in val["valid_flags"] if not v)
            for code, count in val["error_histogram"].items():
                metrics[f"gen/err/{code}"] = count

    attempted = max(stats["attempted"], 1)
    metrics["gen/rejection_rate"] = (stats["dropped"] + n_invalid) / attempted
    metrics["gen/dropped"] = stats["dropped"]
    metrics["gen/face_centroid_consistency"] = float(np.mean([
        face_centroid_consistency(r["coords"], r["node_labels"], r["edge_labels"])
        for r in records])) if records else 0.0

    # Wasserstein/MMD/novelty need >=2 samples on each side (variance-based
    # stats divide by n-1 / pairwise distances collapse with a single point).
    # Skip the arm rather than crash when sampling yields too few buildings.
    if records and datamodule is not None and len(records) >= 2:
        with _stage("generated features (%d buildings)", len(records)):
            gen_X_raw, names = feature_matrix([r["cityjson"] for r in records], cfg.feature_set)
        gen_X = log1p_normalize(gen_X_raw)
        with _stage("test reference features"):
            ref_X_raw, ref_names = reference_features(
                datamodule, "test", cfg.feature_set, cfg.ref_max_samples, seed)

        if len(ref_X_raw) >= 2:
            ref_X = log1p_normalize(ref_X_raw)
            with _stage("wasserstein + mmd (gen=%d, ref=%d)", len(gen_X), len(ref_X)):
                metrics.update(per_feature_wasserstein(gen_X, ref_X, names))
                metrics["gen/mmd"] = kernel_mmd(gen_X, ref_X)

            with _stage("train reference features"):
                train_X_raw, _ = reference_features(
                    datamodule, "train", cfg.feature_set, cfg.ref_max_samples, seed)
            if len(train_X_raw) >= 2:
                with _stage("novelty + uniqueness"):
                    metrics.update({f"gen/{k}": v for k, v in
                                    novelty_uniqueness(gen_X, log1p_normalize(train_X_raw),
                                                        cfg.novelty_tol).items()})

    with _stage("saving records + logging"):
        _save_and_log(records, metrics, loggers, save_dir, cfg)
    logger.info("[gen-eval] finished with %d metrics.", len(metrics))
    return metrics


def _save_and_log(records, metrics, loggers, save_dir, cfg):
    from src.eval.geometry_io import save_records, cityjson_to_obj
    save_dir = Path(save_dir)
    save_records(records, save_dir, cfg.log_n_samples)

    wandb_logger = next((lg for lg in (loggers or [])
                         if type(lg).__name__ == "WandbLogger"), None)
    if wandb_logger is None:
        logger.info("No WandB logger; wrote scalars + files locally only.")
        return
    import wandb
    exp = wandb_logger.experiment
    exp.log(metrics)

    cols = ["sample_id", "num_vertices", "num_faces", "mesh"]
    table = wandb.Table(columns=cols)
    for i, rec in enumerate(records[:cfg.log_n_samples]):
        obj = wandb.Object3D(_io_from_obj(cityjson_to_obj(rec["cityjson"])))
        n_v = int((rec["node_labels"] == VERTEX).sum())
        city_objects = rec["cityjson"]["CityObjects"]
        n_f = len(next(iter(city_objects.values()))["geometry"][0]["boundaries"][0]) if city_objects else 0
        table.add_data(f"gen_{i}", n_v, n_f, obj)
    exp.log({"gen/samples": table})

    art = wandb.Artifact(f"generative_eval_{exp.id}", type="generated_buildings")
    art.add_dir(str(save_dir))
    exp.log_artifact(art)


def _io_from_obj(obj_str):
    import io
    buf = io.StringIO(obj_str)
    buf.name = "sample.obj"  # wandb.Object3D infers format from the name
    return buf


class GenerativeEvalCallback(L.Callback):
    def __init__(self, cfg, save_dir, seed=1234):
        super().__init__()
        self.cfg = cfg
        self.save_dir = Path(save_dir)
        self.seed = seed

    def on_test_end(self, trainer, pl_module):
        if not self.cfg.enabled:
            return
        out_dir = Path(self.cfg.save_dir) if self.cfg.save_dir else self.save_dir / "generative_eval"
        run_generative_eval(pl_module, trainer.datamodule, self.cfg, trainer.loggers,
                            out_dir, self.seed)
