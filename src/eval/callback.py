import logging
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


def reference_features(datamodule, split, feature_set):
    """Test/train graphs -> the same converter -> feature matrix."""
    ds = getattr(datamodule, f"{split}_dataset")
    cjs = []
    for item in ds:
        if isinstance(item, tuple):
            item = item[0]
        coords = item["x"].numpy().astype(float)
        node_classes = item["node_categories"].argmax(-1).numpy()
        edge_classes = item["y"].squeeze(-1).numpy()
        cj = graph_to_cityjson(coords, node_classes, edge_classes)
        if cj:
            cjs.append(cj)
    return feature_matrix(cjs, feature_set)


def run_generative_eval(model, datamodule, cfg, loggers, save_dir):
    """Sample buildings end-to-end and score them. Returns a dict of scalar metrics.

    Standalone (checkpoint-callable) body; the callback is a thin Lightning adapter.
    Wires the validity, distribution, novelty and face-coherence arms together and
    saves/logs the results via `_save_and_log`. Keys are prefixed `gen/`.
    """
    L.seed_everything(cfg.seed)
    records, stats = draw_samples(model, cfg.num_batches, cfg.batch_size)

    metrics = {}
    n_invalid = 0
    if records:
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
        gen_X_raw, names = feature_matrix([r["cityjson"] for r in records], cfg.feature_set)
        gen_X = log1p_normalize(gen_X_raw)
        ref_X_raw, ref_names = reference_features(datamodule, "test", cfg.feature_set)

        if len(ref_X_raw) >= 2:
            ref_X = log1p_normalize(ref_X_raw)
            metrics.update(per_feature_wasserstein(gen_X, ref_X, names))
            metrics["gen/mmd"] = kernel_mmd(gen_X, ref_X)

            train_X_raw, _ = reference_features(datamodule, "train", cfg.feature_set)
            if len(train_X_raw) >= 2:
                metrics.update({f"gen/{k}": v for k, v in
                                novelty_uniqueness(gen_X, log1p_normalize(train_X_raw),
                                                    cfg.novelty_tol).items()})

    _save_and_log(records, metrics, loggers, save_dir, cfg)
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
    def __init__(self, cfg, save_dir):
        super().__init__()
        self.cfg = cfg
        self.save_dir = Path(save_dir)

    def on_test_end(self, trainer, pl_module):
        if not self.cfg.enabled:
            return
        out_dir = Path(self.cfg.save_dir) if self.cfg.save_dir else self.save_dir / "generative_eval"
        run_generative_eval(pl_module, trainer.datamodule, self.cfg, trainer.loggers, out_dir)
