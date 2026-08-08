"""Free-running mesh eval: the standalone body, without a Lightning trainer.

An untrained model emits noise, so nothing here asserts a *good* score. What it
asserts is that the pipeline survives noise -- every generation that decodes to
nothing, fails to close, or fails to merge must produce a number or a nan, not
an exception, because the first real run will be full of exactly those.
"""
import numpy as np
import torch

from src.dataset.mesh_dataset import specials, vocab_size
from src.eval.mesh_eval import eval_indices, run_mesh_eval
from src.models.mesh_transformer import MeshTransformerModule

NUM_BINS = 32
BOS, EOS, PAD = specials(NUM_BINS)


class _Cfg:
    """Stand-in for MeshEvalConfig; the real one is an OmegaConf dataclass."""
    enabled = True
    every_n_epochs = 1
    n_val = n_test = 2
    batch_size = 2
    n_points = 256
    taus = [0.25]
    voxel_m = 0.5
    save_samples = 0


class _Dataset:
    """Two hand-built (cond, tgt) pairs -- one cube each, at two heights."""

    def __init__(self):
        self.ids = ["a", "b"]
        rng = np.random.default_rng(0)
        self.items = []
        for k in range(2):
            n_faces = 4 + k
            self.items.append({
                "cond": torch.from_numpy(rng.integers(0, NUM_BINS, 9 * n_faces)),
                "tgt": torch.cat([torch.tensor([BOS]),
                                  torch.from_numpy(rng.integers(0, NUM_BINS, 9 * n_faces)),
                                  torch.tensor([EOS])]),
                "id": self.ids[k],
                "center": torch.zeros(3),
                "scale": torch.ones(3) * 10.0,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def mesh_pair(self, i):
        """Raw metres pair, as MeshDataset provides for the GT ceiling."""
        verts = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]) * (i + 1)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        return (verts, faces), (verts, faces)


def _model():
    # Seeded: an untrained model's weights decide whether a generation decodes
    # to anything at all, so without this the assertions below depend on
    # whatever consumed the global RNG earlier in the session -- the test
    # passed alone and failed in a full run.
    torch.manual_seed(1)
    return MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2,
                                 num_layers=1, dropout=0.0, max_seq_len=128)


def test_eval_subsample_is_deterministic():
    """A fresh trainer.test must score the same buildings the val curve did.

    The indices are derived from the dataset length and the seed, never cached,
    precisely so nothing has to survive the checkpoint boundary.
    """
    assert np.array_equal(eval_indices(100, 8, seed=42), eval_indices(100, 8, seed=42))
    assert not np.array_equal(eval_indices(100, 8, seed=42),
                              eval_indices(100, 8, seed=7))
    assert len(eval_indices(3, 8, seed=42)) == 3        # count clamps to n
    assert len(eval_indices(0, 8, seed=42)) == 0


def test_run_mesh_eval_on_an_untrained_module():
    """Noise in, numbers out. This is the shape the Lightning adapter logs."""
    metrics = run_mesh_eval(_model(), _Dataset(), np.array([0, 1]), _Cfg(),
                            max_new_tokens=40, seed=1)

    for key in ("chamfer_m", "decode_rate", "eos_rate", "cityjson_rate",
                "watertight_rate"):
        assert key in metrics, key
        assert isinstance(metrics[key], float)
    for key in ("decode_rate", "eos_rate", "cityjson_rate", "watertight_rate"):
        assert 0.0 <= metrics[key] <= 1.0, key
    # vol_iou is legitimately absent when nothing generated was watertight.
    assert "watertight_gt" not in metrics and "watertight_gen" not in metrics


def test_gt_ceiling_metrics_are_reported_on_request():
    """The tokenizer floor, without which no model score can be interpreted."""
    metrics = run_mesh_eval(_model(), _Dataset(), np.array([0, 1]), _Cfg(),
                            max_new_tokens=40, seed=1, gt_ceiling=True)
    assert "gt_rt_chamfer_m" in metrics and metrics["gt_rt_chamfer_m"] >= 0.0
    assert "gt_watertight_rate" in metrics


def test_generation_budget_is_clamped_to_the_positional_limit():
    """max_seq_len is sized from the corpus, not from max_faces, so the two
    disagree whenever no building reaches the face cap -- and forward() raises
    rather than truncating."""
    model = MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2,
                                  num_layers=1, dropout=0.0, max_seq_len=48)
    metrics = run_mesh_eval(model, _Dataset(), np.array([0]), _Cfg(),
                            max_new_tokens=10_000, seed=1)
    assert metrics                      # ran to completion instead of raising


def test_empty_index_set_returns_no_metrics():
    """An empty split must not be an exception on the last epoch of a run."""
    assert run_mesh_eval(_model(), _Dataset(), np.zeros(0, dtype=int), _Cfg(),
                         max_new_tokens=10) == {}


def test_eval_restores_training_mode():
    """It runs from a validation hook mid-fit; leaving the model in eval()
    would silently disable dropout for the rest of training."""
    model = _model()
    model.train()
    run_mesh_eval(model, _Dataset(), np.array([0]), _Cfg(), max_new_tokens=20)
    assert model.training
