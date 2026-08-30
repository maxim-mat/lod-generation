"""The do-nothing floor, logged beside every chamfer.

A chamfer in metres is uninterpretable on its own: 0.2 m is excellent on a
cathedral and worse than useless on a garden shed. `src/eval/lod1_baseline.py`
already answers "what does handing the LOD1 condition back unchanged score?"
over the whole corpus, but a corpus mean is the wrong reference for a curve
computed on a handful of validation buildings at a different point budget.

These tests pin the *matched* version: the same buildings, the same reference
mesh, the same number of sampled points, logged at the same steps so W&B draws
it as a flat line through the panel it belongs to.

The anchor in every case is the degenerate one -- when the LOD1 condition
already IS the target, the floor must be zero, because doing nothing is then a
perfect answer.
"""
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.dataset.mesh_dataset import specials, vocab_size
from src.models.mesh_transformer import MeshTransformerModule
from src.utils.config import Config

NUM_BINS = 32
V = vocab_size(NUM_BINS)
BOS, EOS, PAD = specials(NUM_BINS)

# Two non-degenerate triangles sharing an edge, as coordinate tokens: 9 per
# face, three bins per vertex. Hand-built rather than random because a
# degenerate triangle decodes to nothing and the metric is skipped, which would
# make these tests pass or fail on the RNG.
FACE_A = [0, 0, 0, 16, 0, 0, 0, 16, 0]
FACE_B = [0, 0, 0, 16, 0, 0, 0, 0, 16]
COND_TOKENS = FACE_A + FACE_B
# The same two faces lifted 8 bins in z, so the target is a different mesh a
# known distance away from the condition.
TGT_TOKENS = [t + 8 if i % 3 == 2 else t
              for i, t in enumerate(COND_TOKENS)]


def _model():
    torch.manual_seed(1)
    return MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2,
                                 num_layers=1, dropout=0.0, max_seq_len=128)


def _one_hot(tokens):
    """Logits whose argmax is exactly `tokens`, so the decode is controlled."""
    return torch.nn.functional.one_hot(tokens, V).float() * 30.0


def _tf_batch(cond_tokens):
    """The pieces `_log_tf_chamfer` reads, with scale in metres."""
    cond = torch.tensor([cond_tokens], dtype=torch.long)
    return {"cond": cond, "ids": ["a"],
            "center": torch.zeros(1, 3), "scale": torch.ones(1, 3) * 10.0}


def _log_tf(model, batch, targets, pred):
    """One `_log_tf_chamfer` call, returning what it logged."""
    logged = {}
    model.log = lambda name, value, **kw: logged.__setitem__(name, float(value))
    model._log_tf_chamfer(_one_hot(pred), targets, batch, "val", 0, stride=1)
    return logged


# ----------------------------------------------------------------------
# Teacher-forced chamfer (src/models/mesh_transformer.py)
# ----------------------------------------------------------------------

def test_tf_chamfer_logs_a_floor_beside_the_metric():
    """The reference line: the same key, the same step, one flat series."""
    targets = torch.tensor([TGT_TOKENS], dtype=torch.long)
    pred = torch.tensor([COND_TOKENS], dtype=torch.long)
    logged = _log_tf(_model(), _tf_batch(COND_TOKENS), targets, pred)

    assert "val_tf_chamfer_m" in logged
    assert "val_tf_chamfer_floor_m" in logged
    assert logged["val_tf_chamfer_floor_m"] >= 0.0


def test_tf_chamfer_floor_is_zero_when_the_condition_already_is_the_target():
    """Doing nothing is a perfect answer when LOD1 and LOD2 are the same mesh.

    The anchor for the whole feature: if this is not ~0 the floor is measuring
    something other than the identity prediction -- a wrong reference mesh, a
    frame that was not undone, or the condition decoded with the wrong
    tokenizer.
    """
    targets = torch.tensor([COND_TOKENS], dtype=torch.long)
    pred = torch.tensor([TGT_TOKENS], dtype=torch.long)
    logged = _log_tf(_model(), _tf_batch(COND_TOKENS), targets, pred)

    assert logged["val_tf_chamfer_floor_m"] == pytest.approx(0.0, abs=1e-6)


def test_tf_chamfer_ratio_is_the_metric_over_the_floor():
    """`_rel` is what makes the gap readable without touching the y-axis: 1.0
    is "no better than echoing LOD1", below 1.0 is the model earning its keep."""
    targets = torch.tensor([TGT_TOKENS], dtype=torch.long)
    pred = torch.tensor([COND_TOKENS], dtype=torch.long)
    logged = _log_tf(_model(), _tf_batch(COND_TOKENS), targets, pred)

    assert logged["val_tf_chamfer_floor_m"] > 0.0
    assert logged["val_tf_chamfer_rel"] == pytest.approx(
        logged["val_tf_chamfer_m"] / logged["val_tf_chamfer_floor_m"], rel=1e-6)
    # The prediction here IS the condition, so the model is exactly as good as
    # doing nothing and the ratio has to say so.
    assert logged["val_tf_chamfer_rel"] == pytest.approx(1.0, rel=1e-6)


def test_tf_chamfer_ratio_is_dropped_rather_than_infinite_on_a_zero_floor():
    """An identical LOD1/LOD2 pair divides by zero. A nan or an inf in the
    series poisons the epoch mean for every other building in the split."""
    targets = torch.tensor([COND_TOKENS], dtype=torch.long)
    pred = torch.tensor([TGT_TOKENS], dtype=torch.long)
    logged = _log_tf(_model(), _tf_batch(COND_TOKENS), targets, pred)

    assert "val_tf_chamfer_rel" not in logged


# ----------------------------------------------------------------------
# Free-running eval, autoregressive branch (src/eval/mesh_eval.py)
# ----------------------------------------------------------------------

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
    beam_size = 1
    length_penalty = 1.0


class _ArDataset:
    """Two buildings whose LOD1 condition differs from their LOD2 target."""

    ids = ["a", "b"]

    def __len__(self):
        return 2

    def __getitem__(self, i):
        body = torch.tensor(TGT_TOKENS, dtype=torch.long)
        return {"cond": torch.tensor(COND_TOKENS, dtype=torch.long),
                "tgt": torch.cat([torch.tensor([BOS]), body, torch.tensor([EOS])]),
                "id": self.ids[i],
                "center": torch.zeros(3), "scale": torch.ones(3) * 10.0}

    def mesh_pair(self, i):
        verts = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        return (verts, faces), (verts, faces)


def test_run_mesh_eval_reports_the_lod1_floor():
    """The free-running panel already carries the tokenizer ceiling; without a
    floor a run's position between the two is unknown."""
    from src.eval.mesh_eval import run_mesh_eval

    metrics = run_mesh_eval(_model(), _ArDataset(), np.array([0, 1]), _Cfg(),
                            max_new_tokens=40, seed=1)
    assert "chamfer_floor_m" in metrics
    assert np.isfinite(metrics["chamfer_floor_m"])
    assert metrics["chamfer_floor_m"] > 0.0
    assert "chamfer_rel" in metrics


def test_mesh_eval_floor_keys_survive_the_callback_prefix():
    """`MeshEvalCallback` prefixes everything that is not a `gt_` key, so the
    floor has to arrive on the dashboard as `val_gen_chamfer_floor_m`."""
    from src.eval.mesh_eval import run_mesh_eval

    metrics = run_mesh_eval(_model(), _ArDataset(), np.array([0]), _Cfg(),
                            max_new_tokens=40, seed=1)
    named = [f"val_gen_{k}" if not k.startswith("gt_") else f"val_{k}"
             for k in metrics]
    assert "val_gen_chamfer_floor_m" in named


# ----------------------------------------------------------------------
# Diffusion branch (src/eval/mesh_set_eval.py)
# ----------------------------------------------------------------------

def _diff_cfg(**kw):
    cfg = OmegaConf.structured(Config)
    cfg.config_set = "mesh_diffusion"
    cfg.mesh_data.num_bins = 128
    cfg.mesh_data.max_faces = 32
    cfg.mesh_diffusion.d_model = 32
    cfg.mesh_diffusion.time_dim = 32
    cfg.mesh_diffusion.n_head = 2
    cfg.mesh_diffusion.num_layers = 2
    cfg.mesh_diffusion.noise_steps = 100
    cfg.mesh_diffusion.eval_steps = 3
    cfg.mesh_diffusion.gen_eval_steps = 3
    cfg.mesh_diffusion.slot_budget = 32
    cfg.mesh_diffusion.n_points = 256
    for k, v in kw.items():
        OmegaConf.update(cfg, f"mesh_diffusion.{k}", v)
    return cfg


def _cube():
    v = np.array([[x, y, z] for x in (-0.4, 0.4) for y in (-0.4, 0.4)
                  for z in (-0.4, 0.4)], dtype=float)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 7, 5], [4, 6, 7],
                  [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v, f


class _IdentityDs:
    """LOD1 and LOD2 are the same cube, so the floor is exactly zero."""

    ids = ["cube0", "cube1"]

    def __init__(self):
        v, f = _cube()
        self.tri = v[f]
        self.pairs = [((v, f), (v, f))] * 2

    def __len__(self):
        return 2

    def mesh_pair(self, i):
        return self.pairs[i]

    def __getitem__(self, i):
        x = torch.zeros(12, 10)
        x[:, :9] = torch.from_numpy(self.tri.reshape(12, 9)).float()
        x[:, 9] = 0.5
        return {"x": x, "cond": x.clone(), "id": self.ids[i],
                "center": torch.zeros(3), "scale": torch.ones(3)}


class _OffsetDs(_IdentityDs):
    """LOD1 is the cube shifted in z, so the floor is a known positive number."""

    def __getitem__(self, i):
        item = super().__getitem__(i)
        cond = item["cond"].clone()
        cond[:, 2::3] += 0.05          # z of all three corners
        item["cond"] = cond
        return item


def test_run_mesh_set_eval_reports_the_lod1_floor():
    from src.eval.mesh_set_eval import run_mesh_set_eval
    from src.models.mesh_diffusion_module import MeshDiffusionModule

    cfg = _diff_cfg()
    m = MeshDiffusionModule(cfg).eval()
    out = run_mesh_set_eval(m, _OffsetDs(), [0, 1], cfg, seed=0)
    assert "chamfer_floor_m" in out and np.isfinite(out["chamfer_floor_m"])
    assert out["chamfer_floor_m"] > 0.0


def test_diffusion_floor_lands_on_the_ceiling_when_lod1_already_is_lod2():
    """Same anchor as the AR branch, but it is NOT zero here, and must not be.

    When the condition already is the target, doing nothing is the best this
    representation can do -- so the floor has to land exactly on the `gt_`
    ceiling, which is that same mesh through the same `faces_to_mesh` snap.
    The residual is the snap itself: both the floor and the generated mesh pass
    through the bin grid while the raw LOD2 they are scored against does not,
    and that gap is precisely what the ceiling row exists to report.

    A floor of zero here would mean the condition was being compared against
    something other than what the sample is compared against, which is the one
    way this metric can lie.
    """
    from src.eval.mesh_set_eval import run_mesh_set_eval
    from src.models.mesh_diffusion_module import MeshDiffusionModule

    cfg = _diff_cfg()
    m = MeshDiffusionModule(cfg).eval()
    out = run_mesh_set_eval(m, _IdentityDs(), [0, 1], cfg, seed=0)
    assert out["chamfer_floor_m"] == pytest.approx(out["gt_chamfer_m"], rel=1e-9)
    # Half a bin on a 128-bin grid over a unit box, and nothing more.
    assert out["chamfer_floor_m"] < 1.0 / 128


def test_generative_monitor_reports_the_lod1_floor():
    """Tier 2 is the early-stopping panel, so it is the one that most needs a
    reference: `val_gen_chamfer_m` selects the checkpoint."""
    from src.eval.mesh_set_eval import run_generative_monitor
    from src.models.mesh_diffusion_module import MeshDiffusionModule

    cfg = _diff_cfg()
    m = MeshDiffusionModule(cfg).eval()
    out = run_generative_monitor(m, _OffsetDs(), [0, 1], cfg, seed=0)
    assert "gen_chamfer_floor_m" in out and out["gen_chamfer_floor_m"] > 0.0
    assert "gen_chamfer_rel" in out


def test_generative_monitor_floor_is_cached_across_calls():
    """The floor depends on the data and the indices, never on the weights, so
    recomputing it every validation epoch -- twice, since EMA and live weights
    are both scored -- is pure waste on the hot path."""
    from src.eval.mesh_set_eval import run_generative_monitor
    from src.models.mesh_diffusion_module import MeshDiffusionModule

    cfg = _diff_cfg()
    m = MeshDiffusionModule(cfg).eval()
    floors = {}
    first = run_generative_monitor(m, _OffsetDs(), [0, 1], cfg, seed=0,
                                   floors=floors)
    assert set(floors) == {0, 1}

    # Poison the cache: a second call that recomputes would ignore this and
    # report the real floor again.
    floors[0] = floors[1] = 999.0
    second = run_generative_monitor(m, _OffsetDs(), [0, 1], cfg, seed=0,
                                    floors=floors)
    assert second["gen_chamfer_floor_m"] == pytest.approx(999.0)
    assert first["gen_chamfer_floor_m"] != pytest.approx(999.0)
