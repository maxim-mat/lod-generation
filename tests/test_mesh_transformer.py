"""Smoke checks for the LOD1-conditioned mesh transformer. CPU + random data only."""
import torch

from src.dataset.mesh_dataset import specials, vocab_size
from src.models.mesh_transformer import MeshTransformer, MeshTransformerModule

NUM_BINS = 32           # tiny vocabulary; the tokenizer's bin count is a config knob
V = vocab_size(NUM_BINS)
BOS, EOS, PAD = specials(NUM_BINS)
B, LC, LT = 2, 18, 20   # 2 cond faces, 2 target faces + BOS + EOS


def _net(max_seq_len=64, **kw):
    return MeshTransformer(vocab_size=V, d_model=16, n_head=2, num_layers=1,
                           dropout=0.0, max_seq_len=max_seq_len, **kw)


def _batch():
    cond = torch.randint(0, NUM_BINS, (B, LC))
    body = torch.randint(0, NUM_BINS, (B, LT - 2))
    tgt = torch.cat([torch.full((B, 1), BOS), body, torch.full((B, 1), EOS)], dim=1)
    cond_pad = torch.zeros(B, LC, dtype=torch.bool)
    tgt_pad = torch.zeros(B, LT, dtype=torch.bool)
    # Second sample is shorter: pad its tail so the mask is actually exercised.
    cond[1, -4:], cond_pad[1, -4:] = PAD, True
    tgt[1, -3:], tgt_pad[1, -3:] = PAD, True
    # center/scale are what mesh_collate_fn really emits, and the metres-valued
    # metrics are silently skipped without them.
    return {"cond": cond, "tgt": tgt, "cond_pad_mask": cond_pad,
            "tgt_pad_mask": tgt_pad, "ids": ["a", "b"],
            "center": torch.zeros(B, 3), "scale": torch.ones(B, 3) * 10.0}


def test_forward_logit_shape():
    """Logits predict tgt[:, 1:], so one position shorter than the target."""
    logits = _net()(**{k: _batch()[k] for k in
                       ("cond", "tgt", "cond_pad_mask", "tgt_pad_mask")})
    assert logits.shape == (B, LT - 1, V)
    assert torch.isfinite(logits).all()


def test_forward_is_causal():
    """Changing a late target token must not move an earlier position's logits."""
    net = _net().eval()
    batch = _batch()
    args = {k: batch[k] for k in ("cond", "tgt", "cond_pad_mask", "tgt_pad_mask")}
    with torch.no_grad():
        base = net(**args)
        args["tgt"] = args["tgt"].clone()
        args["tgt"][0, -2] = (args["tgt"][0, -2] + 1) % NUM_BINS
        moved = net(**args)
    assert torch.allclose(base[0, :-3], moved[0, :-3], atol=1e-5)


def test_target_positions_do_not_move_with_condition_padding():
    """The same building must get the same logits whatever else is in its batch.

    Target positions restart at 0, so widening the condition (as a batch
    containing one long LOD1 does) cannot shift the target's position
    embeddings -- at sampling time the condition is unpadded, and a shift here
    would be a train/inference mismatch.
    """
    net = _net().eval()
    batch = _batch()
    args = {k: batch[k] for k in ("cond", "tgt", "cond_pad_mask", "tgt_pad_mask")}

    wider = {**args}
    wider["cond"] = torch.cat([args["cond"], torch.full((B, 7), PAD)], dim=1)
    wider["cond_pad_mask"] = torch.cat(
        [args["cond_pad_mask"], torch.ones(B, 7, dtype=torch.bool)], dim=1)

    with torch.no_grad():
        assert torch.allclose(net(**args), net(**wider), atol=1e-5)


def test_segment_longer_than_max_seq_len_raises():
    """The guard is per segment; a sum-based one would fire on legal batches."""
    net = _net(max_seq_len=LC + 4)          # fits each segment, not their sum
    args = {k: _batch()[k] for k in ("cond", "tgt", "cond_pad_mask", "tgt_pad_mask")}
    with torch.no_grad():
        assert torch.isfinite(net(**args)).all()

    import pytest
    with pytest.raises(ValueError, match="max_seq_len"):
        _net(max_seq_len=LC - 1)(**args)


def test_training_step_returns_finite_scalar():
    model = MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2,
                                  num_layers=1, dropout=0.0, max_seq_len=64)
    model.log = lambda *a, **kw: None
    loss = model.training_step(_batch(), 0)
    assert loss.ndim == 0 and torch.isfinite(loss)

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_padded_targets_are_excluded_from_the_loss():
    """PAD is ignore_index; otherwise short buildings train the model to emit padding."""
    model = MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2,
                                  num_layers=1, dropout=0.0, max_seq_len=64)
    model.log = lambda *a, **kw: None
    torch.manual_seed(0)
    batch = _batch()
    ref = model._shared_step(batch)[0]

    scrambled = {**batch, "tgt": batch["tgt"].clone()}
    scrambled["tgt"][1, -2:] = 0     # rewrite padded slots to a real token id
    torch.manual_seed(0)
    assert torch.isclose(model._shared_step(scrambled)[0], ref)


def test_mesh_config_set_resolves():
    """configs/mesh-train.yaml must survive the structured-schema merge.

    Guards the trap that MeshDataConfig introduces: OmegaConf resolves the
    *whole* Config, so a new MISSING field or an unfilled diffusion field breaks
    every config file, not just the new one.
    """
    from pathlib import Path

    from src.utils.initialization import load_config

    cfg = load_config(Path("configs/mesh-train.yaml"), [])
    assert cfg.config_set == "mesh"
    assert cfg.mesh_data.dataset_dir and cfg.mesh_data.num_bins == 128

    # ...and the diffusion configs keep resolving with the new blocks defaulted.
    levi = load_config(Path("configs/levi1-train.yaml"), [])
    assert levi.config_set == "diffusion" and levi.mesh_data.dataset_dir is None


def test_validation_step_logs_accuracy():
    model = MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2,
                                  num_layers=1, dropout=0.0, max_seq_len=64)
    logged = {}
    model.log = lambda name, value, **kw: logged.__setitem__(name, float(value))
    model.validation_step(_batch(), 0)
    assert 0.0 <= logged["val_token_acc"] <= 1.0
    assert logged["val_loss"] > 0


# ----------------------------------------------------------------------
# Token metrics
# ----------------------------------------------------------------------

def _module():
    return MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2,
                                 num_layers=1, dropout=0.0, max_seq_len=64)


def _one_hot(targets, vocab=V):
    """Logits that predict `targets` exactly."""
    return torch.nn.functional.one_hot(targets.clamp(min=0), vocab).float() * 20.0


def test_token_metrics_are_perfect_on_one_hot_logits():
    """The only way to check the arithmetic without a trained model."""
    model = _module()
    targets = _batch()["tgt"][:, 1:]
    m = model._token_metrics(_one_hot(targets), targets,
                             scale=torch.ones(B, 3) * 10.0)

    assert float(m["token_acc"]) == 1.0
    assert float(m["bin_mae"]) == 0.0
    assert float(m["acc_1bin"]) == 1.0
    assert float(m["coord_mae_m"]) == 0.0
    assert float(m["eos_acc"]) == 1.0
    assert float(m["eos_fp_rate"]) == 0.0


def test_eos_accuracy_sees_the_single_eos_position():
    """Exactly the blindness this metric set exists for.

    One EOS in a long sequence is invisible in aggregate accuracy: get it
    wrong and every other token right, and token_acc barely moves while the
    model has lost the ability to ever stop generating.
    """
    model = _module()
    targets = _batch()["tgt"][:, 1:]
    logits = _one_hot(targets)
    # Break only the EOS positions, predicting bin 0 there instead.
    at_eos = targets == EOS
    logits[at_eos] = 0.0
    logits[at_eos, 0] = 20.0

    m = model._token_metrics(logits, targets)
    assert float(m["eos_acc"]) == 0.0
    assert float(m["token_acc"]) > 0.9        # aggregate accuracy shrugs


def test_bin_mae_separates_off_by_one_from_catastrophic():
    """token_acc scores a 1-bin miss and a 20-bin miss identically; the whole
    point of bin_mae is that it does not.

    Targets are all one mid-grid value so neither perturbation wraps around
    the vocabulary and turns a small error into a huge one.
    """
    model = _module()
    targets = torch.full((B, 9), 5, dtype=torch.long)

    m_near = model._token_metrics(_one_hot(targets + 1), targets)
    m_far = model._token_metrics(_one_hot(targets + 20), targets)

    assert float(m_near["token_acc"]) == float(m_far["token_acc"]) == 0.0
    assert float(m_near["bin_mae"]) == 1.0
    assert float(m_far["bin_mae"]) == 20.0
    assert float(m_near["acc_1bin"]) == 1.0 and float(m_far["acc_1bin"]) == 0.0


def test_coord_mae_converts_bins_to_metres_per_axis():
    """A bin buys a different distance on each axis once the margins make the
    scale anisotropic, so the conversion has to be per axis, not per token."""
    model = _module()
    targets = torch.full((B, 9), 5, dtype=torch.long)   # 3 vertices, 3 axes each
    pred = targets + 2

    # 1 m per bin on x, 0.5 on y, nothing on z.
    scale = torch.tensor([[1.0, 0.5, 0.0]] * B) * (NUM_BINS - 1)
    m = model._token_metrics(_one_hot(pred), targets, scale=scale)

    assert float(m["bin_mae_x"]) == float(m["bin_mae_y"]) == 2.0
    # Positions split evenly across the three axes: (2*1 + 2*0.5 + 0) / 3.
    assert abs(float(m["coord_mae_m"]) - 1.0) < 1e-5


def test_eval_step_logs_the_expected_metric_names():
    """Pins the contract the config's monitor= and the dashboards rely on."""
    model = _module()
    logged = {}
    model.log = lambda name, value, **kw: logged.__setitem__(name, float(value))
    model.validation_step(_batch(), 0)

    assert {"val_loss", "val_ppl", "val_token_acc", "val_bin_mae", "val_acc_1bin",
            "val_bin_mae_x", "val_bin_mae_y", "val_bin_mae_z", "val_coord_mae_m",
            "val_eos_acc", "val_eos_fp_rate"} <= set(logged)
    # Free-running metrics belong to the callback, never to the step.
    assert not [k for k in logged if k.startswith("val_gen_")]
