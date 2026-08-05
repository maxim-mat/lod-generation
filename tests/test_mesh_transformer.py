"""Smoke checks for the LOD1-conditioned mesh transformer. CPU + random data only."""
import torch

from src.dataset.mesh_dataset import BOS, EOS, PAD, vocab_size
from src.models.mesh_transformer import MeshTransformer, MeshTransformerModule

NUM_BINS = 32           # tiny vocabulary; the tokenizer's bin count is a config knob
V = vocab_size(NUM_BINS)
B, LC, LT = 2, 18, 20   # 2 cond faces, 2 target faces + BOS + EOS


def _net(**kw):
    return MeshTransformer(vocab_size=V, d_model=16, n_head=2, num_layers=1,
                           dropout=0.0, max_seq_len=64, **kw)


def _batch():
    cond = torch.randint(0, NUM_BINS, (B, LC))
    body = torch.randint(0, NUM_BINS, (B, LT - 2))
    tgt = torch.cat([torch.full((B, 1), BOS), body, torch.full((B, 1), EOS)], dim=1)
    cond_pad = torch.zeros(B, LC, dtype=torch.bool)
    tgt_pad = torch.zeros(B, LT, dtype=torch.bool)
    # Second sample is shorter: pad its tail so the mask is actually exercised.
    cond[1, -4:], cond_pad[1, -4:] = PAD, True
    tgt[1, -3:], tgt_pad[1, -3:] = PAD, True
    return {"cond": cond, "tgt": tgt, "cond_pad_mask": cond_pad,
            "tgt_pad_mask": tgt_pad, "ids": ["a", "b"]}


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


def test_validation_step_logs_accuracy():
    model = MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2,
                                  num_layers=1, dropout=0.0, max_seq_len=64)
    logged = {}
    model.log = lambda name, value, **kw: logged.__setitem__(name, float(value))
    model.validation_step(_batch(), 0)
    assert 0.0 <= logged["val_token_acc"] <= 1.0
    assert logged["val_loss"] > 0
