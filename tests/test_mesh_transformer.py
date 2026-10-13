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


# ----------------------------------------------------------------------
# Sinusoidal positions (mesh_model.pos_embed)
# ----------------------------------------------------------------------

def _tiny_opt_cfg(max_position_embeddings=8):
    """A local OPT config so the test never needs the hub."""
    from transformers import OPTConfig
    return OPTConfig(vocab_size=V, hidden_size=32, word_embed_proj_dim=16,
                     num_hidden_layers=1, num_attention_heads=4, ffn_dim=64,
                     max_position_embeddings=max_position_embeddings, dropout=0.0)


def _opt_net(pos_embed="learned", max_position_embeddings=8):
    from src.models.mesh_transformer import MeshOPTTransformer
    return MeshOPTTransformer(
        vocab_size=V, opt_config=_tiny_opt_cfg(max_position_embeddings),
        dropout=0.0, pos_embed=pos_embed).eval()


def test_sinusoidal_positions_are_fixed_not_learned():
    """The point of the option is a closed-form position, so it must carry no
    parameters -- otherwise it is just a learned table with extra steps."""
    from src.models.embeddings import SinusoidalPositionEmbedding

    emb = SinusoidalPositionEmbedding(16)
    assert list(emb.parameters()) == []
    out = emb(torch.arange(5))
    assert out.shape == (5, 16)
    # sin(0) = 0, cos(0) = 1 across the whole bank, as in Vaswani et al. §3.5.
    assert torch.allclose(out[0, :8], torch.zeros(8), atol=1e-6)
    assert torch.allclose(out[0, 8:], torch.ones(8), atol=1e-6)
    # Same input, same output: nothing stateful in there.
    assert torch.equal(out, emb(torch.arange(5)))


def test_sinusoidal_scratch_accepts_sequences_past_max_seq_len():
    """A learned table is capped by its row count; the sinusoidal one is not.
    Removing the ceiling rather than raising it is the whole point."""
    import pytest

    args = {k: _batch()[k] for k in ("cond", "tgt", "cond_pad_mask", "tgt_pad_mask")}
    with pytest.raises(ValueError, match="max_seq_len"):
        _net(max_seq_len=8)(**args)

    net = _net(max_seq_len=8, pos_embed="sinusoidal").eval()
    with torch.no_grad():
        logits = net(**args)
    assert logits.shape == (B, LT - 1, V)
    assert torch.isfinite(logits).all()


def test_sinusoidal_scratch_target_positions_do_not_move_with_condition_padding():
    """Same invariant as the learned path: widening the condition with padding
    must not shift the target's positions."""
    net = _net(max_seq_len=8, pos_embed="sinusoidal").eval()
    args = {k: _batch()[k] for k in ("cond", "tgt", "cond_pad_mask", "tgt_pad_mask")}

    wider = {**args}
    wider["cond"] = torch.cat([args["cond"], torch.full((B, 7), PAD)], dim=1)
    wider["cond_pad_mask"] = torch.cat(
        [args["cond_pad_mask"], torch.ones(B, 7, dtype=torch.bool)], dim=1)

    with torch.no_grad():
        assert torch.allclose(net(**args), net(**wider), atol=1e-5)


def test_opt_sinusoidal_accepts_sequences_past_max_position_embeddings():
    """OPT's ceiling lives entirely in `decoder.embed_positions`; swapping that
    module out removes it, so the shared-budget guard must stop firing too."""
    import pytest

    args = {k: _batch()[k] for k in ("cond", "tgt", "cond_pad_mask", "tgt_pad_mask")}
    with pytest.raises(ValueError, match="max_position_embeddings"):
        _opt_net()(**args)

    net = _opt_net(pos_embed="sinusoidal")
    with torch.no_grad():
        logits = net(**args)
    assert logits.shape == (B, LT - 1, V)
    assert torch.isfinite(logits).all()


def test_opt_sinusoidal_keeps_the_mask_derived_positions():
    """OPT derives position ids by cumsum over the attention mask, so padding
    never shifts a real token. A naive `arange` replacement would silently break
    that -- this is the regression it would cause.
    """
    net = _opt_net(pos_embed="sinusoidal", max_position_embeddings=128)
    args = {k: _batch()[k] for k in ("cond", "tgt", "cond_pad_mask", "tgt_pad_mask")}

    wider = {**args}
    wider["cond"] = torch.cat([args["cond"], torch.full((B, 7), PAD)], dim=1)
    wider["cond_pad_mask"] = torch.cat(
        [args["cond_pad_mask"], torch.ones(B, 7, dtype=torch.bool)], dim=1)

    with torch.no_grad():
        assert torch.allclose(net(**args), net(**wider), atol=1e-5)


def test_opt_sinusoidal_replaces_opts_own_table():
    """Belt and braces: the learned table must actually be gone, not merely
    unused, or the ceiling would still be there on the next `from_config`."""
    from transformers.models.opt.modeling_opt import OPTLearnedPositionalEmbedding

    learned = _opt_net().opt.model.decoder.embed_positions
    assert isinstance(learned, OPTLearnedPositionalEmbedding)

    swapped = _opt_net(pos_embed="sinusoidal").opt.model.decoder.embed_positions
    assert not isinstance(swapped, OPTLearnedPositionalEmbedding)
    assert list(swapped.parameters()) == []


def test_unknown_pos_embed_is_rejected():
    """A typo must not silently fall back to the learned table."""
    import pytest
    with pytest.raises(ValueError, match="pos_embed"):
        MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2,
                              num_layers=1, max_seq_len=64, pos_embed="rope")


def test_module_threads_pos_embed_to_both_backbones():
    """The config knob has to reach the network, not just be stored."""
    scratch = MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2,
                                    num_layers=1, dropout=0.0, max_seq_len=8,
                                    pos_embed="sinusoidal")
    args = {k: _batch()[k] for k in ("cond", "tgt", "cond_pad_mask", "tgt_pad_mask")}
    with torch.no_grad():
        assert torch.isfinite(scratch.network(**args)).all()

    opt = MeshTransformerModule(num_bins=NUM_BINS, dropout=0.0, backbone="opt",
                                opt_config=_tiny_opt_cfg(), pos_embed="sinusoidal")
    with torch.no_grad():
        assert torch.isfinite(opt.network(**args)).all()


# ----------------------------------------------------------------------
# Generation budget: what the position scheme lets autoregression reach
# ----------------------------------------------------------------------

def test_scratch_generation_budget_ignores_the_condition():
    """Positions restart per segment on this backbone, so a longer condition
    costs the target nothing."""
    net = _net(max_seq_len=64)
    assert net.generation_budget(0) == 63
    assert net.generation_budget(50) == 63


def test_opt_generation_budget_leaves_room_for_the_condition():
    """OPT's positions run through both halves, so the condition eats into the
    budget. Defaulting to `max_seq_len - 1` overruns it and raises partway
    through sampling -- silently truncating the mesh at whatever token the
    exception landed on.
    """
    net = _opt_net(max_position_embeddings=32)
    assert net.generation_budget(18) == 32 - 18

    module = MeshTransformerModule(num_bins=NUM_BINS, dropout=0.0, backbone="opt",
                                   opt_config=_tiny_opt_cfg(32)).eval()
    cond = torch.randint(0, NUM_BINS, (1, 18))
    with torch.no_grad():
        out = module.generate(cond, max_new_tokens=None, temperature=0.0)
    assert out.shape[1] <= 32 - 18 + 1        # +1 for the BOS the loop starts from


def test_sinusoidal_generation_budget_is_not_capped_by_the_dead_table():
    """The whole point: lifting the ceiling has to reach autoregression too.
    With sinusoidal positions OPT's `max_position_embeddings` is inert, so the
    budget must come from the data-derived `max_seq_len` instead."""
    from src.models.mesh_transformer import MeshOPTTransformer

    net = MeshOPTTransformer(vocab_size=V, opt_config=_tiny_opt_cfg(8),
                             dropout=0.0, pos_embed="sinusoidal",
                             max_seq_len=48).eval()
    assert net.generation_budget(18) == 47, "budget still tied to the dead table"

    cond = torch.randint(0, NUM_BINS, (1, 18))
    module = MeshTransformerModule(num_bins=NUM_BINS, dropout=0.0, backbone="opt",
                                   opt_config=_tiny_opt_cfg(8), max_seq_len=48,
                                   pos_embed="sinusoidal").eval()
    with torch.no_grad():
        out = module.generate(cond, max_new_tokens=20, temperature=0.0)
    # 20 new tokens past an 18-token condition is 38 positions -- more than the
    # learned table would have held, and it must not be clamped down to it.
    assert out.shape[1] > 8


def test_generate_clamps_a_request_past_the_budget():
    """`mesh_eval` asks for `9 * max_faces + 1`, which may exceed what the
    positions hold. One clamp, in the one place that knows both numbers."""
    module = MeshTransformerModule(num_bins=NUM_BINS, dropout=0.0, backbone="opt",
                                   opt_config=_tiny_opt_cfg(32)).eval()
    cond = torch.randint(0, NUM_BINS, (1, 18))
    with torch.no_grad():
        out = module.generate(cond, max_new_tokens=9999, temperature=0.0)
    assert out.shape[1] <= 32 - 18 + 1


def test_condition_that_fills_the_budget_is_rejected_not_silently_empty():
    """A condition at or past the ceiling leaves no room to generate. Returning
    a bare BOS would look like a model that stopped immediately."""
    import pytest
    module = MeshTransformerModule(num_bins=NUM_BINS, dropout=0.0, backbone="opt",
                                   opt_config=_tiny_opt_cfg(8)).eval()
    with torch.no_grad(), pytest.raises(ValueError, match="no room"):
        module.generate(torch.randint(0, NUM_BINS, (1, 18)), temperature=0.0)


# ----------------------------------------------------------------------
# Warmup + cosine (training.warmup_steps)
# ----------------------------------------------------------------------

def _sched_module(**kw):
    return MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2,
                                 num_layers=1, dropout=0.0, max_seq_len=64, **kw)


def _lrs(out, n):
    """The lr this config would use over its first ``n`` optimizer steps."""
    opt = out["optimizer"]
    sched = out["lr_scheduler"]["scheduler"]
    seen = []
    for _ in range(n):
        seen.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return seen


def test_no_warmup_and_no_schedule_still_returns_a_bare_optimizer():
    """The v3 arms ran this way; the new knob must not change them."""
    import torch as _t
    assert isinstance(_sched_module(lr_scheduler="none").configure_optimizers(),
                      _t.optim.Optimizer)


def test_warmup_ramps_on_the_step_interval_not_the_epoch_one():
    """Left on Lightning's default `interval: epoch`, a 50-step warmup ramps
    over 50 EPOCHS -- tens of thousands of steps of ramp. It trains, badly and
    silently, which is the whole reason this is asserted."""
    out = _sched_module(lr=1e-3, lr_scheduler="cosine",
                        warmup_steps=50).configure_optimizers()
    assert isinstance(out, dict)
    assert out["lr_scheduler"]["interval"] == "step"

    lrs = _lrs(out, 60)
    assert lrs[0] < 1e-4, "warmup must start far below the peak"
    assert lrs[:50] == sorted(lrs[:50]), "warmup must be monotonically rising"
    # Peak is the configured lr, reached at the end of the ramp.
    assert abs(max(lrs) - 1e-3) < 1e-5


def test_cosine_anneals_down_from_the_peak_after_the_ramp():
    """Stepped over the WHOLE horizon, which is 1000 when the module is
    detached from a trainer -- annealing is only complete at T_max, so a
    partial walk reads as "no anneal" and says nothing."""
    out = _sched_module(lr=1e-3, lr_scheduler="cosine",
                        warmup_steps=20).configure_optimizers()
    lrs = _lrs(out, 1000)
    peak = max(lrs)
    assert lrs.index(peak) <= 21, "peak must be at the end of warmup, not later"
    assert lrs[-1] < 0.05 * peak, "cosine must anneal to near eta_min"
    # eta_min is 1% of the peak, not zero: a schedule that reaches exactly 0
    # stops training rather than finishing it.
    assert lrs[-1] > 0


def test_warmup_with_a_step_schedule_is_rejected_rather_than_silently_wrong():
    """A step-interval ramp composed with an epoch-interval decay is not a
    schedule anyone chose; refuse it instead of running it."""
    import pytest as _p
    with _p.raises(ValueError, match="warmup"):
        _sched_module(lr_scheduler="step", warmup_steps=50).configure_optimizers()


def test_warmup_is_saved_as_a_hyperparameter():
    """It has to survive the checkpoint, or a resumed run silently re-ramps."""
    m = _sched_module(lr_scheduler="cosine", warmup_steps=123)
    assert m.hparams.warmup_steps == 123


# ----------------------------------------------------------------------
# Weight decay grouping (training.weight_decay)
# ----------------------------------------------------------------------

def _groups(m):
    """(decayed, undecayed) param-id sets from configure_optimizers."""
    out = m.configure_optimizers()
    opt = out["optimizer"] if isinstance(out, dict) else out
    decayed, undecayed = set(), set()
    for g in opt.param_groups:
        (decayed if g["weight_decay"] > 0 else undecayed).update(id(p) for p in g["params"])
    return decayed, undecayed


def test_weight_decay_defaults_to_adamws_own_default():
    """0.01, so leaving it unset reproduces what every run so far actually used."""
    m = _sched_module()
    assert m.hparams.weight_decay == 0.01


def test_embeddings_norms_and_biases_are_excluded_from_decay():
    """Decay shrinks toward zero, which is only a sensible prior for a weight
    matrix. A LayerNorm gain's meaningful default is 1, a bias carries one
    parameter per unit, and an embedding row here is a hot lookup in a
    131-symbol vocabulary with no cold rows to regularize."""
    import torch.nn as nn
    m = _sched_module(lr_scheduler="none", weight_decay=0.01)
    decayed, undecayed = _groups(m)

    excluded = set()
    for mod in m.modules():
        if isinstance(mod, (nn.Embedding, nn.LayerNorm)):
            excluded |= {id(p) for p in mod.parameters(recurse=False)}
    for p in m.parameters():
        if p.dim() < 2:                      # every bias, and every norm gain
            excluded.add(id(p))

    assert excluded, "fixture built no embeddings or norms -- test is vacuous"
    assert excluded <= undecayed, "an embedding, norm or bias is being decayed"
    assert not (excluded & decayed)


def test_every_linear_weight_is_still_decayed():
    """The exclusion must not turn weight decay off altogether -- Linear weights
    are the whole point of the knob.

    Asserted per tensor, not as a share of parameters: at this fixture's
    d_model=16 the embedding table is a third of the model, so a ratio test
    would only be measuring the fixture's width. At the configs' real widths the
    decayed share is 99.43% (scratch) and 99.62% (opt-125m).
    """
    import torch.nn as nn
    m = _sched_module(lr_scheduler="none", weight_decay=0.01)
    decayed, _ = _groups(m)
    linears = [(n, mod.weight) for n, mod in m.named_modules()
               if isinstance(mod, nn.Linear) and mod.weight.requires_grad]
    assert linears, "fixture built no Linear layers -- test is vacuous"
    missing = [n for n, w in linears if id(w) not in decayed]
    assert not missing, f"Linear weights excluded from decay: {missing}"


def test_every_trainable_parameter_lands_in_exactly_one_group():
    """A split that drops a parameter stops training it, silently."""
    m = _sched_module(lr_scheduler="none", weight_decay=0.01)
    decayed, undecayed = _groups(m)
    assert not (decayed & undecayed), "a parameter is in both groups"
    trainable = {id(p) for p in m.parameters() if p.requires_grad}
    assert decayed | undecayed == trainable, "a trainable parameter was dropped"


def test_zero_weight_decay_still_yields_a_usable_optimizer():
    """Turning the knob off must not depend on the grouping code path."""
    m = _sched_module(lr_scheduler="none", weight_decay=0.0)
    decayed, undecayed = _groups(m)
    assert not decayed
    assert undecayed == {id(p) for p in m.parameters() if p.requires_grad}
