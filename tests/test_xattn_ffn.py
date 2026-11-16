"""Gated FFN after each gated cross-attention (Flamingo GATED XATTN-DENSE).

Off by default; on, it must still start exactly at the image-free model.
"""
import pytest
import torch
from transformers import DINOv3ViTConfig, OPTConfig

from src.models.mesh_transformer import MeshOPTTransformer
from tests.test_ortho_fusion import NUM_BINS, S, V, _greedy, batch, net
from tests.test_ortho_module import batch as module_batch
from tests.test_ortho_module import module


def ffn_net(seed=0):
    """`net("xattn")` with the gated FFN on (same tiny OPT and DINOv3)."""
    torch.manual_seed(seed)
    opt = OPTConfig(vocab_size=V, hidden_size=32, word_embed_proj_dim=32, num_hidden_layers=2,
                    num_attention_heads=4, ffn_dim=64, max_position_embeddings=512, dropout=0.0)
    image = {"fusion": "xattn", "name": None, "image_size": S, "beta": 0.1, "xattn_ffn": True,
             "config": DINOv3ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                       num_attention_heads=2, patch_size=16, image_size=S)}
    return MeshOPTTransformer(vocab_size=V, opt_config=opt, dropout=0.0,
                              pos_embed="sinusoidal", max_seq_len=256, num_bins=NUM_BINS,
                              image=image).eval()


def test_default_builds_no_ffn():
    m = net("xattn")
    assert not hasattr(m.xattn, "ffns") or m.xattn.ffns is None
    assert not any(".ffn" in k for k in m.state_dict())


def test_zero_gates_with_ffn_reproduce_the_image_free_model_exactly():
    base, fused = net(None), ffn_net()
    missing, unexpected = fused.load_state_dict(base.state_dict(), strict=False)
    assert unexpected == [] and all(k.startswith(("image_encoder.", "xattn.")) for k in missing)
    assert any(k.startswith("xattn.ffns.") for k in missing)
    cond, tgt, img = batch()
    ctx = fused.encode_image(img, torch.tensor([True, True]))
    with torch.no_grad():
        assert torch.equal(base(cond, tgt), fused(cond, tgt, image_ctx=ctx))


def test_open_ffn_gate_changes_the_output():
    base, fused = net(None), ffn_net()
    fused.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        fused.xattn.ffn_gates.fill_(1.0)
    cond, tgt, img = batch()
    ctx = fused.encode_image(img, torch.tensor([True, True]))
    with torch.no_grad():
        assert not torch.allclose(base(cond, tgt), fused(cond, tgt, image_ctx=ctx))


def test_cached_decoding_matches_uncached_with_ffn():
    m = ffn_net()
    with torch.no_grad():
        m.xattn.gates.fill_(0.7)
        m.xattn.ffn_gates.fill_(0.7)
    cond, _, img = batch()
    assert torch.equal(_greedy(m, cond, img, False), _greedy(m, cond, img, True))


def test_ffn_with_prefix_fusion_is_refused():
    with pytest.raises(ValueError, match="image_xattn_ffn"):
        module(image_fusion="prefix", image_xattn_ffn=True)


def test_ffn_needs_an_image_model():
    with pytest.raises(ValueError, match="image"):
        module(image=False, image_xattn_ffn=True)


def test_module_flag_builds_trainable_undecayed_gates():
    m = module(image_xattn_ffn=True, train_scope="image_only")
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert "network.xattn.ffn_gates" in trainable
    assert any(n.startswith("network.xattn.ffns.") for n in trainable)
    gate = m.network.xattn.ffn_gates
    groups = [g for g in m._param_groups() if any(p is gate for p in g["params"])]
    assert len(groups) == 1 and groups[0]["weight_decay"] == 0.0
    assert m.hparams["image_xattn_ffn"] is True
    assert module().hparams["image_xattn_ffn"] is False


def test_training_step_logs_the_ffn_gates(monkeypatch):
    m = module(image_xattn_ffn=True)
    logged = {}
    monkeypatch.setattr(m, "log", lambda name, value, **kw: logged.__setitem__(name, value))
    m.training_step(module_batch(), 0)
    assert "train_ffn_gate_abs_mean" in logged
    assert {"train_ffn_gate_0", "train_ffn_gate_1"} <= set(logged)
    assert "train_gate_abs_mean" in logged


def test_flag_round_trips_through_a_checkpoint(tmp_path):
    from src.models.mesh_transformer import MeshTransformerModule
    from tests.test_ortho_module import _configs, _save
    m = module(image_xattn_ffn=True).eval()
    with torch.no_grad():
        m.network.xattn.ffn_gates.fill_(0.5)
    path = tmp_path / "ffn.ckpt"
    _save(m, path)
    back = MeshTransformerModule.load_from_checkpoint(str(path), **_configs())
    assert back.network.xattn.ffns is not None
    torch.testing.assert_close(back.network.xattn.ffn_gates, m.network.xattn.ffn_gates)


def test_old_xattn_checkpoint_loads_without_the_ffn(tmp_path):
    from src.models.mesh_transformer import MeshTransformerModule
    from tests.test_ortho_module import _configs, _save
    path = tmp_path / "old.ckpt"
    _save(module().eval(), path, drop=("image_xattn_ffn",))
    back = MeshTransformerModule.load_from_checkpoint(str(path), **_configs())
    assert back.network.xattn.ffns is None
