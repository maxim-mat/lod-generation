"""The Lightning module with an image: dropout, scopes, lr groups, init_from, generate."""
import pytest
import torch
from transformers import DINOv3ViTConfig, OPTConfig

from src.dataset.mesh_dataset import specials, vocab_size
from src.models.mesh_transformer import MeshTransformerModule

NUM_BINS, S = 16, 64
BOS, EOS, PAD = specials(NUM_BINS)


def module(**kw):
    torch.manual_seed(0)
    opt = OPTConfig(vocab_size=vocab_size(NUM_BINS), hidden_size=32, word_embed_proj_dim=32,
                    num_hidden_layers=2, num_attention_heads=4, ffn_dim=64,
                    max_position_embeddings=512, dropout=0.0)
    dino = DINOv3ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=2, patch_size=16, image_size=S)
    common = dict(num_bins=NUM_BINS, backbone="opt", opt_config=opt, dropout=0.0,
                  pos_embed="sinusoidal", max_seq_len=256)
    if kw.pop("image", True):
        common.update(image_encoder_config=dino, image_size=S)
    return MeshTransformerModule(**common, **kw)


def batch(valid=(True, True)):
    g = torch.Generator().manual_seed(1)
    tgt = torch.cat([torch.full((2, 1), BOS), torch.randint(0, NUM_BINS, (2, 18), generator=g),
                     torch.full((2, 1), EOS)], dim=1)
    return {"cond": torch.randint(0, NUM_BINS, (2, 18), generator=g), "tgt": tgt,
            "cond_pad_mask": torch.zeros(2, 18, dtype=torch.bool),
            "tgt_pad_mask": torch.zeros(2, 20, dtype=torch.bool),
            "image": torch.randint(0, 255, (2, 3, S, S), generator=g, dtype=torch.uint8),
            "image_valid": torch.tensor(valid), "ids": ["a", "b"],
            "center": torch.zeros(2, 3), "scale": torch.ones(2, 3)}


def test_image_model_requires_an_image_in_the_batch():
    m = module().eval()
    b = batch()
    del b["image"]
    with pytest.raises(ValueError, match="ortho_dir"):
        m._shared_step(b)


def test_image_only_scope_trains_only_the_image_parameters():
    m = module(train_scope="image_only")
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith(("network.image_encoder.", "network.xattn."))
                             for n in trainable)
    assert not any(n.startswith("network.image_encoder.backbone.") for n in trainable)
    groups = m._param_groups()
    assert sum(len(g["params"]) for g in groups) == len(trainable)


def test_backbone_lr_scale_gives_opt_its_own_learning_rate():
    m = module(lr=1e-3, backbone_lr_scale=0.1)
    lrs = {round(g["lr"], 6) for g in m._param_groups()}
    assert lrs == {1e-3, 1e-4}


def test_image_dropout_only_acts_in_training():
    m = module(image_dropout=1.0)
    seen = []
    m.network.encode_image = lambda img, valid: seen.append(valid.clone()) or \
        type(m.network).encode_image(m.network, img, valid)
    m.train()
    m._shared_step(batch())
    m.eval()
    m._shared_step(batch())
    assert seen[0].tolist() == [False, False] and seen[1].tolist() == [True, True]


def test_init_from_an_image_free_checkpoint_only_misses_image_keys(tmp_path):
    import lightning as L
    base = module(image=False)
    path = tmp_path / "v5.ckpt"
    torch.save({"state_dict": base.state_dict(), "hyper_parameters": dict(base.hparams),
                "pytorch-lightning_version": L.__version__}, path)
    m = module()
    missing = m.load_weights_from(path)
    assert missing and all(k.startswith(("network.image_encoder.", "network.xattn."))
                           for k in missing)
    torch.testing.assert_close(m.network.opt.lm_head.weight, base.network.opt.lm_head.weight)


def test_init_from_refuses_a_checkpoint_from_another_split_scheme(tmp_path):
    import lightning as L
    base = module(image=False)                       # data_split defaults to "random"
    path = tmp_path / "v5.ckpt"
    hp = {k: v for k, v in dict(base.hparams).items() if k != "data_split"}   # pre-2a checkpoint
    torch.save({"state_dict": base.state_dict(), "hyper_parameters": hp,
                "pytorch-lightning_version": L.__version__}, path)
    with pytest.raises(ValueError, match="split"):
        module(data_split="tile").load_weights_from(path)


def test_init_from_refuses_a_different_tokenization(tmp_path):
    import lightning as L
    base = module(image=False, tokenization="amt")
    path = tmp_path / "amt.ckpt"
    torch.save({"state_dict": base.state_dict(), "hyper_parameters": dict(base.hparams),
                "pytorch-lightning_version": L.__version__}, path)
    with pytest.raises(ValueError, match="tokenization"):
        module().load_weights_from(path)


def test_generate_with_images_cached_matches_uncached_including_beams():
    m = module().eval()
    with torch.no_grad():
        m.network.xattn.gates.fill_(0.7)
    b = batch(valid=(True, False))
    for beam in (1, 3):
        kw = dict(image=b["image"], image_valid=b["image_valid"], temperature=0.0,
                  beam_size=beam, max_new_tokens=12)
        off = m.generate(b["cond"], b["cond_pad_mask"], use_cache=False, **kw)
        on = m.generate(b["cond"], b["cond_pad_mask"], use_cache=True, **kw)
        assert torch.equal(off, on), beam


def test_scratch_backbone_with_an_image_is_refused():
    with pytest.raises(ValueError, match="opt"):
        MeshTransformerModule(num_bins=NUM_BINS, backbone="scratch", d_model=16, n_head=2,
                              num_layers=1, max_seq_len=64,
                              image_encoder_config=DINOv3ViTConfig(
                                  hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                  num_attention_heads=2, patch_size=16, image_size=S),
                              image_size=S)
