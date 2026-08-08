"""The transformer driven by VQ-VAE codes instead of raw coordinates (stage 2).

The dataset still emits coordinate tokens -- nothing about `MeshDataset`
changed -- so the module converts a batch into the code vocabulary on the fly
with a frozen VQ-VAE. These check that conversion, since every downstream
number depends on it being exactly right.
"""
import numpy as np
import torch

from src.dataset.mesh_dataset import specials, vocab_size
from src.models.mesh_transformer import MeshTransformerModule
from src.models.mesh_vqvae import MeshVQVAE

NUM_BINS = 32
CODEBOOK, DEPTH = 16, 3
BOS, EOS, PAD = specials(NUM_BINS)
B, F1, F2 = 2, 2, 3          # cond faces, target faces


def _vqvae():
    return MeshVQVAE(num_bins=NUM_BINS, codebook_size=CODEBOOK, depth=DEPTH,
                     d_model=16, n_head=2, num_layers=1, dropout=0.0, max_faces=32)


def _module(**kw):
    return MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2,
                                 num_layers=1, dropout=0.0, max_seq_len=64,
                                 vqvae=_vqvae(), **kw)


def _batch():
    """What `mesh_collate_fn` really emits: coordinate tokens, right-padded.

    Sample 1 is one face shorter on both sides, so the ragged path is exercised.
    """
    cond = torch.randint(0, NUM_BINS, (B, 9 * F1))
    cond_pad = torch.zeros(B, 9 * F1, dtype=torch.bool)
    cond[1, -9:], cond_pad[1, -9:] = PAD, True

    body = torch.randint(0, NUM_BINS, (B, 9 * F2))
    tgt = torch.cat([torch.full((B, 1), BOS), body, torch.full((B, 1), EOS)], dim=1)
    tgt_pad = torch.zeros(B, tgt.shape[1], dtype=torch.bool)
    # Shorter target: EOS moves up by one face, the tail becomes padding.
    tgt[1, 1 + 9 * (F2 - 1)] = EOS
    tgt[1, 2 + 9 * (F2 - 1):] = PAD
    tgt_pad[1, 2 + 9 * (F2 - 1):] = True
    return {"cond": cond, "tgt": tgt, "cond_pad_mask": cond_pad,
            "tgt_pad_mask": tgt_pad, "ids": ["a", "b"],
            "center": torch.zeros(B, 3), "scale": torch.ones(B, 3) * 10.0}


def test_code_vocabulary_is_offset_per_residual_stage():
    """Code 5 at stage 0 is not code 5 at stage 1, so the stages must not collide."""
    m = _module()
    assert m.network.head.out_features == CODEBOOK * DEPTH + 3
    assert (m.bos, m.eos, m.pad) == (CODEBOOK * DEPTH, CODEBOOK * DEPTH + 1,
                                     CODEBOOK * DEPTH + 2)


def test_coordinate_mode_vocabulary_is_untouched():
    """mesh-1 and mesh-2 configs must keep working: no vqvae, no change."""
    m = MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2,
                              num_layers=1, max_seq_len=64)
    assert m.network.head.out_features == vocab_size(NUM_BINS)
    assert m.pad == PAD


def test_prepare_builds_code_sequences_with_bos_eos_and_padding():
    m = _module().eval()
    with torch.no_grad():
        cond, tgt, cond_pad, tgt_pad = m._prepare(_batch())

    # Condition is continuous per-face features -- encoded, never quantized.
    assert cond.dtype.is_floating_point and cond.shape == (B, F1, 16)
    assert cond_pad.shape == (B, F1) and bool(cond_pad[1, -1]) and not bool(cond_pad[0, -1])

    # Target: BOS + depth codes per face + EOS, sample 1 one face shorter.
    assert tgt.shape == (B, 1 + DEPTH * F2 + 1)
    assert (tgt[:, 0] == m.bos).all()
    assert int(tgt[0, 1 + DEPTH * F2]) == m.eos
    assert int(tgt[1, 1 + DEPTH * (F2 - 1)]) == m.eos
    assert int(tgt[1, -1]) == m.pad and bool(tgt_pad[1, -1])
    codes = tgt[0, 1:1 + DEPTH * F2]
    assert int(codes.min()) >= 0 and int(codes.max()) < CODEBOOK * DEPTH


def test_forward_accepts_the_continuous_condition():
    m = _module().eval()
    with torch.no_grad():
        cond, tgt, cond_pad, tgt_pad = m._prepare(_batch())
        logits = m.network(cond, tgt, cond_pad, tgt_pad)
    assert logits.shape == (B, tgt.shape[1] - 1, CODEBOOK * DEPTH + 3)
    assert torch.isfinite(logits).all()


def test_shared_step_loss_is_finite_and_trains_the_transformer_only():
    m = _module()
    loss, _, _ = m._shared_step(_batch())
    assert torch.isfinite(loss) and float(loss) > 0

    loss.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in m.network.parameters()), "transformer got no gradient"
    # The tokenizer is frozen: stage 2 must not move the vocabulary underneath
    # the codes it is learning to predict.
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0
               for p in m.vqvae.parameters()), "frozen VQ-VAE received gradient"


def test_vqvae_parameters_are_frozen_and_excluded_from_the_optimizer():
    m = _module()
    assert not any(p.requires_grad for p in m.vqvae.parameters())
    opt = m.configure_optimizers()
    opt = opt["optimizer"] if isinstance(opt, dict) else opt
    owned = {id(p) for group in opt.param_groups for p in group["params"]}
    assert not any(id(p) in owned for p in m.vqvae.parameters())


def test_decode_tokens_round_trips_a_generated_sequence_to_a_mesh():
    """Un-offset, reshape to [F, depth], through the VQ-VAE decoder."""
    m = _module().eval()
    with torch.no_grad():
        _, tgt, _, _ = m._prepare(_batch())
        verts, faces = m.decode_tokens(tgt[0])
    assert faces.shape == (F2, 3)
    assert verts.ndim == 2 and verts.shape[1] == 3


def test_decode_tokens_survives_a_truncated_sequence():
    """Sampling can stop anywhere; a partial face must be dropped, not crash."""
    m = _module().eval()
    with torch.no_grad():
        verts, faces = m.decode_tokens(torch.tensor([m.bos, 0, 1]))   # 2 of 3 codes
    assert faces.shape == (0, 3) and verts.shape == (0, 3)


def test_generate_emits_code_tokens_within_the_vocabulary():
    m = _module().eval()
    batch = _batch()
    with torch.no_grad():
        cond, _, cond_pad, _ = m._prepare(batch)
        out = m.generate(cond, cond_pad, max_new_tokens=6, temperature=0.0)
    assert out.shape[0] == B and out.shape[1] <= 7      # BOS + at most 6
    assert (out[:, 0] == m.bos).all()
    assert int(out.max()) < CODEBOOK * DEPTH + 3
