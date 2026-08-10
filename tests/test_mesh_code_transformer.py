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
VERTS = 3                    # quantization is per vertex, as in MeshAnything
PER_FACE = VERTS * DEPTH     # 9 code tokens per face
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


def test_code_vocabulary_is_the_shared_codebook_plus_specials():
    """MeshAnything: `vocab_size = self.tokenizer.codebook_size + 3`, one shared
    codebook indexed by every residual stage. Giving each stage its own id block
    tripled this softmax for nothing -- stage identity is already implied by
    position, since the stages are the fastest-varying index."""
    m = _module()
    assert m.network.head.out_features == CODEBOOK + 3
    assert (m.bos, m.eos, m.pad) == (CODEBOOK, CODEBOOK + 1, CODEBOOK + 2)


def test_frozen_tokenizer_stays_in_eval_through_lightning_train():
    """Lightning calls `model.train()` at every training epoch and
    `nn.Module.train` recurses into submodules, which re-enables the frozen
    VQ-VAE's dropout. `_prepare` runs the tokenizer inside `training_step`, so
    that makes the *target labels* stochastic at train time and deterministic at
    val time -- two different label distributions. The `requires_grad` guard in
    `quantize` covers the codebook EMA, not module mode."""
    m = _module()
    for mod in m.vqvae.modules():                # make any leak unmissable
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.5
    m.train()
    assert not m.vqvae.training, "frozen tokenizer was put back into train mode"

    batch = _batch()
    with torch.no_grad():
        labels = [m._prepare(batch)[1] for _ in range(4)]
    assert all(torch.equal(labels[0], t) for t in labels[1:]), \
        "stage-2 target labels move between epochs"


def test_code_tokens_are_embedded_from_the_frozen_codebook():
    """MeshAnything embeds a code by looking up its VQ-VAE vector, not from a
    free table -- `embed_tokens` is literally commented "# not used" and
    `embed_with_vae` does `input_layer(quantize_codebooks[0][ids - 3])`.

    It matters for generalization: codes adjacent in codebook space arrive as
    nearly the same vector, so the transformer starts with the geometry instead
    of having to learn 1024 unrelated rows from 16k buildings.
    """
    m = _module().eval()
    net = m.network
    assert net.token_embed is None, "a free code embedding table is still in use"

    ids = torch.arange(CODEBOOK)[None]
    with torch.no_grad():
        before = net.embed_tokens(ids).clone()
        net.codebook[3] += 5.0            # move one code's geometry
        after = net.embed_tokens(ids)
    moved = (after - before).abs().sum(-1) > 0
    assert bool(moved[0, 3]), "changing a code's vector did not change its embedding"
    assert not bool(moved[0, 4]), "an unrelated code's embedding moved too"


def test_intra_face_slot_has_its_own_embedding():
    """`OPTFacePositionalEmbedding(face_per_token + 3)`: every token also carries
    which of the 9 slots within its face it fills -- which vertex, which residual
    stage. Without it the model has to infer that from absolute position mod 9,
    which we compute ourselves in `_token_metrics` and never tell it."""
    m = _module().eval()
    net = m.network
    assert net.face_pos_embed.num_embeddings == PER_FACE + 3

    # The same code id, one slot apart, must not embed identically.
    ids = torch.zeros(1, PER_FACE + 2, dtype=torch.long)
    with torch.no_grad():
        emb = net.embed_tokens(ids)
    assert not torch.allclose(emb[0, 1], emb[0, 2]), "slot within the face is ignored"
    # ...but a whole face later, the slot repeats, so those two must match.
    assert torch.allclose(emb[0, 1], emb[0, 1 + PER_FACE], atol=1e-6)


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

    # Target: BOS + 3 vertices x depth codes per face + EOS, sample 1 one face
    # shorter. Layout is MeshAnything's `b (nf nv q)`: stage varies fastest.
    assert tgt.shape == (B, 1 + PER_FACE * F2 + 1)
    assert (tgt[:, 0] == m.bos).all()
    assert int(tgt[0, 1 + PER_FACE * F2]) == m.eos
    assert int(tgt[1, 1 + PER_FACE * (F2 - 1)]) == m.eos
    assert int(tgt[1, -1]) == m.pad and bool(tgt_pad[1, -1])
    codes = tgt[0, 1:1 + PER_FACE * F2]
    assert int(codes.min()) >= 0 and int(codes.max()) < CODEBOOK


def test_forward_accepts_the_continuous_condition():
    m = _module().eval()
    with torch.no_grad():
        cond, tgt, cond_pad, tgt_pad = m._prepare(_batch())
        logits = m.network(cond, tgt, cond_pad, tgt_pad)
    assert logits.shape == (B, tgt.shape[1] - 1, CODEBOOK + 3)
    assert torch.isfinite(logits).all()


def test_shared_step_loss_is_finite_and_trains_the_transformer_only():
    m = _module()
    loss, _, _, _ = m._shared_step(_batch())
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
    """Reshape to [F, 3 vertices, depth], through the VQ-VAE decoder."""
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
        partial = torch.tensor([m.bos] + list(range(PER_FACE - 1)))
        verts, faces = m.decode_tokens(partial)
    assert faces.shape == (0, 3) and verts.shape == (0, 3)


def test_decode_tokens_conditions_a_noise_resistant_decoder():
    """Stage 1b fine-tunes the decoder *with* the LOD1 condition injected, so
    stage 2 has to hand it back at decode time. Without this the fine-tuned
    decoder runs at `cond=None` -- a train/inference mismatch that makes the
    fine-tune worse than not doing it."""
    vq = _vqvae()
    vq.conditioned_decoder = True
    m = MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2,
                              num_layers=1, dropout=0.0, max_seq_len=64,
                              vqvae=vq).eval()
    batch = _batch()
    with torch.no_grad():
        cond, tgt, cond_pad, _ = m._prepare(batch)
        plain = m.decode_tokens(tgt[0])[0]
        conditioned = m.decode_tokens(tgt[0], cond=cond[0][~cond_pad[0]])[0]
    assert plain.shape == conditioned.shape
    assert not np.allclose(plain, conditioned), "the condition never reached the decoder"


def test_a_plain_decoder_ignores_a_condition():
    """A stage-1 decoder that never saw a condition has untrained `cond_proj`
    and `segment_embed` weights. Feeding it one would inject random vectors
    into the decoder input, so the flag -- not the caller -- decides."""
    m = _module().eval()
    assert m.vqvae.conditioned_decoder is False
    batch = _batch()
    with torch.no_grad():
        cond, tgt, cond_pad, _ = m._prepare(batch)
        plain = m.decode_tokens(tgt[0])[0]
        offered = m.decode_tokens(tgt[0], cond=cond[0][~cond_pad[0]])[0]
    assert np.array_equal(plain, offered), "an untrained condition path was used"


def test_generate_emits_code_tokens_within_the_vocabulary():
    m = _module().eval()
    batch = _batch()
    with torch.no_grad():
        cond, _, cond_pad, _ = m._prepare(batch)
        out = m.generate(cond, cond_pad, max_new_tokens=6, temperature=0.0)
    assert out.shape[0] == B and out.shape[1] <= 7      # BOS + at most 6
    assert (out[:, 0] == m.bos).all()
    assert int(out.max()) < CODEBOOK + 3
