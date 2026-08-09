"""Contract checks for the VQ-VAE mesh tokenizer. CPU + random data only.

An untrained codebook cannot reconstruct anything, so nothing here asserts
reconstruction *quality* -- that is the training gate in the design spec
(recon chamfer <= 0.15 m against the coordinate tokenizer's 0.107 m floor).
What is asserted is the contract stage 2 will build on: shapes, code ranges,
gradient flow through the straight-through estimator, and the empty-input
behaviour that lets an aggregator nanmean over a batch.
"""
import numpy as np
import torch

from src.models.mesh_vqvae import MeshVQVAE, MeshVQVAEModule

NUM_BINS = 32           # tiny grid; the real bin count is a config knob
CODEBOOK, DEPTH = 64, 3
B, F = 2, 5             # 2 meshes of 5 faces


def _vqvae(**kw):
    kw = {"num_bins": NUM_BINS, "codebook_size": CODEBOOK, "depth": DEPTH,
          "d_model": 16, "n_head": 2, "num_layers": 1, "dropout": 0.0, **kw}
    return MeshVQVAE(**kw)


def _coords(b=B, f=F):
    """[b, f, 9] discretized face coordinates, as `quantize` would produce."""
    return torch.randint(0, NUM_BINS, (b, f, 9))


def test_encode_shape():
    """One continuous embedding per face -- this is the LOD1 condition path."""
    z = _vqvae().encode(_coords())
    assert z.shape == (B, F, 16)
    assert torch.isfinite(z).all()


def test_tokenize_shape_and_range():
    """`depth` codes per face, every one a valid codebook index."""
    codes = _vqvae().tokenize(_coords())
    assert codes.shape == (B, F, DEPTH)
    assert codes.dtype == torch.long
    assert int(codes.min()) >= 0 and int(codes.max()) < CODEBOOK


def test_decoder_logit_shape():
    """Nine coordinate distributions per face, over the bin vocabulary."""
    model = _vqvae()
    logits = model.decode(model.quantize(model.encode(_coords()))[0])
    assert logits.shape == (B, F, 9, NUM_BINS)
    assert torch.isfinite(logits).all()


def test_detokenize_returns_a_mesh_with_every_face():
    """Codes in, (verts, faces) out, one triangle per code row."""
    model = _vqvae().eval()
    verts, faces = model.detokenize(model.tokenize(_coords(b=1))[0])
    assert faces.shape == (F, 3)
    assert verts.ndim == 2 and verts.shape[1] == 3
    # detokenize dequantizes, so coordinates land back in the unit box.
    assert float(np.abs(verts).max()) <= 0.5 + 1e-9


def test_detokenize_of_nothing_is_the_empty_mesh():
    """Matches the coordinate detokenizer: empty pair, never an exception."""
    verts, faces = _vqvae().eval().detokenize(torch.zeros((0, DEPTH), dtype=torch.long))
    assert verts.shape == (0, 3) and faces.shape == (0, 3)


def test_out_of_range_codes_are_clamped_not_raised():
    """A stage-2 transformer can emit any integer; the decoder must survive it."""
    model = _vqvae().eval()
    codes = torch.full((F, DEPTH), CODEBOOK + 99, dtype=torch.long)
    verts, faces = model.detokenize(codes)
    assert faces.shape == (F, 3)


def test_straight_through_gradients_reach_the_encoder():
    """Quantization is not differentiable; without the STE the encoder is orphaned."""
    model = _vqvae()
    z = model.encode(_coords())
    z_q, _, _ = model.quantize(z)
    z_q.sum().backward()
    grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert grads, "no encoder parameter received a gradient"
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_commitment_loss_is_finite_and_positive():
    """Zero commitment loss on random input would mean the quantizer is a no-op."""
    model = _vqvae()
    _, loss, _ = model.quantize(model.encode(_coords()))
    assert torch.isfinite(loss) and float(loss) > 0


def test_usage_counts_distinct_codes():
    """The diagnostic that says whether `codebook_size` is mis-set."""
    model = _vqvae()
    stats = model.codebook_stats(torch.tensor([[0, 0, 1, 1, 2]]).expand(1, 5))
    assert stats["distinct"] == 3
    # Perplexity of a 3-way split weighted 2/2/1 sits between 1 and 3.
    assert 1.0 < stats["perplexity"] < 3.0 + 1e-6


def test_codebook_is_seeded_from_data_not_the_origin():
    """Guards `kmeans_init`. Entries sitting near the origin are unreachable
    from a transformer output: every candidate distance equals ||z|| to within
    rounding, argmin picks on noise, and the rows that win the first step are
    the only ones ever updated (a hand-rolled quantizer froze at 28 live entries
    of 1024 this way, run mesh-vqvae-2)."""
    def nearest(model, z):
        """Mean distance from each encoder output to its closest stage-0 entry."""
        d = torch.cdist(z.flatten(0, -2), model.quantizer.codebooks[0])
        return float(d.min(dim=-1).values.mean())

    model = _vqvae().train()
    z = model.encode(_coords())
    before = nearest(model, z)
    model.quantize(z)
    # Entries are drawn from z itself, so the nearest one sits on top of it.
    assert nearest(model, z) < 0.1 * before


def test_rotation_trick_changes_the_gradient_but_not_the_forward_pass():
    """`rotation_trick` has to reach ResidualVQ to mean anything. It rotates the
    gradient from the code to the encoder output instead of copy-pasting it
    (Fifty et al., ICLR 2025, arXiv:2410.06424), so the forward value is
    identical by construction and only the backward pass moves."""
    def run(rotation_trick):
        torch.manual_seed(0)
        model = _vqvae(rotation_trick=rotation_trick).train()
        z = model.encode(_coords()).detach().requires_grad_(True)
        z_q, _, _ = model.quantize(z)
        z_q.square().sum().backward()
        return z_q.detach(), z.grad

    (q_ste, g_ste), (q_rot, g_rot) = run(False), run(True)
    assert torch.allclose(q_ste, q_rot, atol=1e-6), "the forward pass must not move"
    assert not torch.allclose(g_ste, g_rot, atol=1e-6), "rotation_trick never reached the quantizer"


def test_a_frozen_codebook_never_moves():
    """Stage 2 holds the tokenizer as a submodule, so Lightning's per-epoch
    `model.train()` recurses in and undoes the `eval()` it was constructed with.
    An EMA codebook updates on `training` alone, so without the freeze it would
    rewrite the vocabulary the transformer is being trained to predict, mid-run.
    """
    model = _vqvae().train()
    z = model.encode(_coords())
    model.quantize(z)                       # seeds while still trainable
    for p in model.parameters():
        p.requires_grad_(False)

    before = model.quantizer.codebooks.clone()
    model.quantize(z.detach())
    assert torch.equal(model.quantizer.codebooks, before)


def test_padded_faces_come_back_as_negative_one():
    """The quantizer marks padding rather than assigning it a code. Anything
    counting occupancy or writing meshes has to skip those, not clamp them."""
    model = _vqvae().eval()
    pad = torch.zeros(B, F, dtype=torch.bool)
    pad[1, -2:] = True
    _, _, codes = model.quantize(model.encode(_coords(), pad), pad)
    assert (codes[1, -2:] == -1).all()
    assert (codes[0] >= 0).all()


def test_codebook_stats_reports_each_stage_separately():
    """Pooling the stages hides the classic residual-VQ failure: stage 0 healthy
    while every later stage has collapsed onto one entry."""
    codes = torch.tensor([[0, 5, 5], [1, 5, 5], [2, 5, 5], [3, 5, 5]])   # [4 faces, 3 stages]
    stats = _vqvae().codebook_stats(codes)
    assert stats["distinct_per_stage"] == [4, 1, 1]
    assert stats["perplexity_per_stage"][1] == 1.0
    assert stats["distinct"] == 5           # pooled {0,1,2,3,5}, as before


def test_tokenize_is_deterministic_in_eval():
    """Stage 2 trains on these codes; they must not move between epochs."""
    model, coords = _vqvae().eval(), _coords()
    with torch.no_grad():
        assert torch.equal(model.tokenize(coords), model.tokenize(coords))


def test_module_training_step_returns_a_finite_loss():
    """Reconstruction + commitment, over a face batch from the datamodule."""
    module = MeshVQVAEModule(num_bins=NUM_BINS, codebook_size=CODEBOOK, depth=DEPTH,
                             d_model=16, n_head=2, num_layers=1, dropout=0.0)
    loss = module.training_step({"coords": _coords(), "pad_mask": torch.zeros(B, F, dtype=torch.bool)}, 0)
    assert torch.isfinite(loss) and float(loss) > 0


def test_loss_is_reconstruction_plus_the_commitment_term_only():
    """The codebook is EMA-updated, not trained, so nothing else reaches the
    objective. Run mesh-vqvae-1 also summed in a raw codebook MSE, which carried
    ~85% of `val_loss` and made its best checkpoint the untrained epoch 0."""
    module = MeshVQVAEModule(num_bins=NUM_BINS, codebook_size=CODEBOOK, depth=DEPTH,
                             d_model=16, n_head=2, num_layers=1, dropout=0.0).eval()
    batch = {"coords": _coords(), "pad_mask": torch.zeros(B, F, dtype=torch.bool)}
    with torch.no_grad():
        loss, metrics = module._shared_step(batch)
    assert torch.allclose(loss, metrics["recon_ce"] + metrics["vq_loss"], atol=1e-6)


def test_padded_faces_do_not_contribute_to_the_loss():
    """Buildings have different face counts; padding must not train the decoder."""
    module = MeshVQVAEModule(num_bins=NUM_BINS, codebook_size=CODEBOOK, depth=DEPTH,
                             d_model=16, n_head=2, num_layers=1, dropout=0.0).eval()
    coords, pad = _coords(), torch.zeros(B, F, dtype=torch.bool)
    pad[1, -2:] = True
    batch = {"coords": coords.clone(), "pad_mask": pad}
    with torch.no_grad():
        base = module.training_step(batch, 0)
        # Scribble over the padded slots only: the loss must not notice.
        batch["coords"][1, -2:] = (coords[1, -2:] + 7) % NUM_BINS
        after = module.training_step(batch, 0)
    assert torch.allclose(base, after, atol=1e-6)
