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


def test_vq_weight_scales_the_quantizer_term_only():
    """`recon_ce` is cross-entropy in nats; `vq_loss` is a raw MSE in d_model
    space summed over `depth` stages. Summing them unweighted let the codebook
    term carry ~85% of the objective (run mesh-vqvae-1), so it needs a knob."""
    module = MeshVQVAEModule(num_bins=NUM_BINS, codebook_size=CODEBOOK, depth=DEPTH,
                             d_model=16, n_head=2, num_layers=1, dropout=0.0,
                             vq_weight=0.1).eval()
    batch = {"coords": _coords(), "pad_mask": torch.zeros(B, F, dtype=torch.bool)}
    with torch.no_grad():
        loss, metrics = module._shared_step(batch)
    assert torch.allclose(loss, metrics["recon_ce"] + 0.1 * metrics["vq_loss"], atol=1e-6)


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
