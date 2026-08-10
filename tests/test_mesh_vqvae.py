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
VERTS = 3               # quantization is per vertex, as in MeshAnything
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
    """`depth` codes per *vertex*, every one a valid codebook index."""
    codes = _vqvae().tokenize(_coords())
    assert codes.shape == (B, F, VERTS, DEPTH)
    assert codes.dtype == torch.long
    assert int(codes.min()) >= 0 and int(codes.max()) < CODEBOOK


def test_tokens_per_face_matches_the_paper():
    """MeshAnything quantizes per vertex: `face_per_token = num_quantizers * 3`.

    Nine tokens per face, the same count as raw coordinates -- the codebook buys
    a learned vocabulary, not compression. Quantizing one feature per face
    instead squeezes 63 bits of coordinate through 30 bits of code, which is
    what put run mesh-3's reconstruction ceiling at 0.246 m against the
    coordinate tokenizer's 0.107 m.
    """
    assert _vqvae().tokens_per_face == VERTS * DEPTH == 9


def test_codebook_dim_is_independent_of_the_model_width():
    """MeshGPT: `project_dim_codebook = Linear(curr_dim, dim_codebook * nvf)`,
    with `dim_codebook=192` against a model `dim` of 512. The codes live in
    their own, narrower space; tying them to d_model is a coincidence, not a
    design."""
    model = _vqvae(d_model=16, codebook_dim=8)
    assert model.quantizer.codebooks.shape[-1] == 8
    assert model.vert_proj.out_features == VERTS * 8
    assert model.face_proj_down.in_features == VERTS * 8
    codes = model.tokenize(_coords())
    assert codes.shape == (B, F, VERTS, DEPTH)


def _shared_vertex_mesh():
    """Two faces sharing an edge: vertices 0 and 1 appear in both.

    Coordinates, not indices -- that is all the tokenizer carries, so the
    sharing has to be recovered from the numbers.
    """
    a, b = [1, 1, 1], [2, 2, 2]
    f0 = a + b + [3, 3, 3]
    f1 = a + b + [4, 4, 4]
    return torch.tensor([[f0, f1]], dtype=torch.long)      # [1, 2, 9]


def test_vertex_ids_dedupe_within_a_mesh():
    """Faces are coordinates; the shared vertex has to be found by value."""
    ids, counts = _vqvae().vertex_ids(_shared_vertex_mesh())
    assert int(counts[0]) == 4, "expected 4 unique vertices across the two faces"
    assert ids.shape == (1, 2, VERTS)
    assert int(ids[0, 0, 0]) == int(ids[0, 1, 0]), "shared vertex a got two ids"
    assert int(ids[0, 0, 1]) == int(ids[0, 1, 1]), "shared vertex b got two ids"
    assert int(ids[0, 0, 2]) != int(ids[0, 1, 2]), "distinct vertices got one id"


def test_a_shared_vertex_emits_the_same_code_in_every_face():
    """MeshGPT quantizes *unique* vertices (`scatter_mean`) and gathers the
    codes back per face (`get_at('b [n] q, b nf nvf -> b (nf nvf) q')`).

    Two consequences, and the second is the one that was missed: the decoded
    coordinate for a shared vertex is identical across the faces that touch it,
    so `canonicalize`'s exact-equality merge cannot crack the mesh -- and the
    token sequence becomes highly redundant, which is a regularity stage 2 can
    learn instead of memorizing.
    """
    model = _vqvae().eval()
    coords = _shared_vertex_mesh()
    codes = model.tokenize(coords)
    assert torch.equal(codes[0, 0, 0], codes[0, 1, 0])
    assert torch.equal(codes[0, 0, 1], codes[0, 1, 1])


def test_code_ids_are_one_space_disambiguated_by_position():
    """MeshAnything's `vocab_size = codebook_size + 3` -- ids do not carry the
    residual stage, position does. Offsetting each stage into its own id block
    tripled the stage-2 softmax to encode what the [..., depth] layout already
    says, and `get_output_from_indices` reads the stage off that axis.

    The codebook is shared across stages, as in both references, so an id means
    one vector -- which is what lets stage 2 embed a code by looking it up.
    """
    model = _vqvae().eval()
    codes = model.tokenize(_coords())
    assert int(codes.max()) < CODEBOOK, "ids escaped a single codebook_size space"

    # The same id at a different stage must decode to a different vector, which
    # is the whole reason position has to carry the stage.
    a = model.lookup(torch.zeros(1, 1, VERTS, DEPTH, dtype=torch.long))
    b = model.lookup(torch.tensor([0] + [1] * (DEPTH - 1))
                     .expand(1, 1, VERTS, DEPTH).contiguous())
    assert not torch.allclose(a, b)


def test_decoder_logit_shape():
    """Nine coordinate distributions per face, over the bin vocabulary."""
    model = _vqvae()
    logits = model.decode(model.quantize(model.encode(_coords()), *model.vertex_ids(_coords()))[0])
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
    empty = torch.zeros((0, VERTS, DEPTH), dtype=torch.long)
    verts, faces = _vqvae().eval().detokenize(empty)
    assert verts.shape == (0, 3) and faces.shape == (0, 3)


def test_out_of_range_codes_are_clamped_not_raised():
    """A stage-2 transformer can emit any integer; the decoder must survive it."""
    model = _vqvae().eval()
    codes = torch.full((F, VERTS, DEPTH), CODEBOOK + 99, dtype=torch.long)
    verts, faces = model.detokenize(codes)
    assert faces.shape == (F, 3)


def test_straight_through_gradients_reach_the_encoder():
    """Quantization is not differentiable; without the STE the encoder is orphaned."""
    model, coords = _vqvae(), _coords()
    z = model.encode(coords)
    z_q, _, _ = model.quantize(z, *model.vertex_ids(coords))
    z_q.sum().backward()
    grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert grads, "no encoder parameter received a gradient"
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_commitment_loss_is_finite_and_positive():
    """Zero commitment loss on random input would mean the quantizer is a no-op."""
    model = _vqvae()
    c = _coords()
    _, loss, _ = model.quantize(model.encode(c), *model.vertex_ids(c))
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
    torch.manual_seed(0)
    model = _vqvae().train()
    coords = _coords()
    z = model.encode(coords)
    model.quantize(z, *model.vertex_ids(coords))

    # Compare *norms*, not nearest-neighbour distance. Distance depends on how
    # many features there are per codebook entry -- with fewer points than
    # entries kmeans parks one on top of each and the ratio collapses to ~0,
    # which made an earlier version of this test measure the test's own
    # dimensions rather than the model. Norm is regime-independent: entries
    # seeded from data carry the data's scale, an origin-initialized book
    # carries ~1/codebook_size.
    v = model.vert_proj(z).reshape(-1, model.codebook_dim)
    assert (float(model.quantizer.codebooks.norm(dim=-1).mean())
            > 0.3 * float(v.norm(dim=-1).mean()))


def test_rotation_trick_changes_the_gradient_but_not_the_forward_pass():
    """`rotation_trick` has to reach ResidualVQ to mean anything. It rotates the
    gradient from the code to the encoder output instead of copy-pasting it
    (Fifty et al., ICLR 2025, arXiv:2410.06424), so the forward value is
    identical by construction and only the backward pass moves."""
    def run(rotation_trick):
        torch.manual_seed(0)
        model = _vqvae(rotation_trick=rotation_trick).train()
        coords = _coords()
        z = model.encode(coords).detach().requires_grad_(True)
        z_q, _, codes = model.quantize(z, *model.vertex_ids(coords))
        z_q.square().sum().backward()
        return z_q.detach(), z.grad, codes

    (q_ste, g_ste, c_ste), (q_rot, g_rot, c_rot) = run(False), run(True)
    # The decision is what must not move. The value carries float32 rounding
    # from the rotation arithmetic -- ~1e-5 once a shared codebook changes the
    # reduction order -- so it is compared at a tolerance that means "same
    # code", not "same bits".
    assert torch.equal(c_ste, c_rot), "the forward pass picked different codes"
    assert torch.allclose(q_ste, q_rot, atol=1e-4), "the forward pass must not move"
    assert not torch.allclose(g_ste, g_rot, atol=1e-6), "rotation_trick never reached the quantizer"


def test_a_frozen_codebook_never_moves():
    """Stage 2 holds the tokenizer as a submodule, so Lightning's per-epoch
    `model.train()` recurses in and undoes the `eval()` it was constructed with.
    An EMA codebook updates on `training` alone, so without the freeze it would
    rewrite the vocabulary the transformer is being trained to predict, mid-run.
    """
    model, coords = _vqvae().train(), _coords()
    z = model.encode(coords)
    model.quantize(z, *model.vertex_ids(coords))   # seeds while still trainable
    for p in model.parameters():
        p.requires_grad_(False)

    before = model.quantizer.codebooks.clone()
    model.quantize(z.detach(), *model.vertex_ids(coords))
    assert torch.equal(model.quantizer.codebooks, before)


def test_padded_faces_come_back_as_negative_one():
    """The quantizer marks padding rather than assigning it a code. Anything
    counting occupancy or writing meshes has to skip those, not clamp them."""
    model = _vqvae().eval()
    pad = torch.zeros(B, F, dtype=torch.bool)
    pad[1, -2:] = True
    c = _coords()
    _, _, codes = model.quantize(model.encode(c, pad), *model.vertex_ids(c, pad), pad)
    # Every vertex of a padded face, at every residual stage.
    assert codes[1, -2:].shape == (2, VERTS, DEPTH)
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


def test_decoder_reads_the_condition():
    """Noise-resistant decoder, arXiv:2406.10163 section 4.2: the shape condition
    is injected into the VQ-VAE decoder so it can *correct* imperfect codes
    rather than only smooth them. If the logits ignore the condition, the whole
    fine-tune stage is a no-op."""
    model = _vqvae().eval()
    with torch.no_grad():
        z_q = model.quantize(model.encode(_coords()), *model.vertex_ids(_coords()))[0]
        a = model.decode(z_q, cond=model.encode(_coords()))
        b = model.decode(z_q, cond=model.encode(_coords()))
        plain = model.decode(z_q)
    assert a.shape == plain.shape == (B, F, 9, NUM_BINS)
    assert not torch.allclose(a, b), "two different conditions gave one answer"


def test_sample_temp_perturbs_codes_only_while_training():
    """Gumbel noise on the codebook logits is the paper's augmentation for the
    fine-tune, so it must vanish at eval -- otherwise every reported
    reconstruction number becomes a random variable."""
    model, coords = _vqvae(), _coords()
    with torch.no_grad():
        model.train()
        model.quantize(model.encode(coords), *model.vertex_ids(coords))   # seed kmeans_init
        z = model.encode(coords)
        noisy = [model.quantize(z, *model.vertex_ids(coords), sample_temp=1e3)[2] for _ in range(2)]
        assert not torch.equal(*noisy), "sample_temp never reached the quantizer"

        model.eval()
        z = model.encode(coords)
        calm = [model.quantize(z, *model.vertex_ids(coords), sample_temp=1e3)[2] for _ in range(2)]
        assert torch.equal(*calm), "noise leaked into eval"


def test_noise_resistant_step_trains_the_decoder_and_nothing_else():
    """The paper fine-tunes the decoder only: the encoder and the codebook stay
    put, or the vocabulary stage 2 already trained against would move."""
    module = MeshVQVAEModule(num_bins=NUM_BINS, codebook_size=CODEBOOK, depth=DEPTH,
                             d_model=16, n_head=2, num_layers=1, dropout=0.0,
                             noise_resistant=True, noise_temp=1.0)
    batch = {"coords": _coords(), "pad_mask": torch.zeros(B, F, dtype=torch.bool),
             "cond": _coords(), "cond_pad_mask": torch.zeros(B, F, dtype=torch.bool)}
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()

    net = module.network
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in net.decoder.parameters()), "decoder got no gradient"
    for name, part in (("encoder", net.encoder), ("coord_embed", net.coord_embed),
                       ("vert_proj", net.vert_proj), ("pos_embed", net.pos_embed)):
        assert not any(p.requires_grad for p in part.parameters()), f"{name} is trainable"

    # A frozen weight the optimizer still owns is a moment buffer per codebook
    # entry, carried for a parameter that can never move.
    opt = module.configure_optimizers()
    opt = opt["optimizer"] if isinstance(opt, dict) else opt
    owned = {id(p) for group in opt.param_groups for p in group["params"]}
    assert not any(id(p) in owned for p in net.encoder.parameters())


def test_noise_resistant_leaves_the_codebook_alone():
    """The fine-tune trains the decoder, so `quantize`'s freeze can no longer key
    off "no parameter wants a gradient" -- it has to ask whether anything can
    move the encoder. Otherwise the EMA rewrites the vocabulary stage 2 was
    trained against, mid-fine-tune."""
    module = MeshVQVAEModule(num_bins=NUM_BINS, codebook_size=CODEBOOK, depth=DEPTH,
                             d_model=16, n_head=2, num_layers=1, dropout=0.0,
                             noise_resistant=True, noise_temp=1.0)
    batch = {"coords": _coords(), "pad_mask": torch.zeros(B, F, dtype=torch.bool),
             "cond": _coords(), "cond_pad_mask": torch.zeros(B, F, dtype=torch.bool)}
    book = module.network.quantizer.layers[0]._codebook
    module.training_step(batch, 0)                  # seeds, then would drift
    before, sizes = module.network.quantizer.codebooks.clone(), book.cluster_size.clone()
    # kmeans_init runs even under a frozen EMA, so there is a real codebook here
    # to hold still -- without this the assertions below would pass on zeros.
    assert float(before.abs().sum()) > 0

    for _ in range(3):
        module.training_step(batch, 0)

    # `cluster_size` is the EMA state, so this is the exact signal: it moves by
    # 3.1 in a step of ordinary stage-1 training and by exactly 0 here, which
    # means `update_codebook` -- and with it the EMA update and dead-code
    # replacement -- never ran.
    assert torch.equal(book.cluster_size, sizes)
    # The entries themselves settle by ~2e-5 once after the first step and then
    # do not move again (measured constant over 8 further steps), so this is a
    # tolerance on that one-time settle, not room for drift.
    assert torch.allclose(module.network.quantizer.codebooks, before, atol=1e-4)


def _nr_module(**kw):
    return MeshVQVAEModule(num_bins=NUM_BINS, codebook_size=CODEBOOK, depth=DEPTH,
                           d_model=16, n_head=2, num_layers=1, dropout=0.0,
                           noise_resistant=True, noise_temp=1e3, **kw)


def _nr_batch():
    return {"coords": _coords(), "pad_mask": torch.zeros(B, F, dtype=torch.bool),
            "cond": _coords(), "cond_pad_mask": torch.zeros(B, F, dtype=torch.bool)}


def _warm(module, batch):
    """One pass to get `kmeans_init` out of the way.

    That init is stochastic and rewrites the codebook, so anything measuring
    determinism or RNG hygiene has to happen after it -- in a real run it fires
    on the first training batch, long before any validation.
    """
    module._eval_metrics(batch)
    return module


def test_noise_resistant_reports_reconstruction_from_noised_codes():
    """`val_recon_ce` is measured with clean argmin codes -- the Gumbel noise is
    gated on training mode, so validation never sees it. Monitoring that would
    early-stop the fine-tune exactly when it begins trading clean-code accuracy
    for the robustness it exists to buy, so the noisy figure is reported too."""
    module, batch = _nr_module().eval(), _nr_batch()
    _, metrics = _warm(module, batch)._eval_metrics(batch)
    assert "noisy_recon_ce" in metrics
    assert torch.isfinite(metrics["noisy_recon_ce"])
    # Only that it differs, not that it is worse: an untrained decoder emits
    # near-uniform logits whatever codes it is handed, so the *direction* is
    # meaningless here. Differing at all is what proves the noise reached the
    # quantizer; `test_sample_temp_perturbs_codes_only_while_training` pins the
    # mechanism.
    assert not torch.equal(metrics["noisy_recon_ce"], metrics["recon_ce"])


def test_the_noisy_metric_is_not_a_random_variable():
    """Seeded, so the early-stopping curve compares like with like. An unseeded
    draw would make every epoch's value partly noise and the patience counter
    fire on sampling luck."""
    module, batch = _nr_module().eval(), _nr_batch()
    _warm(module, batch)
    a = module._eval_metrics(batch)[1]["noisy_recon_ce"]
    b = module._eval_metrics(batch)[1]["noisy_recon_ce"]
    assert torch.equal(a, b)


def test_measuring_the_noisy_metric_does_not_disturb_training_rng():
    """It runs the quantizer in train mode to un-gate the noise, so it has to
    put both the module mode and the global RNG stream back."""
    module, batch = _nr_module().eval(), _nr_batch()
    _warm(module, batch)
    quantizer = module.network.quantizer
    torch.manual_seed(7)
    before_state = torch.random.get_rng_state()
    module._eval_metrics(batch)
    assert torch.equal(torch.random.get_rng_state(), before_state), "RNG stream moved"
    assert not quantizer.training, "quantizer left in train mode"


def test_the_noisy_pass_does_not_move_the_codebook():
    """Un-gating the noise means running the quantizer in train mode, which is
    also what un-gates the EMA. The freeze has to hold through it."""
    module = _nr_module()
    book = module.network.quantizer.layers[0]._codebook
    module._eval_metrics(_nr_batch())            # seeds
    sizes = book.cluster_size.clone()
    module._eval_metrics(_nr_batch())
    assert torch.equal(book.cluster_size, sizes)


def test_plain_stage_one_reports_no_noisy_metric():
    """Nothing to be robust to when the codes are argmin anyway."""
    module = MeshVQVAEModule(num_bins=NUM_BINS, codebook_size=CODEBOOK, depth=DEPTH,
                             d_model=16, n_head=2, num_layers=1, dropout=0.0)
    _, metrics = module._eval_metrics({"coords": _coords(),
                                       "pad_mask": torch.zeros(B, F, dtype=torch.bool)})
    assert "noisy_recon_ce" not in metrics


def test_noise_resistant_needs_a_condition():
    """Fine-tuning without one would silently train an unconditioned decoder --
    the paper's whole point is that the condition is what corrects bad codes."""
    module = MeshVQVAEModule(num_bins=NUM_BINS, codebook_size=CODEBOOK, depth=DEPTH,
                             d_model=16, n_head=2, num_layers=1, dropout=0.0,
                             noise_resistant=True)
    try:
        module.training_step({"coords": _coords(),
                              "pad_mask": torch.zeros(B, F, dtype=torch.bool)}, 0)
    except ValueError:
        return
    raise AssertionError("a conditionless noise-resistant step was accepted")


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
