"""Denoiser shape and masking contracts.

No learning here -- these are the invariants that make a wrong wiring fail in
seconds instead of after an epoch: shapes survive a round trip, padded faces
cannot influence real ones, and dropping the condition is a legal call rather
than a crash.
"""
import pytest
import torch

from src.models.mesh_set_modules import timestep_embedding
from src.models.mesh_set_unet import ConditionalMeshUNet


from src.models.mesh_set_transformer import MeshSetTransformer


def _tf(pos_embed="none", **kw):
    return MeshSetTransformer(d_model=32, n_head=2, num_layers=2,
                              time_dim=32, pos_embed=pos_embed, **kw)


def _wake(net):
    """Undo the zero-init on the output head.

    Both denoisers zero-initialise `outc` on purpose, so an untrained model
    emits exactly 0 for every input. Every claim of the form "input change X
    does / does not reach the output" is then vacuous -- the negative version
    fails outright, and the positive version passes without testing anything.
    """
    torch.nn.init.normal_(net.outc.weight, std=0.05)
    torch.nn.init.normal_(net.outc.bias, std=0.05)
    return net


def _inputs(b=2, f=16, fc=8):
    x = torch.randn(b, 10, f)
    t = torch.rand(b)
    cond = torch.randn(b, 10, fc)
    mask = torch.zeros(b, f, dtype=torch.bool)
    mask[:, : f - 3] = True
    cond_mask = torch.ones(b, fc, dtype=torch.bool)
    return x, t, cond, mask, cond_mask


def test_timestep_embedding_shape_and_range():
    emb = timestep_embedding(torch.rand(5), 128)
    assert emb.shape == (5, 128)
    assert emb.abs().max() <= 1.0


def test_unet_preserves_shape():
    net = ConditionalMeshUNet(base=16, cond_dim=32, time_dim=32, n_head=2)
    x, t, cond, mask, cond_mask = _inputs()
    assert net(x, t, cond, mask, cond_mask).shape == x.shape


def test_unet_requires_length_divisible_by_eight():
    net = ConditionalMeshUNet(base=16, cond_dim=32, time_dim=32, n_head=2)
    x, t, cond, mask, cond_mask = _inputs(f=13)
    with pytest.raises(ValueError, match="divisible by 8"):
        net(x, t, cond, mask, cond_mask)


def test_unet_accepts_a_dropped_condition():
    net = ConditionalMeshUNet(base=16, cond_dim=32, time_dim=32, n_head=2)
    x, t, _, mask, _ = _inputs()
    assert net(x, t, None, mask, None).shape == x.shape


def test_output_is_invariant_to_batch_padding():
    """The property that matters: a building's prediction must not depend on
    which other buildings share its batch.

    Two bugs used to break this, both worth ~0.2 of a coordinate channel (~23
    bins of mean drift). `GroupNorm` pooled its mean and variance over the face
    axis *including* pad slots, so every real face was scaled by statistics
    that moved with the padded width -- reaching face 0 at 256 slots from the
    boundary, far outside any receptive field. And `SinusoidalFacePositions`
    divided the face index by the padded length, so face 5 was encoded
    differently at F_pad 24 than at F_pad 64.
    """
    from src.dataset.mesh_set_dataset import mesh_set_collate_fn

    def _item(n, seed):
        g = torch.Generator().manual_seed(seed)
        x = torch.zeros(n, 10)
        x[:, :9] = torch.rand(n, 9, generator=g) - 0.5
        x[:, 9] = 0.5
        c = torch.zeros(6, 10)
        c[:, :9] = torch.rand(6, 9, generator=g) - 0.5
        c[:, 9] = 0.5
        return {"x": x, "cond": c, "id": "a",
                "center": torch.zeros(3), "scale": torch.ones(3)}

    torch.manual_seed(0)
    target = _item(20, 1)
    alone = mesh_set_collate_fn([target], 8)                    # F_pad = 24
    crowded = mesh_set_collate_fn([target, _item(60, 2)], 8)    # F_pad = 64
    t = torch.zeros(1) + 0.5

    nets = {
        "unet": _wake(ConditionalMeshUNet(base=16, cond_dim=32, time_dim=32,
                                          n_head=2)).eval(),
        "tf-none": _wake(_tf(pos_embed="none")).eval(),
        "tf-sinusoidal": _wake(_tf(pos_embed="sinusoidal")).eval(),
    }
    for name, net in nets.items():
        with torch.no_grad():
            a = net(alone["x"], t, alone["cond"], alone["x_mask"], alone["cond_mask"])
            b = net(crowded["x"][:1], t, crowded["cond"][:1],
                    crowded["x_mask"][:1], crowded["cond_mask"][:1])
        drift = (a[0, :, :20] - b[0, :, :20]).abs().max().item()
        assert drift < 1e-5, f"{name}: batch composition moved a real face by {drift:.2e}"


def test_masked_groupnorm_ignores_padded_faces():
    """The norm's statistics must be computed over real faces only."""
    from src.models.mesh_set_modules import MaskedGroupNorm

    torch.manual_seed(0)
    gn = MaskedGroupNorm(4, 16)
    h = torch.randn(1, 16, 64)
    mask = torch.zeros(1, 64, dtype=torch.bool)
    mask[:, :20] = True
    h2 = h.clone()
    h2[:, :, 20:] = 99.0                       # garbage in the pad tail only
    with torch.no_grad():
        a, b = gn(h, mask), gn(h2, mask)
    assert torch.allclose(a[:, :, :20], b[:, :, :20], atol=1e-6)


def test_sinusoidal_positions_are_absolute():
    """Face i must encode the same however wide the batch was padded."""
    from src.models.mesh_set_modules import SinusoidalFacePositions

    pos = SinusoidalFacePositions(32, scale=200)
    assert torch.allclose(pos(24, torch.device("cpu"))[5],
                          pos(64, torch.device("cpu"))[5], atol=1e-7)


def test_unet_discrete_head_shapes():
    net = ConditionalMeshUNet(base=16, cond_dim=32, time_dim=32, n_head=2,
                              out_bins=128)
    x, t, cond, mask, cond_mask = _inputs()
    logits, presence = net(x, t, cond, mask, cond_mask)
    assert logits.shape == (2, 9, 128, 16)
    assert presence.shape == (2, 16)


def test_transformer_preserves_shape():
    x, t, cond, mask, cond_mask = _inputs()
    assert _tf()(x, t, cond, mask, cond_mask).shape == x.shape


def test_transformer_accepts_any_length():
    net = _tf()
    x, t, cond, mask, cond_mask = _inputs(f=13)      # not a multiple of 8
    assert net(x, t, cond, mask, cond_mask).shape == x.shape


def test_no_pe_transformer_is_permutation_equivariant():
    torch.manual_seed(0)
    net = _wake(_tf(pos_embed="none")).eval()
    x, t, cond, mask, cond_mask = _inputs(f=16)
    mask[:] = True                                    # no padding to confuse it
    perm = torch.randperm(16)
    with torch.no_grad():
        a = net(x, t, cond, mask, cond_mask)[:, :, perm]
        b = net(x[:, :, perm], t, cond, mask[:, perm], cond_mask)
    assert torch.allclose(a, b, atol=1e-5)


def test_pe_transformer_is_not_permutation_equivariant():
    torch.manual_seed(0)
    net = _wake(_tf(pos_embed="sinusoidal")).eval()
    x, t, cond, mask, cond_mask = _inputs(f=16)
    mask[:] = True
    perm = torch.randperm(16)
    with torch.no_grad():
        a = net(x, t, cond, mask, cond_mask)[:, :, perm]
        b = net(x[:, :, perm], t, cond, mask[:, perm], cond_mask)
    assert not torch.allclose(a, b, atol=1e-5)


def test_transformer_is_invariant_to_condition_order():
    """Cross-attention reads the condition as a set; the LOD1 face order must
    not change the output, which is what makes a length-mismatched condition
    legitimate in the first place."""
    torch.manual_seed(0)
    net = _wake(_tf()).eval()
    x, t, cond, mask, cond_mask = _inputs(fc=8)
    perm = torch.randperm(8)
    with torch.no_grad():
        a = net(x, t, cond, mask, cond_mask)
        b = net(x, t, cond[:, :, perm], mask, cond_mask[:, perm])
    assert torch.allclose(a, b, atol=1e-5)


def test_transformer_discrete_head_shapes():
    x, t, cond, mask, cond_mask = _inputs()
    logits, presence = _tf(out_bins=128)(x, t, cond, mask, cond_mask)
    assert logits.shape == (2, 9, 128, 16)
    assert presence.shape == (2, 16)


# --- adaLN-Zero time conditioning --------------------------------------------

def _tf_adaln(**kw):
    from src.models.mesh_set_transformer import MeshSetTransformer
    net = MeshSetTransformer(d_model=64, n_head=4, num_layers=3, time_dim=32,
                             dropout=0.0, **kw)
    return net.eval()


def test_adaln_blocks_start_as_the_identity():
    """Zero-init modulation means every gate is 0, so no block contributes.

    That is the "Zero" in adaLN-Zero (Peebles & Xie, arXiv:2212.09748 sec 3.2):
    the residual stream reaches the head untouched at step 0.
    """
    net = _tf_adaln(time_cond="adaln", pos_embed="none")
    torch.nn.init.normal_(net.outc.weight, std=0.1)   # undo the head zero-init
    x = torch.randn(2, 10, 16)
    t = torch.rand(2)
    with torch.no_grad():
        got = net(x, t, None, None, None)
        h = net.inp(x)
        want = net.outc(net.norm(h.permute(0, 2, 1)).permute(0, 2, 1))
    assert torch.allclose(got, want, atol=1e-6)


def test_adaln_carries_the_timestep_once_the_gates_are_open():
    net = _tf_adaln(time_cond="adaln", pos_embed="none")
    torch.nn.init.normal_(net.outc.weight, std=0.1)
    for blk in net.blocks:                            # open the zero-init gates
        torch.nn.init.normal_(blk.emb[1].weight, std=0.05)
    x = torch.randn(2, 10, 16)
    with torch.no_grad():
        lo = net(x, torch.zeros(2), None, None, None)
        hi = net(x, torch.ones(2), None, None, None)
    assert (lo - hi).abs().max() > 1e-3, "adaLN block ignores t"


def test_additive_time_conditioning_is_untouched():
    """The default path must be bit-identical -- the U-Net shares these blocks."""
    torch.manual_seed(0)
    a = _tf_adaln(time_cond="additive", pos_embed="none")
    torch.manual_seed(0)
    b = _tf_adaln(pos_embed="none")
    # Seed each head separately: the two init calls would otherwise draw from
    # different RNG states and the comparison would fail on the head alone.
    torch.manual_seed(1)
    torch.nn.init.normal_(a.outc.weight, std=0.1)
    torch.manual_seed(1)
    torch.nn.init.normal_(b.outc.weight, std=0.1)
    x = torch.randn(2, 10, 16)
    t = torch.rand(2)
    with torch.no_grad():
        assert torch.equal(a(x, t, None, None, None), b(x, t, None, None, None))


def test_adaln_keeps_permutation_equivariance():
    """Modulation is per-sample, not per-face, so a3's premise must survive."""
    net = _tf_adaln(time_cond="adaln", pos_embed="none")
    torch.nn.init.normal_(net.outc.weight, std=0.1)
    for blk in net.blocks:
        torch.nn.init.normal_(blk.emb[1].weight, std=0.05)
    x = torch.randn(1, 10, 16)
    t = torch.rand(1)
    perm = torch.randperm(16)
    with torch.no_grad():
        a = net(x, t, None, None, None)[:, :, perm]
        b = net(x[:, :, perm], t, None, None, None)
    assert (a - b).abs().max() < 1e-5
