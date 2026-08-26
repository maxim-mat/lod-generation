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


def test_unet_padding_leaks_through_groupnorm():
    """KNOWN DEFECT, pinned so it cannot be mistaken for correctness.

    This test previously asserted the opposite and passed -- but only because
    the output head is zero-initialised, so both sides were identically 0. Woken
    up, the U-Net moves a real face's output by O(0.1) in response to padding
    alone, at ANY distance from the pad boundary: at F=512 with 256 real faces,
    face 0 still moves. No convolution receptive field reaches that far.

    The path is `GroupNorm`, which normalises over (channel group x face axis)
    and so folds padded positions into the mean and variance every real face is
    then scaled by. Consequence: a building's prediction depends on which other
    buildings share its batch. Fix is a mask-aware norm (statistics over real
    positions only) or a per-face channel norm; both change trained-model
    semantics, so neither is applied here without a decision.
    """
    torch.manual_seed(0)
    net = _wake(ConditionalMeshUNet(base=16, cond_dim=32, time_dim=32, n_head=2)).eval()
    x, t, cond, mask, cond_mask = _inputs()
    with torch.no_grad():
        a = net(x, t, cond, mask, cond_mask)
        x2 = x.clone()
        x2[:, :, ~mask[0]] = 99.0            # scribble on the padding
        b = net(x2, t, cond, mask, cond_mask)
    leak = (a - b).abs().max().item()
    assert leak > 1e-3, (
        "the U-Net no longer leaks padding into real faces -- if that was "
        "deliberate, delete this test and tighten it to allclose instead")


def test_transformer_padding_is_exactly_masked():
    """Both transformers must be bit-exact under a scribbled pad tail.

    This is the property the U-Net lacks, and it is why `denoiser: transformer`
    is the only backbone whose output does not depend on batch composition.
    """
    torch.manual_seed(0)
    for pe in ("none", "sinusoidal"):
        net = _wake(_tf(pos_embed=pe)).eval()   # dropout off: this is an exactness claim
        x, t, cond, mask, cond_mask = _inputs()
        with torch.no_grad():
            a = net(x, t, cond, mask, cond_mask)
            x2 = x.clone()
            x2[:, :, ~mask[0]] = 99.0
            b = net(x2, t, cond, mask, cond_mask)
        # Only the REAL slots. A padded query still produces an output and that
        # output legitimately changes -- it is masked from the coordinate loss
        # and dropped by presence in `faces_to_mesh`. The claim is that no real
        # face moves.
        real = mask[0]
        assert torch.allclose(a[:, :, real], b[:, :, real], atol=1e-6), (
            f"pos_embed={pe} leaked padding into a real face")


def test_sinusoidal_positions_depend_on_the_padded_width():
    """KNOWN DEFECT, pinned. `SinusoidalFacePositions` normalises position by
    the *padded* length, so face i is encoded as i/(F_pad-1) -- and F_pad is the
    batch maximum rounded to 8. The same face therefore gets a different
    positional encoding depending on which buildings share its batch.

    Fix is to normalise by a fixed constant (mesh_data.max_faces) or to use
    absolute positions; both change trained-model semantics.
    """
    from src.models.mesh_set_modules import SinusoidalFacePositions

    pos = SinusoidalFacePositions(32)
    narrow = pos(24, torch.device("cpu"))
    wide = pos(64, torch.device("cpu"))
    assert not torch.allclose(narrow[5], wide[5], atol=1e-6)


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
