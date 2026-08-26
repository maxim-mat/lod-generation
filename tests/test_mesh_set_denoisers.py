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


def test_padded_faces_do_not_change_real_outputs():
    torch.manual_seed(0)
    net = ConditionalMeshUNet(base=16, cond_dim=32, time_dim=32, n_head=2).eval()
    x, t, cond, mask, cond_mask = _inputs()
    with torch.no_grad():
        a = net(x, t, cond, mask, cond_mask)
        x2 = x.clone()
        x2[:, :, ~mask[0]] = 99.0            # scribble on the padding
        b = net(x2, t, cond, mask, cond_mask)
    # Convolution has a receptive field, so only assert on faces far from the
    # padded tail; attention is the part that must be exactly masked.
    assert torch.allclose(a[:, :, :4], b[:, :, :4], atol=1e-4)


def test_unet_discrete_head_shapes():
    net = ConditionalMeshUNet(base=16, cond_dim=32, time_dim=32, n_head=2,
                              out_bins=128)
    x, t, cond, mask, cond_mask = _inputs()
    logits, presence = net(x, t, cond, mask, cond_mask)
    assert logits.shape == (2, 9, 128, 16)
    assert presence.shape == (2, 16)


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
