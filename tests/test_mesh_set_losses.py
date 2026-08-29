"""Loss contracts for the face-set diffusion branch.

The properties that matter: padding never contributes a coordinate gradient,
presence contributes everywhere, and the Hungarian loss is invariant to a
permutation of the prediction slots while the MSE loss is not. That last pair
is the whole reason both exist.
"""
import numpy as np
import pytest
import torch

from src.models.mesh_set_losses import (
    discrete_ce_loss,
    hungarian_loss,
    hungarian_match,
    masked_mean_per_sample,
    masked_mse_loss,
)


def _batch(n_real=3, width=8, batch=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    target = torch.zeros(batch, 10, width)
    target[:, :9] = torch.rand(batch, 9, width, generator=g) - 0.5
    mask = torch.zeros(batch, width, dtype=torch.bool)
    mask[:, :n_real] = True
    target[:, 9] = torch.where(mask, 0.5, -0.5)
    target[:, :9] = target[:, :9] * mask[:, None, :]
    return target, mask


def test_masked_mse_ignores_padded_coordinates():
    target, mask = _batch()
    pred = target.clone()
    base = masked_mse_loss(pred, target, mask)
    pred[:, :9, mask[0].sum():] += 100.0        # garbage in padding only
    assert torch.isclose(masked_mse_loss(pred, target, mask), base)
    assert base.item() == pytest.approx(0.0, abs=1e-7)


def test_masked_mse_counts_padded_presence():
    target, mask = _batch()
    pred = target.clone()
    pred[:, 9, mask[0].sum():] = 0.5            # claims padding is a real face
    assert masked_mse_loss(pred, target, mask).item() > 0.0


def test_presence_weight_scales_only_presence():
    target, mask = _batch()
    pred = target.clone()
    pred[:, 9] += 0.1
    a = masked_mse_loss(pred, target, mask, presence_weight=1.0)
    b = masked_mse_loss(pred, target, mask, presence_weight=2.0)
    assert b.item() == pytest.approx(2.0 * a.item(), rel=1e-5)


def test_mse_is_not_permutation_invariant():
    target, mask = _batch(n_real=4, width=8, batch=1)
    perm = torch.tensor([3, 2, 1, 0, 4, 5, 6, 7])
    shuffled = target[:, :, perm]
    assert masked_mse_loss(shuffled, target, mask).item() > 1e-4


def test_hungarian_is_permutation_invariant():
    target, mask = _batch(n_real=4, width=8, batch=1)
    perm = torch.tensor([3, 2, 1, 0, 4, 5, 6, 7])
    shuffled = target[:, :, perm]
    loss = hungarian_loss(shuffled, target, mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_hungarian_match_recovers_the_permutation():
    target, mask = _batch(n_real=4, width=8, batch=1)
    perm = torch.tensor([3, 2, 1, 0, 4, 5, 6, 7])
    assignment = hungarian_match(target[:, :, perm], target, mask)
    # Prediction slot i holds target perm[i], so it must be assigned perm[i].
    assert assignment[0, :4].tolist() == perm[:4].tolist()


def test_hungarian_equals_mse_when_already_aligned():
    target, mask = _batch(n_real=4, width=8, batch=2, seed=3)
    pred = target + 0.01
    assert hungarian_loss(pred, target, mask).item() == pytest.approx(
        masked_mse_loss(pred, target, mask).item(), rel=1e-4)


def test_hungarian_handles_an_all_padding_row():
    target, mask = _batch(n_real=4, width=8, batch=2)
    mask[1] = False
    target[1, 9] = -0.5
    loss = hungarian_loss(target.clone(), target, mask)
    assert torch.isfinite(loss)


def test_discrete_ce_ignores_padded_bins():
    b, k, w = 1, 128, 8
    bins = torch.randint(0, k, (b, 9, w))
    mask = torch.zeros(b, w, dtype=torch.bool)
    mask[:, :3] = True
    logits = torch.zeros(b, 9, k, w)
    logits.scatter_(2, bins.unsqueeze(2), 20.0)        # confident and correct
    presence_logits = torch.where(mask, 10.0, -10.0)[:, None].squeeze(1)
    loss_a = discrete_ce_loss(logits, bins, mask, presence_logits, mask.float())
    logits[:, :, :, 3:] = 0.0                          # wreck the padded slots
    loss_b = discrete_ce_loss(logits, bins, mask, presence_logits, mask.float())
    assert loss_a.item() == pytest.approx(loss_b.item(), rel=1e-5)


# --- per-sample means and min-SNR reweighting --------------------------------

def test_masked_mean_per_sample_divides_by_channels_too():
    """Mean over real faces AND the nine channels -> ones average to one.

    The regression this pins: `_shared_step`'s inline coord_err_bins divided by
    `mask.sum()` alone, so a nine-channel error read nine times too large (378
    bins on a 128-bin grid).
    """
    _, mask = _batch(n_real=3, width=8, batch=2)
    per_element = torch.ones(2, 9, 8)
    out = masked_mean_per_sample(per_element, mask)
    assert out.shape == (2,)
    assert torch.allclose(out, torch.ones(2))


def test_masked_mean_per_sample_ignores_padding():
    _, mask = _batch(n_real=3, width=8, batch=2)
    per_element = torch.zeros(2, 9, 8)
    per_element[:, :, 3:] = 100.0                  # padding only
    assert torch.allclose(masked_mean_per_sample(per_element, mask),
                          torch.zeros(2))


def test_uniform_min_snr_weight_leaves_the_loss_alone():
    """The weight reweights the batch; it must not rescale the gradient.

    A constant weight is a no-op, so flipping min_snr_gamma on cannot silently
    change the effective learning rate.
    """
    target, mask = _batch()
    pred = target + 0.1
    base = masked_mse_loss(pred, target, mask)
    for c in (0.5, 1.0, 7.0):
        w = torch.full((target.shape[0],), c)
        assert torch.isclose(masked_mse_loss(pred, target, mask, weight=w), base,
                             rtol=1e-5)


def test_min_snr_weight_selects_between_samples():
    """Weight [1, 0] must give exactly the loss of sample 0 alone."""
    target, mask = _batch(batch=2)
    pred = target.clone()
    pred[0] += 0.3
    pred[1] += 2.0
    only_first = masked_mse_loss(pred[:1], target[:1], mask[:1])
    weighted = masked_mse_loss(pred, target, mask,
                               weight=torch.tensor([1.0, 0.0]))
    assert torch.isclose(weighted, only_first, rtol=1e-5)
