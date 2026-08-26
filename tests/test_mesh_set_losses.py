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
