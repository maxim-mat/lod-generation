"""Losses over padded face sets.

Two regimes, and the choice between them is forced by the model, not by taste
(see the plan's D4). With a canonical face order the model can be told which
slot to fill and a slot-to-slot MSE is correct. Without one -- which is the
only honest setting for a permutation-equivariant model -- the target is a
*set*, and the loss has to find the correspondence first. That is the DETR
construction (Carion et al., arXiv:2005.12872 section 3.1), with the presence
channel playing the role of their no-object class.

Convention throughout: tensors are [B, C, F] channels-first with C = 10
(9 coordinates + presence), and `mask` is [B, F] bool with True at real faces.
Coordinate terms are masked; the presence term never is, because predicting
"absent" at a padded slot is exactly what the model has to learn.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

N_COORD = 9
PRESENCE = 9


def _masked_mean(per_element, mask):
    """Mean of ``[B, C, F]`` over real faces only. ``clamp`` guards empty rows."""
    m = mask[:, None, :].to(per_element.dtype)
    return (per_element * m).sum() / (m.sum() * per_element.shape[1]).clamp(min=1.0)


def masked_mse_loss(pred, target, mask, presence_weight=1.0):
    """Slot-to-slot squared error, coordinates masked, presence not.

    Args:
        pred, target: ``[B, 10, F]``.
        mask: ``[B, F]`` bool, True at real faces.
        presence_weight: multiplier on the presence channel's contribution.
            Face count moves the geometric metrics far more than a fraction of
            a bin on one vertex does, so this is a real knob and not cosmetic.

    Returns:
        Tensor: scalar.
    """
    coord = _masked_mean((pred[:, :N_COORD] - target[:, :N_COORD]) ** 2, mask)
    presence = ((pred[:, PRESENCE] - target[:, PRESENCE]) ** 2).mean()
    return coord + presence_weight * presence


@torch.no_grad()
def hungarian_match(pred, target, mask, match_presence_weight=1.0):
    """Optimal assignment of prediction slots to target slots, per batch item.

    Cost is mean absolute error over the nine coordinates plus a weighted
    presence term. L1 rather than L2 inside the matcher on DETR's grounds: the
    squared cost lets one far-off coordinate dominate an otherwise good match,
    and the matcher's job is correspondence, not calibration.

    Padded *target* slots are still matched -- something has to absorb the
    surplus prediction slots, and the surplus must learn to predict absence.

    Args:
        pred, target: ``[B, 10, F]``.
        mask: ``[B, F]`` bool. Unused for the assignment itself; kept in the
            signature so callers cannot accidentally pass an unmasked pair to
            the loss and a masked one to the matcher.
        match_presence_weight: presence weight inside the matching cost only.

    Returns:
        LongTensor: ``[B, F]`` where entry ``i`` is the target index assigned to
        prediction slot ``i``.
    """
    b, _, f = pred.shape
    # [B, F_pred, F_tgt]: broadcast prediction slots against target slots.
    p = pred.permute(0, 2, 1)          # [B, F, 10]
    t = target.permute(0, 2, 1)
    coord_cost = (p[:, :, None, :N_COORD] - t[:, None, :, :N_COORD]).abs().mean(-1)
    pres_cost = (p[:, :, None, PRESENCE] - t[:, None, :, PRESENCE]).abs()
    cost = (coord_cost + match_presence_weight * pres_cost).cpu().numpy()

    out = torch.empty((b, f), dtype=torch.long)
    for i in range(b):
        rows, cols = linear_sum_assignment(cost[i])
        out[i, torch.from_numpy(rows)] = torch.from_numpy(cols)
    return out.to(pred.device)


def hungarian_loss(pred, target, mask, presence_weight=1.0,
                   match_presence_weight=1.0):
    """Set loss: match first, then the same masked MSE on the matched pairs.

    The matcher runs under `no_grad` -- the assignment is a discrete decision
    and is treated as a constant, exactly as DETR does. Gradients flow only
    through the reordered squared error.

    Args:
        pred, target: ``[B, 10, F]``.
        mask: ``[B, F]`` bool, True at real *target* faces.
        presence_weight: as `masked_mse_loss`.
        match_presence_weight: presence weight inside the matching cost.

    Returns:
        Tensor: scalar.
    """
    assignment = hungarian_match(pred, target, mask, match_presence_weight)
    idx = assignment[:, None, :].expand(-1, target.shape[1], -1)
    # Reorder the *target* onto the prediction slots, and the mask with it, so
    # the masked mean below counts the same real faces it always did.
    target_m = torch.gather(target, 2, idx)
    mask_m = torch.gather(mask, 1, assignment)
    return masked_mse_loss(pred, target_m, mask_m, presence_weight)


def discrete_ce_loss(logits, target_bins, mask, presence_logits,
                     presence_target, presence_weight=1.0):
    """Cross-entropy over coordinate bins, plus binary presence.

    The D3PM arm's x0-parameterisation: nine independent categorical heads over
    `num_bins` classes each, one per coordinate channel.

    Args:
        logits: ``[B, 9, K, F]`` unnormalised bin scores.
        target_bins: ``[B, 9, F]`` int64 in ``[0, K)``.
        mask: ``[B, F]`` bool, True at real faces.
        presence_logits: ``[B, F]`` unnormalised.
        presence_target: ``[B, F]`` float in {0, 1}.
        presence_weight: multiplier on the presence term.

    Returns:
        Tensor: scalar.
    """
    b, c, k, f = logits.shape
    ce = F.cross_entropy(
        logits.permute(0, 2, 1, 3).reshape(b, k, c * f),
        target_bins.reshape(b, c * f),
        reduction="none",
    ).reshape(b, c, f)
    coord = _masked_mean(ce, mask)
    presence = F.binary_cross_entropy_with_logits(
        presence_logits, presence_target.to(presence_logits.dtype))
    return coord + presence_weight * presence
