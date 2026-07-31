"""Class balancing for the discrete losses, and the de-biasing it forces.

Off dominates both categorical targets -- 73.7% of node slots and 99.0% of node
pairs on The Hague LOD2 -- so an unweighted mean spends almost all of the
gradient on padding, exactly as it did for the coordinate MSE. GroundSurface,
the class that defines the base plane, is 0.67% of node slots.

The catch the MSE does not have: weighting a *classification* by target class
moves its optimum. For weights w, the minimiser of the weighted cross entropy is
p_hat(c) ∝ w_c · p(c), and `sample_zs_from_zt_and_pred` consumes
softmax(pred) directly as p(x0 | xt). Left uncorrected, balancing would tilt
every generated graph toward the rare classes. So the weights must be divided
back out before sampling, which for logits means subtracting log(w).
"""
import math

import pytest
import torch

from src.models.diffusion import CityJSONDiffusionModule

N = 6


def _model(x_marginals=None, e_marginals=None, class_balance=0.5,
           off_ce_weight=1.0):
    return CityJSONDiffusionModule(
        num_node_classes=5, num_edge_classes=3, hidden_dim=8, edge_dim=4,
        global_dim=4, n_head=2, num_layers=1, T=10, n_max=N,
        x_marginals=x_marginals, e_marginals=e_marginals,
        class_balance=class_balance, off_ce_weight=off_ce_weight)


# The Hague LOD2 at n_max=150, measured.
X_MARG = torch.tensor([0.1594, 0.0067, 0.0173, 0.0802, 0.7365])
E_MARG = torch.tensor([0.98998, 0.003352, 0.006668])


# --- weight construction --------------------------------------------------

def test_no_balancing_leaves_the_loss_proper():
    m = _model(X_MARG, E_MARG, class_balance=0.0, off_ce_weight=1.0)
    assert torch.allclose(m.x_ce_weight, torch.ones(5))
    assert torch.allclose(m.e_ce_weight, torch.ones(3))


def test_full_balancing_equalises_every_class_contribution():
    """beta=1 makes pi_c * w_c identical across classes."""
    m = _model(X_MARG, E_MARG, class_balance=1.0, off_ce_weight=1.0)
    contribution = X_MARG * m.x_ce_weight
    assert torch.allclose(contribution, contribution[0].expand(5), rtol=1e-4)


def test_off_weight_is_controllable_independently_of_the_real_classes():
    balanced = _model(X_MARG, E_MARG, class_balance=0.5, off_ce_weight=1.0)
    damped = _model(X_MARG, E_MARG, class_balance=0.5, off_ce_weight=0.25)
    assert damped.x_ce_weight[-1] == pytest.approx(
        float(balanced.x_ce_weight[-1]) * 0.25, rel=1e-5)
    # the real classes are untouched by it
    assert torch.allclose(damped.x_ce_weight[:-1], balanced.x_ce_weight[:-1])
    assert damped.e_ce_weight[0] == pytest.approx(
        float(balanced.e_ce_weight[0]) * 0.25, rel=1e-5)


def test_the_rare_ground_class_gains_weight_over_the_common_ones():
    m = _model(X_MARG, E_MARG, class_balance=0.5)
    ground, vertex, off = m.x_ce_weight[1], m.x_ce_weight[0], m.x_ce_weight[4]
    assert ground > vertex > off


def test_a_zero_frequency_class_does_not_blow_up():
    marg = torch.tensor([0.5, 0.0, 0.0, 0.2, 0.3])
    m = _model(marg, E_MARG, class_balance=1.0)
    assert torch.isfinite(m.x_ce_weight).all()


def test_uniform_marginals_give_uniform_weights():
    m = _model(None, None, class_balance=1.0)
    assert torch.allclose(m.x_ce_weight, m.x_ce_weight[0].expand(5))


# --- the de-biasing that keeps sampling correct ---------------------------

def test_debias_recovers_the_true_posterior_from_a_weighted_optimum():
    """A perfectly trained weighted model emits p ∝ w·p_true; undo it exactly."""
    m = _model(X_MARG, E_MARG, class_balance=1.0)
    p_true = torch.tensor([[[0.5, 0.1, 0.05, 0.3, 0.05]]])
    weighted = p_true * m.x_ce_weight
    weighted = weighted / weighted.sum(-1, keepdim=True)

    logits = weighted.log()
    recovered = torch.softmax(m._debias(logits, m.x_ce_weight), dim=-1)
    assert torch.allclose(recovered, p_true, atol=1e-6)


def test_debias_is_a_noop_without_balancing():
    m = _model(X_MARG, E_MARG, class_balance=0.0, off_ce_weight=1.0)
    logits = torch.randn(2, N, 5)
    assert torch.allclose(m._debias(logits, m.x_ce_weight), logits, atol=1e-6)


def test_sampling_debiases_the_network_output(monkeypatch):
    """The chain must consume p_true, not the balanced surrogate."""
    m = _model(X_MARG, E_MARG, class_balance=1.0)
    seen = {}

    def spy(pos_t, X_t, E_t, pred_pos, pred_X, pred_E, t_int, s_int, node_mask):
        seen.setdefault("pred_X", pred_X.clone())
        return pos_t, X_t, E_t

    monkeypatch.setattr(m.noise, "sample_zs_from_zt_and_pred", spy)
    raw = {}
    real_forward = m.forward

    def capture(*a, **k):
        out = real_forward(*a, **k)
        raw.setdefault("X", out[2].clone())
        return out

    monkeypatch.setattr(m, "forward", capture)
    m.sample(batch_size=1)

    expected = m._debias(raw["X"], m.x_ce_weight)
    assert torch.allclose(seen["pred_X"], expected, atol=1e-5)
    assert not torch.allclose(seen["pred_X"], raw["X"], atol=1e-3)


# --- the losses actually use the weights ----------------------------------

def test_weighted_node_loss_reacts_more_to_a_rare_class_error():
    """Getting Ground wrong must cost more than getting Off wrong, once balanced.

    Measured on a two-slot batch: a single-slot batch cannot show this, because
    a weighted mean over one element divides the weight straight back out.
    """
    plain = _model(X_MARG, E_MARG, class_balance=0.0, off_ce_weight=1.0)
    balanced = _model(X_MARG, E_MARG, class_balance=1.0, off_ce_weight=1.0)

    target = torch.zeros(1, 2, 5)
    target[0, 0, 1] = 1.0            # slot 0 is Ground
    target[0, 1, 4] = 1.0            # slot 1 is Off

    def delta(model, slot):
        base = torch.zeros(1, 2, 5)
        worse = base.clone()
        worse[0, slot, target[0, slot].argmax()] = -4.0    # push the truth down
        return float(model._node_ce(worse, target)) - float(model._node_ce(base, target))

    assert delta(plain, 0) == pytest.approx(delta(plain, 1), rel=1e-5)
    assert delta(balanced, 0) > delta(balanced, 1) * 5


def test_off_edge_class_is_damped_not_the_vertex_face_class():
    """EDGE_OFF is index 0, unlike OFF which is the last node class.

    Scaling the last entry damped EDGE_VF -- the rarest, most informative edge
    class -- while leaving the 99% off-class untouched.
    """
    from src.dataset.dataset import EDGE_OFF, EDGE_VF

    balanced = _model(X_MARG, E_MARG, class_balance=0.5, off_ce_weight=1.0)
    damped = _model(X_MARG, E_MARG, class_balance=0.5, off_ce_weight=0.25)

    assert damped.e_ce_weight[EDGE_OFF] == pytest.approx(
        float(balanced.e_ce_weight[EDGE_OFF]) * 0.25, rel=1e-5)
    assert damped.e_ce_weight[EDGE_VF] == pytest.approx(
        float(balanced.e_ce_weight[EDGE_VF]), rel=1e-5)
