"""Debiased Sinkhorn divergence between point clouds, in metres.

Entropic optimal transport solved in the log domain (Cuturi, "Sinkhorn
Distances: Lightspeed Computation of Optimal Transport", arXiv:1306.0895). The
entropic bias is removed with the Sinkhorn divergence
``S = OT(a, b) - OT(a, a) / 2 - OT(b, b) / 2``, using epsilon-scaling and
averaged dual updates (Feydy et al., "Interpolating between Optimal Transport
and MMDs using Sinkhorn Divergences", arXiv:1810.08278).

Cost is ``|x - y|^2 / 2``, so ``sqrt(2 S)`` tends to the 2-Wasserstein distance
as the blur shrinks and is exactly ``d`` for a copy translated by ``d``.

Torch rather than numpy: three n x n log-domain solves per pair are too slow in
CPU numpy at n = 2048, and `mesh_metrics` is numpy-only by its own rule.
"""
import math

import numpy as np
import torch

from src.eval.mesh_metrics import sample_surface


def _eps_schedule(diameter, blur, scaling=0.5):
    """Epsilons ``(diameter * scaling^k)^2`` stepping down to ``blur^2``.

    Starting at the blur is correct but slow: a cold solve at small epsilon
    needs on the order of diameter^2 / epsilon iterations. Annealing from the
    diameter reaches the same fixed point in a handful of steps per octave.
    """
    diameter = max(float(diameter), blur)
    steps = int(math.ceil(math.log(diameter / blur) / math.log(1.0 / scaling)))
    return [(diameter * scaling ** k) ** 2 for k in range(steps)] + [blur ** 2]


def _ot_eps(x, y, schedule, max_iters, tol):
    """Entropic OT cost ``<a, f> + <b, g>`` at the dual optimum, uniform weights."""
    log_a = torch.full((len(x),), -math.log(len(x)), dtype=x.dtype, device=x.device)
    log_b = torch.full((len(y),), -math.log(len(y)), dtype=x.dtype, device=x.device)
    cost = 0.5 * torch.cdist(x, y).pow(2)
    f = torch.zeros(len(x), dtype=x.dtype, device=x.device)
    g = torch.zeros(len(y), dtype=x.dtype, device=x.device)

    def step(eps, f, g):
        f_new = -eps * torch.logsumexp(log_b[None, :] + (g[None, :] - cost) / eps, dim=1)
        g_new = -eps * torch.logsumexp(log_a[:, None] + (f[:, None] - cost) / eps, dim=0)
        # Averaged (symmetric) updates: the plain alternating scheme oscillates.
        return 0.5 * (f + f_new), 0.5 * (g + g_new)

    for eps in schedule[:-1]:
        f, g = step(eps, f, g)
    for _ in range(max_iters):
        f_new, g_new = step(schedule[-1], f, g)
        delta = max(float((f_new - f).abs().max()), float((g_new - g).abs().max()))
        f, g = f_new, g_new
        if delta < tol:
            break
    return float((log_a.exp() * f).sum() + (log_b.exp() * g).sum())


def sinkhorn_divergence(x, y, blur=0.1, max_iters=500, tol=1e-6, device="cpu"):
    """``sqrt(2 S_eps(x, y))`` in the units of the input.

    Args:
        x, y: [n, 3] and [m, 3] numpy arrays in the same units (metres here).
        blur (float): entropic length scale, ``eps = blur ** 2``.
        max_iters (int): iteration cap at the final epsilon.
        tol (float): stop when no dual potential moves more than this, in
            squared input units.
        device: torch device for the solve. Always float64.

    Returns:
        float: ``nan`` if either cloud is empty. Identical inputs return exactly
        0, because the three solves then run on bit-identical tensors.
    """
    x = np.asarray(x, dtype=float).reshape(-1, 3)
    y = np.asarray(y, dtype=float).reshape(-1, 3)
    if len(x) == 0 or len(y) == 0:
        return float("nan")

    # One shared origin, then float64: at EPSG:28992 magnitudes (y ~ 450,000)
    # float32 squared distances cannot resolve a metre.
    origin = np.minimum(x.min(axis=0), y.min(axis=0))
    xt = torch.as_tensor(x - origin, dtype=torch.float64, device=device)
    yt = torch.as_tensor(y - origin, dtype=torch.float64, device=device)
    extent = np.maximum(x.max(axis=0), y.max(axis=0)) - origin
    schedule = _eps_schedule(np.linalg.norm(extent), blur)

    s = (_ot_eps(xt, yt, schedule, max_iters, tol)
         - 0.5 * _ot_eps(xt, xt, schedule, max_iters, tol)
         - 0.5 * _ot_eps(yt, yt, schedule, max_iters, tol))
    return math.sqrt(2.0 * max(s, 0.0))


def mesh_sinkhorn(gen, ref, n_points=2048, blur=0.1, max_iters=500, tol=1e-6,
                  seed=0, device="cpu"):
    """`sinkhorn_divergence` between area-weighted surface samples of two meshes.

    Both meshes are sampled with the SAME seed. Identical meshes then give
    identical clouds and score exactly 0; independent seeds would leave a
    self-score of sampling noise on the order of the sample spacing.

    Returns:
        float: ``nan`` when either mesh has no positive surface area.
    """
    p = sample_surface(*gen, n=n_points, seed=seed)
    q = sample_surface(*ref, n=n_points, seed=seed)
    if len(p) == 0 or len(q) == 0:
        return float("nan")
    return sinkhorn_divergence(p, q, blur=blur, max_iters=max_iters, tol=tol,
                               device=device)
