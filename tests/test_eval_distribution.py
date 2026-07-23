import numpy as np
from src.eval.distribution import log1p_normalize, per_feature_wasserstein, kernel_mmd

rng = np.random.default_rng(0)


def test_log1p_normalize_matches_expm1_inverse():
    X = np.array([[0.0, 9.0]])
    assert np.allclose(log1p_normalize(X), np.log1p(X))


def test_identical_distributions_zero_distance():
    A = rng.normal(size=(200, 3))
    w = per_feature_wasserstein(A, A.copy(), ["a", "b", "c"])
    assert w["wass/mean"] < 1e-9
    assert kernel_mmd(A, A.copy()) < 1e-6


def test_shift_increases_distance_monotonically():
    A = rng.normal(size=(200, 2))
    near = per_feature_wasserstein(A, A + 0.5, ["a", "b"])["wass/mean"]
    far = per_feature_wasserstein(A, A + 2.0, ["a", "b"])["wass/mean"]
    assert far > near > 0
    assert kernel_mmd(A, A + 2.0) > kernel_mmd(A, A + 0.5)
