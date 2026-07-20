"""Distribution-realism distances between generated and reference feature sets.

log(1+x) normalization per 3dSAGER (arXiv:2511.06300, sec 3.1).
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


def log1p_normalize(X):
    return np.log1p(np.asarray(X, dtype=float))


def per_feature_wasserstein(gen, ref, names):
    gen, ref = np.asarray(gen, dtype=float), np.asarray(ref, dtype=float)
    out, acc = {}, []
    for j, name in enumerate(names):
        d = wasserstein_distance(gen[:, j], ref[:, j])
        out[f"wass/{name}"] = float(d)
        acc.append(d)
    out["wass/mean"] = float(np.mean(acc)) if acc else 0.0
    return out


def _median_bandwidth(pooled):
    d = cdist(pooled, pooled)
    med = np.median(d[np.triu_indices_from(d, k=1)])
    return float(med) if med > 0 else 1.0


def kernel_mmd(gen, ref):
    """Unbiased RBF-kernel MMD^2 with median-heuristic bandwidth."""
    gen, ref = np.asarray(gen, dtype=float), np.asarray(ref, dtype=float)
    gamma = 1.0 / (2.0 * _median_bandwidth(np.vstack([gen, ref])) ** 2)
    k = lambda a, b: np.exp(-gamma * cdist(a, b) ** 2)  # noqa: E731
    m, n = len(gen), len(ref)
    kxx = (k(gen, gen).sum() - np.trace(k(gen, gen))) / (m * (m - 1))
    kyy = (k(ref, ref).sum() - np.trace(k(ref, ref))) / (n * (n - 1))
    kxy = k(gen, ref).mean()
    return float(kxx + kyy - 2 * kxy)
