import numpy as np
from scipy.spatial.distance import cdist


def novelty_uniqueness(gen, train, tol):
    gen, train = np.asarray(gen, dtype=float), np.asarray(train, dtype=float)
    if len(gen) == 0:
        return {"novelty": 0.0, "uniqueness": 0.0}

    nn_train = cdist(gen, train).min(axis=1) if len(train) else np.full(len(gen), np.inf)
    novelty = float((nn_train > tol).mean())

    if len(gen) == 1:
        uniqueness = 1.0
    else:
        d = cdist(gen, gen)
        np.fill_diagonal(d, np.inf)
        uniqueness = float((d.min(axis=1) > tol).mean())
    return {"novelty": novelty, "uniqueness": uniqueness}
