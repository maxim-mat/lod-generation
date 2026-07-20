import numpy as np
from src.eval.novelty import novelty_uniqueness


def test_identical_to_train_is_not_novel():
    train = np.array([[0.0, 0.0], [1.0, 1.0]])
    gen = np.array([[0.0, 0.0]])          # exactly a train building
    out = novelty_uniqueness(gen, train, tol=0.1)
    assert out["novelty"] == 0.0


def test_all_distinct_is_unique():
    train = np.array([[0.0, 0.0]])
    gen = np.array([[5.0, 5.0], [9.0, 9.0]])
    out = novelty_uniqueness(gen, train, tol=0.1)
    assert out["uniqueness"] == 1.0
    assert out["novelty"] == 1.0


def test_duplicates_lower_uniqueness():
    train = np.array([[0.0, 0.0]])
    gen = np.array([[5.0, 5.0], [5.0, 5.0]])  # mutual duplicates
    out = novelty_uniqueness(gen, train, tol=0.1)
    assert out["uniqueness"] == 0.0


def test_empty_gen_returns_zeros():
    train = np.array([[0.0, 0.0]])
    gen = np.zeros((0, 2))
    out = novelty_uniqueness(gen, train, tol=0.1)
    assert out == {"novelty": 0.0, "uniqueness": 0.0}


def test_single_gen_is_unique():
    train = np.array([[0.0, 0.0]])
    gen = np.array([[5.0, 5.0]])  # only one generated sample, far from train
    out = novelty_uniqueness(gen, train, tol=0.1)
    assert out["uniqueness"] == 1.0


def test_empty_train_is_fully_novel():
    train = np.zeros((0, 2))
    gen = np.array([[5.0, 5.0]])
    out = novelty_uniqueness(gen, train, tol=0.1)
    assert out["novelty"] == 1.0
