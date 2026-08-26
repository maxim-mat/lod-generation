"""Face-set representation: canonical rotation, Morton order, batch padding.

Tiny synthetic meshes, CPU only. These assertions are what every downstream
loss and denoiser rests on: a face must survive rotation unchanged as a set,
the order must be a permutation, and padding must not leak into the mask.
"""
import numpy as np
import pytest
import torch

from src.dataset.mesh_set_dataset import (
    faces_to_array,
    mesh_set_collate_fn,
    morton_order,
    rotate_faces_canonical,
)

NUM_BINS = 128


def _tri(a, b, c):
    return np.array([[a, b, c]], dtype=float)


def test_rotate_is_cyclic_and_winding_preserving():
    v0, v1, v2 = [0.4, 0.4, 0.4], [-0.1, 0.0, 0.1], [0.2, -0.3, -0.4]
    tri = _tri(v0, v1, v2)
    out = rotate_faces_canonical(tri, NUM_BINS)
    # Same three corners, as a set.
    assert {tuple(x) for x in out[0]} == {tuple(v0), tuple(v1), tuple(v2)}
    # Smallest by (z, y, x) leads: v2 has z=-0.4, the lowest.
    assert np.allclose(out[0, 0], v2)
    # Winding preserved: the cyclic successor of v2 is still v0.
    assert np.allclose(out[0, 1], v0)


def test_rotate_is_idempotent():
    tri = _tri([0.4, 0.4, 0.4], [-0.1, 0.0, 0.1], [0.2, -0.3, -0.4])
    once = rotate_faces_canonical(tri, NUM_BINS)
    assert np.allclose(rotate_faces_canonical(once, NUM_BINS), once)


def test_rotate_equalises_the_three_spellings():
    v = [[0.4, 0.4, 0.4], [-0.1, 0.0, 0.1], [0.2, -0.3, -0.4]]
    spellings = [_tri(v[0], v[1], v[2]), _tri(v[1], v[2], v[0]), _tri(v[2], v[0], v[1])]
    outs = [rotate_faces_canonical(s, NUM_BINS) for s in spellings]
    assert np.allclose(outs[0], outs[1])
    assert np.allclose(outs[0], outs[2])


def test_morton_order_is_a_permutation_and_deterministic():
    rng = np.random.default_rng(0)
    tri = rng.uniform(-0.5, 0.5, size=(17, 3, 3))
    perm = morton_order(tri, NUM_BINS)
    assert sorted(perm.tolist()) == list(range(17))
    assert np.array_equal(perm, morton_order(tri, NUM_BINS))


def test_morton_order_groups_nearby_faces():
    # Two tight clusters far apart; sorting must not interleave them.
    lo = np.full((5, 3, 3), -0.45) + np.linspace(0, 0.01, 5)[:, None, None]
    hi = np.full((5, 3, 3), 0.45) + np.linspace(0, 0.01, 5)[:, None, None]
    tri = np.concatenate([lo, hi])[[0, 5, 1, 6, 2, 7, 3, 8, 4, 9]]  # interleaved
    perm = morton_order(tri, NUM_BINS)
    is_hi = (tri[perm][:, 0, 0] > 0)
    # Sorted output must be all-low then all-high, never alternating.
    assert is_hi.tolist() == sorted(is_hi.tolist())


def test_faces_to_array_expands_indices():
    verts = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]])
    faces = np.array([[0, 1, 2]])
    out = faces_to_array(verts, faces)
    assert out.shape == (1, 3, 3)
    assert np.allclose(out[0, 1], [1.0, 0, 0])


def test_state_quantized_snaps_the_continuous_channels():
    """Under state: quantized the float channels must already sit on grid
    points, so a regression readout and a categorical one are scored against an
    identical target -- which is what makes the b3-vs-b4 control readable."""
    from src.dataset.mesh_dataset import dequantize
    from src.dataset.mesh_set_dataset import _pack

    rng = np.random.default_rng(0)
    tri = rng.uniform(-0.5, 0.5, size=(6, 3, 3))
    x, bins = _pack(tri, NUM_BINS, "morton", state="quantized")
    assert bins is not None and bins.shape == (6, 9)
    assert np.allclose(x[:, :9], dequantize(bins, NUM_BINS), atol=1e-6)


def test_state_continuous_emits_no_bins():
    from src.dataset.mesh_set_dataset import _pack

    x, bins = _pack(np.zeros((4, 3, 3)), NUM_BINS, "morton", state="continuous")
    assert bins is None and x.shape == (4, 10)


def test_unknown_state_raises():
    from src.dataset.mesh_set_dataset import _pack

    with pytest.raises(ValueError, match="Unknown state"):
        _pack(np.zeros((2, 3, 3)), NUM_BINS, "morton", state="onehotish")


def test_collate_pads_to_multiple_of_eight_and_sets_presence():
    items = [
        {"x": torch.zeros(5, 10), "cond": torch.zeros(3, 10),
         "id": "a", "center": torch.zeros(3), "scale": torch.ones(3)},
        {"x": torch.zeros(11, 10), "cond": torch.zeros(4, 10),
         "id": "b", "center": torch.zeros(3), "scale": torch.ones(3)},
    ]
    for it in items:                       # real faces carry presence +0.5
        it["x"][:, 9] = 0.5
        it["cond"][:, 9] = 0.5
    out = mesh_set_collate_fn(items, multiple_of=8)
    assert out["x"].shape == (2, 10, 16)   # max 11 -> 16
    assert out["x_mask"][0].sum() == 5 and out["x_mask"][1].sum() == 11
    # Pad slots: presence -0.5, coords 0.
    assert torch.allclose(out["x"][0, 9, 5:], torch.full((11,), -0.5))
    assert torch.allclose(out["x"][0, :9, 5:], torch.zeros(9, 11))
    # Condition padded independently, also to a multiple of 8.
    assert out["cond"].shape == (2, 10, 8)


def test_bins_padding_describes_the_same_point_as_the_coordinate_padding():
    """Unused slots must look the same whichever channel an arm reads.

    `x` pads to 0.0 (the box centre) and `x_bins` used to pad to bin 0 (a box
    corner), so `state: onehot` and `state: bins` -- which build their state
    from `x_bins` -- marked unused slots somewhere completely different from
    the continuous arms. Harmless while those slots were masked out of the
    model; they are DETR no-object slots now, so their content is real input.
    """
    from src.dataset.mesh_dataset import dequantize, quantize

    items = [{"x": torch.zeros(3, 10), "cond": torch.zeros(2, 10), "id": "a",
              "center": torch.zeros(3), "scale": torch.ones(3),
              "x_bins": torch.full((3, 9), 64, dtype=torch.long)}]
    items[0]["x"][:, 9] = 0.5
    out = mesh_set_collate_fn(items, width=16, num_bins=NUM_BINS)

    pad = ~out["x_mask"][0]
    pad_coord = float(out["x"][0, 0][pad][0])
    pad_bin = int(out["x_bins"][0, 0][pad][0])
    assert pad_bin == int(quantize(np.zeros((1, 3)), NUM_BINS)[0, 0])
    assert float(dequantize(np.array([[pad_bin] * 3]), NUM_BINS)[0, 0]) == \
        pytest.approx(pad_coord, abs=1.0 / (NUM_BINS - 1))
