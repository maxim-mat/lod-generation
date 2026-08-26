"""Scaffold masking: MeshWeaver's fix (3), as a sampler-side projection.

LOD1 already bounds where an LOD2 vertex can legally sit. These tests pin the
three things that make that bound useful rather than harmful: it must contain
the LOD1 surface, it must be dilated enough to admit the ridge that rises above
LOD1, and it must not fire early, when x_t is still noise.
"""
import numpy as np
import pytest
import torch

from src.models.mesh_scaffold import lod1_scaffold, make_bin_masker, make_projector


def _prism_cond(b=1, fc=8):
    """A box occupying the lower half of the normalized cube."""
    v = np.array([[x, y, z] for x in (-0.3, 0.3) for y in (-0.3, 0.3)
                  for z in (-0.4, 0.0)], dtype=float)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 7, 5], [4, 6, 7],
                  [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6]])
    tri = v[f]
    cond = torch.zeros(b, 10, fc)
    cond[:, :9, : len(tri)] = torch.from_numpy(tri.reshape(len(tri), 9)).float().T
    cond[:, 9, : len(tri)] = 0.5
    mask = torch.zeros(b, fc, dtype=torch.bool)
    mask[:, : len(tri)] = True
    return cond, mask


def test_scaffold_contains_the_lod1_surface():
    cond, mask = _prism_cond()
    sc = lod1_scaffold(cond, mask, voxel=1 / 32, dilate=0)
    g = sc.shape[-1]
    # A point on the LOD1 wall must be inside.
    idx = ((torch.tensor([0.3, 0.0, -0.2]) + 0.5) * (g - 1)).round().long()
    assert sc[0, idx[0], idx[1], idx[2]]


def test_dilation_admits_the_ridge_above_lod1():
    cond, mask = _prism_cond()
    tight = lod1_scaffold(cond, mask, voxel=1 / 32, dilate=0)
    loose = lod1_scaffold(cond, mask, voxel=1 / 32, dilate=3)
    assert loose.sum() > tight.sum()
    g = loose.shape[-1]
    ridge = ((torch.tensor([0.0, 0.0, 0.05]) + 0.5) * (g - 1)).round().long()
    assert not tight[0, ridge[0], ridge[1], ridge[2]]
    assert loose[0, ridge[0], ridge[1], ridge[2]]


def test_projector_is_a_noop_above_the_threshold():
    cond, mask = _prism_cond()
    sc = lod1_scaffold(cond, mask)
    proj = make_projector(sc, apply_below_t=0.5)
    x = torch.randn(1, 10, 8) * 3.0
    assert torch.allclose(proj(0.9, x), x)


def test_projector_pulls_outliers_inside_below_the_threshold():
    cond, mask = _prism_cond()
    sc = lod1_scaffold(cond, mask, dilate=3)
    proj = make_projector(sc, apply_below_t=0.5)
    x = torch.zeros(1, 10, 8)
    x[:, :9] = 5.0                          # far outside the box
    x[:, 9] = 0.5
    out = proj(0.1, x)
    assert out[:, :9].abs().max().item() <= 0.5 + 1e-6
    assert not torch.allclose(out[:, :9], x[:, :9])


def test_projector_leaves_presence_alone():
    cond, mask = _prism_cond()
    proj = make_projector(lod1_scaffold(cond, mask), apply_below_t=0.5)
    x = torch.zeros(1, 10, 8)
    x[:, 9] = -0.3
    assert torch.allclose(proj(0.1, x)[:, 9], x[:, 9])


def test_projector_leaves_already_legal_points_untouched():
    cond, mask = _prism_cond()
    sc = lod1_scaffold(cond, mask, dilate=3)
    proj = make_projector(sc, apply_below_t=0.5)
    x = torch.zeros(1, 10, 8)
    x[:, :9] = -0.2                          # inside the prism
    assert torch.allclose(proj(0.1, x)[:, :9], x[:, :9], atol=1e-6)


def test_bin_masker_moves_illegal_bins_and_keeps_legal_ones():
    cond, mask = _prism_cond()
    sc = lod1_scaffold(cond, mask, dilate=3)
    masker = make_bin_masker(sc, num_bins=128, apply_below_t=0.5)
    bins = torch.full((1, 9, 8), 127, dtype=torch.long)   # corner of the box
    out = masker(0.1, bins)
    assert out.shape == bins.shape and out.dtype == torch.long
    assert (out != bins).any()
