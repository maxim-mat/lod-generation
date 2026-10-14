"""Unit tests for the exact CSG symmetric-difference volume.

Hand-built solids with analytic volumes only -- never the real dataset. These
assertions carry no resolution tolerance: manifold3d is exact, so a box against
a half-height box is 500 m^3 and nothing else.

The one thing here that is genuinely easy to get wrong, and is pinned by
`test_precision_survives_rd_new_coordinates`, is coordinate magnitude. This
corpus lives in EPSG:28992 where y is ~450,000, and manifold3d's float32 `Mesh`
resolves that to ~3 cm. Everything must be translated to a local origin first.
"""
import numpy as np
import pytest

from src.analysis.exact_volume import (
    ManifoldError, exact_pair_volumes, to_manifold,
)


# --- fixtures -------------------------------------------------------------

def box_mesh(size=(1.0, 1.0, 1.0), at=(0.0, 0.0, 0.0)):
    """(verts, faces) of an axis-aligned box, outward-wound, watertight."""
    s, o = np.asarray(size, dtype=float), np.asarray(at, dtype=float)
    v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
                 dtype=float) * s + o
    # Corner order is (x,y,z) little-endian: 0=000 1=001 2=010 3=011 ...
    # Wound OUTWARD, so the enclosed volume is positive -- exact CSG rejects an
    # inside-out solid, so this is load-bearing rather than decorative.
    f = np.array([
        [0, 3, 2], [0, 1, 3],      # x = lo
        [4, 7, 5], [4, 6, 7],      # x = hi
        [0, 5, 1], [0, 4, 5],      # y = lo
        [2, 7, 6], [2, 3, 7],      # y = hi
        [0, 6, 4], [0, 2, 6],      # z = lo
        [1, 7, 3], [1, 5, 7],      # z = hi
    ], dtype=np.int64)
    return v, f


def gable_mesh(size=(1.0, 1.0, 1.0), ridge=1.5, at=(0.0, 0.0, 0.0)):
    """A box with a ridge -- a roof that a flat LOD1 does not have."""
    s, o = np.asarray(size, dtype=float), np.asarray(at, dtype=float)
    v, f = box_mesh(size, at)
    # Lift the top face into a ridge running along y.
    top = v[:, 2] > o[2] + s[2] / 2
    v = v.copy()
    v[top, 2] = o[2] + ridge
    v[top & (v[:, 0] > o[0] + s[0] / 2), 2] = o[2] + s[2]
    return v, f


def hollow_box():
    """A 3 m box with a 1 m cavity -- volume 27 - 1 = 26."""
    ov, of = box_mesh((3.0, 3.0, 3.0))
    iv, if_ = box_mesh((1.0, 1.0, 1.0), at=(1.0, 1.0, 1.0))
    return np.vstack([ov, iv]), np.vstack([of, if_[:, ::-1] + len(ov)])


# --- exactness ------------------------------------------------------------

def test_identical_solids_have_exactly_zero_difference():
    m = box_mesh((10.0, 6.0, 4.0))
    out = exact_pair_volumes(m, m)
    assert out["symdiff_m3"] == pytest.approx(0.0, abs=1e-9)
    assert out["rel_symdiff"] == pytest.approx(0.0, abs=1e-12)
    assert out["iou"] == pytest.approx(1.0, abs=1e-12)


def test_volumes_are_exact_not_approximate():
    """240 m^3, to machine precision -- the backend is exact, not sampled."""
    out = exact_pair_volumes(box_mesh((10.0, 6.0, 4.0)), box_mesh((10.0, 6.0, 4.0)))
    assert out["vol_lod1_m3"] == pytest.approx(240.0, rel=1e-12)


def test_half_height_box_differs_by_exactly_the_missing_volume():
    out = exact_pair_volumes(box_mesh((10.0, 10.0, 10.0)), box_mesh((10.0, 10.0, 5.0)))
    assert out["symdiff_m3"] == pytest.approx(500.0, rel=1e-9)
    assert out["intersection_m3"] == pytest.approx(500.0, rel=1e-9)
    assert out["union_m3"] == pytest.approx(1000.0, rel=1e-9)


def test_inclusion_exclusion_is_consistent():
    """|A| + |B| == |A u B| + |A n B|, and symdiff == union - intersection."""
    out = exact_pair_volumes(box_mesh((10.0, 6.0, 4.0)),
                             gable_mesh((10.0, 6.0, 4.0), ridge=7.0))
    lhs = out["vol_lod1_m3"] + out["vol_lod2_m3"]
    assert lhs == pytest.approx(out["union_m3"] + out["intersection_m3"], rel=1e-12)
    assert out["symdiff_m3"] == pytest.approx(
        out["union_m3"] - out["intersection_m3"], rel=1e-12)


def test_a_cavity_is_respected():
    """Connectivity that changes enclosed volume -- 27 m^3 solid vs 26 hollow."""
    out = exact_pair_volumes(box_mesh((3.0, 3.0, 3.0)), hollow_box())
    assert out["vol_lod1_m3"] == pytest.approx(27.0, rel=1e-9)
    assert out["vol_lod2_m3"] == pytest.approx(26.0, rel=1e-9)
    assert out["symdiff_m3"] == pytest.approx(1.0, rel=1e-9)


# --- metric properties ----------------------------------------------------

def test_distance_is_symmetric():
    a, b = box_mesh((10.0, 6.0, 4.0)), gable_mesh((10.0, 6.0, 4.0), ridge=7.0)
    assert exact_pair_volumes(a, b)["symdiff_m3"] == pytest.approx(
        exact_pair_volumes(b, a)["symdiff_m3"], rel=1e-12)


def test_distance_obeys_the_triangle_inequality():
    a, b, c = (box_mesh((1, 1, 1.0)), box_mesh((1, 1, 0.6)), box_mesh((1, 1, 0.2)))
    ab = exact_pair_volumes(a, b)["symdiff_m3"]
    bc = exact_pair_volumes(b, c)["symdiff_m3"]
    ac = exact_pair_volumes(a, c)["symdiff_m3"]
    assert ac <= ab + bc + 1e-9


def test_disjoint_solids_share_nothing():
    out = exact_pair_volumes(box_mesh(at=(0, 0, 0)), box_mesh(at=(5, 0, 0)))
    assert out["intersection_m3"] == pytest.approx(0.0, abs=1e-9)
    assert out["iou"] == pytest.approx(0.0, abs=1e-12)
    assert out["rel_symdiff"] == pytest.approx(1.0, rel=1e-12)


# --- the precision trap ---------------------------------------------------

def test_result_is_invariant_to_world_placement():
    """The same two solids must score the same wherever they sit on Earth.

    The end-to-end property that matters. Note it does NOT, on its own, pin the
    local-origin translation in `to_manifold`: at float64 this passes with that
    translation removed (measured deviation 1.8e-12). What makes it hold is the
    `Mesh64` backend, which `test_world_coordinates_do_not_coarsen_the_tolerance`
    pins directly.
    """
    at_origin = exact_pair_volumes(box_mesh((10.0, 10.0, 10.0)),
                                   box_mesh((10.0, 10.0, 5.0)))
    rd = (71800.39, 448978.99, 1.297)
    in_the_field = exact_pair_volumes(box_mesh((10.0, 10.0, 10.0), at=rd),
                                      box_mesh((10.0, 10.0, 5.0), at=rd))
    assert in_the_field["symdiff_m3"] == pytest.approx(
        at_origin["symdiff_m3"], rel=1e-9)
    assert in_the_field["symdiff_m3"] == pytest.approx(500.0, rel=1e-9)


def test_world_coordinates_do_not_coarsen_the_tolerance():
    """The real regression guard: `Mesh64`, not `Mesh`.

    manifold3d derives its vertex-merge tolerance from the bounding box, so
    coordinate magnitude sets it. At EPSG:28992 magnitudes the float32 `Mesh`
    yields a ~5e-2 m tolerance -- it welds vertices 5 cm apart, and symmetric
    difference (a small difference of two large near-equal volumes) drifts by
    up to 4.4% on real buildings. `Mesh64` keeps it near 4.5e-7 m.

    Swapping the backend back to float32 fails here even though every other
    test in this file would still pass.
    """
    rd = (71800.39, 448978.99, 1.297)
    v, f = box_mesh((10.0, 10.0, 10.0), at=rd)
    assert to_manifold(v, f, origin=np.zeros(3)).get_tolerance() < 1e-5


def test_both_solids_share_one_origin():
    """Translating each by its own centre would destroy their relative position.

    Two disjoint boxes must stay disjoint; a per-solid origin would stack them.
    """
    out = exact_pair_volumes(box_mesh((2, 2, 2), at=(0, 0, 0)),
                             box_mesh((2, 2, 2), at=(50, 0, 0)))
    assert out["intersection_m3"] == pytest.approx(0.0, abs=1e-9)


# --- degenerate input -----------------------------------------------------

def test_an_empty_target_is_maximally_different_not_an_error():
    """`convert_to_lod1` is known to pass roofless objects through."""
    empty = (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    out = exact_pair_volumes(box_mesh((10.0, 6.0, 4.0)), empty)
    assert out["vol_lod2_m3"] == 0.0
    assert out["rel_symdiff"] == 1.0
    assert out["ok"] is False


def test_two_empty_solids_are_defined_as_identical():
    """0/0 -- nan would poison the histogram this feeds."""
    empty = (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    out = exact_pair_volumes(empty, empty)
    assert out["rel_symdiff"] == 0.0
    assert out["iou"] == 1.0


def test_inverted_winding_is_reported_not_silently_negated():
    """`fix_winding` is known to randomise on open meshes.

    An inside-out solid must not quietly contribute a negative volume to a
    distribution; the row is marked instead.
    """
    v, f = box_mesh((10.0, 6.0, 4.0))
    out = exact_pair_volumes((v, f), (v, f[:, ::-1]))
    assert out["ok"] is False


def test_a_non_manifold_input_is_flagged_rather_than_crashing():
    """One bad object must not kill a corpus pass."""
    v, f = box_mesh((10.0, 6.0, 4.0))
    torn = f[:-2]                      # holes in the shell: no longer closed
    out = exact_pair_volumes((v, f), (v, torn))
    assert out["ok"] is False
    assert np.isfinite(out["symdiff_m3"])


def test_to_manifold_raises_on_a_torn_shell():
    """The low-level helper is strict; `exact_pair_volumes` is what degrades."""
    v, f = box_mesh((10.0, 6.0, 4.0))
    with pytest.raises(ManifoldError):
        to_manifold(v, f[:-2], origin=np.zeros(3))
