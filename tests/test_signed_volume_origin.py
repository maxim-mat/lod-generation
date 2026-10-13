"""`signed_volume` must not depend on where the origin is.

The divergence sum ``1/6 sum(p . N)`` is origin-independent only when the
surface is *closed* -- the translation terms cancel against each other. On an
open surface they do not, and the residual scales with |p|. The corpus is in
EPSG:7415 (RD), so |p| ~ 82,000 / 454,000 and the residual swamps the volume:
one real building came out at 7,439,739 m3 against a 3,553 m3 bounding box.

Centring on the centroid first bounds that to the object's own size. On a
closed mesh it provably changes nothing, which the first test pins.
"""
import numpy as np
import pytest

from src.analysis.cityobject_analysis import signed_volume as an_signed_volume
from src.dataset.mesh_dataset import fix_winding
from src.dataset.mesh_dataset import signed_volume as md_signed_volume
from src.eval.building_features import signed_volume as bf_signed_volume

# Unit cube, outward wound.
CUBE_V = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                   [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
CUBE_F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                   [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                   [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
# The same cube as CityJSON faces (each face a list of rings).
CUBE_RINGS = [[[0, 3, 2, 1]], [[4, 5, 6, 7]], [[0, 1, 5, 4]],
              [[1, 2, 6, 5]], [[2, 3, 7, 6]], [[3, 0, 4, 7]]]

FAR = np.array([82_000.0, 454_000.0, 30.0])     # a real RD coordinate


def test_closed_mesh_is_unaffected_by_translation():
    """The property that makes centring free."""
    here = md_signed_volume(CUBE_V, CUBE_F)
    there = md_signed_volume(CUBE_V + FAR, CUBE_F)
    assert here == pytest.approx(1.0)
    assert there == pytest.approx(1.0, rel=1e-6)


def test_closed_rings_are_unaffected_by_translation():
    here = an_signed_volume(CUBE_RINGS, CUBE_V)
    there = an_signed_volume(CUBE_RINGS, CUBE_V + FAR)
    assert abs(here) == pytest.approx(1.0)
    assert abs(there) == pytest.approx(1.0, rel=1e-6)
    assert bf_signed_volume(CUBE_V + FAR, CUBE_RINGS) == pytest.approx(1.0, rel=1e-6)


def open_cube():
    """The cube with one *wall* removed.

    A wall, not the lid: the residual is ``t . sum(area vectors)``, and the
    missing face's normal is what that sum points along. Removing the
    horizontal bottom leaves it along z, which a translation in x/y is
    orthogonal to -- the bug hides. The gap has to face the direction the
    coordinates are large in.
    """
    keep = [i for i in range(len(CUBE_F)) if i not in (4, 5)]   # drops the -y wall
    return CUBE_V.copy(), CUBE_F[keep].copy()


def test_open_mesh_stays_the_same_order_of_magnitude_when_translated():
    v, f = open_cube()
    near = md_signed_volume(v, f)
    far = md_signed_volume(v + FAR, f)
    # Not equal -- an open surface has no well-defined volume - but it must not
    # scale with the coordinate origin.
    assert abs(far) < 10.0, f"open-mesh volume exploded with the origin: {far}"
    assert abs(near) < 10.0


def test_open_rings_stay_bounded_when_translated():
    rings = [r for i, r in enumerate(CUBE_RINGS) if i != 2]      # drops the -y wall
    far = an_signed_volume(rings, CUBE_V + FAR)
    assert abs(far) < 10.0, f"open-shell volume exploded with the origin: {far}"
    assert abs(bf_signed_volume(CUBE_V + FAR, rings)) < 10.0


def test_fix_winding_decision_survives_translation():
    """The concrete downstream harm: the outward flip is chosen on the sign."""
    v, f = open_cube()
    assert np.array_equal(np.asarray(fix_winding(v, f)),
                          np.asarray(fix_winding(v + FAR, f)))


def test_sign_still_reports_inward_orientation():
    """Centring must not cost the thing the sign is actually for."""
    assert md_signed_volume(CUBE_V, CUBE_F) > 0
    assert md_signed_volume(CUBE_V, CUBE_F[:, ::-1]) < 0
    assert md_signed_volume(CUBE_V + FAR, CUBE_F[:, ::-1]) < 0
