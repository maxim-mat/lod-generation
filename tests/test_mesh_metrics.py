"""Paired geometric metrics on shapes whose answer is known by hand.

Every mesh here is an axis-aligned box, so the expected IoU / distance can be
worked out on paper and a failure means the implementation is wrong rather than
that a tolerance drifted.
"""
import numpy as np
import pytest

from src.eval.mesh_metrics import (
    chamfer_distance,
    f_score,
    footprint_iou,
    hausdorff_p95,
    is_watertight_mesh,
    mesh_metrics,
    roof_height_error,
    sample_surface,
    surface_distances,
    volumetric_iou,
)


def _box(lo=(0.0, 0.0, 0.0), hi=(2.0, 2.0, 2.0)):
    """Closed axis-aligned box, 8 vertices / 12 outward-wound triangles."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    verts = np.array([[x, y, z] for x in (lo[0], hi[0])
                      for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    # Vertex index = 4*ix + 2*iy + iz. Winding is outward on every face, which
    # `test_the_box_helper_is_outward_wound` pins -- an inward box still passes
    # is_watertight (it is consistently oriented) but inverts roof and ground.
    faces = np.array([
        [0, 1, 3], [0, 3, 2],      # x = lo, normal -x
        [4, 6, 7], [4, 7, 5],      # x = hi, normal +x
        [0, 4, 5], [0, 5, 1],      # y = lo, normal -y
        [2, 3, 7], [2, 7, 6],      # y = hi, normal +y
        [0, 2, 6], [0, 6, 4],      # z = lo, normal -z (ground)
        [1, 5, 7], [1, 7, 3],      # z = hi, normal +z (roof)
    ], dtype=np.int64)
    return verts, faces


def _normals(verts, faces):
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = np.cross(b - a, c - a)
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def _box_area(lo=(0.0, 0.0, 0.0), hi=(2.0, 2.0, 2.0)):
    d = np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)
    return 2.0 * (d[0] * d[1] + d[1] * d[2] + d[0] * d[2])


# ----------------------------------------------------------------------
# Sampling and point-set distances
# ----------------------------------------------------------------------

def test_the_box_helper_is_outward_wound():
    """Every other test trusts this. An inward box is still watertight, so
    is_watertight would not catch it -- it would just silently swap the roof
    and the ground and make the height metrics read zero."""
    verts, faces = _box()
    n = _normals(verts, faces)
    assert n[-2:, 2].min() > 0.99          # last two faces point up: the roof
    assert n[-4:-2, 2].max() < -0.99       # the two before: the ground
    # Outward overall: divergence theorem gives +volume, not -volume.
    a = verts[faces[:, 0]]
    assert float(np.sum(a * np.cross(verts[faces[:, 1]] - a,
                                     verts[faces[:, 2]] - a))) / 6.0 > 0


def test_surface_sampling_is_seeded():
    """Metrics compared across epochs must not move because the RNG moved."""
    box = _box()
    assert np.array_equal(sample_surface(*box, n=256, seed=3),
                          sample_surface(*box, n=256, seed=3))
    assert not np.array_equal(sample_surface(*box, n=256, seed=3),
                              sample_surface(*box, n=256, seed=4))


def test_chamfer_of_a_shape_against_itself_is_exactly_zero():
    """The property that makes the number readable.

    Distances are point-to-*surface*, not point-to-point-cloud: only the query
    side is sampled, and the target is the triangles themselves. So a sampled
    point of a mesh lies exactly on that mesh and the distance is 0, at any
    sample count.

    This replaced a point-cloud/point-cloud version whose self-comparison was
    ~sqrt(area/n) -- 0.129 m at n=4096 on real buildings, which was more than
    half of the measured LOD1->LOD2 baseline and made every chamfer below about
    0.15 m unresolvable.
    """
    box = _box()
    for n in (256, 4096):
        d_ab, d_ba = surface_distances(box, box, n=n)
        assert chamfer_distance(d_ab, d_ba) == pytest.approx(0.0, abs=1e-9)


def test_self_chamfer_does_not_depend_on_sample_count():
    """The old floor scaled as 1/sqrt(n); this must not scale at all."""
    box = _box()
    vals = [chamfer_distance(*surface_distances(box, box, n=n))
            for n in (256, 1024, 4096)]
    assert max(vals) == pytest.approx(0.0, abs=1e-9)


def test_chamfer_tracks_a_known_translation():
    """The test with teeth: self-comparison alone passes on a constant."""
    box = _box()
    moved = (box[0] + np.array([1.0, 0.0, 0.0]), box[1])
    d_ab, d_ba = surface_distances(box, moved, n=8192)
    # Two boxes offset by half their width: most of the surface is 1 m away,
    # but the overlapping side walls are much closer, so only bound it.
    assert 0.3 < chamfer_distance(d_ab, d_ba) < 1.0

    far = (box[0] + np.array([50.0, 0.0, 0.0]), box[1])
    d_ab, d_ba = surface_distances(box, far, n=4096)
    assert 48.0 < chamfer_distance(d_ab, d_ba) < 52.0


def test_f_score_of_a_shape_against_itself_is_one():
    """Exactly 1.0 at any threshold, because every distance is exactly 0.

    It used to take a 25 cm threshold to absorb ~8 cm of sampling noise, and
    tightening tau below that floor dropped it under 1.0. Point-to-surface
    distances have no floor to absorb.
    """
    box = _box()
    d_ab, d_ba = surface_distances(box, box, n=4096)
    for tau in (0.25, 0.01, 1e-6):
        precision, recall, f1 = f_score(d_ab, d_ba, tau=tau)
        assert (precision, recall, f1) == (1.0, 1.0, 1.0)

    # Teeth: a genuinely displaced surface must fall short.
    moved = (box[0] + np.array([0.5, 0.0, 0.0]), box[1])
    assert f_score(*surface_distances(box, moved, n=4096), tau=0.01)[2] < 1.0


def test_hausdorff_p95_ignores_a_small_detached_spike():
    """Why p95 and not max: a partial decode leaves stray far-off geometry.

    The spike is a *separate* sliver, not a moved vertex -- moving a vertex
    drags every triangle touching it and deforms a third of the surface, which
    is not an outlier at all. Sampling is area-weighted, so a sliver draws
    almost no points and p95 stays put while the max does not.
    """
    verts, faces = _box()
    spiked_v = np.vstack([verts, [[1.0, 1.0, 9.0], [1.02, 1.0, 9.0], [1.0, 1.02, 9.0]]])
    spiked_f = np.vstack([faces, [[8, 9, 10]]])
    d_ab, d_ba = surface_distances((spiked_v, spiked_f), (verts, faces), n=8192)

    assert hausdorff_p95(d_ab, d_ba) < 1.0
    assert max(d_ab.max(), d_ba.max()) > 6.0     # a true Hausdorff would report this


# ----------------------------------------------------------------------
# Voxel IoU
# ----------------------------------------------------------------------

def test_volume_iou_of_a_box_against_itself_is_one():
    box = _box()
    assert volumetric_iou(box, box, voxel_m=0.1) == 1.0


def test_volume_iou_of_half_overlapping_boxes_is_one_third():
    """Self-IoU cannot catch a parity-fill that fills the wrong side; this can.

    [0,2]^3 against [1,3]x[0,2]^2 overlap in a 1x2x2 slab: 4 / (8 + 8 - 4) = 1/3.
    """
    a = _box()
    b = _box(lo=(1.0, 0.0, 0.0), hi=(3.0, 2.0, 2.0))
    assert abs(volumetric_iou(a, b, voxel_m=0.05) - 1.0 / 3.0) < 0.02


def test_volume_iou_is_nan_on_a_non_watertight_mesh():
    """A parity fill leaks through a hole, so the honest answer is 'undefined'."""
    verts, faces = _box()
    open_box = (verts, faces[:-2])               # drop the roof
    assert not is_watertight_mesh(open_box[1])
    assert np.isnan(volumetric_iou(open_box, _box(), voxel_m=0.1))


def test_footprint_iou_ignores_height():
    """It answers 'is the building in the right place', not 'is it the right size'."""
    flat = _box(hi=(2.0, 2.0, 1.0))
    tall = _box(hi=(2.0, 2.0, 10.0))
    assert footprint_iou(flat, tall, voxel_m=0.05) == 1.0


def test_footprint_iou_works_without_a_roof():
    """The whole reason it exists: it is defined where volumetric IoU is not."""
    verts, faces = _box()
    open_box = (verts, faces[:-2])
    assert np.isnan(volumetric_iou(open_box, _box(), voxel_m=0.1))
    assert footprint_iou(open_box, _box(), voxel_m=0.05) == 1.0


# ----------------------------------------------------------------------
# Domain metrics and degeneracy
# ----------------------------------------------------------------------

def test_roof_height_error_on_a_raised_box():
    low = _box(hi=(2.0, 2.0, 2.0))
    high = _box(hi=(2.0, 2.0, 3.5))
    max_err, mean_err = roof_height_error(low, high)
    assert abs(max_err - 1.5) < 1e-9
    assert abs(mean_err - 1.5) < 1e-9        # both roofs are flat, so they agree


def test_metrics_on_an_empty_mesh_are_nan():
    """A generation that decoded to nothing must not crash the eval loop."""
    empty = (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    out = mesh_metrics(empty, _box(), n_points=256)

    assert out["watertight_gen"] == 0.0 and out["watertight_gt"] == 1.0
    for key in ("chamfer_m", "hausdorff_p95_m", "vol_iou", "footprint_iou",
                "roof_max_z_err_m"):
        assert np.isnan(out[key]), key


def test_mesh_metrics_reports_every_expected_key():
    """Guards the contract the Lightning logging depends on."""
    out = mesh_metrics(_box(), _box(), taus=(0.25, 0.5), n_points=512)
    assert set(out) == {
        "chamfer_m", "hausdorff_p95_m", "fscore_25cm", "fscore_50cm",
        "vol_iou", "footprint_iou", "roof_max_z_err_m", "roof_mean_z_err_m",
        "watertight_gen", "watertight_gt",
    }
    assert out["vol_iou"] == 1.0 and out["footprint_iou"] == 1.0
