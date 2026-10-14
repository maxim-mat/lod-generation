"""Orthophoto crops land in the same normalized frame as the mesh tokens.

A marker painted at a known RD point must come back at the image pixel whose
normalized (u, v) is that point pushed through `normalize_to_unit_box` -- rows
along +y, columns along +x. A missing north-up flip or a transposed axis moves
the marker to the wrong quadrant, which is what these tests pin.
"""
import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from src.dataset.mesh_dataset import normalize_to_unit_box
from src.dataset.ortho import OrthoReader, crop_bounds, pixel_to_uv

GSD = 0.08
X0, Y1 = 1000.0, 2080.0          # raster upper-left corner, metres (RD-like)
PX = 1000                        # 80 m x 80 m raster
S, BETA = 64, 0.1


def _box(x0, y0, x1, y1, z1):
    """LOD1 box vertices in metres; only the bbox matters for the frame."""
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (0.0, z1)])


def _frame(verts):
    _, center, scale = normalize_to_unit_box(verts, margin_lo=[0, 0, 0],
                                             margin_hi=[0, 0, 0.1])
    return center, scale


def _write_raster(path, markers):
    """Lossless 3-band raster, black except for square markers {(x, y): value}."""
    data = np.zeros((3, PX, PX), np.uint8)
    for (x, y), value in markers.items():
        col, row = int((x - X0) / GSD), int((Y1 - y) / GSD)
        data[:, row - 3:row + 4, col - 3:col + 4] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=PX, height=PX, count=3,
                       dtype="uint8", crs="EPSG:28992", compress="deflate",
                       transform=from_origin(X0, Y1, GSD, GSD)) as dst:
        dst.write(data)


def _peak(img, mask=None):
    band = img[0].numpy().astype(float)
    if mask is not None:
        band[mask] = -1
    return np.unravel_index(np.argmax(band), band.shape)


@pytest.fixture
def scene(tmp_path):
    # An off-centre building: footprint 15 m x 20 m, 6 m tall -> base 20 m.
    center, scale = _frame(_box(1020.0, 2030.0, 1035.0, 2050.0, 6.0))
    ne, sw = (1036.0, 2049.0), (1018.0, 2031.0)
    path = tmp_path / "The Hague" / "cleaner" / "ortho" / "Source B" / "t.tif"
    _write_raster(path, {ne: 255, sw: 200})
    return path.parent.parent, center, scale, ne, sw


def _expected_uv(point, center, scale):
    return ((point[0] - center[0]) / scale[0], (point[1] - center[1]) / scale[1])


def test_crop_bounds_is_the_normalized_square_plus_margin():
    center, scale = np.array([10.0, 20.0, 3.0]), np.array([4.0, 4.0, 4.4])
    assert crop_bounds(center, scale, 0.25) == pytest.approx((7.0, 17.0, 13.0, 23.0))


def test_pixel_to_uv_spans_the_margin_symmetrically():
    step = (1 + 2 * BETA) / S
    assert pixel_to_uv(0, 0, S, BETA) == pytest.approx((-0.6 + step / 2, -0.6 + step / 2))
    assert pixel_to_uv(S - 1, S - 1, S, BETA) == pytest.approx((0.6 - step / 2, 0.6 - step / 2))
    # j drives u (x), i drives v (y)
    u, v = pixel_to_uv(0, S - 1, S, BETA)
    assert u > 0 > v


def test_markers_land_at_their_normalized_coordinates(scene):
    root, center, scale, ne, sw = scene
    img = OrthoReader(root, size=S, beta=BETA).crop("Source B/t.tif", center, scale)

    assert img.dtype == torch.uint8 and tuple(img.shape) == (3, S, S)
    tol = (1 + 2 * BETA) / S                                   # one output pixel

    i, j = _peak(img)
    assert i > S / 2 and j > S / 2, "NE marker must be in the +y/+x quadrant"
    assert pixel_to_uv(i, j, S, BETA) == pytest.approx(_expected_uv(ne, center, scale), abs=tol)

    mask = np.zeros((S, S), bool)
    mask[max(i - 4, 0):i + 5, max(j - 4, 0):j + 5] = True
    i2, j2 = _peak(img, mask)
    assert i2 < S / 2 and j2 < S / 2, "SW marker must be in the -y/-x quadrant"
    assert pixel_to_uv(i2, j2, S, BETA) == pytest.approx(_expected_uv(sw, center, scale), abs=tol)


def test_window_off_the_raster_is_zero_padded_and_counted(scene):
    root, _, _, _, _ = scene
    # A building straddling the raster's west edge: its window starts 2 m west of X0.
    center, scale = _frame(_box(999.0, 2040.0, 1009.0, 2050.0, 4.0))
    reader = OrthoReader(root, size=S, beta=BETA)
    img = reader.crop("Source B/t.tif", center, scale).numpy()

    assert reader.n_padded == 1
    xmin = crop_bounds(center, scale, BETA)[0]
    outside_cols = int((X0 - xmin) / (crop_bounds(center, scale, BETA)[2] - xmin) * S)
    assert outside_cols >= 1
    assert not img[:, :, :outside_cols].any()


def test_reader_pickles_without_open_handles(scene):
    import pickle

    root, center, scale, _, _ = scene
    reader = OrthoReader(root, size=S, beta=BETA)
    reader.crop("Source B/t.tif", center, scale)
    clone = pickle.loads(pickle.dumps(reader))
    assert tuple(clone.crop("Source B/t.tif", center, scale).shape) == (3, S, S)
