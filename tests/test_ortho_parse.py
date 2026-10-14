"""Per-file orthophoto mosaics: seams, format, sparsity, coverage and the subset run.

Raw tiles here are real 1 km tiles at a coarse ground sampling distance (GSD), so
the tile arithmetic is the production one while each tile is only a few MB.
"""
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.ortho_parse import main, mosaic_file, parse_subset
from src.ortho_tiles import tile_name
from tests.test_ortho_tiles import write_tile

YEAR, GSD = 2022, 0.5            # 2000 px per 1 km tile
TILE_PX = int(1000 / GSD)


def texture(x, y):
    """Smooth, seam-crossing RGB pattern as a function of world coordinates."""
    r = (x - 80000) * 0.1
    g = (y - 450000) * 0.1
    b = 128 + 60 * np.sin(x / 7.0) * np.cos(y / 9.0)
    return np.clip(np.stack([r, g, b]), 0, 255).astype(np.uint8)


def write_raw_tile(raw_dir, xmin, ymax, noise=False):
    xs = xmin + (np.arange(TILE_PX) + 0.5) * GSD
    ys = ymax - (np.arange(TILE_PX) + 0.5) * GSD
    data = texture(*np.meshgrid(xs, ys))
    if noise:
        data = np.random.default_rng(int(xmin + ymax)).integers(0, 256, data.shape, dtype=np.uint8)
    path = raw_dir / str(YEAR) / tile_name(xmin, ymax, YEAR)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=TILE_PX, height=TILE_PX, count=3,
                       dtype="uint8", crs="EPSG:28992", compress="deflate", tiled=True,
                       transform=from_origin(xmin, ymax, GSD, GSD)) as dst:
        dst.write(data)
    return path


def read_world(path, x, y):
    """RGB at world (x, y) from a north-up raster."""
    with rasterio.open(path) as src:
        row, col = src.index(x, y)
        return src.read(window=((row, row + 1), (col, col + 1)))[:, 0, 0]


def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return 10 * np.log10(255 ** 2 / mse)


@pytest.fixture
def raw(tmp_path):
    raw_dir = tmp_path / "The Hague" / "raw" / "ortho"
    for x in (80000, 81000):
        write_raw_tile(raw_dir, x, 451000)
    return raw_dir


def test_mosaic_across_a_tile_seam_is_continuous_and_grid_snapped(raw, tmp_path):
    out = tmp_path / "m.tif"
    window = (80950.3, 450400.1, 81070.7, 450520.9)     # straddles x = 81 000
    stats = mosaic_file({"a": window}, raw / str(YEAR), out, YEAR, gsd=GSD)

    assert stats["coverage"] == {"a": pytest.approx(1.0)}
    with rasterio.open(out) as src:
        x0, y1 = src.transform.c, src.transform.f
        assert x0 / GSD == pytest.approx(round(x0 / GSD)) and x0 <= window[0]
        assert y1 / GSD == pytest.approx(round(y1 / GSD)) and y1 >= window[3]
        img = src.read()
        xs = x0 + (np.arange(src.width) + 0.5) * GSD
        ys = y1 - (np.arange(src.height) + 0.5) * GSD
    expect = texture(*np.meshgrid(xs, ys))
    assert psnr(img, expect) > 35
    # The two columns either side of the seam carry the source values.
    for x in (80999.75, 81000.25):
        np.testing.assert_allclose(read_world(out, x, 450460.0),
                                   texture(np.array(x), np.array(450460.25)), atol=6)


def test_mosaic_format_is_tiled_ycbcr_jpeg_with_overviews(raw, tmp_path):
    out = tmp_path / "m.tif"
    mosaic_file({"a": (80100.0, 450100.0, 80700.0, 450700.0)}, raw / str(YEAR), out, YEAR, gsd=GSD)
    with rasterio.open(out) as src:
        assert src.crs.to_epsg() == 28992
        assert src.compression.name.lower() == "jpeg"
        assert src.photometric.name.lower() == "ycbcr"
        assert src.block_shapes[0] == (256, 256)
        assert src.overviews(1) == [2, 4, 8, 16]


def test_mosaic_writes_only_blocks_under_a_window(tmp_path):
    raw_dir = tmp_path / "raw"
    for x in (80000, 81000):
        write_raw_tile(raw_dir, x, 451000, noise=True)
    small = {"a": (80010.0, 450010.0, 80040.0, 450040.0),
             "b": (81940.0, 450940.0, 81970.0, 450970.0)}
    sparse = mosaic_file(small, raw_dir / str(YEAR), tmp_path / "sparse.tif", YEAR, gsd=GSD)
    dense = mosaic_file({**small, "c": (80010.0, 450010.0, 81970.0, 450970.0)},
                        raw_dir / str(YEAR), tmp_path / "dense.tif", YEAR, gsd=GSD)
    assert sparse["bytes"] < 0.2 * dense["bytes"]
    # Unwritten blocks read back as zeros.
    assert not read_world(tmp_path / "sparse.tif", 81000.0, 450500.0).any()


def test_missing_raw_tile_lowers_coverage_and_reads_zero(tmp_path):
    raw_dir = tmp_path / "raw"
    write_raw_tile(raw_dir, 80000, 451000)                # 81 000 tile absent
    out = tmp_path / "m.tif"
    stats = mosaic_file({"a": (80900.0, 450400.0, 81100.0, 450600.0)}, raw_dir / str(YEAR),
                        out, YEAR, gsd=GSD)
    assert stats["coverage"]["a"] == pytest.approx(0.5)
    assert stats["missing_tiles"] == [tile_name(81000, 451000, YEAR)]
    assert not read_world(out, 81050.0, 450500.0).any()


def _subset(root):
    datum = {"a": 2022, "b": 2020}
    boxes = {"a": (80100, 450100, 80120, 450130, 8), "b": (80500, 450500, 80510, 450512, 5)}
    for lod in ("LOD1", "LOD2"):
        write_tile(root / lod / "Source B" / "10-1-2.city.json", boxes, datum=datum)
    return root


def test_parse_subset_mirrors_cityjson_names_and_reports_provenance(raw, tmp_path):
    subset = _subset(tmp_path / "The Hague" / "mini_cleaner")
    out = subset / "ortho"
    totals = parse_subset(subset, raw, out, year=YEAR, beta_max=0.25, lod_in="LOD1",
                          lod_out="LOD2", margin_lo=[0, 0, 0], margin_hi=[0, 0, 0.1],
                          gsd=GSD, workers=1)

    assert (out / "Source B" / "10-1-2.tif").is_file()
    assert totals["files_written"] == 1 and totals["buildings"] == 2
    assert totals["pw_datum"] == {"2022": 1, "2020": 1}
    assert totals["partial_coverage"] == {}
    assert set(totals["need_beta_frac_over"]) == {"0.05", "0.1", "0.25"}


def test_main_writes_manifest_and_readme_and_refuses_a_non_empty_out(raw, tmp_path):
    subset = _subset(tmp_path / "The Hague" / "mini_cleaner")
    out = subset / "ortho"
    argv = [str(subset), "--raw", str(raw), "--out", str(out), "--gsd", str(GSD), "--workers", "1"]
    main(argv)

    manifest = json.loads((out / "ortho_manifest.json").read_text(encoding="utf-8"))
    assert {"created", "git_commit", "source", "args", "totals"} <= set(manifest)
    assert manifest["args"]["beta_max"] == 0.25 and manifest["args"]["lod_in"] == "LOD1"
    readme = (out / "README.md").read_text(encoding="utf-8")
    for needle in ("2022", "CC-BY 4.0", "b3_pw_datum", "2020: 1"):
        assert needle in readme

    with pytest.raises(SystemExit):
        main(argv)
