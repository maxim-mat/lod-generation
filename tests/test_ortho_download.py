"""Downloading raw orthophoto tiles: planning, resume, validation, retries, provenance.

No network: `opener` and `sleep` are injected. Tiles are tiny real GeoTIFFs
anchored like production tiles (1 km, named after the upper-left corner).
"""
import io
import json
import urllib.error

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.ortho_download import download_tile, main, plan_downloads
from src.ortho_tiles import tile_name
from tests.test_ortho_tiles import write_tile

YEAR, TILE_PX = 2022, 100            # 10 m pixels -> a real 1 km tile in 30 kB


def tile_bytes(xmin, ymax, px=TILE_PX):
    buf = io.BytesIO()
    with rasterio.MemoryFile() as mem:
        with mem.open(driver="GTiff", width=px, height=px, count=3, dtype="uint8",
                      crs="EPSG:28992", transform=from_origin(xmin, ymax, 1000 / px, 1000 / px)) as dst:
            dst.write(np.full((3, px, px), 7, np.uint8))
        buf.write(mem.read())
    return buf.getvalue()


class FakeResponse:
    def __init__(self, body):
        self.headers = {"Content-Length": str(len(body))}
        self._buf = io.BytesIO(body)

    def read(self, n=-1):
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    """Serves `bodies[url]`; fails the first `fail` calls; records methods."""

    def __init__(self, bodies, fail=0):
        self.bodies, self.fail, self.calls = bodies, fail, []

    def __call__(self, req, timeout=None):
        self.calls.append((req.get_method(), req.full_url))
        if self.fail:
            self.fail -= 1
            raise urllib.error.URLError("connection reset")
        if req.full_url not in self.bodies:
            raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, None)
        body = self.bodies[req.full_url]
        if req.get_method() == "HEAD":
            return _Head(len(body))
        return FakeResponse(body)


class _Head(FakeResponse):
    def __init__(self, n):
        super().__init__(b"")
        self.headers = {"Content-Length": str(n)}


URL = "https://example.test/tile.tif"


def test_complete_file_is_skipped_after_a_head(tmp_path):
    body = tile_bytes(81000, 455000)
    dest = tmp_path / tile_name(81000, 455000, YEAR)
    dest.write_bytes(body)
    opener = FakeOpener({URL: body})
    assert download_tile(URL, dest, (81000, 455000), opener=opener, tile_px=TILE_PX) == "skipped"
    assert [m for m, _ in opener.calls] == ["HEAD"]


def test_download_validates_and_lands_atomically(tmp_path):
    body = tile_bytes(81000, 455000)
    dest = tmp_path / "2022" / tile_name(81000, 455000, YEAR)
    status = download_tile(URL, dest, (81000, 455000), opener=FakeOpener({URL: body}),
                           tile_px=TILE_PX)
    assert status == "downloaded" and dest.read_bytes() == body
    assert not list(dest.parent.glob("*.part"))


def test_transient_errors_are_retried_with_growing_waits(tmp_path):
    body = tile_bytes(81000, 455000)
    waits = []
    status = download_tile(URL, tmp_path / "t.tif", (81000, 455000),
                           opener=FakeOpener({URL: body}, fail=2), sleep=waits.append,
                           tile_px=TILE_PX)
    assert status == "downloaded"
    assert len(waits) == 2 and waits[1] > waits[0]


@pytest.mark.parametrize("body", [b"<html>not a tiff</html>", "wrong_origin"])
def test_invalid_body_fails_and_leaves_nothing_behind(tmp_path, body):
    if body == "wrong_origin":
        body = tile_bytes(82000, 455000)               # valid GeoTIFF, wrong place
    dest = tmp_path / "t.tif"
    with pytest.raises(RuntimeError):
        download_tile(URL, dest, (81000, 455000), opener=FakeOpener({URL: body}),
                      sleep=lambda s: None, retries=2, tile_px=TILE_PX)
    assert not dest.exists() and not list(tmp_path.glob("*.part"))


def _subset(root, boxes, datum=None):
    for lod in ("LOD1", "LOD2"):
        write_tile(root / lod / "Source B" / "t.city.json", boxes,
                   datum=datum or {k: 2022 for k in boxes})
    return root


def test_plan_covers_each_window_and_unions_subsets(tmp_path):
    hague = tmp_path / "The Hague"
    # "a" straddles the x = 81 000 line; "b" sits inside one tile far away.
    full = _subset(hague / "cleaner", {"a": (80990, 454400, 81010, 454420, 6),
                                       "b": (83400, 452400, 83410, 452410, 3)})
    mini = _subset(hague / "mini_cleaner", {"b": (83400, 452400, 83410, 452410, 3)})
    tiles, pw = plan_downloads([full, mini], beta_max=0.25, lod_in="LOD1", lod_out="LOD2",
                               margin_lo=[0, 0, 0], margin_hi=[0, 0, 0.1], years=[2022, 2023])
    assert tiles == {2022: {(80000, 455000), (81000, 455000), (83000, 453000)}}
    assert pw == {"2022": 2}          # unique buildings: "b" is in both subsets


def _index(path):
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"BLOK": "3", "PERCEEL": "Perceel 1"},
         "geometry": {"type": "MultiPolygon", "coordinates": [[[
             [70000, 440000], [90000, 440000], [90000, 470000], [70000, 470000], [70000, 440000]]]]}}]}
    path.write_text(json.dumps(fc), encoding="utf-8")
    return path


def _urls(year, *tiles):
    """2022/2023 URLs of `tiles` in the one-block (perceel 1, blok 3) test index."""
    return {f"https://basisdata.nl/hwh-ortho/{year}/Ortho/1/03/beelden_tif_tegels/"
            + tile_name(x, y, year): tile_bytes(x, y) for x, y in tiles}


def test_main_downloads_each_buildings_year_writes_provenance_and_resumes(tmp_path):
    hague = tmp_path / "The Hague"
    subset = _subset(hague / "cleaner", {"a": (80100, 454100, 80110, 454110, 4),
                                         "b": (81100, 454100, 81110, 454110, 4)},
                     datum={"a": 2023, "b": 2020})
    out = hague / "raw" / "ortho"
    index = _index(tmp_path / "BM.geojson")
    opener = FakeOpener({**_urls(2023, (80000, 455000)), **_urls(2022, (81000, 455000))})
    argv = [str(subset), "--out", str(out), "--years", "2022", "2023",
            "--sheet-index", f"2022={index}", "--sheet-index", f"2023={index}",
            "--tile-px", str(TILE_PX), "--plan-workers", "1"]

    totals = main(argv, opener=opener, sleep=lambda s: None)
    assert totals["downloaded"] == 2 and totals["failed"] == {}
    assert totals["per_year"]["2023"]["tiles_planned"] == 1
    assert (out / "2023" / tile_name(80000, 455000, 2023)).is_file()
    assert (out / "2022" / tile_name(81000, 455000, 2022)).is_file()
    assert not (out / "2022" / tile_name(80000, 455000, 2022)).exists()   # "a" is a 2023 building

    manifest = json.loads((out / "download_manifest.json").read_text(encoding="utf-8"))
    assert {"created", "git_commit", "sheet_index", "args", "totals"} <= set(manifest)
    assert len(manifest["sheet_index"]["2023"]["sha256"]) == 64
    assert manifest["args"]["years"] == [2022, 2023]
    readme = (out / "README.md").read_text(encoding="utf-8")
    for needle in ("2022", "2023", "CC-BY 4.0", "b3_pw_datum",
                   "2023: 1 (50.0%) -> 2023 imagery", "2020: 1 (50.0%) -> 2022 imagery"):
        assert needle in readme, needle

    again = main(argv, opener=opener, sleep=lambda s: None)
    assert again["skipped"] == 2 and again["downloaded"] == 0


def test_a_403_falls_back_to_the_next_candidate_block(tmp_path):
    from src.ortho_download import download_all
    from src.ortho_tiles import load_sheet_index, tile_url
    index = load_sheet_index({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"BLOK": "3", "PERCEEL": "Perceel 1"},
         "geometry": {"type": "MultiPolygon", "coordinates": [[[
             [80000, 450000], [81600, 450000], [81600, 460000], [80000, 460000], [80000, 450000]]]]}},
        {"type": "Feature", "properties": {"BLOK": "12", "PERCEEL": "Perceel 2"},
         "geometry": {"type": "MultiPolygon", "coordinates": [[[
             [81600, 450000], [90000, 450000], [90000, 460000], [81600, 460000], [81600, 450000]]]]}}]})
    good = tile_url(81000, 455000, 2, 12, YEAR)
    opener = FakeOpener({good: tile_bytes(81000, 455000)})   # centre block's URL -> 403
    stats = download_all({(81000, 455000)}, index, tmp_path, YEAR, opener=opener,
                         sleep=lambda s: None, tile_px=TILE_PX)
    assert stats["downloaded"] == 1 and stats["failed"] == {}
    # A 403 is final for that URL: one HEAD, no retries, then the next block.
    assert [m for m, u in opener.calls if u != good] == ["HEAD"]


def test_dry_run_downloads_nothing(tmp_path):
    hague = tmp_path / "The Hague"
    subset = _subset(hague / "cleaner", {"a": (80100, 454100, 80110, 454110, 4)})
    opener = FakeOpener({})
    index = _index(tmp_path / "BM2022.geojson")
    totals = main([str(subset), "--out", str(hague / "raw" / "ortho"), "--years", "2022", "2023",
                   "--sheet-index", f"2022={index}", "--sheet-index", f"2023={index}",
                   "--dry-run", "--plan-workers", "1"], opener=opener)
    assert totals["tiles_planned"] == {"2022": 1, "2023": 0} and opener.calls == []
    assert not (hague / "raw" / "ortho" / "2022").exists()
