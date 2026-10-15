"""Spatial-correlation statistics on synthetic fields whose answer is known."""
import numpy as np
import pytest

from src.analysis.spatial_correlation import (building_metrics, correlogram, icc1,
                                              nearest_train, permutation_null)
from tests.test_ortho_tiles import write_tile


def test_building_metrics_reads_footprint_volume_and_centroid(tmp_path):
    for lod, h in (("LOD1", 6), ("LOD2", 6)):
        write_tile(tmp_path / lod / "t.city.json", {"a": (100, 200, 110, 208, h)})
    (x, y, area, vol, faces, vol1), = building_metrics(tmp_path / "LOD1" / "t.city.json",
                                                       tmp_path / "LOD2" / "t.city.json").values()
    assert (x, y) == pytest.approx((105.0, 204.0))
    assert area == pytest.approx(80.0) and vol == pytest.approx(480.0)
    assert faces == 12                    # six quads -> twelve LOD2 triangles
    assert vol1 == pytest.approx(480.0)   # LOD1 volume, for the LOD1-residual targets


def test_icc_is_one_for_identical_members_and_zero_for_iid_noise():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(200), 25)
    clustered = rng.normal(size=200)[groups]
    assert icc1(clustered, groups) == pytest.approx(1.0, abs=1e-9)
    assert abs(icc1(rng.normal(size=groups.size), groups)) < 0.02


def test_permutation_null_centres_on_zero_and_rejects_real_clustering():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(100), 20)
    values = rng.normal(size=100)[groups] + rng.normal(size=groups.size)
    null = permutation_null(values, groups, n_perm=200, seed=2)
    assert abs(null.mean()) < 0.02
    assert (null >= icc1(values, groups)).mean() == 0.0


def test_correlogram_decays_with_distance_on_a_smooth_field():
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 4000, size=(6000, 2))
    values = np.sin(xy[:, 0] / 300) + np.cos(xy[:, 1] / 300) + 0.3 * rng.normal(size=6000)
    table = correlogram(xy, values, edges=[0, 50, 200, 1000, 3000], n_anchors=1500, seed=4)
    corr = table["corr"].to_numpy()
    assert corr[0] > 0.7 and corr[-1] < corr[0] - 0.5
    assert (table["n_pairs"] > 0).all()


def test_nearest_train_is_closer_and_more_similar_under_a_random_split():
    from src.dataset.splits import random_split_indices, tile_split_indices
    rng = np.random.default_rng(5)
    xy = rng.uniform(0, 4000, size=(8000, 2))
    values = np.sin(xy[:, 0] / 300) + 0.2 * rng.normal(size=8000)
    tiles = [f"{int(x // 500)}-{int(y // 500)}" for x, y in xy]
    r_train, r_val, r_test = random_split_indices(len(xy), (0.8, 0.1, 0.1), 0)
    (t_train, t_val, t_test), _ = tile_split_indices(tiles, (0.8, 0.1, 0.1), 0)
    rand = nearest_train(xy, values, r_train, r_val + r_test)
    tile = nearest_train(xy, values, t_train, t_val + t_test)
    assert rand["median_dist_m"] < tile["median_dist_m"]
    assert rand["corr"] > tile["corr"]


def test_main_writes_the_report_for_a_tiny_subset(tmp_path):
    import json

    from src.analysis.spatial_correlation import main
    root = tmp_path / "mini_cleaner"
    for t in range(10):
        boxes = {f"t{t}-b{b}": (100 * t + 20 * b, 0, 100 * t + 20 * b + 8 + b, 10, 4 + t)
                 for b in range(4)}
        for lod in ("LOD1", "LOD2"):
            write_tile(root / lod / "Source B" / f"10-{t}-0.city.json", boxes)
    main([str(root), "--out", str(tmp_path / "out"), "--n-perm", "5", "--workers", "1"])
    rep = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert rep["n_buildings"] == 40 and rep["n_tiles"] == 10 and rep["git_commit"]
    assert set(rep["metrics"]) == {"footprint_m2", "volume_m3", "n_faces"}
    # Every tile repeats the same footprints, but height is set per tile.
    assert rep["metrics"]["footprint_m2"]["icc1"] < 0.1
    assert rep["metrics"]["volume_m3"]["icc1"] > 0.5
    assert (tmp_path / "out" / "report.md").read_text(encoding="utf-8").startswith(
        "# Spatial correlation (40 buildings, 10 tiles)")


def test_correlogram_memory_stays_bounded_on_a_dense_city():
    # cleaner is ~258k buildings in ~100 km2, so a 4 km radius holds ~100k
    # neighbours per anchor. Materialising them all at once ran out of memory.
    import tracemalloc
    rng = np.random.default_rng(6)
    xy = rng.uniform(0, 2500, size=(100_000, 2))          # every pair within 4 km
    values = rng.normal(size=len(xy))
    tracemalloc.start()
    correlogram(xy, values, n_anchors=100, seed=7)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 150e6, f"peak {peak / 1e6:.0f} MB"


def test_building_table_rows_come_in_file_order_whatever_the_worker_count(tmp_path):
    # report()'s random split depends on row order, so it must not depend on
    # which worker finished first.
    from src.analysis.spatial_correlation import building_table
    root = tmp_path / "mini_cleaner"
    for t in range(8):
        n = 40 if t == 0 else 2                     # the first file is the slow one
        boxes = {f"t{t}-b{b}": (100 * t + 2 * b, 0, 100 * t + 2 * b + 1, 1, 3) for b in range(n)}
        for lod in ("LOD1", "LOD2"):
            write_tile(root / lod / "Source B" / f"10-{t}-0.city.json", boxes)
    table = building_table(root, workers=4)
    assert table["tile"].tolist() == sorted(table["tile"])
