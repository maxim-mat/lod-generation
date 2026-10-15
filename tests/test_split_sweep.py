"""Block-size sweep: grid/tile/random splits scored on synthetic fields whose answer is known."""
import json

import numpy as np
import pandas as pd
import pytest

from src.analysis.split_sweep import add_targets, sweep
from tests.test_ortho_tiles import write_tile


def _table(n=3000, seed=0):
    """Buildings on a 4 km square whose roofs vary smoothly in space."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 4000, size=(n, 2))
    fp = np.exp(rng.normal(4.5, 0.5, n))
    v1 = fp * rng.uniform(5, 15, n)
    field = np.sin(xy[:, 0] / 300) + np.cos(xy[:, 1] / 300)
    return pd.DataFrame({
        "id": [f"b{i}" for i in range(n)],
        "tile": [f"{int(x // 1000)}-{int(y // 1000)}" for x, y in xy],
        "x": xy[:, 0], "y": xy[:, 1], "footprint_m2": fp, "lod1_volume_m3": v1,
        "volume_m3": v1 * np.exp(0.2 + 0.1 * field),
        "n_faces": np.round(np.exp(2 + 0.3 * np.log(fp) + 0.5 * field)).clip(4),
    })


def test_targets_factor_out_lod1():
    t = _table()
    t["n_faces"] = np.exp(1.0 + 0.4 * np.log(t["footprint_m2"]) + 0.2 * np.log(t["lod1_volume_m3"]))
    out = add_targets(t)
    assert out["faces_resid"].abs().max() < 1e-9          # fully explained by LOD1
    assert np.allclose(out["roof_ratio"], np.log(t["volume_m3"] / t["lod1_volume_m3"]))


def test_sweep_has_a_row_per_scheme_size_buffer_and_seed():
    rows = sweep(_table(), sizes=[250, 1000], buffers=[0, 50], seeds=[0, 1])
    counts = rows.groupby("scheme").size().to_dict()
    assert counts == {"random": 2, "tile": 4, "grid": 8}   # random takes no buffer
    for col in ("block_m", "buffer_m", "seed", "n_groups", "n_eval_groups", "n_train", "n_eval",
                "train_lost_frac", "dist_median_m", "dist_min_m", "within_50m",
                "corr_faces_resid", "mad_roof_ratio"):
        assert col in rows, col


def test_buffer_guarantees_the_minimum_distance():
    rows = sweep(_table(), sizes=[250], buffers=[0, 50], seeds=[0])
    buffered = rows[rows["buffer_m"] == 50]
    assert (buffered["dist_min_m"] >= 50).all() and (buffered["within_50m"] == 0).all()
    assert (buffered["train_lost_frac"] > 0).all()


def test_bigger_blocks_push_eval_buildings_further_from_train():
    rows = sweep(_table(), sizes=[100, 1000], buffers=[0], seeds=[0])
    med = rows.set_index(["scheme", "block_m"])["dist_median_m"]
    assert med[("random", 0)] < med[("grid", 100)] < med[("grid", 1000)]
    corr = rows.set_index(["scheme", "block_m"])["corr_faces_resid"]
    assert corr[("random", 0)] > corr[("grid", 1000)]


def _subset(tmp_path):
    root = tmp_path / "mini_cleaner"
    for t in range(6):
        boxes = {f"t{t}-b{b}": (60 * t + 12 * b, 0, 60 * t + 12 * b + 8 + b, 10, 4 + t)
                 for b in range(5)}
        for lod in ("LOD1", "LOD2"):
            write_tile(root / lod / "Source B" / f"10-{t}-0.city.json", boxes)
    return root


def test_main_writes_csvs_and_run_metadata(tmp_path):
    from src.analysis.split_sweep import main
    out = tmp_path / "out"
    main([str(_subset(tmp_path)), "--out", str(out), "--sizes", "50", "100", "--buffers", "0", "10",
          "--seeds", "0", "--n-anchors", "20", "--workers", "1"])
    rows = pd.read_csv(out / "split_sweep.csv")
    assert len(rows) == 1 + 2 + 4                          # random, tile x 2 buffers, grid 2 x 2
    corr = pd.read_csv(out / "correlogram.csv")
    assert {"target", "lo", "hi", "n_pairs", "corr"} <= set(corr.columns)
    assert len(pd.read_csv(out / "buildings.csv")) == 30
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["git_commit"] and run["args"]["sizes"] == [50.0, 100.0]


def test_main_reuses_a_saved_building_table(tmp_path):
    from src.analysis.split_sweep import main
    out = tmp_path / "out"
    main([str(_subset(tmp_path)), "--out", str(out), "--sizes", "50", "--buffers", "0",
          "--seeds", "0", "--n-anchors", "20", "--workers", "1"])
    again = tmp_path / "again"
    main(["--table", str(out / "buildings.csv"), "--out", str(again), "--sizes", "50",
          "--buffers", "0", "--seeds", "0", "--n-anchors", "20"])
    pd.testing.assert_frame_equal(pd.read_csv(out / "split_sweep.csv"),
                                  pd.read_csv(again / "split_sweep.csv"))


def test_plot_writes_figures_from_the_saved_csvs(tmp_path):
    from src.analysis.plot_split_sweep import main as plot_main
    from src.analysis.split_sweep import main
    out = tmp_path / "out"
    main([str(_subset(tmp_path)), "--out", str(out), "--sizes", "50", "100", "--buffers", "0", "10",
          "--seeds", "0", "1", "--n-anchors", "20", "--workers", "1"])
    written = plot_main([str(out)])
    assert written and all(p.suffix == ".png" and p.stat().st_size > 0 for p in written)


def test_sweep_refuses_an_empty_size_list():
    with pytest.raises(ValueError, match="sizes"):
        sweep(_table(n=200), sizes=[], buffers=[0], seeds=[0])
