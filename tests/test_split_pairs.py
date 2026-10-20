"""Split explorer precompute: splits, shortlist, pair metrics, match selection, CLI.

Synthetic boxes only -- never the real dataset.
"""
import json

import numpy as np
import pandas as pd
import pytest

import src.analysis.split_pairs as sp
from src.analysis.split_pairs import (CRITERIA, SPLITS, best_matches, centred, pair_metrics,
                                      sample_test, shortlist, split_parts)
from src.dataset.splits import split_name
from tests.test_splits import _line_subset

RATIOS = (0.8, 0.1, 0.1)
TRIS = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                 [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])


def box(x0=0.0, y0=0.0, z0=0.0, w=10.0, d=10.0, h=6.0):
    """Closed, outward-wound box as ``(verts, faces)``."""
    v = np.array([[0, 0, 0], [w, 0, 0], [w, d, 0], [0, d, 0],
                  [0, 0, h], [w, 0, h], [w, d, h], [0, d, h]], float) + [x0, y0, z0]
    return v, TRIS.copy()


# ---------------------------------------------------------------- splits

def test_split_names_match_split_name():
    assert set(SPLITS) == {"random", "tile", "tile-buffer50m", "grid500m-buffer50m",
                           "grid1000m-buffer50m"}
    assert all(name == split_name(*p) for name, p in SPLITS.items())


@pytest.mark.parametrize("scheme,block,buffer", [("random", None, 0.0), ("tile", None, 0.0),
                                                 ("tile", None, 150.0), ("grid", 200, 150.0)])
def test_split_parts_matches_the_datamodule(tmp_path, scheme, block, buffer):
    from src.dataset.mesh_datamodule import MeshDataModule
    dm = MeshDataModule(_line_subset(tmp_path), lod_in="LOD1", lod_out="LOD2", batch_size=1,
                        seed=3, train_val_test_split=(0.5, 0.25, 0.25), split=scheme,
                        split_block_m=block, split_buffer_m=buffer)
    dm.setup()
    parts = split_parts(dm.full_dataset, scheme, (0.5, 0.25, 0.25), 3, block, buffer)
    assert parts == [list(dm.train_dataset.indices), list(dm.val_dataset.indices),
                     list(dm.test_dataset.indices)]


def test_split_parts_random_never_reads_xy():
    class Stub:
        files = ["t"] * 20

        def __len__(self):
            return 20

        @property
        def xy(self):
            raise AssertionError("random split read xy")
    assert sum(map(len, split_parts(Stub(), "random", RATIOS, 42))) == 20


# ---------------------------------------------------------------- sample, shortlist

def test_sample_is_seeded_and_sorted():
    a = sample_test(list(range(100, 200)), 10, 42)
    assert a == sample_test(list(range(100, 200)), 10, 42) == sorted(a)
    assert len(set(a)) == 10 and set(a) <= set(range(100, 200))
    assert a != sample_test(list(range(100, 200)), 10, 0)


def test_sample_takes_all_when_fewer():
    assert sample_test([5, 3, 9], 1000, 42) == [3, 5, 9]


def test_shortlist_is_nearest_first():
    xy = np.array([[0, 0], [1, 0], [5, 0], [2, 0], [10, 0]], float)
    idx, dist = shortlist(xy, train_idx=[1, 2, 3, 4], query_idx=[0], k=3)
    assert idx.tolist() == [[1, 3, 2]] and dist.tolist() == [[1.0, 2.0, 5.0]]


def test_shortlist_clips_k():
    xy = np.array([[0, 0], [1, 0], [2, 0]], float)
    idx, dist = shortlist(xy, train_idx=[1, 2], query_idx=[0], k=50)
    assert idx.shape == dist.shape == (1, 2)


# ---------------------------------------------------------------- metrics

def test_centred_puts_footprint_at_origin_and_ground_at_zero():
    v, f = centred(box(81000, 455000, 3))
    assert v[:, :2].mean(axis=0) == pytest.approx((0.0, 0.0))
    assert v[:, 2].min() == pytest.approx(0.0)
    assert f.tolist() == TRIS.tolist()


def test_identical_boxes_score_zero():
    m = pair_metrics(box(), box(81000, 455000, 7), 256)
    assert m["vol_ok"] and m["symdiff_m3"] == pytest.approx(0.0, abs=1e-6)
    assert m["chamfer_m"] == pytest.approx(0.0, abs=1e-6)


def test_taller_box_scores_its_extra_volume():
    m = pair_metrics(box(h=6), box(500, 0, 0, h=8), 256)
    assert m["vol_ok"] and m["symdiff_m3"] == pytest.approx(200.0, rel=1e-6)
    assert m["rel_symdiff"] == pytest.approx(200.0 / 800.0, rel=1e-6)
    assert m["chamfer_m"] > 0


def test_pair_metrics_survives_a_failure(monkeypatch):
    def boom(a, b):
        raise RuntimeError("csg")
    monkeypatch.setattr(sp, "exact_pair_volumes", boom)
    m = pair_metrics(box(), box(h=8), 256)
    assert not m["vol_ok"] and np.isnan(m["symdiff_m3"]) and np.isfinite(m["chamfer_m"])


# ---------------------------------------------------------------- match selection

def _cands(vol_ok=(True, True, True)):
    return pd.DataFrame({
        "split": ["random"] * 3, "test_id": ["t"] * 3, "cand_id": ["near", "same", "alike"],
        "rank": [0, 1, 2], "distance_m": [5.0, 40.0, 90.0],
        "symdiff_m3": [300.0, 10.0, 50.0], "rel_symdiff": [0.3, 0.01, 0.05],
        "vol_ok": list(vol_ok), "chamfer_m": [1.0, 0.4, 0.1]})


def test_best_matches_picks_each_criterion():
    best = best_matches(_cands()).set_index("criterion")["cand_id"].to_dict()
    assert set(CRITERIA) == {"distance", "volume", "chamfer"}
    assert best == {"distance": "near", "volume": "same", "chamfer": "alike"}


def test_best_matches_skips_volume_when_none_ok():
    best = best_matches(_cands(vol_ok=(False, False, False)))
    assert set(best["criterion"]) == {"distance", "chamfer"}


def test_best_matches_ignores_a_not_ok_volume_winner():
    best = best_matches(_cands(vol_ok=(True, False, True))).set_index("criterion")
    assert best.loc["volume", "cand_id"] == "alike"
