"""Sanity and baseline check of every post-hoc metric on real buildings.

Two readings from the same scores:
  * sanity -- each mesh against itself must score ~0 on every metric; anything
    else is a metric bug, not a model finding;
  * baseline -- LOD1 against its own LOD2 is the do-nothing prediction. Its
    numbers are printed (pytest -s) as the floor a generation must beat.

Reads ONE tile per LOD (max_files=1), never the datamodule. Meshes pass through
the same tokenizer round trip the eval scores against. Skipped when the corpus
is not on disk.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dataset.mesh_dataset import MeshDataset, detokenize
from src.eval.posthoc_eval import METRICS, score_object
from src.utils.config import SinkhornConfig

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "The Hague" / "cleaner"
NUM_BINS = 128
N_OBJECTS = 8
N_POINTS = 4096
# Smaller than the eval default so the test stays under a minute on CPU. The
# self-score assertions do not depend on it: identical inputs give exactly 0.
SINK = SinkhornConfig(n_points=512, blur_m=0.1, max_iters=200, tol=1e-6)

pytestmark = pytest.mark.skipif(
    not ((DATA / "LOD1").is_dir() and (DATA / "LOD2").is_dir()),
    reason=f"real corpus not present under {DATA}")

DISTANCES = ("chamfer_m", "hausdorff_p95_m", "hausdorff_max_m")


@pytest.fixture(scope="module")
def pairs():
    # The mesh-v5 runs' data settings: LOD1 in, LOD2 out, 128 bins, +z margin 0.1.
    ds = MeshDataset(DATA, lod_in="LOD1", lod_out="LOD2", num_bins=NUM_BINS,
                     margin_lo=[0.0, 0.0, 0.0], margin_hi=[0.0, 0.0, 0.1],
                     max_faces=200, max_files=1, tokenization="coord")
    assert len(ds) >= N_OBJECTS, f"only {len(ds)} pairs in the first tile"
    out = []
    for i in range(N_OBJECTS):
        item = ds[i]
        centre, scale = item["center"].numpy(), item["scale"].numpy()
        v1, f1 = detokenize(item["cond"].numpy(), NUM_BINS)
        v2, f2 = detokenize(item["tgt"].numpy(), NUM_BINS)
        out.append((ds.ids[i], (v1 * scale + centre, f1), (v2 * scale + centre, f2)))
    return out


@pytest.fixture(scope="module")
def scores(pairs):
    def s(a, b):
        return score_object(a, {"x": b}, METRICS, n_points=N_POINTS, sinkhorn_cfg=SINK)
    return [{"id": oid, "lod1_self": s(l1, l1), "lod2_self": s(l2, l2),
             "lod1_lod2": s(l1, l2), "lod2_lod1": s(l2, l1)}
            for oid, l1, l2 in pairs]


@pytest.mark.parametrize("which", ["lod1_self", "lod2_self"])
def test_every_metric_scores_a_mesh_against_itself_as_zero(scores, which):
    for r in scores:
        row = r[which]
        for key in DISTANCES:
            value = row[f"{key}_vs_x"]
            # Not exactly 0: the proximity query runs on world coordinates.
            assert np.isfinite(value) and value < 1e-6, (r["id"], key, value)
        roof = row["roof_chamfer_m_vs_x"]
        assert np.isnan(roof) or roof < 1e-6, (r["id"], roof)
        assert row["sinkhorn_m_vs_x"] < 0.1 * SINK.blur_m, r["id"]
        if row["vol_ok_vs_x"]:
            assert row["rel_symdiff_vs_x"] < 1e-9, r["id"]


def test_volume_checks_are_not_vacuous(scores):
    assert any(r["lod2_self"]["vol_ok_vs_x"] for r in scores)


def test_lod1_against_lod2_is_a_well_formed_baseline(scores):
    keys = DISTANCES + ("roof_chamfer_m", "sinkhorn_m", "symdiff_m3", "rel_symdiff")
    for r in scores:
        pair, own = r["lod1_lod2"], r["lod1_self"]
        assert np.isfinite(pair["chamfer_m_vs_x"]) and np.isfinite(pair["sinkhorn_m_vs_x"])
        for key in keys:
            value = pair[f"{key}_vs_x"]
            if not np.isfinite(value):
                continue
            # Zero is allowed: 31.8% of cleaner pairs are effectively identical.
            assert value >= 0.0, (r["id"], key)
            base = own.get(f"{key}_vs_x", np.nan)
            if np.isfinite(base):
                assert value >= base - 1e-6, (r["id"], key)
        assert pair["hausdorff_max_m_vs_x"] >= pair["hausdorff_p95_m_vs_x"]
        assert pair["chamfer_m_vs_x"] <= pair["hausdorff_max_m_vs_x"]


def test_swapping_the_arguments(scores):
    """Exact CSG and same-seed Sinkhorn agree tightly after a swap. The chamfer
    family only agrees statistically: `surface_distances` samples the two sides
    with seeds `seed` and `seed + 1`, so a swap resamples both clouds.
    `hausdorff_max_m` is not compared -- a max over samples is not stable."""
    for r in scores:
        a, b = r["lod1_lod2"], r["lod2_lod1"]
        if a["vol_ok_vs_x"] and b["vol_ok_vs_x"]:
            # 1e-3, measured: manifold3d's boolean is commutative to the bit on
            # 7 of the 8 pairs in this tile; one differs by 5.2e-5 relative,
            # which is 7e-6 of the building's volume amplified by the
            # cancellation a symmetric difference performs (see
            # src/analysis/exact_volume.py). A real directional bug would be
            # percent-level and still fails here.
            assert a["symdiff_m3_vs_x"] == pytest.approx(b["symdiff_m3_vs_x"],
                                                          rel=1e-3, abs=1e-6)
        assert a["sinkhorn_m_vs_x"] == pytest.approx(b["sinkhorn_m_vs_x"],
                                                      rel=0.02, abs=0.01)
        assert a["chamfer_m_vs_x"] == pytest.approx(b["chamfer_m_vs_x"],
                                                     rel=0.05, abs=0.01)
        # 25%, from measurement rather than taste: p95 is a percentile of 4096
        # sampled distances, and re-running the SAME direction under four seeds
        # moves it by up to 13.9% on this tile (chamfer moves <= 10%). The
        # largest swap gap seen is 13.3%, i.e. inside that noise, so a tighter
        # bound would test the sampler's luck instead of the metric.
        assert a["hausdorff_p95_m_vs_x"] == pytest.approx(b["hausdorff_p95_m_vs_x"],
                                                           rel=0.25, abs=0.02)


def test_print_the_do_nothing_baseline(scores):
    table = pd.DataFrame([{"id": r["id"], **{k.removesuffix("_vs_x"): v
                                            for k, v in r["lod1_lod2"].items()}}
                          for r in scores]).set_index("id")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\nLOD1 vs LOD2 (do-nothing baseline), per object:")
        print(table.to_string())
        print("\nmeans:")
        print(table.select_dtypes("number").mean().to_string())
    assert len(table) == N_OBJECTS
