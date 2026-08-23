"""Selection-ladder logic for `src.select_cityjson`.

Only the pure parts: fence arithmetic and which ids survive. Scanning and
writing are exercised by running the script -- they are I/O over a 1.5 GB
corpus and have no branch worth pinning here.
"""
import numpy as np
import pytest

from src.select_cityjson import DROP_DEFECTS, select, tukey_fences, tukey_inliers


def row(oid, n_verts, defects="", source="Source B"):
    return {"source": source, "tile": "t.json", "id": oid,
            "n_verts": n_verts, "n_faces": 6, "defects": defects}


# ----------------------------------------------------------------------
# tukey_fences
# ----------------------------------------------------------------------

def test_raw_fences_match_the_textbook_definition():
    x = np.arange(1, 101)                     # q1=25.75, q3=75.25, iqr=49.5
    lo, hi = tukey_fences(x, k=1.5, transform="raw")
    assert lo == pytest.approx(25.75 - 1.5 * 49.5)
    assert hi == pytest.approx(75.25 + 1.5 * 49.5)


def test_log_fences_come_back_in_original_units():
    x = np.array([10.0, 100.0, 1000.0])
    lo, hi = tukey_fences(x, k=1.5, transform="log")
    # log10 -> [1, 2, 3]; q1=1.5, q3=2.5, iqr=1 -> fences [0, 4]
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(10_000.0)


def test_log_transform_is_far_less_aggressive_on_skewed_counts():
    """The reason `log` is the default -- raw fences cut into the bulk."""
    rng = np.random.default_rng(0)
    x = np.round(10 ** rng.normal(1.2, 0.5, 20_000)).clip(8)
    raw_lo, raw_hi = tukey_fences(x, transform="raw")
    log_lo, log_hi = tukey_fences(x, transform="log")
    assert ((x < raw_lo) | (x > raw_hi)).mean() > ((x < log_lo) | (x > log_hi)).mean()


def test_unknown_transform_raises():
    with pytest.raises(ValueError, match="fence transform"):
        tukey_fences([1.0, 2.0], transform="sqrt")


def test_empty_input_raises():
    with pytest.raises(ValueError, match="no values"):
        tukey_fences([], transform="raw")


# ----------------------------------------------------------------------
# select
# ----------------------------------------------------------------------

def test_unpaired_objects_are_dropped_before_anything_else():
    rows = [row("a", 20), row("b", 20)]
    keep, stats = select(rows, {"a"})
    assert keep == {"a"}
    assert stats["lod2_objects"] == 2
    assert stats["after_lod1_pairing"] == 1


def test_every_drop_defect_removes_its_object():
    rows = [row(d, 20, defects=d) for d in DROP_DEFECTS] + [row("ok", 20)]
    keep, stats = select(rows, {r["id"] for r in rows})
    assert keep == {"ok"}
    assert stats["after_defect_drop"] == 1


def test_disconnected_is_kept():
    rows = [row("a", 20, defects="disconnected"), row("b", 20)]
    keep, stats = select(rows, {"a", "b"})
    assert keep == {"a", "b"}
    assert stats["kept_disconnected"] == 1


def test_disconnected_alongside_a_real_defect_still_drops():
    rows = [row("a", 20, defects="disconnected|open_shell"), row("b", 20)]
    keep, _ = select(rows, {"a", "b"})
    assert keep == {"b"}


def test_vertex_fence_drops_the_high_tail():
    rows = [row(f"n{i}", 20) for i in range(50)] + [row("huge", 100_000)]
    keep, stats = select(rows, {r["id"] for r in rows})
    assert "huge" not in keep
    assert stats["dropped_above_fence"] == 1
    assert stats["after_vertex_fence"] == 50


def test_defect_breakdown_counts_only_the_paired_population():
    rows = [row("a", 20, defects="open_shell"),      # paired, counted
            row("b", 20, defects="open_shell")]      # unpaired, not counted
    _, stats = select(rows, {"a"})
    assert stats["defect_breakdown"] == {"open_shell": 1}


def test_no_survivors_returns_empty_without_computing_fences():
    keep, stats = select([row("a", 20, defects="open_shell")], {"a"})
    assert keep == set()
    assert "fence_lo_verts" not in stats


def test_inliers_keep_values_sitting_exactly_on_the_fence():
    """`10 ** log10(v)` does not round-trip; comparing in log space does."""
    x = np.array([10.0, 100.0, 1000.0])
    assert tukey_inliers(x, transform="log").all()


def test_inliers_with_zero_iqr_keep_everything_at_that_value():
    x = np.array([20.0] * 50)
    assert tukey_inliers(x, transform="log").all()
