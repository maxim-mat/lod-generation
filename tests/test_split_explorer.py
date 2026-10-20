"""Pure helpers of the split explorer (the Streamlit UI itself is not tested)."""
import pandas as pd

from src.split_explorer import METRIC_LABELS, matches


def _pairs():
    return pd.DataFrame({
        "split": ["random", "random", "random", "tile", "tile", "random"],
        "criterion": ["distance", "distance", "chamfer", "distance", "distance", "distance"],
        "test_id": ["a", "b", "a", "c", "d", "e"],
        "distance_m": [9.0, 3.0, 50.0, 200.0, 100.0, 6.0],
        "symdiff_m3": [1.0] * 6, "chamfer_m": [0.5] * 6})


def test_matches_filters_and_sorts_by_the_criterion_column():
    m = matches(_pairs(), "random", "distance")
    assert m["test_id"].tolist() == ["b", "e", "a"]
    assert m.index.tolist() == [0, 1, 2]


def test_matches_can_sort_by_another_metric_descending():
    m = matches(_pairs(), "tile", "distance", sort_by="distance_m", ascending=False)
    assert m["test_id"].tolist() == ["c", "d"]


def test_every_metric_column_has_a_label():
    assert {"distance_m", "symdiff_m3", "rel_symdiff", "chamfer_m"} <= set(METRIC_LABELS)
