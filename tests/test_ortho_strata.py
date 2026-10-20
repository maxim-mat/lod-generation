import pandas as pd

from src.analysis.ortho_strata import stratify
from tests.test_ortho_tiles import write_tile


def test_stratify_groups_by_point_cloud_year_and_height(tmp_path):
    root = tmp_path / "The Hague" / "mini_cleaner"
    write_tile(root / "LOD2" / "Source B" / "t.city.json",
               {"a": (0, 0, 10, 10, 6), "b": (20, 0, 30, 10, 25), "c": (40, 0, 50, 10, 8)},
               datum={"a": 2023, "b": 2023, "c": 2020})
    runs = {"true": pd.DataFrame({"object_id": ["a", "b", "c"], "m": [1.0, 3.0, 5.0]}),
            "shuffled": pd.DataFrame({"object_id": ["a", "b", "c"], "m": [2.0, 4.0, 6.0]})}
    out = stratify(root, runs, "m")
    row = out[(out["mode"] == "true") & (out["stratum"] == "pw_datum") & (out["value"] == "2023")]
    assert row["n"].item() == 2 and row["mean"].item() == 2.0
    tall = out[(out["mode"] == "shuffled") & (out["stratum"] == "height_bin") & (out["value"] == "20-40")]
    assert tall["mean"].item() == 4.0


def test_stratify_compares_on_common_ids_and_reports_the_imagery_gap(tmp_path):
    root = tmp_path / "The Hague" / "mini_cleaner"
    write_tile(root / "LOD2" / "Source B" / "t.city.json",
               {"a": (0, 0, 10, 10, 6), "b": (20, 0, 30, 10, 6), "c": (40, 0, 50, 10, 6)},
               datum={"a": 2023, "b": 2020, "c": 2014})
    runs = {"v5": pd.DataFrame({"object_id": ["a", "b", "c"], "m": [1.0, 2.0, 9.0]}),
            "true": pd.DataFrame({"object_id": ["a", "b"], "m": [0.5, 1.5]})}   # "c" skipped
    out = stratify(root, runs, "m", assignment={"a": 2023, "b": 2022, "c": 2022})
    v5_all = out[(out["mode"] == "v5") & (out["stratum"] == "all")]
    assert v5_all["n"].item() == 2 and v5_all["mean"].item() == 1.5        # "c" excluded
    gaps = out[(out["mode"] == "true") & (out["stratum"] == "imagery_gap")]
    assert dict(zip(gaps["value"], gaps["n"])) == {"0": 1, "2": 1}
