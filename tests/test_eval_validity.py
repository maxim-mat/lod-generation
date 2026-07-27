import json
from pathlib import Path

from src.eval.validity import _to_cityjsonseq, parse_val3dity_report

FIX = Path(__file__).parent / "fixtures" / "val3dity_report.json"


def test_parse_report_counts_valid_and_errors():
    report = json.loads(FIX.read_text())
    out = parse_val3dity_report(report)
    assert len(out["valid_flags"]) == 2
    assert out["valid_fraction"] == 0.5
    assert sum(out["error_histogram"].values()) >= 1
    # error codes must be extracted as integer codes (stringified for dict keys),
    # not str()'d whole error dicts.
    assert "302" in out["error_histogram"]


def test_seq_is_header_plus_features():
    """val3dity's stdin reader wants CityJSONSeq: one CityJSON header line, then
    CityJSONFeature lines -- not a stream of whole CityJSON documents."""
    cjs = [{"type": "CityJSON", "CityObjects": {f"b{i}": {"type": "Building"}},
            "vertices": [[0, 0, 0]]} for i in range(2)]
    lines = [json.loads(l) for l in _to_cityjsonseq(cjs).splitlines()]
    assert len(lines) == 3
    assert lines[0]["type"] == "CityJSON" and "transform" in lines[0]
    assert [l["type"] for l in lines[1:]] == ["CityJSONFeature"] * 2
    assert [l["id"] for l in lines[1:]] == ["b0", "b1"]
