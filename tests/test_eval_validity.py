import json
from pathlib import Path

from src.eval.validity import parse_val3dity_report

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
