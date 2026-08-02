import json
import subprocess
from pathlib import Path

from src.eval import validity
from src.eval.validity import _to_cityjsonseq, check_validity, parse_val3dity_report

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
    """val3dity's .jsonl reader wants CityJSONSeq: one CityJSON header line, then
    CityJSONFeature lines -- not a sequence of whole CityJSON documents."""
    cjs = [{"type": "CityJSON", "CityObjects": {f"b{i}": {"type": "Building"}},
            "vertices": [[0, 0, 0]]} for i in range(2)]
    lines = [json.loads(l) for l in _to_cityjsonseq(cjs).splitlines()]
    assert len(lines) == 3
    assert lines[0]["type"] == "CityJSON" and "transform" in lines[0]
    assert [l["type"] for l in lines[1:]] == ["CityJSONFeature"] * 2
    assert [l["id"] for l in lines[1:]] == ["b0", "b1"]


def test_input_is_a_jsonl_file_not_stdin(monkeypatch):
    """val3dity's stdin branch returns before it reaches the --report block, so a
    piped run exits 0, writes no report, and the whole validity arm goes None.
    The input must be an on-disk .jsonl path."""
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        # inspect while check_validity's TemporaryDirectory is still alive
        seen["input"] = Path(argv[1]).read_text(encoding="utf-8")
        seen["piped"] = kwargs.get("input")
        Path(argv[3]).write_text(FIX.read_text(), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(validity.subprocess, "run", fake_run)
    cjs = [{"CityObjects": {"b0": {"type": "Building"}}, "vertices": [[0, 0, 0]]}]
    out = check_validity(cjs, val3dity_path="val3dity-stub")

    assert seen["argv"][1].endswith(".jsonl")
    assert seen["argv"][1] != "stdin"
    assert seen["piped"] is None
    assert json.loads(seen["input"].splitlines()[1])["id"] == "b0"
    assert out["valid_fraction"] == 0.5  # report actually got parsed
