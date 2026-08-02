"""3D validity via val3dity (Ledoux, "val3dity: a software to validate 3D
GIS primitives according to the international standards", 2018), gated on
the binary being on PATH.
"""
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_val3dity_report(report):
    """Flatten a val3dity --report (v2.x JSON) into per-feature validity + an
    error code histogram.

    Each feature has "validity" (bool) and "errors": a list of
    {"code": int, "description": str} dicts. Histogram keys are the
    stringified error codes (not the whole error dict) so counts are
    meaningful.
    """
    features = report.get("features", [])
    flags, hist = [], {}
    for feat in features:
        flags.append(bool(feat.get("validity", False)))
        for err in feat.get("errors", []):
            code = str(err["code"]) if isinstance(err, dict) else str(err)
            hist[code] = hist.get(code, 0) + 1
    valid_fraction = float(sum(flags) / len(flags)) if flags else 0.0
    return {"valid_flags": flags, "error_histogram": hist, "valid_fraction": valid_fraction}


_SEQ_HEADER = {
    "type": "CityJSON", "version": "1.1",
    "transform": {"scale": [1.0, 1.0, 1.0], "translate": [0.0, 0.0, 0.0]},
    "CityObjects": {}, "vertices": [],
}


def _to_cityjsonseq(cjs):
    """CityJSONSeq for val3dity: a CityJSON header line, then one CityJSONFeature
    per line. Writing whole CityJSON documents per line instead makes val3dity
    reject every line after the first.
    """
    lines = [json.dumps(_SEQ_HEADER)]
    for cj in cjs:
        lines.append(json.dumps({
            "type": "CityJSONFeature",
            "id": next(iter(cj["CityObjects"])),
            "CityObjects": cj["CityObjects"],
            "vertices": cj["vertices"],
        }))
    return "\n".join(lines)


def check_validity(cjs, val3dity_path=None):
    """Run val3dity on CityJSON objects `cjs`; None if the binary is absent or
    the run fails (the validity arm is optional, so it must not kill the eval).
    """
    exe = val3dity_path or shutil.which("val3dity")
    if not exe:
        logger.warning("val3dity not found; skipping the validity arm.")
        return None
    with tempfile.TemporaryDirectory() as tmp:
        # A .jsonl *file*, not "stdin": val3dity's stdin branch returns before it
        # ever reaches the --report block, so a piped run exits 0, writes no
        # report, and only prints '"<id>" [codes]' lines to stdout. File input
        # goes through the same CityJSONSeq reader but writes the JSON report.
        src = Path(tmp) / "input.jsonl"
        src.write_text(_to_cityjsonseq(cjs), encoding="utf-8")
        report = Path(tmp) / "report.json"
        proc = subprocess.run(
            [exe, str(src), "--report", str(report)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not report.exists():
            logger.error("val3dity failed (rc=%d): %s", proc.returncode,
                         (proc.stderr or proc.stdout).strip()[:2000])
            return None
        return parse_val3dity_report(json.loads(report.read_text(encoding="utf-8")))
