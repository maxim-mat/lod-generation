"""3D validity via val3dity (Ledoux, "val3dity: a software to validate 3D
GIS primitives according to the international standards", 2018), gated on
the binary being on PATH.
"""
import json
import logging
import shutil
import subprocess

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


def _to_cityjsonseq(cjs):
    """One JSON object per line (CityJSONSeq) for val3dity stdin streaming."""
    return "\n".join(json.dumps(cj) for cj in cjs)


def check_validity(cjs, val3dity_path=None):
    """Run val3dity on CityJSON objects `cjs`; None if the binary is absent."""
    exe = val3dity_path or shutil.which("val3dity")
    if not exe:
        logger.warning("val3dity not found; skipping the validity arm.")
        return None
    proc = subprocess.run(
        [exe, "stdin", "--report"], input=_to_cityjsonseq(cjs),
        capture_output=True, text=True, check=True,
    )
    return parse_val3dity_report(json.loads(proc.stdout))
