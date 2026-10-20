#!/usr/bin/env python3
"""Streamlit browser: each sampled test building beside its nearest training building.

Answers "under this split, does training hold a near-copy of this test building
close by?" -- the pair-by-pair version of the aggregate numbers in
`outputs/split_sweep`. "Nearest" is selectable: footprint-centroid distance,
exact LOD2 volume symmetric difference, or LOD2 chamfer, the last two searched
among the K geographically nearest training buildings (see
`src.analysis.split_pairs`, which precomputes everything shown here).

Loading is lazy on purpose: the CSVs hold only scalars and names, and a
CityJSON tile is read from disk only when a pair that uses it is shown.

Usage:
    python -m src.analysis.split_pairs --config configs/mesh-v5-full-opt-coord.yaml
    streamlit run src/split_explorer.py
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:                    # `streamlit run` puts src/ on the path, not the root
    sys.path.insert(0, str(REPO))

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402
import plotly.graph_objects as go                                    # noqa: E402
import streamlit as st                                               # noqa: E402

from src.visualize_cityjson import cityjson_figure, load_cityjson    # noqa: E402

DEFAULT_OUT = REPO / "outputs" / "split_pairs"
DEFAULT_DATA = REPO / "data" / "The Hague" / "cleaner"

# Same validated slots as identity_explorer: the selected split carries the
# series colour, the random-split reference a muted ink; the legend and direct
# labels name both, so identity never rests on colour alone.
LIGHT = {"series": "#2a78d6", "muted": "#8a8984", "ink2": "#52514e"}
DARK = {"series": "#3987e5", "muted": "#9a9990", "ink2": "#c3c2b7"}

CRITERION_COLUMN = {"distance": "distance_m", "volume": "symdiff_m3", "chamfer": "chamfer_m"}
METRIC_LABELS = {"distance_m": "distance (m)", "symdiff_m3": "volume sym. diff. (m³)",
                 "rel_symdiff": "relative sym. diff.", "chamfer_m": "chamfer (m)",
                 "test_faces": "test faces", "cand_faces": "match faces"}


def matches(pairs, split, criterion, sort_by=None, ascending=True):
    """The split's matches under one criterion, sorted (default: by that criterion's column)."""
    m = pairs[(pairs["split"] == split) & (pairs["criterion"] == criterion)]
    return m.sort_values(sort_by or CRITERION_COLUMN[criterion], ascending=ascending,
                         kind="stable").reset_index(drop=True)


def palette():
    try:
        return DARK if st.get_option("theme.base") == "dark" else LIGHT
    except Exception:
        return LIGHT


@st.cache_data
def read_csv(path):
    """Scalars and names only -- never geometry."""
    return pd.read_csv(path)


@st.cache_data(max_entries=6)                    # tiles are 4-14 MB, so keep a few
def read_tile(path):
    """One CityJSON tile, read only when a pair from it is shown."""
    return load_cityjson(path)


def draw_object(data_root, tile, oid, shade, title, slot):
    path = Path(data_root) / "LOD2" / tile
    if not path.exists():
        st.warning(f"missing tile: LOD2/{tile}")
        return
    cj, verts = read_tile(str(path))
    obj = (cj.get("CityObjects") or {}).get(oid)
    if obj is None:
        st.warning(f"{oid} not in LOD2/{tile}")
        return
    st.plotly_chart(cityjson_figure([obj], verts, title=title, shade=shade),
                    width="stretch", key=f"{slot}-{tile}-{oid}")


def ecdf_figure(series, col, label, log_x):
    """Cumulative share of sampled test buildings at or below each value, one line per split.

    ECDFs rather than overlaid histograms: two distributions on one axis stay
    readable, and the medians can be read straight off the 50% line.
    """
    fig = go.Figure()
    for name, values, colour in series:
        xs = np.sort(values[np.isfinite(values)])
        if log_x:
            xs = xs[xs > 0]
        if not len(xs):
            continue
        ys = np.arange(1, len(xs) + 1) / len(xs) * 100
        fig.add_scatter(x=xs, y=ys, mode="lines", name=name,
                        line=dict(color=colour, width=2),
                        hovertemplate=f"{name}<br>{label} %{{x:.3g}}<br>"
                                      "%{y:.1f}% at or below<extra></extra>")
        fig.add_annotation(x=np.log10(xs[len(xs) // 2]) if log_x else xs[len(xs) // 2],
                           y=50, text=f"{name}: median {xs[len(xs) // 2]:.3g}",
                           showarrow=False, xanchor="left", yanchor="bottom",
                           font=dict(color=col["ink2"], size=11))
    fig.update_layout(height=280, margin=dict(l=8, r=8, t=36, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=col["ink2"], size=12), hovermode="closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_yaxes(title_text="% of sampled test buildings", range=[0, 100],
                     gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    fig.update_xaxes(title_text=label + (" (log)" if log_x else ""),
                     type="log" if log_x else "linear",
                     gridcolor="rgba(128,128,128,0.10)", zeroline=False)
    return fig


def _fmt(row, column):
    value = row.get(column)
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.3g}" if isinstance(value, float) else str(value)


def main():
    st.set_page_config(page_title="Split explorer", layout="wide")
    st.title("Test building vs its nearest training building")

    with st.sidebar:
        out_dir = Path(st.text_input("split_pairs output", str(DEFAULT_OUT)))
        data_root = Path(st.text_input("dataset root", str(DEFAULT_DATA)))
    pairs_path, cands_path = out_dir / "pairs.csv", out_dir / "candidates.csv"
    if not pairs_path.exists():
        st.error(f"Not found: {pairs_path}. Run "
                 "`python -m src.analysis.split_pairs --config configs/mesh-v5-full-opt-coord.yaml`.")
        return
    pairs = read_csv(str(pairs_path))
    run = (json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
           if (out_dir / "run.json").exists() else {})

    splits = list(dict.fromkeys(pairs["split"]))
    with st.sidebar:
        split = st.selectbox("split", splits, key="split",
                             index=splits.index("grid1000m-buffer50m")
                             if "grid1000m-buffer50m" in splits else 0)
        criterion = st.radio("nearest by", list(CRITERION_COLUMN), horizontal=True,
                             key="criterion",
                             help="distance: footprint centroids. volume / chamfer: the "
                                  f"most similar of the {run.get('k', 'K')} nearest training "
                                  "buildings, compared after moving both to a common origin.")
        sort_by = st.selectbox("sort test buildings by",
                               ["distance_m", "symdiff_m3", "rel_symdiff", "chamfer_m"],
                               index=["distance_m", "symdiff_m3", "rel_symdiff",
                                      "chamfer_m"].index(CRITERION_COLUMN[criterion]),
                               format_func=METRIC_LABELS.get)
        descending = st.checkbox("descending", value=False)
        shade = st.checkbox("shade faces", value=True)
        if run:
            st.caption(f"K = {run.get('k')}, chamfer points {run.get('n_points')}, "
                       f"seed {run.get('seed')}, commit `{str(run.get('git_commit'))[:8]}`")

    col = palette()
    column = CRITERION_COLUMN[criterion]
    label = METRIC_LABELS[column]
    current = matches(pairs, split, criterion, sort_by, not descending)
    series = [(split, current[column].to_numpy(float), col["series"])]
    if split != "random" and "random" in splits:
        series.append(("random", matches(pairs, "random", criterion)[column].to_numpy(float),
                       col["muted"]))
    # Log only by default for distance: a zero volume / chamfer is an exact copy,
    # the most telling case, and a log axis would silently drop it.
    log_x = st.checkbox("log x-axis", value=criterion == "distance",
                        help="Values of exactly 0 are not drawn on a log axis.")
    st.plotly_chart(ecdf_figure(series, col, label, log_x), width="stretch")
    st.caption(f"{len(current)} sampled test buildings with a {criterion} match under "
               f"`{split}`" + ("; grey is the random split for reference."
                               if len(series) > 1 else "."))

    if current.empty:
        st.info("No matches for this split and criterion.")
        return
    left, right = st.columns([1, 3])
    with left:
        pick = st.selectbox(
            "test building", range(len(current)), key="pick",
            format_func=lambda i: f"{current.at[i, 'test_id']}  ·  "
                                  f"{METRIC_LABELS[sort_by]} {_fmt(current.iloc[i], sort_by)}")
    row = current.iloc[pick]
    with right:
        cells = st.columns(5)
        for cell, (name, value) in zip(cells, [
                ("distance", f"{_fmt(row, 'distance_m')} m"),
                ("volume sym. diff.", f"{_fmt(row, 'symdiff_m3')} m³"),
                ("relative sym. diff.", _fmt(row, "rel_symdiff")),
                ("chamfer", f"{_fmt(row, 'chamfer_m')} m"),
                ("faces (test / match)", f"{_fmt(row, 'test_faces')} / {_fmt(row, 'cand_faces')}")]):
            cell.metric(name, value)
        if not bool(row.get("vol_ok", True)):
            st.caption("Volume undefined: one of the two LOD2s is not a closed solid.")

    a, b = st.columns(2)
    with a:
        draw_object(data_root, row["test_tile"], row["test_id"], shade,
                    f"TEST  {row['test_id']}", "test")
    with b:
        draw_object(data_root, row["cand_tile"], row["cand_id"], shade,
                    f"TRAIN ({criterion}-nearest, rank {int(row['rank'])})  {row['cand_id']}",
                    "train")

    if cands_path.exists():
        with st.expander(f"all {run.get('k', 'K')} shortlisted training buildings"):
            cands = read_csv(str(cands_path))
            mine = cands[(cands["split"] == split) & (cands["test_id"] == row["test_id"])]
            st.dataframe(mine.drop(columns=["split", "test_id", "test_tile"]),
                         hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
