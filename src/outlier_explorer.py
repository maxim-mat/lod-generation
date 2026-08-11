#!/usr/bin/env python3
"""Streamlit browser for the CityObject analysis outliers.csv.

Pick a check, pick a flagged building, see it. Beats spamming notebook cells.

Usage:
    streamlit run src/outlier_explorer.py
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:                    # `streamlit run` puts src/ on the path, not the root
    sys.path.insert(0, str(REPO))

import pandas as pd                                                  # noqa: E402
import streamlit as st                                               # noqa: E402

from src.analysis.cityobject_analysis import (                       # noqa: E402
    iter_faces, object_defects, shell_edge_defects, solid_faces)
from src.visualize_cityjson import cityjson_figure, load_cityjson    # noqa: E402

DEFAULT_CSV = REPO / "outputs" / "cityobject_analysis" / "outliers.csv"
# The LOD1/LOD1_synth/LOD2 folders live under a variant dir, not the city root.
DEFAULT_DATA = REPO / "data" / "The Hague" / "full"

st.set_page_config(page_title="CityObject outliers", layout="wide")


@st.cache_data
def read_outliers(path):
    df = pd.read_csv(path)
    df["oid"] = df["object_id"].str.split(":", n=1).str[-1]
    # Written by the analysis as metric="n_vertices"; older csvs lack it.
    df["n_vertices"] = pd.to_numeric(df.get("value"), errors="coerce")
    return df


@st.cache_data(max_entries=6)                    # tiles are 5-7 MB, so keep a few
def read_tile(path):
    return load_cityjson(path)


@st.cache_resource
def _to_wgs84():
    from pyproj import Transformer       # both sources are EPSG:7415 (RD New + NAP)
    return Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)


def real_world_reference(oid, points):
    """``(bag_pand_id, lat, lon)`` so a flagged object can be found on a map.

    Both id styles carry the 16-digit BAG pand identifier: Source A's
    ``bag_0518100000223148`` and 3DBAG's ``NL.IMBAG.Pand.1783100000041423-0``.
    """
    m = re.search(r"\d{16}", oid)
    lon, lat = _to_wgs84().transform(points[:, 0].mean(), points[:, 1].mean())
    return (m.group() if m else None), lat, lon


st.title("CityObject outlier explorer")

with st.sidebar:
    csv_path = Path(st.text_input("outliers.csv", str(DEFAULT_CSV)))
    data_root = Path(st.text_input("dataset root", str(DEFAULT_DATA)))
    lod = st.radio("geometry to show", ["LOD2", "LOD1_synth", "LOD1", "raw"], horizontal=True)
    show_normals = st.checkbox("draw face normals", value=True)
    shade = st.checkbox("shade faces", value=False)
    highlight = st.checkbox("highlight closure defects", value=True)

if not csv_path.exists():
    st.error(f"Not found: {csv_path}. Run `python -m src.analysis.cityobject_analysis`.")
    st.stop()

df = read_outliers(csv_path)
left, right = st.columns([1, 2])

with left:
    counts = df["check"].value_counts()
    check = st.selectbox("check", counts.index,
                         format_func=lambda c: f"{c}  ({counts[c]})")
    subset = df[df["check"] == check]

    source = st.selectbox("source", ["(all)"] + sorted(subset["source"].unique()))
    if source != "(all)":
        subset = subset[subset["source"] == source]

    has_counts = subset["n_vertices"].notna().any()
    if has_counts:
        # Smallest first: big buildings are slow to draw and hard to read, and
        # the interesting defect is usually just as visible on a small one.
        biggest = int(subset["n_vertices"].max())
        cap = st.slider("max LOD2 vertices", 8, biggest, biggest) if biggest > 8 else biggest
        subset = subset[subset["n_vertices"] <= cap].sort_values("n_vertices")
    else:
        st.info("No vertex counts in this csv — re-run the analysis to sort by size.")

    st.caption(f"{len(subset)} flagged objects (csv caps each check at 500)")
    sizes = dict(zip(subset["object_id"], subset["n_vertices"]))

    def label_of(s):
        n = sizes.get(s)
        suffix = f"  —  {int(n)} verts" if pd.notna(n) else ""
        return f"{s.split(':', 1)[-1][:34]}{suffix}"

    labels = subset["object_id"].tolist()
    if not labels:
        st.warning("Nothing left after filtering; raise the vertex cap.")
        st.stop()
    picked = st.selectbox("building", labels, format_func=label_of)

    row = subset[subset["object_id"] == picked].iloc[0]
    oid, fname, src = row["oid"], row["file"], row["source"]

tile = data_root / lod / src / fname
cj, verts, obj = None, None, None
if tile.exists():
    cj, verts = read_tile(str(tile))
    obj = (cj.get("CityObjects") or {}).get(oid)

with left:
    # Recomputed, never read from the csv: outliers.csv caps every check at 500
    # rows, so a busy check silently drops most of the objects that trip it.
    st.markdown("**checks this object trips** _(recomputed)_")
    if obj is None:
        st.caption(f"unavailable — no {lod} geometry loaded")
    else:
        st.write(", ".join(object_defects(obj, verts)) or "none")

    st.markdown("**file**")
    st.code(f"{src}/{fname}", language=None)

    if obj is not None:
        ids = sorted({i for g in obj.get("geometry") or []
                      for ring, _ in iter_faces(g) for i in ring})
        if ids:
            pand, lat, lon = real_world_reference(oid, verts[ids])
            st.markdown("**real-world reference**")
            if pand:
                st.code(pand, language=None)
            st.markdown(
                f"[map](https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f})"
                f" · [street view](https://www.google.com/maps/@?api=1"
                f"&map_action=pano&viewpoint={lat:.6f},{lon:.6f})")

with right:
    if not tile.exists():
        st.warning(f"{lod} has no {src}/{fname} (expected for LOD1 where the "
                   f"source ships no lod-1 geometry).")
        st.stop()
    if obj is None:
        st.warning(f"{oid} is not present in {lod} — the LOD folders do not "
                   f"always share an object set.")
        st.stop()

    edges = {}
    solid = next((g for g in obj.get("geometry") or []
                  if g.get("type") == "Solid"), None)
    if highlight and solid:
        unpaired, reused = shell_edge_defects(solid_faces(solid))
        edges = {"open boundary": unpaired, "inconsistent winding": reused}

    st.plotly_chart(
        cityjson_figure([obj], verts, title=f"{oid}  [{lod}]",
                        show_normals=show_normals, shade=shade,
                        highlight_edges=edges),
        width="stretch")
    st.caption(f"{len(obj.get('geometry') or [])} geometry / lod tags: "
               f"{[g.get('lod') for g in obj.get('geometry') or []]}")
