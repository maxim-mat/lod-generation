#!/usr/bin/env python3
"""Streamlit explorer: one building at a time, its 3D LOD1/LOD2 and its orthophoto.

Three views of the selected building:
  * 3D -- LOD2 and LOD1 meshes (the model's target and condition);
  * crop -- the orthophoto exactly as `MeshDataset(ortho_dir=...)` returns it, in
    the normalized frame of the mesh tokens (+x right, +y up), with the LOD2
    footprint and the unit box drawn on top, so alignment can be checked by eye;
  * map -- the footprint outlined on a web basemap, with the same crop as a
    toggleable, georeferenced overlay.

The crop uses the frame (`lod_in`, margins) recorded in the mosaics'
`ortho_manifest.json`, so it is the dataset's crop, not an approximation of it.

Loading is lazy: nothing is read at start-up beyond the list of file names. A
CityJSON file is parsed only when one of its buildings is selected, and only the
selected building's window is read from its mosaic.

Usage:
    streamlit run src/ortho_explorer.py
    streamlit run src/ortho_explorer.py -- --subset "data/The Hague/mini_cleaner"
"""
import argparse
import base64
import io
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:                    # `streamlit run` puts src/ on the path, not the root
    sys.path.insert(0, str(REPO))

import matplotlib.image                                              # noqa: E402
import numpy as np                                                   # noqa: E402
import plotly.graph_objects as go                                    # noqa: E402
import pydeck as pdk                                                 # noqa: E402
import streamlit as st                                               # noqa: E402
from pyproj import Transformer                                       # noqa: E402

from src.dataset.mesh_dataset import (normalize_to_unit_box,         # noqa: E402
                                      parse_cityjson_file_to_meshes)
from src.dataset.ortho import OrthoReader, crop_bounds               # noqa: E402
from src.visualize_cityjson import _faces_with_types, cityjson_figure, load_cityjson  # noqa: E402

DEFAULT_SUBSET = REPO / "data" / "The Hague" / "mini_cleaner"
FOOTPRINT = [235, 104, 52]                       # the rule colour of the other explorers
_TO_WGS84 = Transformer.from_crs(28992, 4326, always_xy=True)


# ----------------------------------------------------------------------
# Pure helpers (tested in tests/test_ortho_explorer.py)
# ----------------------------------------------------------------------

def footprint_rings(obj, verts):
    """Exterior rings ``[(N, 2)]`` in RD metres of an object's ground surfaces.

    Uses the ``GroundSurface`` semantics; without them, the faces lying entirely
    at the object's lowest z.
    """
    verts = np.asarray(verts, float)
    faces = [f for geom in obj.get("geometry", []) for f in _faces_with_types(geom)]
    ground = [rings[0] for rings, stype in faces if stype == "GroundSurface"]
    if not ground and faces:
        zmin = min(verts[i, 2] for rings, _ in faces for i in rings[0])
        ground = [rings[0] for rings, _ in faces
                  if all(abs(verts[i, 2] - zmin) < 1e-3 for i in rings[0])]
    return [verts[ring, :2] for ring in ground]


def to_uv(xy, center, scale):
    """RD x/y to the dataset's normalized frame."""
    return (np.asarray(xy, float) - np.asarray(center)[:2]) / np.asarray(scale)[:2]


def to_lonlat(xy):
    """RD New x/y ``(N, 2)`` to WGS84 lon/lat ``(N, 2)``.

    pyproj's default RD transformation, not the official RDNAPTRANS grid: about
    a metre off in absolute terms, but footprint and overlay go through the same
    transform, so they stay aligned with each other.
    """
    xy = np.asarray(xy, float)
    lon, lat = _TO_WGS84.transform(xy[:, 0], xy[:, 1])
    return np.column_stack([lon, lat])


def window_corners(bounds):
    """``[bottom-left, top-left, top-right, bottom-right]`` lon/lat of an RD box.

    Four corners rather than a lon/lat bbox: the RD grid is slightly rotated
    against WGS84, and deck.gl's BitmapLayer takes the corners as given.
    """
    xmin, ymin, xmax, ymax = bounds
    return to_lonlat([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]]).tolist()


def north_up(crop):
    """Mesh-order crop ``[3, S, S]`` (row index along +y) to a north-up ``[S, S, 3]`` image."""
    return np.ascontiguousarray(crop.numpy().transpose(1, 2, 0)[::-1])


# ----------------------------------------------------------------------
# Lazy, cached loaders
# ----------------------------------------------------------------------

@st.cache_data
def list_files(root):
    """Relative paths of the subset's LOD2 files. Names only -- nothing is parsed."""
    lod2 = Path(root) / "LOD2"
    return sorted(p.relative_to(lod2).as_posix() for p in lod2.rglob("*.json"))


@st.cache_data
def read_manifest(ortho_root):
    path = Path(ortho_root) / "ortho_manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


@st.cache_data(max_entries=8)
def read_file(path):
    """One CityJSON file and its world vertices, parsed on first use."""
    return load_cityjson(path)


@st.cache_data(max_entries=8)
def read_meshes(path):
    """Triangulated meshes of one file -- what the dataset normalises."""
    return parse_cityjson_file_to_meshes(path)


@st.cache_resource
def ortho_reader(root, size, beta):
    return OrthoReader(root, size=size, beta=beta)


def png_data_url(image):
    buf = io.BytesIO()
    matplotlib.image.imsave(buf, image, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------------
# Views
# ----------------------------------------------------------------------

def crop_figure(crop, beta, rings_uv):
    """The model-input crop on normalized axes, footprint and unit box on top."""
    size = crop.shape[-1]
    step = (1 + 2 * beta) / size
    lo = -0.5 - beta + step / 2
    fig = go.Figure(go.Image(z=crop.numpy().transpose(1, 2, 0), x0=lo, dx=step, y0=lo, dy=step,
                             hoverinfo="skip"))
    box = [-0.5, 0.5, 0.5, -0.5, -0.5]
    fig.add_scatter(x=box, y=box[1:] + box[:1], mode="lines", name="unit box",
                    line=dict(color="white", dash="dash", width=1))
    for k, ring in enumerate(rings_uv):
        closed = np.vstack([ring, ring[:1]])
        fig.add_scatter(x=closed[:, 0], y=closed[:, 1], mode="lines", name="LOD2 footprint",
                        showlegend=k == 0, line=dict(color="rgb(%d,%d,%d)" % tuple(FOOTPRINT),
                                                     width=2))
    fig.update_xaxes(title_text="u (+x, east)", range=[-0.5 - beta, 0.5 + beta],
                     zeroline=False, showgrid=False)
    # Image traces default to a reversed y axis; the crop's row index runs +y.
    fig.update_yaxes(title_text="v (+y, north)", range=[-0.5 - beta, 0.5 + beta],
                     autorange=False, scaleanchor="x", zeroline=False, showgrid=False)
    fig.update_layout(height=460, margin=dict(l=8, r=8, t=8, b=8),
                      legend=dict(orientation="h", y=-0.15))
    return fig


def map_deck(rings_xy, bounds, image, overlay, opacity):
    """Footprint outline on a basemap, the crop as an optional georeferenced overlay."""
    layers = []
    if overlay:
        # Quoted on purpose: pydeck sends any other string to deck.gl as a JS
        # expression, and "data:image/png;..." then fails to parse in the browser.
        layers.append(pdk.Layer("BitmapLayer", image=f'"{png_data_url(image)}"',
                                bounds=window_corners(bounds), opacity=opacity))
    layers.append(pdk.Layer(
        "PolygonLayer", data=[{"polygon": to_lonlat(r).tolist()} for r in rings_xy],
        get_polygon="polygon", filled=False, stroked=True, get_line_color=FOOTPRINT,
        line_width_min_pixels=2))
    lon, lat = to_lonlat([[(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2]])[0]
    span = bounds[2] - bounds[0]
    zoom = float(np.clip(np.log2(40075016 * np.cos(np.radians(lat)) / (span * 2.2)) - 8, 14, 21))
    return pdk.Deck(layers=layers, map_provider="carto", map_style=pdk.map_styles.CARTO_LIGHT,
                    initial_view_state=pdk.ViewState(longitude=lon, latitude=lat, zoom=zoom),
                    tooltip=False)


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------

def _select(file, oid):
    st.session_state["file"], st.session_state["oid"] = file, oid


def _random_building(root, files):
    file = random.choice(files)
    ids = sorted(read_file(str(Path(root) / "LOD2" / file))[0]["CityObjects"])
    _select(file, random.choice(ids))


def _step(ids, delta):
    i = ids.index(st.session_state["oid"]) if st.session_state.get("oid") in ids else 0
    st.session_state["oid"] = ids[(i + delta) % len(ids)]


def main():
    st.set_page_config(page_title="Building + orthophoto", layout="wide")
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    args, _ = parser.parse_known_args()

    root = Path(st.sidebar.text_input("Subset", str(args.subset)))
    ortho_root = root / "ortho"
    files = list_files(str(root))
    manifest = read_manifest(str(ortho_root))
    if not files:
        st.error(f"No LOD2 CityJSON under {root}.")
        return
    if manifest is None:
        st.error(f"No ortho_manifest.json under {ortho_root}; run `python -m src.ortho_parse`.")
        return
    built = manifest["args"]

    if "file" not in st.session_state:
        _random_building(root, files)
    st.sidebar.button("Random building", on_click=_random_building, args=(root, files),
                      width="stretch")
    file = st.sidebar.selectbox("File", files, key="file")
    cj2, verts2 = read_file(str(root / "LOD2" / file))
    ids = sorted(cj2["CityObjects"])
    if st.session_state.get("oid") not in ids:
        st.session_state["oid"] = ids[0]
    left, right = st.sidebar.columns(2)
    left.button("Previous", on_click=_step, args=(ids, -1), width="stretch")
    right.button("Next", on_click=_step, args=(ids, 1), width="stretch")
    oid = st.sidebar.selectbox(f"Building ({len(ids)} in file)", ids, key="oid")

    st.sidebar.divider()
    beta = st.sidebar.slider("Crop margin (normalized)", 0.0, float(built["beta_max"]),
                             min(0.1, float(built["beta_max"])), 0.01)
    size = st.sidebar.select_slider("Crop size (px)", [128, 256, 512, 768, 1024], 512,
                                    help="The training default is 256.")
    shade = st.sidebar.toggle("Shaded 3D", True)
    overlay = st.sidebar.toggle("Orthophoto overlay on map", True)
    opacity = st.sidebar.slider("Overlay opacity", 0.0, 1.0, 0.85, 0.05, disabled=not overlay)

    lod1_path = root / built["lod_in"] / file
    meshes1 = read_meshes(str(lod1_path)) if lod1_path.is_file() else {}
    if oid not in meshes1:
        st.warning(f"`{oid}` has no {built['lod_in']} mesh, so it has no frame and no crop.")
        return
    _, center, scale = normalize_to_unit_box(meshes1[oid][0], margin_lo=built["margin_lo"],
                                             margin_hi=built["margin_hi"])
    bounds = crop_bounds(center, scale, beta)
    mosaic = file.split(".")[0] + ".tif"
    crop = ortho_reader(str(ortho_root), size, beta).crop(mosaic, center, scale)
    obj2 = cj2["CityObjects"][oid]
    rings = footprint_rings(obj2, verts2)

    datum = (obj2.get("attributes") or {}).get("b3_pw_datum")
    st.subheader(oid)
    st.caption(f"`{file}` · window {bounds[2] - bounds[0]:.1f} m square · "
               f"{(bounds[2] - bounds[0]) / size * 100:.1f} cm/px at {size} px · "
               f"point cloud {datum or 'unknown'} vs imagery {built['year']}")
    if datum is not None and str(datum) != str(built["year"]):
        st.warning(f"LOD2 comes from a {datum} point cloud; the photo is {built['year']}. "
                   "The building may have changed in between.")

    col3d, colcrop = st.columns([3, 2])
    with col3d:
        tab2, tab1 = st.tabs(["LOD2 (target)", f"{built['lod_in']} (condition)"])
        with tab2:
            st.plotly_chart(cityjson_figure([obj2], verts2, shade=shade), width="stretch",
                            key=f"lod2-{file}-{oid}")
        with tab1:
            cj1, verts1 = read_file(str(lod1_path))
            obj1 = cj1["CityObjects"].get(oid)
            if obj1 is None:
                st.info(f"No {built['lod_in']} object.")
            else:
                st.plotly_chart(cityjson_figure([obj1], verts1, shade=shade), width="stretch",
                                key=f"lod1-{file}-{oid}")
    with colcrop:
        st.markdown("**Model input crop** (normalized frame)")
        st.plotly_chart(crop_figure(crop, beta, [to_uv(r, center, scale) for r in rings]),
                        width="stretch", key=f"crop-{file}-{oid}-{beta}-{size}")
        if ortho_reader(str(ortho_root), size, beta).n_padded:
            st.caption("Some crops this session reached past their mosaic (zero-filled).")

    st.markdown("**Map**: footprint outline, orthophoto overlay toggle in the sidebar")
    st.pydeck_chart(map_deck(rings, bounds, north_up(crop), overlay, opacity), height=520)
    st.caption("Imagery: Beeldmateriaal Nederland (CC-BY 4.0). Basemap: CARTO, OpenStreetMap "
               "contributors. RD to WGS84 via pyproj (about 1 m absolute); footprint and overlay "
               "share that transform.")


if __name__ == "__main__":
    main()
