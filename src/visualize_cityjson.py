#!/usr/bin/env python3
"""Plotly views of raw CityJSON geometry.

Lifted out of ``notebooks/visualize_cityjson.ipynb`` so the outlier explorer
and any other caller share one implementation instead of copying cells around.

Faces are coloured by semantic surface type, and normals can be drawn, because
the checks these views exist to explain are mostly orientation ones.
"""
import json

import numpy as np
import plotly.graph_objects as go

from src.filter_cityjson import world_vertices

# Validated categorical palette, same family as src/visualize_levi.py.
SEMANTIC_COLORS = {
    "GroundSurface": "#2a78d6",
    "RoofSurface": "#eda100",
    "WallSurface": "#898781",
    "OuterFloorSurface": "#008300",
    "OuterCeilingSurface": "#e87ba4",
    None: "#d62728",                 # unlabelled -- stands out on purpose
}
DEFAULT_COLOR = "#d62728"

# Closure defects are invisible in a wireframe -- every undirected edge is still
# drawn -- so they get their own loud overlay.
DEFECT_COLORS = {"open boundary": "#ff2d55", "inconsistent winding": "#b14aed"}


def load_cityjson(path):
    """``(cityjson_dict, world_vertices)`` for a file on disk."""
    with open(path, "r", encoding="utf-8") as fh:
        cj = json.load(fh)
    return cj, world_vertices(cj)


def _faces_with_types(geom):
    """``(outer_ring, semantic_type)`` for each surface of one geometry."""
    boundaries = geom.get("boundaries", []) or []
    gtype = geom.get("type")
    sem = geom.get("semantics") or {}
    surfaces, values = sem.get("surfaces") or [], sem.get("values") or []

    if gtype == "Solid":
        faces = [f for shell in boundaries for f in shell]
        vals = [v for shell_vals in values for v in shell_vals] if values else []
    elif gtype in ("MultiSurface", "CompositeSurface"):
        faces, vals = boundaries, values
    else:
        return []

    out = []
    for i, face in enumerate(faces):
        if not face or len(face[0]) < 3:
            continue
        stype = None
        if i < len(vals) and vals[i] is not None and vals[i] < len(surfaces):
            s = surfaces[vals[i]]
            stype = s.get("type") if s else None
        out.append((face[0], stype))
    return out


def _normal(points):
    n = np.cross(points, np.roll(points, -1, axis=0)).sum(axis=0)
    mag = np.linalg.norm(n)
    return None if mag < 1e-12 else n / mag


def cityjson_figure(city_objects, vertices, title="", show_normals=False,
                    shade=False, highlight_edges=None):
    """Wireframe (optionally shaded) view of CityObjects, coloured by semantics.

    ``city_objects`` is an iterable of CityObject dicts; ``vertices`` must
    already be in world coordinates (see :func:`load_cityjson`).
    ``highlight_edges`` maps a label to ``(a, b)`` vertex-index pairs drawn as a
    thick overlay -- used to show where a solid fails to close.
    """
    # Batched by semantic type, with NaN breaks between rings: one trace per
    # face would mean 5000+ traces on the larger outliers and a browser that
    # cannot rotate them.
    lines, normals, mesh = {}, {}, {}

    for obj in city_objects:
        for geom in obj.get("geometry", []) or []:
            for ring, stype in _faces_with_types(geom):
                pts = vertices[ring]
                label = stype or "unlabelled"
                gap = np.full((1, 3), np.nan)
                lines.setdefault(label, []).append(np.vstack([pts, pts[:1], gap]))

                if shade and len(ring) >= 3:
                    tris, base = mesh.setdefault(label, ([], []))[0], None
                    verts_so_far = mesh[label][1]
                    base = sum(len(p) for p in verts_so_far)
                    verts_so_far.append(pts)
                    tris.extend((base, base + k, base + k + 1)
                                for k in range(1, len(ring) - 1))

                if show_normals:
                    n = _normal(pts)
                    if n is None:
                        continue
                    centre = pts.mean(axis=0)
                    span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
                    tip = centre + n * max(span * 0.3, 0.5)
                    normals.setdefault(label, []).append(
                        np.vstack([centre, tip, np.full((1, 3), np.nan)]))

    fig = go.Figure()
    for label, chunks in lines.items():
        pts = np.vstack(chunks)
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
            line=dict(color=SEMANTIC_COLORS.get(label, DEFAULT_COLOR), width=3),
            name=label, legendgroup=label, hovertext=label, hoverinfo="text"))

    for label, (tris, chunks) in mesh.items():
        pts = np.vstack(chunks)
        tris = np.asarray(tris)
        fig.add_trace(go.Mesh3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
            color=SEMANTIC_COLORS.get(label, DEFAULT_COLOR), opacity=0.35,
            name=label, legendgroup=label, showlegend=False, hoverinfo="skip"))

    for label, chunks in normals.items():
        pts = np.vstack(chunks)
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
            line=dict(color=SEMANTIC_COLORS.get(label, DEFAULT_COLOR), width=5),
            name=f"{label} normals", legendgroup=label, showlegend=False,
            hoverinfo="skip"))

    for label, pairs in (highlight_edges or {}).items():
        if not pairs:
            continue
        gap = np.full((1, 3), np.nan)
        seg = np.vstack([np.vstack([vertices[a], vertices[b], gap]) for a, b in pairs])
        fig.add_trace(go.Scatter3d(
            x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode="lines",
            line=dict(color=DEFECT_COLORS.get(label, DEFAULT_COLOR), width=9),
            name=f"{label} ({len(pairs)})", hovertext=label, hoverinfo="text"))

    fig.update_layout(title=title, scene=dict(aspectmode="data"),
                      margin=dict(l=0, r=0, b=0, t=40), height=680,
                      legend=dict(itemsizing="constant"))
    return fig
