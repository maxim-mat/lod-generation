"""Exploratory: Plotly views of triangle meshes, shared by the mesh notebooks.

Throwaway visualization helpers -- exempt from TDD and reproducibility
requirements (project instructions). Lifted out of `notebooks/visualize_mesh.ipynb`
so `notebooks/test-mesh.ipynb` can use the same two figures.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.filter_cityjson import world_vertices
from src.post_process.post_process import mesh_to_cityjson
from src.visualize_cityjson import cityjson_figure


def mesh_figure(verts, faces, title="", color="#2a78d6"):
    """Shaded triangle mesh with its wireframe on top.

    Hovering any triangle names it. Plotly resolves a 3D hover to the nearest
    vertex, so the surface is built from *unshared* vertices -- three per
    triangle -- and each carries its own triangle's text. That triples the
    vertex array (14k on the largest object here, which plotly handles fine)
    and is invisible under ``flatshading``, but it is the only way a Mesh3d
    can answer "which face is this".
    """
    v, tri = np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)
    fig = go.Figure()
    if len(tri):
        corner = v[tri.reshape(-1)]                  # [3F, 3], one set per tri
        idx = np.arange(len(tri) * 3).reshape(-1, 3)
        tri_text = [f"tri {t} · v {a},{b},{c}"
                    for t, (a, b, c) in enumerate(tri) for _ in range(3)]
        fig.add_trace(go.Mesh3d(
            x=corner[:, 0], y=corner[:, 1], z=corner[:, 2],
            i=idx[:, 0], j=idx[:, 1], k=idx[:, 2],
            color=color, opacity=0.45, flatshading=True,
            hovertext=tri_text, hoverinfo="text"))
        # a-b-c-a then a NaN break, so all triangles are one trace
        seg = np.full((len(tri) * 5, 3), np.nan)
        for k, col in enumerate([0, 1, 2, 0]):
            seg[k::5] = v[tri[:, col]]
        seg_text = [""] * (len(tri) * 5)
        for t, (a, b, c) in enumerate(tri):
            for k, vid in enumerate((a, b, c, a)):
                seg_text[t * 5 + k] = f"tri {t} · vertex {vid}"
        fig.add_trace(go.Scatter3d(
            x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode="lines",
            line=dict(color=color, width=2), showlegend=False,
            hovertext=seg_text, hoverinfo="text"))
    fig.update_layout(title=title, scene=dict(aspectmode="data"))
    return fig


def cityjson_figure_from_mesh(verts, faces):
    """``(figure, n_surfaces)`` for the mesh put back through the CityJSON writer.

    `mesh_to_cityjson` merges coplanar triangles into polygons and labels them
    Ground/Roof/Wall, so this is the same geometry as `mesh_figure` shows but as
    the pipeline would actually store it -- which is where a merge that fused or
    shattered the wrong planes becomes visible. Returns an empty figure and 0
    when the writer rejects the mesh, matching its ``{}``-on-failure contract.
    """
    cj = mesh_to_cityjson(verts, faces)
    if not cj:
        return go.Figure(), 0
    obj = next(iter(cj["CityObjects"].values()))
    return (cityjson_figure([obj], world_vertices(cj), shade=True),
            len(obj["geometry"][0]["boundaries"][0]))


def side_by_side(figs, titles, height=680):
    """Put already-built 3D figures next to each other, sharing one camera.

    ``figs`` is a list of figures, or a list of rows of figures for a grid;
    ``titles`` is flat either way, in row-major order.
    """
    rows = list(figs) if isinstance(figs[0], (list, tuple)) else [figs]
    ncols = max(len(row) for row in rows)

    out = make_subplots(rows=len(rows), cols=ncols, subplot_titles=titles,
                        specs=[[{"type": "scene"}] * ncols for _ in rows])
    # make_subplots numbers scenes row-major, so one running counter names them.
    n = 0
    for r, row in enumerate(rows, start=1):
        for c, fig in enumerate(row, start=1):
            n += 1
            for trace in fig.data:
                out.add_trace(trace, row=r, col=c)
            out.layout["scene" if n == 1 else f"scene{n}"].aspectmode = "data"

    out.update_layout(height=height * len(rows), margin=dict(l=0, r=0, b=0, t=60),
                      legend=dict(itemsizing="constant"))
    return out
