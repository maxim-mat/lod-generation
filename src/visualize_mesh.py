"""Exploratory: Plotly views of triangle meshes, shared by the mesh notebooks.

Throwaway visualization helpers -- exempt from TDD and reproducibility
requirements (project instructions). Lifted out of `notebooks/visualize_mesh.ipynb`
so `notebooks/test-mesh.ipynb` can use the same two figures.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def mesh_figure(verts, faces, title="", color="#2a78d6"):
    """Shaded triangle mesh with its wireframe on top."""
    v, tri = np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)
    fig = go.Figure()
    if len(tri):
        fig.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
            color=color, opacity=0.45, flatshading=True, hoverinfo="skip"))
        # a-b-c-a then a NaN break, so all triangles are one trace
        seg = np.full((len(tri) * 5, 3), np.nan)
        for k, col in enumerate([0, 1, 2, 0]):
            seg[k::5] = v[tri[:, col]]
        fig.add_trace(go.Scatter3d(
            x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode="lines",
            line=dict(color=color, width=2), showlegend=False, hoverinfo="skip"))
    fig.update_layout(title=title, scene=dict(aspectmode="data"))
    return fig


def side_by_side(figs, titles, height=680):
    """Put already-built 3D figures next to each other, sharing one camera."""
    out = make_subplots(rows=1, cols=len(figs), subplot_titles=titles,
                        specs=[[{"type": "scene"}] * len(figs)])
    for col, fig in enumerate(figs, start=1):
        for trace in fig.data:
            out.add_trace(trace, row=1, col=col)
        out.layout["scene" if col == 1 else f"scene{col}"].aspectmode = "data"
    out.update_layout(height=height, margin=dict(l=0, r=0, b=0, t=60),
                      legend=dict(itemsizing="constant"))
    return out
