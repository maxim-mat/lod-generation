"""Exploratory: interactive Plotly views of Levi graphs sampled from the dataset.

Throwaway visualization script -- exempt from TDD and reproducibility
requirements (project instructions). Samples a few buildings with 30-50 vertex
nodes and writes one self-contained HTML per building.

Usage:
    python -m src.visualize_levi --dataset-dir "data/The Hague" --lod 2 --n 3
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from src.dataset.dataset import (
    EDGE_VF, EDGE_VV, NODE_CLASS_NAMES, VERTEX, parse_cityjson_file_to_graphs,
)

# Validated categorical palette (dataviz reference palette, light mode);
# Off nodes never occur in raw (unpadded) graphs but keep a muted slot anyway.
NODE_COLORS = {0: "#2a78d6", 1: "#008300", 2: "#e87ba4", 3: "#eda100", 4: "#898781"}
EDGE_COLORS = {EDGE_VV: "#1baf7a", EDGE_VF: "#eb6834"}
EDGE_NAMES = {EDGE_VV: "vertex-vertex", EDGE_VF: "vertex-face"}


def sample_graphs(dataset_dir, lod, n, min_v, max_v):
    lod_dir = next(d for d in Path(dataset_dir).iterdir()
                   if d.is_dir() and d.name.lower() == f"lod{lod}")
    found = []
    for root, _, files in os.walk(lod_dir):
        for f in sorted(files):
            if not f.lower().endswith((".json", ".city.json")):
                continue
            for g in parse_cityjson_file_to_graphs(Path(root) / f).values():
                n_vertices = int((g["node_labels"] == VERTEX).sum())
                if min_v <= n_vertices <= max_v:
                    found.append(g)
                    if len(found) >= n:
                        return found
    return found


def display_positions(g):
    """Vertices at their coords; face nodes drawn at their members' centroid."""
    pos = g["x"].numpy().astype(float).copy()
    labels = g["node_labels"].numpy()
    ei, ea = g["edge_index"].numpy(), g["edge_attr"].numpy()
    for f in np.flatnonzero(labels != VERTEX):
        members = ei[1][(ei[0] == f) & (ea == EDGE_VF)]
        if len(members):
            pos[f] = pos[members].mean(axis=0)
    return pos


def levi_figure(g):
    labels = g["node_labels"].numpy()
    coords = g["x"].numpy()
    pos = display_positions(g)
    fig = go.Figure()

    # --- edges: one line trace per class, plus invisible midpoints for hover
    ei, ea = g["edge_index"].numpy(), g["edge_attr"].numpy()
    keep = ei[0] < ei[1]                      # each undirected edge once
    pairs, kinds = ei[:, keep].T, ea[keep]
    for cls in (EDGE_VV, EDGE_VF):
        xs, ys, zs, mid, hover = [], [], [], [], []
        for (u, v) in pairs[kinds == cls]:
            xs += [pos[u, 0], pos[v, 0], None]
            ys += [pos[u, 1], pos[v, 1], None]
            zs += [pos[u, 2], pos[v, 2], None]
            mid.append((pos[u] + pos[v]) / 2)
            hover.append(f"edge {u}–{v}<br>class: {EDGE_NAMES[cls]}")
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines", name=f"{EDGE_NAMES[cls]} edge",
            line=dict(color=EDGE_COLORS[cls], width=3),
            hoverinfo="skip",
        ))
        if mid:
            mid = np.array(mid)
            fig.add_trace(go.Scatter3d(
                x=mid[:, 0], y=mid[:, 1], z=mid[:, 2], mode="markers",
                marker=dict(size=6, color="rgba(0,0,0,0)"),
                hovertext=hover, hoverinfo="text",
                showlegend=False,
            ))

    # --- nodes: one marker trace per class present
    for cls in sorted(set(labels.tolist())):
        ids = np.flatnonzero(labels == cls)
        name = NODE_CLASS_NAMES[cls]
        if cls == VERTEX:
            hover = [
                f"node {i} · {name}<br>"
                f"x={coords[i, 0]:.2f} y={coords[i, 1]:.2f} z={coords[i, 2]:.2f}"
                for i in ids
            ]
            marker = dict(size=8, color=NODE_COLORS[cls])
        else:
            hover = [f"node {i} · face: {name}<br>(no coords; drawn at centroid)"
                     for i in ids]
            marker = dict(size=10, color=NODE_COLORS[cls], symbol="diamond")
        fig.add_trace(go.Scatter3d(
            x=pos[ids, 0], y=pos[ids, 1], z=pos[ids, 2], mode="markers",
            name=f"{name} node", marker=marker,
            hovertext=hover, hoverinfo="text",
        ))

    n_vertices = int((labels == VERTEX).sum())
    fig.update_layout(
        title=f"{g['id']} — {n_vertices} vertices, {len(labels) - n_vertices} faces",
        paper_bgcolor="#fcfcfb",
        font=dict(family='system-ui, "Segoe UI", sans-serif', color="#0b0b0b"),
        legend=dict(itemsizing="constant"),
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default="data/The Hague")
    ap.add_argument("--lod", type=int, default=2)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--min-v", type=int, default=30)
    ap.add_argument("--max-v", type=int, default=50)
    ap.add_argument("--out-dir", default="outputs/levi_viz")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graphs = sample_graphs(args.dataset_dir, args.lod, args.n, args.min_v, args.max_v)
    if not graphs:
        raise SystemExit("No graphs found in the requested vertex range.")
    for g in graphs:
        safe_id = re.sub(r"[^\w.-]", "_", g["id"])
        out = out_dir / f"{safe_id}.html"
        levi_figure(g).write_html(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
