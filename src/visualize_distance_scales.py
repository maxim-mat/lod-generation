"""Exploratory EDA (throwaway): train-split pairwise-distance distributions.

Motivation in docs/superpowers/specs/2026-07-19-distance-featurization-scale-design.md.
The network feeds raw pairwise distance through a single Linear (`lin_dist1`),
which is weak on a wide, multi-scale distance distribution. This plots that
distribution, split by node-pair type and by edge class, so an RBF/Fourier
range (r_max, centre spacing) can be set from data rather than a molecular
default -- or so we can decide the issue is scale, not featurization.

Distances are in the MODEL's coordinate space (raw metres / coord_scale), i.e.
exactly what `lin_dist1` sees.

    python visualize_distance_scales.py --config configs/train.yaml

Exploratory: no tests, no reproducibility contract -- it reads data and draws.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from src.utils.initialization import load_config
from src.utils.setup_utils import create_datamodule

# Levi node classes: 0=vertex, 1..3=faces (ground/roof/wall), 4=off/padding.
# Edge classes: 0=off, 1=vertex-vertex, 2=vertex-face.
VERTEX = 0


def _collect(train_dataset, coord_scale):
    """Upper-triangular pairwise distances (model units) bucketed by pair type."""
    buckets = {k: [] for k in ("v-v", "v-f", "f-f", "v-v edge", "v-f edge")}
    for item in train_dataset:
        if isinstance(item, tuple):          # multi-LOD yields a tuple; use LOD 0
            item = item[0]
        mask = item["node_mask"].bool()
        pos = item["x"][mask].double() / coord_scale         # [n, 3], model units
        cls = item["node_categories"][mask].argmax(dim=-1)   # [n]
        edge = item["y"].squeeze(-1)[mask][:, mask].long()   # [n, n] edge classes
        n = pos.shape[0]
        if n < 2:
            continue

        d = torch.cdist(pos, pos)                            # [n, n]
        iu, ju = torch.triu_indices(n, n, offset=1)          # unique pairs, no diag
        dij = d[iu, ju]
        is_v_i, is_v_j = cls[iu] == VERTEX, cls[ju] == VERTEX
        n_vertices = is_v_i.int() + is_v_j.int()             # 2=v-v, 1=v-f, 0=f-f
        eij = edge[iu, ju]

        buckets["v-v"].append(dij[n_vertices == 2])
        buckets["v-f"].append(dij[n_vertices == 1])
        buckets["f-f"].append(dij[n_vertices == 0])
        buckets["v-v edge"].append(dij[eij == 1])
        buckets["v-f edge"].append(dij[eij == 2])

    return {k: torch.cat(v).numpy() if v else np.array([]) for k, v in buckets.items()}


def _summary(name, d):
    if d.size == 0:
        return f"  {name:<10} (empty)"
    q = np.percentile(d, [0, 50, 95, 100])
    return (f"  {name:<10} n={d.size:>8}  min={q[0]:.3f}  median={q[1]:.3f}  "
            f"p95={q[2]:.3f}  max={q[3]:.3f}  max/median={q[3]/max(q[1],1e-9):.1f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("../configs/train.yaml"))
    ap.add_argument("--out", type=Path, default=Path("outputs/distance_scales.html"))
    args = ap.parse_args()

    cfg = load_config(args.config, [])
    dm = create_datamodule(cfg)
    dm.setup()
    coord_scale = cfg.data.coord_scale or dm.compute_coord_scale()
    print(f"coord_scale = {coord_scale:.4f} m/unit  (distances below are in model units)")

    buckets = _collect(dm.train_dataset, coord_scale)
    for k in ("v-v", "v-f", "f-f", "v-v edge", "v-f edge"):
        print(_summary(k, buckets[k]))

    # Panel A: all pairs by node-type. Panel B: edges only, by edge class.
    # Probability-density norm so shapes compare despite very different counts.
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "All pairs by node type", "Edge pairs by edge class"))
    for name in ("v-v", "v-f", "f-f"):
        d = buckets[name]
        if d.size:
            fig.add_trace(go.Histogram(x=d, name=name, histnorm="probability density",
                                       opacity=0.6, nbinsx=80), row=1, col=1)
    for name in ("v-v edge", "v-f edge"):
        d = buckets[name]
        if d.size:
            fig.add_trace(go.Histogram(x=d, name=name, histnorm="probability density",
                                       opacity=0.6, nbinsx=80), row=1, col=2)
    fig.update_layout(barmode="overlay", title_text=(
        "Train-split pairwise distances (model coordinate units) — "
        "sets the RBF/Fourier range r_max"))
    fig.update_xaxes(title_text="distance (model units)")
    fig.update_yaxes(title_text="density")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
