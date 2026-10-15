"""Figures from a `src.analysis.split_sweep` output directory.

Reads only the saved CSVs (`split_sweep.csv`, `correlogram.csv`), so a figure
can always be traced to the numbers behind it. Writes four PNGs next to them:
  * correlogram.png -- pair correlation vs distance per target: sets the buffer;
  * separation.png  -- val/test-to-nearest-train distance vs block size;
  * similarity.png  -- how alike that nearest training building still is;
  * cost.png        -- training data lost to the buffer, and val/test blocks.
Lines are the mean over seeds; bands span min..max.

Usage:
    python -m src.analysis.plot_split_sweep outputs/split_sweep
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Reference palette (dataviz skill), light mode. Targets are identities ->
# categorical slots in fixed order; buffers are ordered -> one blue ramp.
TARGET_COLORS = {"log_footprint": "#2a78d6", "log_volume": "#eb6834", "log_faces": "#1baf7a",
                 "roof_ratio": "#eda100", "faces_resid": "#e87ba4"}
TARGET_LABELS = {"log_footprint": "log footprint (LOD1)", "log_volume": "log volume (LOD2)",
                 "log_faces": "log face count (LOD2)",
                 "roof_ratio": "roof ratio: log LOD2/LOD1 volume",
                 "faces_resid": "face count beyond LOD1 (residual)"}
BUFFER_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                         "#e1e0d9", "#c3c2b7")


def _style(ax, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)


def _figure(ncols=1, width=7.0):
    fig, axes = plt.subplots(1, ncols, figsize=(width, 4.2), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes[0]


def _buffer_lines(ax, rows, column):
    """Grid rows: one line per buffer (mean over seeds, min..max band)."""
    grid = rows[rows["scheme"] == "grid"]
    buffers = sorted(grid["buffer_m"].unique())
    for k, buf in enumerate(buffers):
        g = grid[grid["buffer_m"] == buf].groupby("block_m")[column].agg(["mean", "min", "max"])
        color = BUFFER_RAMP[min(k, len(BUFFER_RAMP) - 1)]
        ax.fill_between(g.index, g["min"], g["max"], color=color, alpha=0.15, linewidth=0)
        ax.plot(g.index, g["mean"], color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"grid, buffer {buf:g} m")
    ax.set_xscale("log")
    sizes = sorted(grid["block_m"].unique())
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s:g}" for s in sizes])


def _reference(ax, rows, column, scheme, buffer_m=0):
    ref = rows[(rows["scheme"] == scheme) & (rows["buffer_m"] == buffer_m)][column].mean()
    if np.isfinite(ref):
        ax.axhline(ref, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.annotate(f"{scheme} split", (1.0, ref), xycoords=("axes fraction", "data"),
                    xytext=(-4, 3), textcoords="offset points", ha="right", va="bottom",
                    color=INK2, fontsize=8)


def _legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=8, labelcolor=INK2, **kw)
    return leg


def plot_correlogram(corr, path):
    fig, (ax,) = _figure(width=7.5)
    for name, color in TARGET_COLORS.items():
        c = corr[(corr["target"] == name) & (corr["n_pairs"] > 0)]
        mid = np.sqrt(np.maximum(c["lo"], 1) * c["hi"])          # geometric bin centre
        ax.plot(mid, c["corr"], color=color, linewidth=2, marker="o", markersize=4,
                markeredgecolor=SURFACE, markeredgewidth=1, label=TARGET_LABELS[name])
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xscale("log")
    _style(ax, "distance between building centroids (m)", "correlation of building pairs")
    ax.set_title("Pair correlation by distance: the buffer goes where it flattens",
                 color=INK, fontsize=11, loc="left")
    _legend(ax, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_separation(rows, path):
    fig, (ax,) = _figure()
    _buffer_lines(ax, rows, "dist_median_m")
    _reference(ax, rows, "dist_median_m", "random")
    _reference(ax, rows, "dist_median_m", "tile")
    ax.set_yscale("log")
    _style(ax, "grid block side (m)", "median distance to nearest train building (m)")
    ax.set_title("How far val/test buildings are from training", color=INK, fontsize=11,
                 loc="left")
    _legend(ax, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_similarity(rows, path):
    targets = ["faces_resid", "roof_ratio"]
    fig, axes = _figure(ncols=len(targets), width=11)
    for ax, name in zip(axes, targets):
        _buffer_lines(ax, rows, f"corr_{name}")
        _reference(ax, rows, f"corr_{name}", "random")
        _reference(ax, rows, f"corr_{name}", "tile")
        _style(ax, "grid block side (m)", "correlation with nearest train building")
        ax.set_title(TARGET_LABELS[name], color=INK, fontsize=10, loc="left")
    _legend(axes[0], loc="upper right")
    fig.suptitle("How alike the nearest training building still is (lower = better separated)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_cost(rows, path):
    fig, (left, right) = _figure(ncols=2, width=11)
    _buffer_lines(left, rows, "train_lost_frac")
    left.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _style(left, "grid block side (m)", "training buildings dropped by the buffer")
    left.set_title("Cost: training data lost", color=INK, fontsize=10, loc="left")
    _legend(left, loc="upper right")

    grid = rows[(rows["scheme"] == "grid") & (rows["buffer_m"] == rows["buffer_m"].min())]
    g = grid.groupby("block_m")["n_eval_groups"].mean()
    right.plot(g.index, g.values, color=BUFFER_RAMP[1], linewidth=2, marker="o", markersize=5,
               markeredgecolor=SURFACE, markeredgewidth=1.5)
    _reference(right, rows, "n_eval_groups", "tile")
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xticks(list(g.index))
    right.set_xticklabels([f"{s:g}" for s in g.index])
    _style(right, "grid block side (m)", "val + test blocks")
    right.set_title("Independent val/test units", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path, help="a split_sweep output directory")
    args = parser.parse_args(argv)
    try:
        rows = pd.read_csv(args.out / "split_sweep.csv")
        corr = pd.read_csv(args.out / "correlogram.csv")
    except OSError as e:
        raise SystemExit(f"{args.out} is not a split_sweep output directory: {e}")
    written = []
    for name, fn, data in (("correlogram", plot_correlogram, corr),
                           ("separation", plot_separation, rows),
                           ("similarity", plot_similarity, rows),
                           ("cost", plot_cost, rows)):
        path = args.out / f"{name}.png"
        fn(data, path)
        written.append(path)
    return written


if __name__ == "__main__":
    main()
