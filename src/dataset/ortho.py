"""Orthophoto crops aligned to the mesh transformer's normalized frame.

The mesh tokens live in the frame `normalize_to_unit_box` builds from the LOD1
mesh: ``u = (x - cx) / sx``, ``v = (y - cy) / sy``, with the building spanning
[-0.5, 0.5]. A crop covers ``[-0.5 - beta, 0.5 + beta]`` on both axes of that
same frame and is resampled to a fixed ``size x size``, so every image shares one
normalized extent and patch (i, j) always means the same (u, v) -- whatever the
building's size in metres.

Array axes follow the mesh, not the raster: row index increases with +y and
column index with +x. GeoTIFFs are stored north-up (row 0 is the *largest* y),
hence the vertical flip in `OrthoReader.crop`.

This module is the single source of the window arithmetic; `src.ortho_parse`
imports `crop_bounds` so the mosaics it writes always cover what is read here.
"""
import collections
import logging
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds

logger = logging.getLogger(__name__)


def crop_bounds(center, scale, beta):
    """World box ``(xmin, ymin, xmax, ymax)`` covering normalized [-0.5-beta, 0.5+beta]².

    Args:
        center, scale: the building's frame from `normalize_to_unit_box` ([3] each;
            only x and y are used).
        beta: margin beyond the unit box, in normalized units.
    """
    hx = (0.5 + beta) * float(scale[0])
    hy = (0.5 + beta) * float(scale[1])
    cx, cy = float(center[0]), float(center[1])
    return cx - hx, cy - hy, cx + hx, cy + hy


def pixel_to_uv(i, j, size, beta):
    """Normalized ``(u, v)`` of the centre of output pixel (row i, column j).

    The inverse of the crop: column j maps to u (x), row i to v (y), both
    increasing with the index.
    """
    step = (1.0 + 2.0 * beta) / size
    return -0.5 - beta + (j + 0.5) * step, -0.5 - beta + (i + 0.5) * step


class OrthoReader:
    """Reads aligned per-building crops out of per-file orthophoto mosaics.

    Handles are opened lazily and kept in a small LRU, and are dropped on
    pickling: DataLoader workers on Windows are spawned and receive a pickled
    copy, so each worker opens its own handles on first use.

    Args:
        root: directory holding the mosaics (``<subset>/ortho``).
        size: output side length in pixels.
        beta: margin beyond the unit box, in normalized units.
        max_open: open-handle budget per process.
    """

    def __init__(self, root, size=256, beta=0.1, max_open=32):
        self.root = Path(root)
        self.size = int(size)
        self.beta = float(beta)
        self.max_open = int(max_open)
        self.n_padded = 0          # crops that reached past their mosaic
        self._open = collections.OrderedDict()

    def _source(self, rel_name):
        src = self._open.pop(rel_name, None)
        if src is None:
            src = rasterio.open(self.root / rel_name)
            while len(self._open) >= self.max_open:
                self._open.popitem(last=False)[1].close()
        self._open[rel_name] = src
        return src

    def crop(self, rel_name, center, scale):
        """``uint8 [3, size, size]`` tensor of the building's window, mesh-axis order.

        Args:
            rel_name: mosaic path relative to `root`, e.g. ``"Source B/x.tif"``.
            center, scale: the building's frame from `normalize_to_unit_box`.

        Parts of the window outside the mosaic come back as zeros and are
        counted in `n_padded`, rather than raising mid-epoch.
        """
        size = self.size
        out = np.zeros((3, size, size), np.uint8)
        src = self._source(rel_name)

        win = from_bounds(*crop_bounds(center, scale, self.beta), transform=src.transform)
        c0, r0 = win.col_off, win.row_off
        c1, r1 = c0 + win.width, r0 + win.height
        cc0, rr0 = max(c0, 0.0), max(r0, 0.0)
        cc1, rr1 = min(c1, float(src.width)), min(r1, float(src.height))
        if (cc0, rr0, cc1, rr1) != (c0, r0, c1, r1):
            self.n_padded += 1

        if cc1 > cc0 and rr1 > rr0:
            # Destination span of the clipped window inside the full canvas.
            fx, fy = size / win.width, size / win.height
            dc0, dc1 = int(round((cc0 - c0) * fx)), int(round((cc1 - c0) * fx))
            dr0, dr1 = int(round((rr0 - r0) * fy)), int(round((rr1 - r0) * fy))
            if dc1 > dc0 and dr1 > dr0:
                out[:, dr0:dr1, dc0:dc1] = src.read(
                    (1, 2, 3), window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0),
                    out_shape=(3, dr1 - dr0, dc1 - dc0), resampling=Resampling.bilinear)

        # North-up raster rows -> +y rows. The copy is required: torch rejects
        # negative strides.
        return torch.from_numpy(np.ascontiguousarray(out[:, ::-1, :]))

    def close(self):
        while self._open:
            self._open.popitem()[1].close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_open"] = collections.OrderedDict()
        return state
