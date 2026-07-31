#!/usr/bin/env python3
"""Quantify a LOD split: technical validity, sanity vs raw, model-relevant stats.

Reads CityJSON directly and never invokes the datamodule -- that is verified
separately. Vertex and face counts match the parser exactly rather than
approximately: the Levi vertex-node set is the distinct vertex ids over outer
rings, so replicating ``dataset._iter_faces`` semantics here reproduces both
node counts without torch.

``_iter_faces`` is *replicated*, not imported, on purpose: this analysis must be
able to detect a regression in the parser, which it cannot do if it shares the
parser's code.

Three families of output, written to ``outputs/cityobject_analysis``:

  1. splitting technical validity -- did the splitter do what it claims
  2. sanity vs raw -- did the pathological distributions actually go away
  3. model-relevant distributions -- LOD2-only statistics for denoiser design

Usage:
    python -m src.analysis.cityobject_analysis "data/The Hague" --tiles 4
    python -m src.analysis.cityobject_analysis "data/The Hague"
"""
import argparse
import csv
import json
import logging
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

# CityGML boundary-surface classes. Anything else is an unrecognised label.
SEMANTIC_TYPES = frozenset({
    "GroundSurface", "WallSurface", "RoofSurface",
    "OuterFloorSurface", "OuterCeilingSurface", "ClosureSurface",
})
# Orientation predicates apply only to these three; OuterFloor/OuterCeiling
# legitimately share orientations with Roof/Ground (balcony tops, overhang
# undersides) and are deliberately exempt.
ORIENTED_TYPES = ("GroundSurface", "RoofSurface", "WallSurface")

# Measured over 3 tiles per source: every ground face is exactly -1, Source B
# walls are exactly 0 and Source A's worst is 0.027 (1.56 deg). 0.05 sits clear
# of that tessellation noise while still catching genuine breakage. Re-check per
# source -- a photogrammetric or LoD3 supplier can have real battered facades.
WALL_NZ_MAX = 0.05
GROUND_NZ_MAX = -0.999
PLANARITY_MAX = 0.02            # out-of-plane deviation as a fraction of ring diameter
SMALL_OBJECT_VERTICES = 10      # the population that motivated this analysis
MAX_OUTLIER_ROWS = 500          # per check, to keep outliers.csv openable
RESERVOIR = 400_000             # per continuous metric
SEED = 20260730                 # reproducibility: fixes the reservoir sampling

SUPPORTED_GEOMETRY = ("Solid", "MultiSurface", "CompositeSurface")


# ==============================================================================
# Geometry primitives
# ==============================================================================

def newell_vector(points):
    """Newell area vector: 2 * A * n for a planar ring, translation invariant."""
    p = np.asarray(points, dtype=float)
    return np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)


def face_area(points):
    return float(np.linalg.norm(newell_vector(points)) / 2.0)


def face_normal(points):
    """Unit normal, or None when the ring is degenerate (collinear/zero area)."""
    n = newell_vector(points)
    mag = np.linalg.norm(n)
    return None if mag < 1e-12 else n / mag


def planarity_ratio(points):
    """Max out-of-plane deviation as a fraction of the ring's diameter."""
    p = np.asarray(points, dtype=float)
    if len(p) < 4:
        return 0.0                       # a triangle is planar by construction
    n = face_normal(p)
    if n is None:
        return 0.0
    dev = float(np.abs((p - p.mean(axis=0)) @ n).max())
    diameter = float(np.linalg.norm(p.max(axis=0) - p.min(axis=0)))
    return dev / diameter if diameter > 1e-12 else 0.0


def iter_faces(geom):
    """Yield ``(outer_ring, semantic_type_or_None)`` -- mirrors dataset._iter_faces.

    Only Solid / MultiSurface / CompositeSurface contribute; every other type
    yields nothing, exactly as the parser does. Rings shorter than 3 are skipped.
    """
    boundaries = geom.get("boundaries", []) or []
    gtype = geom.get("type")
    sem = geom.get("semantics") or {}
    surfaces = sem.get("surfaces") or []
    values = sem.get("values") or []

    if gtype == "Solid":
        faces = [f for shell in boundaries for f in shell]
        vals = [v for shell_vals in values for v in shell_vals] if values else []
    elif gtype in ("MultiSurface", "CompositeSurface"):
        faces, vals = boundaries, values
    else:
        return

    for i, face in enumerate(faces):
        if not face or len(face[0]) < 3:
            continue
        stype = None
        if i < len(vals) and vals[i] is not None and vals[i] < len(surfaces):
            s = surfaces[vals[i]]
            stype = s.get("type") if s else None
        yield face[0], stype


def solid_faces(geom):
    """Full faces (outer ring + holes) of a Solid, for closure and volume."""
    if geom.get("type") != "Solid":
        return []
    return [f for shell in geom.get("boundaries", []) for f in shell]


# ==============================================================================
# Wireframe topology
# ==============================================================================

def wireframe_edges(rings):
    """Undirected vertex-vertex edges implied by ring adjacency."""
    edges = set()
    for ring in rings:
        for i in range(len(ring)):
            a, b = ring[i], ring[(i + 1) % len(ring)]
            if a != b:
                edges.add((a, b) if a < b else (b, a))
    return edges


def vertex_degrees(rings):
    """Distinct wireframe neighbours per vertex."""
    adj = defaultdict(set)
    for a, b in wireframe_edges(rings):
        adj[a].add(b)
        adj[b].add(a)
    return {v: len(nb) for v, nb in adj.items()}


def connected_components(rings):
    adj = defaultdict(set)
    for a, b in wireframe_edges(rings):
        adj[a].add(b)
        adj[b].add(a)
    seen, components = set(), 0
    for start in adj:
        if start in seen:
            continue
        components += 1
        stack = [start]
        seen.add(start)
        while stack:
            for nxt in adj[stack.pop()]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return components


def shell_edge_defects(faces):
    """``(unpaired, reused)`` directed edges of a shell.

    ``unpaired`` -- no oppositely-wound twin, so the surface is open there.
    ``reused``   -- traversed twice in the *same* direction, so two faces
    disagree about which side faces out; the shell is closed but not orientable.

    A wireframe view shows neither: both defects leave every *undirected* edge
    drawn exactly as it would be on a clean solid, which is why visually
    perfect buildings still fail closure.
    """
    used = Counter()
    for face in faces:
        for ring in face:
            for i in range(len(ring)):
                used[(ring[i], ring[(i + 1) % len(ring)])] += 1
    unpaired = [e for e in used if (e[1], e[0]) not in used]
    reused = [e for e, n in used.items() if n > 1]
    return unpaired, reused


def is_watertight(faces):
    """Closed *and* consistently oriented: no holes, no repeated direction."""
    unpaired, reused = shell_edge_defects(faces)
    return not unpaired and not reused


def signed_volume(faces, verts):
    """Divergence theorem; positive iff the shell is outward-oriented.

    Newell summed over a face's outer ring *and* its holes gives 2 * A_net * n,
    oppositely wound holes subtracting themselves, so the integral collapses to
    1/6 * sum(p . N).
    """
    v = np.asarray(verts, dtype=float)
    total = 0.0
    for face in faces:
        area_vec = np.zeros(3)
        for ring in face:
            area_vec = area_vec + newell_vector(v[ring])
        total += float(np.dot(v[face[0][0]], area_vec)) / 6.0
    return total


def count_coincident(verts, ids):
    """Distinct vertex ids sharing a position -- two Levi nodes at one point."""
    if not len(ids):
        return 0
    pts = np.asarray(verts, dtype=float)[list(ids)]
    unique = np.unique(np.round(pts, 6), axis=0)
    return len(pts) - len(unique)


# ==============================================================================
# Per-object measurement
# ==============================================================================

def _faces_and_types(obj):
    out = []
    for geom in obj.get("geometry", []) or []:
        out.extend(iter_faces(geom))
    return out


def object_metrics(obj, verts):
    """Per-object statistics, or None when the object yields no faces.

    ``n_vertices`` and ``n_faces`` are the Levi vertex-node and face-node counts.
    """
    faces = _faces_and_types(obj)
    if not faces:
        return None
    rings = [r for r, _ in faces]
    active = sorted({v for r in rings for v in r})
    pts = np.asarray(verts, dtype=float)[active]

    com = pts.mean(axis=0)
    extent = pts.max(axis=0) - pts.min(axis=0)
    ground_z = [np.asarray(verts, dtype=float)[r][:, 2].mean()
                for r, t in faces if t == "GroundSurface"]

    geoms = obj.get("geometry", []) or []
    solid = next((g for g in geoms if g.get("type") == "Solid"), None)
    sfaces = solid_faces(solid) if solid else []

    m = {
        "n_vertices": len(active),
        "n_faces": len(faces),
        "n_levi_nodes": len(active) + len(faces),
        "com": com,
        "extent": extent,
        "diameter": float(np.linalg.norm(extent)),
        "ground_level": float(np.mean(ground_z)) if ground_z else float(pts[:, 2].min()),
        "n_components": connected_components(rings),
        "n_coincident": count_coincident(verts, active),
        "radial": np.linalg.norm(pts - com, axis=1),
        "degrees": list(vertex_degrees(rings).values()),
        "ring_lengths": [len(r) for r in rings],
        "faces_per_vertex": list(Counter(v for r in rings for v in set(r)).values()),
        "semantic_types": Counter(t for _, t in faces),
        "watertight": is_watertight(sfaces) if sfaces else False,
        "volume": signed_volume(sfaces, verts) if sfaces else 0.0,
    }

    v = np.asarray(verts, dtype=float)
    m["points"] = pts
    m["edge_lengths"] = [float(np.linalg.norm(v[a] - v[b]))
                         for a, b in wireframe_edges(rings)]
    m["normals"] = [(t, face_normal(v[r])) for r, t in faces]
    m["planarity"] = [planarity_ratio(v[r]) for r in rings]
    return m


def object_defects(obj, verts):
    """Names of every geometry-breaking predicate this object trips."""
    faces = _faces_and_types(obj)
    if not faces:
        return []
    v = np.asarray(verts, dtype=float)
    found = set()

    present = {t for _, t in faces}
    for name, label in (("missing_ground", "GroundSurface"),
                        ("missing_roof", "RoofSurface"),
                        ("missing_wall", "WallSurface")):
        if label not in present:
            found.add(name)
    if any(t is not None and t not in SEMANTIC_TYPES for t in present):
        found.add("unknown_semantic_label")
    if None in present:
        found.add("null_semantic_label")

    for ring, stype in faces:
        if len(set(ring)) != len(ring):
            found.add("repeated_ring_vertex")
        if len(set(ring)) < 3:
            found.add("degenerate_ring")
        n = face_normal(v[ring])
        if n is None:
            found.add("collinear_face")
            continue
        if stype == "GroundSurface" and n[2] > GROUND_NZ_MAX:
            found.add("ground_normal_up")
        elif stype == "RoofSurface" and n[2] <= 0:
            found.add("roof_normal_down")
        elif stype == "WallSurface" and abs(n[2]) > WALL_NZ_MAX:
            found.add("wall_not_vertical")
        if planarity_ratio(v[ring]) > PLANARITY_MAX:
            found.add("non_planar_face")

    if connected_components([r for r, _ in faces]) > 1:
        found.add("disconnected")

    solid = next((g for g in (obj.get("geometry") or []) if g.get("type") == "Solid"), None)
    if solid:
        sfaces = solid_faces(solid)
        # Split, because the two failures need different repairs: a hole needs a
        # face added, a same-direction edge needs a face rewound.
        unpaired, reused = shell_edge_defects(sfaces)
        if unpaired:
            found.add("open_shell")
        if reused:
            found.add("non_manifold_edge")
        if signed_volume(sfaces, verts) <= 0:
            found.add("non_positive_volume")
    return sorted(found)


# ==============================================================================
# Streaming aggregation
# ==============================================================================

class Reservoir:
    """Fixed-size uniform sample of a stream, so full corpus fits in memory."""

    def __init__(self, size=RESERVOIR, seed=SEED):
        self.size, self.seen, self.items = size, 0, []
        self._rng = random.Random(seed)

    def extend(self, values):
        for x in values:
            self.seen += 1
            if len(self.items) < self.size:
                self.items.append(x)
            else:
                j = self._rng.randrange(self.seen)
                if j < self.size:
                    self.items[j] = x

    def array(self):
        return np.asarray(self.items, dtype=float)


def world_vertices(cj):
    """Vertex array in world coordinates, applying ``transform`` when present."""
    v = np.asarray(cj["vertices"], dtype=float)
    t = cj.get("transform")
    if t:
        v = v * np.asarray(t["scale"], dtype=float) + np.asarray(t["translate"], dtype=float)
    return v


class Accumulator:
    """Everything the figures need, gathered in one streaming pass."""

    def __init__(self):
        self.per_object = defaultdict(list)      # key -> list of scalars
        self.reservoirs = defaultdict(Reservoir)
        self.counters = defaultdict(Counter)
        self.outliers = defaultdict(list)
        self.defect_counts = Counter()
        self.n_objects = 0

    def note_outlier(self, check, source, fname, oid, metric="", value=""):
        self.defect_counts[check] += 1
        rows = self.outliers[check]
        if len(rows) < MAX_OUTLIER_ROWS:
            rows.append((check, source, fname, f"{fname}:{oid}", metric, value))


def scan_file(path, source, acc, kind, collect_full=True):
    """Fold one CityJSON file into ``acc``. ``kind`` is 'raw' or a LOD folder."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cj = json.load(fh)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return 0
    if cj.get("type") != "CityJSON":
        return 0

    verts = world_vertices(cj)
    n_seen = 0

    for oid, obj in (cj.get("CityObjects") or {}).items():
        for geom in obj.get("geometry", []) or []:
            gt = geom.get("type")
            acc.counters[f"{kind}_geom_types"][gt] += 1
            if gt not in SUPPORTED_GEOMETRY:
                acc.counters[f"{kind}_unhandled"][gt] += 1

        m = object_metrics(obj, verts)
        if m is None:
            continue
        n_seen += 1
        acc.n_objects += 1

        acc.counters[f"{kind}_vertex_count"][m["n_vertices"]] += 1
        acc.counters[f"{kind}_face_count"][m["n_faces"]] += 1
        acc.counters[f"{kind}_levi_nodes"][m["n_levi_nodes"]] += 1
        acc.counters[f"{kind}_degree"].update(m["degrees"])
        acc.counters[f"{kind}_ring_length"].update(m["ring_lengths"])
        acc.counters[f"{kind}_faces_per_vertex"].update(m["faces_per_vertex"])
        acc.counters[f"{kind}_semantic"].update(
            {str(t): n for t, n in m["semantic_types"].items()})

        # attribution of the population that motivated the analysis
        if m["n_vertices"] < SMALL_OBJECT_VERTICES:
            for geom in obj.get("geometry", []) or []:
                acc.counters[f"{kind}_small_attrib"][
                    (source, obj.get("type"), geom.get("type"), str(geom.get("lod")))] += 1

        if not collect_full:
            continue

        acc.per_object["n_vertices"].append(m["n_vertices"])
        acc.per_object["n_faces"].append(m["n_faces"])
        acc.per_object["n_levi_nodes"].append(m["n_levi_nodes"])
        acc.per_object["com_x"].append(m["com"][0])
        acc.per_object["com_y"].append(m["com"][1])
        acc.per_object["com_z"].append(m["com"][2])
        acc.per_object["ground_level"].append(m["ground_level"])
        acc.per_object["diameter"].append(m["diameter"])
        acc.per_object["extent_x"].append(m["extent"][0])
        acc.per_object["extent_y"].append(m["extent"][1])
        acc.per_object["extent_z"].append(m["extent"][2])
        acc.per_object["volume"].append(m["volume"])
        acc.per_object["n_components"].append(m["n_components"])
        acc.per_object["n_coincident"].append(m["n_coincident"])
        acc.per_object["watertight"].append(1.0 if m["watertight"] else 0.0)
        acc.per_object["source"].append(source)

        acc.reservoirs["edge_length"].extend(m["edge_lengths"])
        acc.reservoirs["radial"].extend(m["radial"].tolist())
        acc.reservoirs["planarity"].extend(m["planarity"])
        if len(m["points"]) > 1:
            # KD-tree, not a pairwise matrix: some objects carry 2000+ vertices
            # and the O(n^2) form dominates the whole run.
            dists, _ = cKDTree(m["points"]).query(m["points"], k=2)
            acc.reservoirs["nn_distance"].extend(dists[:, 1].tolist())

        for stype, n in m["normals"]:
            if n is None or stype not in ORIENTED_TYPES:
                continue
            acc.reservoirs[f"nz_{stype}"].extend([float(n[2])])

        for check in object_defects(obj, verts):
            # Carry the vertex count so the explorer can sort by size -- the
            # worst offenders run to 1700+ faces and are painful to render.
            acc.note_outlier(check, source, path.name, oid,
                             "n_vertices", m["n_vertices"])

    return n_seen


# ==============================================================================
# Plotting
# ==============================================================================

import plotly.graph_objects as go                                    # noqa: E402
from plotly.subplots import make_subplots                            # noqa: E402

# Validated categorical palette, matching src/visualize_levi.py.
COLORS = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#898781"]
OUT_KW = dict(include_plotlyjs="directory", full_html=True)


def _qq_points(values, log=False):
    """Sample vs theoretical-normal quantiles, on log(values) when asked."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if log:
        v = v[v > 0]
        v = np.log(v)
    if len(v) < 3:
        return np.array([]), np.array([])
    if len(v) > 5000:                       # thin for a readable, light figure
        v = np.sort(v)[np.linspace(0, len(v) - 1, 5000).astype(int)]
    else:
        v = np.sort(v)
    from scipy.special import ndtri
    p = (np.arange(len(v)) + 0.5) / len(v)
    return ndtri(p), v


def distribution_figure(series, title, xlabel, log_ref=True, log_x=False, nbins=80):
    """Histogram plus a Q-Q panel.

    The Q-Q is a *shape diagnostic*, not an outlier detector: against a normal
    reference these long-tailed distributions deviate everywhere. Read it for
    straightness (is this lognormal?) and for kinks (a break means a mixture,
    most likely Source A vs Source B).
    """
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.62, 0.38],
        subplot_titles=(xlabel, f"Q-Q vs {'log-' if log_ref else ''}normal"))

    for i, (name, values) in enumerate(series.items()):
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        colour = COLORS[i % len(COLORS)]
        fig.add_trace(go.Histogram(x=np.log10(v[v > 0]) if log_x else v,
                                   name=name, nbinsx=nbins, opacity=0.65,
                                   marker_color=colour), row=1, col=1)
        tq, sq = _qq_points(v, log=log_ref)
        if len(tq):
            fig.add_trace(go.Scatter(x=tq, y=sq, mode="markers", name=f"{name} Q-Q",
                                     marker=dict(size=3, color=colour),
                                     showlegend=False), row=1, col=2)
            lo, hi = np.percentile(sq, [25, 75])
            tlo, thi = np.percentile(tq, [25, 75])
            if thi > tlo:
                slope = (hi - lo) / (thi - tlo)
                ref = slope * (tq - tlo) + lo
                fig.add_trace(go.Scatter(x=tq, y=ref, mode="lines", showlegend=False,
                                         line=dict(color=colour, dash="dot", width=1)),
                              row=1, col=2)

    fig.update_layout(title=title, barmode="overlay", template="plotly_white",
                      height=430, legend=dict(orientation="h", y=-0.18))
    fig.update_xaxes(title_text=f"log10({xlabel})" if log_x else xlabel, row=1, col=1)
    fig.update_yaxes(title_text="objects", type="log", row=1, col=1)
    fig.update_xaxes(title_text="theoretical quantile", row=1, col=2)
    fig.update_yaxes(title_text="log(sample)" if log_ref else "sample", row=1, col=2)
    return fig


def counter_figure(series, title, xlabel, log_y=True, max_x=None):
    """Bar chart from integer-keyed Counters (exact, no sampling)."""
    fig = go.Figure()
    for i, (name, counter) in enumerate(series.items()):
        if not counter:
            continue
        keys = sorted(k for k in counter if isinstance(k, (int, float)))
        if max_x is not None:
            keys = [k for k in keys if k <= max_x]
        fig.add_trace(go.Bar(x=keys, y=[counter[k] for k in keys], name=name,
                             opacity=0.7, marker_color=COLORS[i % len(COLORS)]))
    fig.update_layout(title=title, barmode="overlay", template="plotly_white",
                      height=430, xaxis_title=xlabel, legend=dict(orientation="h", y=-0.18))
    fig.update_yaxes(title_text="count", type="log" if log_y else "linear")
    return fig


def category_figure(series, title, xlabel):
    """Bar chart over string categories."""
    fig = go.Figure()
    labels = sorted({k for c in series.values() for k in c}, key=str)
    for i, (name, counter) in enumerate(series.items()):
        fig.add_trace(go.Bar(x=[str(k) for k in labels], y=[counter.get(k, 0) for k in labels],
                             name=name, marker_color=COLORS[i % len(COLORS)]))
    fig.update_layout(title=title, barmode="group", template="plotly_white",
                      height=430, xaxis_title=xlabel, yaxis_title="count",
                      yaxis_type="log", legend=dict(orientation="h", y=-0.18))
    return fig


def ecdf_figure(series, title, xlabel):
    fig = go.Figure()
    for i, (name, values) in enumerate(series.items()):
        v = np.sort(np.asarray(values, dtype=float))
        if not len(v):
            continue
        if len(v) > 20000:
            v = v[np.linspace(0, len(v) - 1, 20000).astype(int)]
        fig.add_trace(go.Scatter(x=v, y=np.linspace(0, 1, len(v)), mode="lines",
                                 name=name, line=dict(color=COLORS[i % len(COLORS)])))
    fig.update_layout(title=title, template="plotly_white", height=430,
                      xaxis_title=xlabel, xaxis_type="log", yaxis_title="cumulative fraction",
                      legend=dict(orientation="h", y=-0.18))
    return fig


# ==============================================================================
# Corpus walk
# ==============================================================================

def _tiles(folder, limit):
    files = sorted(p for p in folder.rglob("*.json") if p.is_file())
    return files[:limit] if limit else files


def scan_folder(base, label, acc, tiles=None, collect_full=True):
    """Scan every source subfolder of ``base/label`` into ``acc``."""
    root = base / label
    if not root.is_dir():
        return {}
    counts = {}
    for source in sorted(d for d in root.iterdir() if d.is_dir()):
        n = 0
        for path in _tiles(source, tiles):
            n += scan_file(path, source.name, acc, f"{label}:{source.name}",
                           collect_full=collect_full)
        counts[source.name] = n
        logger.info("%s/%s: %d objects", label, source.name, n)
    return counts


def pairing_and_calibration(base, tiles=None):
    """Per-file id-set agreement across LOD folders, and synth:LOD2 volume ratio."""
    rows, ratios, mismatches = [], [], []
    lod2_root = base / "LOD2"
    if not lod2_root.is_dir():
        return rows, np.array([]), mismatches

    for source in sorted(d for d in lod2_root.iterdir() if d.is_dir()):
        for path in _tiles(source, tiles):
            rel = path.relative_to(lod2_root)
            sets, cjs = {}, {}
            for label in ("LOD1", "LOD2", "LOD1_synth"):
                p = base / label / rel
                if not p.exists():
                    sets[label] = None
                    continue
                cj = json.loads(p.read_text(encoding="utf-8"))
                cjs[label] = cj
                sets[label] = set(cj.get("CityObjects") or {})

            rows.append({
                "source": source.name, "file": path.name,
                "n_lod2": len(sets["LOD2"] or ()),
                "n_lod1": len(sets["LOD1"]) if sets["LOD1"] is not None else -1,
                "n_synth": len(sets["LOD1_synth"]) if sets["LOD1_synth"] is not None else -1,
                "synth_matches": sets["LOD1_synth"] == sets["LOD2"]
                                 if sets["LOD1_synth"] is not None else None,
                "lod1_matches": sets["LOD1"] == sets["LOD2"]
                                if sets["LOD1"] is not None else None,
            })
            if rows[-1]["synth_matches"] is False:
                mismatches.append(f"{path.name}: LOD1_synth ids != LOD2 ids")
            if sets["LOD1"] is not None and rows[-1]["lod1_matches"] is False:
                mismatches.append(f"{path.name}: LOD1 ids != LOD2 ids")

            if "LOD1_synth" in cjs and "LOD2" in cjs:
                v2, v1 = world_vertices(cjs["LOD2"]), world_vertices(cjs["LOD1_synth"])
                for oid, o2 in cjs["LOD2"]["CityObjects"].items():
                    o1 = cjs["LOD1_synth"]["CityObjects"].get(oid)
                    if not o1:
                        continue
                    a = abs(signed_volume(solid_faces(o1["geometry"][0]), v1))
                    b = abs(signed_volume(solid_faces(o2["geometry"][0]), v2))
                    if b > 1e-6:
                        ratios.append(a / b)
    return rows, np.array(ratios, dtype=float), mismatches


def counter_to_array(counter):
    keys = np.array([k for k in counter if isinstance(k, (int, float))], dtype=float)
    if not len(keys):
        return np.array([])
    return np.repeat(keys, [counter[int(k)] for k in keys])


def _pct(values, qs=(1, 25, 50, 75, 99, 99.9)):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return "n/a"
    return "  ".join(f"p{q}={np.percentile(v, q):.4g}" for q in qs)


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# ==============================================================================
# Report assembly
# ==============================================================================

def _by_source(acc, key, sources):
    """Per-object metric split by source, as ``{source: ndarray}``."""
    src = np.array(acc.per_object["source"])
    vals = np.array(acc.per_object[key], dtype=float)
    return {s: vals[src == s] for s in sources if (src == s).any()}


def build_report(acc, base, out_dir, tiles, sources, pair_rows, ratios, mismatches,
                 raw_counts, lod_counts):
    out_dir.mkdir(parents=True, exist_ok=True)
    figures, notes = {}, []

    def cnt(label, source, key):
        return acc.counters[f"{label}:{source}_{key}"]

    # ---- family 1: splitting technical validity --------------------------
    ok_synth = sum(1 for r in pair_rows if r["synth_matches"])
    ok_lod1 = sum(1 for r in pair_rows if r["lod1_matches"])
    n_lod1 = sum(1 for r in pair_rows if r["lod1_matches"] is not None)
    figures["split_pairing_integrity"] = category_figure(
        {"id sets agree": Counter({"LOD1_synth vs LOD2": ok_synth, "LOD1 vs LOD2": ok_lod1}),
         "files checked": Counter({"LOD1_synth vs LOD2": len(pair_rows), "LOD1 vs LOD2": n_lod1})},
        "Object-id set agreement across LOD folders", "comparison")
    notes.append(f"pairing: {ok_synth}/{len(pair_rows)} files have LOD1_synth ids == LOD2 ids; "
                 f"{ok_lod1}/{n_lod1} have LOD1 ids == LOD2 ids")
    if mismatches:
        notes.append(f"  MISMATCHES: {mismatches[:5]}")

    wt = np.array(acc.per_object["watertight"], dtype=float)
    vol = np.array(acc.per_object["volume"], dtype=float)
    figures["split_solid_validity"] = category_figure(
        {"LOD2 solids": Counter({"watertight": int(wt.sum()),
                                 "not watertight": int((1 - wt).sum()),
                                 "volume > 0": int((vol > 0).sum()),
                                 "volume <= 0": int((vol <= 0).sum())})},
        "LOD2 solid validity", "check")
    notes.append(f"solid validity: watertight {100 * wt.mean():.3f}%, "
                 f"positive volume {100 * (vol > 0).mean():.3f}% of {len(wt)} objects")

    if len(ratios):
        figures["split_volume_calibration"] = distribution_figure(
            {"LOD1_synth / LOD2": ratios},
            "Synthetic LOD1 volume relative to its LOD2 "
            "(target 1.0725 median / 1.1033 mean, from real 3DBAG lod1.2)",
            "volume ratio", log_ref=True)
        notes.append(f"synth:LOD2 volume ratio median {np.median(ratios):.4f} "
                     f"mean {ratios.mean():.4f} (target 1.0725 / 1.1033), n={len(ratios)}")

    # ---- family 2: sanity vs raw -----------------------------------------
    vseries, fseries, dseries, eseries = {}, {}, {}, {}
    for s in sources:
        for label in ("raw", "LOD2"):
            c = cnt(label, s, "vertex_count")
            if c:
                vseries[f"{label} {s}"] = c
                fseries[f"{label} {s}"] = cnt(label, s, "face_count")
                dseries[f"{label} {s}"] = cnt(label, s, "degree")
                eseries[f"{label} {s}"] = counter_to_array(c)
    figures["raw_vs_lod2_vertex_count"] = counter_figure(
        vseries, "Levi vertex-node count per object, raw vs LOD2", "vertices", max_x=400)
    figures["raw_vs_lod2_vertex_ecdf"] = ecdf_figure(
        eseries, "Levi vertex-node count, ECDF", "vertices")
    figures["raw_vs_lod2_face_count"] = counter_figure(
        fseries, "Levi face-node count per object, raw vs LOD2", "faces", max_x=200)
    figures["raw_vs_lod2_degree"] = counter_figure(
        dseries, "Wireframe vertex degree, raw vs LOD2", "degree", max_x=20)

    for s in sources:
        raw_c = cnt("raw", s, "vertex_count")
        lod2_c = cnt("LOD2", s, "vertex_count")
        small = sum(n for k, n in raw_c.items() if k < SMALL_OBJECT_VERTICES)
        lod2_small = sum(n for k, n in lod2_c.items() if k < SMALL_OBJECT_VERTICES)
        notes.append(f"{s}: objects with <{SMALL_OBJECT_VERTICES} vertices -- "
                     f"raw {small} of {sum(raw_c.values())}, "
                     f"LOD2 {lod2_small} of {sum(lod2_c.values())}")

    attrib = Counter()
    for s in sources:
        for k, n in cnt("raw", s, "small_attrib").items():
            attrib[" | ".join(str(x) for x in k)] += n
    figures["raw_small_object_attribution"] = category_figure(
        {"raw objects below the threshold": attrib},
        f"Raw objects with <{SMALL_OBJECT_VERTICES} vertices",
        "source | object type | geometry type | lod")
    for k, n in attrib.most_common(6):
        notes.append(f"  small-object attribution: {k} -> {n}")

    unhandled = Counter()
    for s in sources:
        unhandled.update(cnt("raw", s, "unhandled"))
    figures["raw_unhandled_geometry_types"] = category_figure(
        {"dropped silently by _iter_faces": unhandled or Counter({"(none)": 0})},
        "Raw geometry types the parser drops silently", "geometry type")
    notes.append(f"geometry types unsupported by _iter_faces: {dict(unhandled) or 'none'}")

    sem = {s: cnt("LOD2", s, "semantic") for s in sources}
    figures["lod2_semantic_label_mix"] = category_figure(
        sem, "LOD2 face semantic labels (full CityGML set)", "semantic type")
    for s in sources:
        notes.append(f"{s} LOD2 semantic labels: {dict(sem[s])}")

    presence = Counter({c: acc.defect_counts.get(c, 0)
                        for c in ("missing_ground", "missing_roof", "missing_wall")})
    figures["lod2_semantic_presence"] = category_figure(
        {"objects missing a class": presence},
        "LOD2 objects missing Ground / Roof / Wall", "missing class")

    for stype, name in (("RoofSurface", "lod2_roof_orientation"),
                        ("GroundSurface", "lod2_ground_orientation"),
                        ("WallSurface", "lod2_wall_orientation")):
        vals = acc.reservoirs[f"nz_{stype}"].array()
        if len(vals):
            figures[name] = distribution_figure(
                {stype: vals}, f"LOD2 {stype} normal z-component", "normal z", log_ref=False)
            notes.append(f"{stype} n_z: min {vals.min():.6f} median {np.median(vals):.6f} "
                         f"max {vals.max():.6f} (n={acc.reservoirs[f'nz_{stype}'].seen})")

    # ---- family 3: model-relevant distributions --------------------------
    for name, key, title, logref in (
            ("model_com_xy", "com_x", "Centre of mass, x", False),
            ("model_com_z", "com_z", "Centre of mass, z", False),
            ("model_ground_level", "ground_level", "Ground level (absolute z)", False),
            ("model_diameter", "diameter", "Object diameter", True),
            ("model_vertex_count", "n_vertices", "Levi vertex nodes", True),
            ("model_face_count", "n_faces", "Levi face nodes", True),
            ("model_levi_node_count", "n_levi_nodes", "Total Levi nodes", True)):
        series = _by_source(acc, key, sources)
        if series:
            figures[name] = distribution_figure(series, f"LOD2 {title}", title, log_ref=logref)
            notes.append(f"{title}: " + "  ".join(f"[{s}] {_pct(v)}" for s, v in series.items()))

    figures["model_axis_extent"] = distribution_figure(
        {ax: np.array(acc.per_object[f"extent_{ax}"], dtype=float) for ax in "xyz"},
        "LOD2 per-axis extent", "extent (m)", log_ref=True)
    notes.append("axis extent: " + "  ".join(
        f"[{ax}] {_pct(acc.per_object['extent_' + ax])}" for ax in "xyz"))

    for name, key, title, logref in (
            ("model_radial_distance", "radial", "Vertex distance from centre of mass", True),
            ("model_edge_length", "edge_length", "Wireframe edge length", True),
            ("model_nn_distance", "nn_distance", "Nearest-neighbour vertex distance", True),
            ("model_planarity", "planarity", "Face out-of-plane deviation / diameter", True)):
        vals = acc.reservoirs[key].array()
        if len(vals):
            figures[name] = distribution_figure({title: vals}, f"LOD2 {title}", title,
                                                log_ref=logref)
            notes.append(f"{title} (n={acc.reservoirs[key].seen}): {_pct(vals)}")

    node_classes, edge_classes = Counter(), Counter()
    for s in sources:
        node_classes["VERTEX"] += sum(k * n for k, n in cnt("LOD2", s, "vertex_count").items())
        for t, n in cnt("LOD2", s, "semantic").items():
            node_classes[t if t != "None" else "unlabelled"] += n
        edge_classes["EDGE_VV"] += sum(k * n // 2 for k, n in cnt("LOD2", s, "degree").items())
        edge_classes["EDGE_VF"] += sum(k * n for k, n in cnt("LOD2", s, "ring_length").items())
    figures["model_node_class_balance"] = category_figure(
        {"Levi nodes": node_classes}, "LOD2 Levi node class balance", "node class")
    figures["model_edge_class_balance"] = category_figure(
        {"Levi edges (undirected)": edge_classes}, "LOD2 Levi edge class balance", "edge class")
    total_nodes = max(1, sum(node_classes.values()))
    notes.append("node class balance: " + "  ".join(
        f"{k} {100 * v / total_nodes:.2f}%" for k, v in node_classes.most_common()))
    notes.append(f"edge class balance: {dict(edge_classes)}")

    figures["model_ring_length"] = counter_figure(
        {s: cnt("LOD2", s, "ring_length") for s in sources},
        "LOD2 vertices per face", "ring length", max_x=30)
    figures["model_faces_per_vertex"] = counter_figure(
        {s: cnt("LOD2", s, "faces_per_vertex") for s in sources},
        "LOD2 faces per vertex", "faces per vertex", max_x=20)

    comp = Counter(int(c) for c in acc.per_object["n_components"])
    coin = Counter(int(c) for c in acc.per_object["n_coincident"])
    figures["model_degeneracy"] = counter_figure(
        {"connected components": comp, "coincident vertices": coin},
        "LOD2 degeneracy: wireframe components and coincident vertices", "count", max_x=20)
    notes.append(f"components: {dict(comp.most_common(5))}; objects with coincident "
                 f"vertices: {sum(n for k, n in coin.items() if k > 0)}")

    # ---- write out --------------------------------------------------------
    for name, fig in figures.items():
        fig.write_html(str(out_dir / f"{name}.html"), **OUT_KW)

    with open(out_dir / "outliers.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["check", "source", "file", "object_id", "metric", "value"])
        for check in sorted(acc.outliers):
            w.writerows(acc.outliers[check])

    lines = [
        "CityObject analysis", "=" * 62,
        f"generated       : {datetime.now(timezone.utc).isoformat()}",
        f"git commit      : {_git_commit()}",
        f"dataset         : {base}",
        f"tiles per source: {tiles or 'all'}",
        f"raw objects     : {raw_counts}",
        f"lod2 objects    : {lod_counts}",
        f"figures written : {len(figures)}",
        "", "DEFECTS (geometry-breaking; ids in outliers.csv)", "-" * 62,
    ]
    if acc.defect_counts:
        lines += [f"  {check:26s} {n}" for check, n in acc.defect_counts.most_common()]
    else:
        lines.append("  none")
    lines += ["", "FINDINGS", "-" * 62] + [f"  {n}" for n in notes]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return figures


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", help="Dataset root holding raw/, LOD1/, LOD2/, LOD1_synth/.")
    ap.add_argument("--out", default="outputs/cityobject_analysis", help="Output folder.")
    ap.add_argument("--tiles", type=int, default=None,
                    help="Sample this many files per source (development).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base = Path(args.dataset_dir).resolve()
    if not base.is_dir():
        sys.exit(f"Dataset folder not found: {base}")
    out_dir = Path(args.out).resolve()

    acc = Accumulator()
    logger.info("scanning raw ...")
    raw_counts = scan_folder(base, "raw", acc, args.tiles, collect_full=False)
    logger.info("scanning LOD2 ...")
    lod_counts = scan_folder(base, "LOD2", acc, args.tiles, collect_full=True)

    logger.info("checking pairing and volume calibration ...")
    pair_rows, ratios, mismatches = pairing_and_calibration(base, args.tiles)

    sources = sorted(raw_counts) or sorted(lod_counts)
    logger.info("building figures ...")
    figures = build_report(acc, base, out_dir, args.tiles, sources,
                           pair_rows, ratios, mismatches, raw_counts, lod_counts)
    logger.info("%d figures + summary.txt + outliers.csv -> %s", len(figures), out_dir)


if __name__ == "__main__":
    main()
