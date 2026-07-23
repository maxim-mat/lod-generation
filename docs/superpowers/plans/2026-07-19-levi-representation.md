# Levi Graph Representation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shift the dataset representation to Levi graphs (faces become nodes), make `graph_to_cityjson` its exact inverse with CCW-outward face rings, and add a Plotly visualization script.

**Architecture:** `parse_cityjson_file_to_graphs` emits vertex nodes (class 0, coords) followed by face nodes (classes 1–3, zero coords) with 3-class edges (off / vertex-vertex / vertex-face). `graph_to_cityjson` reconstructs each face from its vertex-face neighbors, ordering rings by walking the vertex-vertex cycle and orienting them CCW-viewed-from-outside. Model wiring is config-only (5 node / 3 edge classes as constructor args).

**Tech Stack:** Python, numpy, torch, Lightning, plotly (already in requirements), pytest.

## Global Constraints

- Node classes: `0=vertex, 1=ground-face, 2=roof-face, 3=wall-face, 4=off`. Edge classes: `0=off, 1=vertex-vertex, 2=vertex-face`. Constants live in `src/dataset/dataset.py`.
- Vertex coordinates are the only continuous features; face and off nodes have zero coords.
- No centering: `normalize_coords` flag kept, defaults `False` everywhere.
- `coord_scale` contract unchanged. `node_mask` now means "vertex node" (the coordinate-carrying nodes) — its two consumers (`_centre_positions`, coord metrics/scale) want exactly that set.
- `CityJSONDiffusionModule(num_node_classes=5, ..., num_edge_classes=3, ...)` — both constructor args; module-level `NUM_EDGE_CLASSES` constant removed.
- Conversion functions contain no geometry regularization (must stay exactly invertible); `straighten_face`/`regularize_building_geometry` remain as separate optional post-processing.
- Blast radius (all LOW, verified via gitnexus impact): parse→`CityJSONDataset.__init__` only; `graph_to_cityjson`→`diffusion.generate_cityjson` only; `compute_marginals`→`train`→`main`.

---

### Task 1: Levi parsing (dataset side) + parse tests

**Files:**
- Modify: `src/dataset/dataset.py` (constants, `parse_cityjson_file_to_graphs`, `_pad_graph`)
- Create: `tests/test_levi_roundtrip.py` (fixtures + parse tests; round-trip tests come in Task 2)

**Interfaces:**
- Produces constants: `VERTEX=0, GROUND=1, ROOF=2, WALL=3, OFF=4`, `NUM_NODE_CLASSES=5`, `NODE_CLASS_NAMES`, `SURFACE_TO_CLASS`, `EDGE_OFF=0, EDGE_VV=1, EDGE_VF=2`, `NUM_EDGE_CLASSES=3`.
- Produces raw graph dict: `{"id": str, "x": FloatTensor [N,3], "node_labels": LongTensor [N], "edge_index": LongTensor [2,E], "edge_attr": LongTensor [E], "type": str}` — vertices first (0..Nv-1), then face nodes in boundary order.
- Produces padded item: `x [n_max,3]`, `node_categories [n_max,5]` one-hot, `y [n_max,n_max,1]` int labels {0,1,2}, `node_mask [n_max]` 1 for vertex nodes.

- [ ] **Step 1: Write failing parse tests with synthetic fixtures**

`tests/test_levi_roundtrip.py`:

```python
"""Levi representation: parsing and the parse <-> graph_to_cityjson inverse pair."""
import json

import numpy as np
import pytest
import torch

from src.dataset.dataset import (
    EDGE_VF, EDGE_VV, GROUND, ROOF, VERTEX, WALL,
    parse_cityjson_file_to_graphs,
)

# ---------------------------------------------------------------- fixtures

CUBE_VERTICES = [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
]
# outward, CCW seen from outside
CUBE_FACES = [
    ([0, 3, 2, 1], "GroundSurface"),
    ([4, 5, 6, 7], "RoofSurface"),
    ([0, 1, 5, 4], "WallSurface"),
    ([1, 2, 6, 5], "WallSurface"),
    ([2, 3, 7, 6], "WallSurface"),
    ([3, 0, 4, 7], "WallSurface"),
]

# Gable-roof house: ridge along x, pentagon gable end walls.
HOUSE_VERTICES = [
    [0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0],
    [0, 0, 2], [4, 0, 2], [4, 3, 2], [0, 3, 2],
    [0, 1.5, 3], [4, 1.5, 3],
]
HOUSE_FACES = [
    ([0, 3, 2, 1], "GroundSurface"),
    ([0, 1, 5, 4], "WallSurface"),
    ([2, 3, 7, 6], "WallSurface"),
    ([0, 4, 8, 7, 3], "WallSurface"),   # x=0 gable pentagon
    ([1, 2, 6, 9, 5], "WallSurface"),   # x=4 gable pentagon
    ([4, 5, 9, 8], "RoofSurface"),
    ([8, 9, 6, 7], "RoofSurface"),
]


def make_cityjson(vertices, faces, with_semantics=True):
    surfaces = [{"type": "GroundSurface"}, {"type": "RoofSurface"}, {"type": "WallSurface"}]
    sem_map = {"GroundSurface": 0, "RoofSurface": 1, "WallSurface": 2}
    geometry = {
        "type": "Solid",
        "lod": "2",
        "boundaries": [[[ring] for ring, _ in faces]],
    }
    if with_semantics:
        geometry["semantics"] = {
            "surfaces": surfaces,
            "values": [[sem_map[s] for _, s in faces]],
        }
    return {
        "type": "CityJSON",
        "version": "1.1",
        "CityObjects": {"b1": {"type": "Building", "geometry": [geometry]}},
        "vertices": vertices,
    }


def write_and_parse(cj, tmp_path, name="b.city.json"):
    p = tmp_path / name
    p.write_text(json.dumps(cj), encoding="utf-8")
    graphs = parse_cityjson_file_to_graphs(p)
    assert len(graphs) == 1
    return next(iter(graphs.values()))


# ---------------------------------------------------------------- parse tests

def test_parse_cube_levi_structure(tmp_path):
    g = write_and_parse(make_cityjson(CUBE_VERTICES, CUBE_FACES), tmp_path)
    labels = g["node_labels"]

    assert g["x"].shape == (14, 3)              # 8 vertices + 6 faces
    assert (labels[:8] == VERTEX).all()
    assert (labels[8:] != VERTEX).all()
    assert (labels == GROUND).sum() == 1
    assert (labels == ROOF).sum() == 1
    assert (labels == WALL).sum() == 4
    # face nodes carry no coordinates
    assert torch.all(g["x"][8:] == 0)
    # vertex coords are raw metres, uncentered
    assert torch.allclose(
        g["x"][:8], torch.tensor(CUBE_VERTICES, dtype=torch.float32)
    )
    # cube: 12 undirected vv edges, 24 undirected vf edges, stored both directions
    assert (g["edge_attr"] == EDGE_VV).sum() == 24
    assert (g["edge_attr"] == EDGE_VF).sum() == 48
    # every face node touches exactly its ring's vertices
    ei, ea = g["edge_index"], g["edge_attr"]
    for f_i, (ring, _) in enumerate(CUBE_FACES):
        f_node = 8 + f_i
        members = sorted(ei[1][(ei[0] == f_node) & (ea == EDGE_VF)].tolist())
        assert members == sorted(ring)


def test_parse_infers_semantics_from_normals_when_absent(tmp_path):
    g = write_and_parse(
        make_cityjson(CUBE_VERTICES, CUBE_FACES, with_semantics=False), tmp_path
    )
    labels = g["node_labels"][8:]
    assert labels.tolist() == [GROUND, ROOF, WALL, WALL, WALL, WALL]


def test_pad_graph_shapes_and_masks(tmp_path):
    from src.dataset.dataset import NUM_NODE_CLASSES, OFF, CityJSONDataset

    g = write_and_parse(make_cityjson(CUBE_VERTICES, CUBE_FACES), tmp_path)
    ds = CityJSONDataset.__new__(CityJSONDataset)   # bypass folder scanning
    ds.n_max = 20
    item = ds._pad_graph(g)

    assert item["x"].shape == (20, 3)
    assert item["node_categories"].shape == (20, NUM_NODE_CLASSES)
    assert item["y"].shape == (20, 20, 1)
    assert item["node_categories"][:8].argmax(-1).tolist() == [VERTEX] * 8
    assert (item["node_categories"][14:].argmax(-1) == OFF).all()
    # node_mask marks coordinate-carrying (vertex) nodes only
    assert item["node_mask"].tolist() == [1.0] * 8 + [0.0] * 12
    # dense edge labels match the sparse representation
    ei, ea = g["edge_index"], g["edge_attr"]
    dense = item["y"].squeeze(-1)
    assert (dense[ei[0], ei[1]] == ea.float()).all()
    assert dense.sum() == ea.float().sum()          # nothing else set
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `python -m pytest tests/test_levi_roundtrip.py -v`
Expected: FAIL / ImportError (`EDGE_VF` etc. not defined).

- [ ] **Step 3: Implement constants + parse + `_pad_graph` in `src/dataset/dataset.py`**

Replace `NUM_NODE_CLASSES = 2 ...` block with:

```python
# Levi graph classes. Nodes: vertices carry coordinates; each face of the
# building becomes its own node whose class is its surface semantic; OFF pads.
VERTEX, GROUND, ROOF, WALL, OFF = 0, 1, 2, 3, 4
NUM_NODE_CLASSES = 5
NODE_CLASS_NAMES = ("Vertex", "GroundSurface", "RoofSurface", "WallSurface", "Off")
SURFACE_TO_CLASS = {"GroundSurface": GROUND, "RoofSurface": ROOF, "WallSurface": WALL}
# Edges: ring adjacency between vertices, membership between a face and its vertices.
EDGE_OFF, EDGE_VV, EDGE_VF = 0, 1, 2
NUM_EDGE_CLASSES = 3
```

Add helpers above `parse_cityjson_file_to_graphs`:

```python
def surface_class_from_normal(ring_coords):
    """Fallback face class when the file has no semantics.

    Same snap thresholds as `straighten_face`: |nz| > 0.9 horizontal
    (roof up / ground down), < 0.1 wall, sloped otherwise (roof if up).
    Relies on the CityJSON convention that exterior rings wind CCW seen
    from outside, i.e. outward normals.
    """
    n = np.zeros(3)
    for i in range(len(ring_coords)):
        n += np.cross(ring_coords[i], ring_coords[(i + 1) % len(ring_coords)])
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return WALL
    nz = n[2] / norm
    if abs(nz) > 0.9:
        return ROOF if nz > 0 else GROUND
    if abs(nz) < 0.1:
        return WALL
    return ROOF if nz > 0 else WALL


def _iter_faces(geom):
    """Yield (outer_ring, semantic_type_or_None) for each surface of a geometry."""
    boundaries = geom.get("boundaries", [])
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
```

Rewrite the per-object body of `parse_cityjson_file_to_graphs` (file loading and
transform handling stay as they are):

```python
def parse_cityjson_file_to_graphs(filepath, normalize_coords=False):
    """Parses a single CityJSON file into a dict of Levi building graphs.

    Node order: the building's vertices (class VERTEX, 3D coords) followed by
    one node per face outer ring (classes GROUND/ROOF/WALL, zero coords).
    Edges: EDGE_VV ring adjacency, EDGE_VF face membership; both directions.
    Face semantics come from the file, falling back to the face normal.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        cj = json.load(f)

    v_raw = np.array(cj["vertices"], dtype=float)
    if "transform" in cj:
        scale = np.array(cj["transform"]["scale"])
        translate = np.array(cj["transform"]["translate"])
        v_raw = v_raw * scale + translate

    graphs = {}
    for obj_id, city_obj in cj.get("CityObjects", {}).items():
        geom_list = city_obj.get("geometry", [])
        if not geom_list:
            continue

        faces = []  # (original-index ring, node class)
        for geom in geom_list:
            for ring, stype in _iter_faces(geom):
                cls = SURFACE_TO_CLASS.get(stype)
                if cls is None:
                    cls = surface_class_from_normal(v_raw[ring])
                faces.append((ring, cls))
        if not faces:
            continue

        active = sorted({vid for ring, _ in faces for vid in ring})
        idx_map = {old: new for new, old in enumerate(active)}
        n_vertices = len(active)

        coords = v_raw[active]
        if normalize_coords:
            coords = coords - get_base_center(geom_list, v_raw, set(active))

        x = np.zeros((n_vertices + len(faces), 3))
        x[:n_vertices] = coords
        node_labels = [VERTEX] * n_vertices + [cls for _, cls in faces]

        edges = {}
        for f_i, (ring, _) in enumerate(faces):
            f_node = n_vertices + f_i
            for k in range(len(ring)):
                u, v = idx_map[ring[k]], idx_map[ring[(k + 1) % len(ring)]]
                edges[(u, v)] = EDGE_VV
                edges[(v, u)] = EDGE_VV
                edges[(f_node, u)] = EDGE_VF
                edges[(u, f_node)] = EDGE_VF

        edge_index = np.array(list(edges.keys()), dtype=np.int64).T
        edge_attr = np.array(list(edges.values()), dtype=np.int64)

        graphs[obj_id] = {
            "id": obj_id,
            "x": torch.tensor(x, dtype=torch.float32),
            "node_labels": torch.tensor(node_labels, dtype=torch.long),
            "edge_index": torch.tensor(edge_index, dtype=torch.long),
            "edge_attr": torch.tensor(edge_attr, dtype=torch.long),
            "type": city_obj.get("type", "Unknown"),
        }
    return graphs
```

Rewrite `_pad_graph` (add `import torch.nn.functional as F` at the top of the file):

```python
    def _pad_graph(self, graph):
        """
        Pads a variable-size Levi graph to fixed N_max size as dense tensors.

        Returns a dict with:
            "x":               [N_max, 3]       — coords (zero for face/off nodes)
            "node_categories": [N_max, 5]       — one-hot over
                                                  (vertex, ground, roof, wall, off)
            "y":               [N_max, N_max, 1] — edge class labels
                                                  (0=off, 1=vertex-vertex, 2=vertex-face)
            "node_mask":       [N_max]           — 1 for vertex (coordinate-carrying)
                                                   nodes; consumed by coordinate
                                                   centring/metrics/scale only
            "id", "type":      str
        """
        x = graph["x"]
        N = x.size(0)
        N_max = self.n_max

        x_padded = torch.zeros((N_max, 3), dtype=torch.float32)
        x_padded[:N] = x

        labels_padded = torch.full((N_max,), OFF, dtype=torch.long)
        labels_padded[:N] = graph["node_labels"]
        node_categories = F.one_hot(labels_padded, NUM_NODE_CLASSES).float()

        node_mask = (labels_padded == VERTEX).float()

        y = torch.zeros((N_max, N_max, 1), dtype=torch.float32)
        edge_index = graph["edge_index"]
        if edge_index.numel() > 0:
            y[edge_index[0], edge_index[1], 0] = graph["edge_attr"].float()

        return {
            "x": x_padded,
            "node_categories": node_categories,
            "y": y,
            "node_mask": node_mask,
            "id": graph["id"],
            "type": graph["type"],
        }
```

Also update the `_pad_graph` reference in `__init__`'s node counting: `graph["x"].size(0)` still works (total node count) — no change needed there.

- [ ] **Step 4: Run tests, verify they pass**

Run: `python -m pytest tests/test_levi_roundtrip.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dataset/dataset.py tests/test_levi_roundtrip.py
git commit -m "feat: parse CityJSON into Levi graphs (vertex/face nodes, 3 edge classes)"
```

---

### Task 2: `graph_to_cityjson` inverse + round-trip tests

**Files:**
- Modify: `src/post_process/post_process.py` (rewrite `graph_to_cityjson`, add `_order_ring`/`_orient_outward`/`_newell_normal`, delete `find_cycles_dfs`)
- Modify: `src/post_process/__init__.py` (drop `find_cycles_dfs` from `__all__`/imports)
- Test: `tests/test_levi_roundtrip.py` (append)

**Interfaces:**
- Consumes: Task 1's constants and raw graph dict.
- Produces: `graph_to_cityjson(coords, node_classes, edge_classes, building_id="generated_building") -> dict` — numpy-compatible arrays `[N,3]` float, `[N]` int, `[N,N]` int; returns `{}` when no reconstructable face exists. Rings CCW viewed from outside.

- [ ] **Step 1: Append failing round-trip tests**

```python
# ---------------------------------------------------------------- round trip

from src.dataset.dataset import NUM_EDGE_CLASSES, NUM_NODE_CLASSES  # noqa: E402


def graph_to_dense(g):
    n = g["x"].shape[0]
    edge = np.zeros((n, n), dtype=np.int64)
    ei, ea = g["edge_index"].numpy(), g["edge_attr"].numpy()
    edge[ei[0], ei[1]] = ea
    return g["x"].numpy().astype(float), g["node_labels"].numpy(), edge


def canonical_graph(g):
    """Index-permutation-invariant view: coord-keyed vertices, faces, vv edges."""
    coords, labels, edge = graph_to_dense(g)
    key = lambda i: tuple(round(c, 6) for c in coords[i])
    vertex_ids = np.flatnonzero(labels == VERTEX)
    faces = []
    for f in np.flatnonzero(labels != VERTEX):
        members = frozenset(key(v) for v in vertex_ids if edge[f, v] == EDGE_VF)
        faces.append((int(labels[f]), members))
    vv = frozenset(
        frozenset((key(u), key(v)))
        for u in vertex_ids for v in vertex_ids
        if u < v and edge[u, v] == EDGE_VV
    )
    return frozenset(key(v) for v in vertex_ids), sorted(faces, key=repr), vv


def solid_volume(cj):
    """Signed volume via divergence theorem; positive iff rings are CCW-outward."""
    obj = next(iter(cj["CityObjects"].values()))
    verts = np.asarray(cj["vertices"], dtype=float)
    vol = 0.0
    for face in obj["geometry"][0]["boundaries"][0]:
        ring = verts[face[0]]
        for i in range(1, len(ring) - 1):
            vol += np.dot(ring[0], np.cross(ring[i], ring[i + 1]))
    return vol / 6.0


@pytest.mark.parametrize(
    "vertices, faces, volume",
    [(CUBE_VERTICES, CUBE_FACES, 1.0), (HOUSE_VERTICES, HOUSE_FACES, 30.0)],
    ids=["cube", "gable-house"],
)
def test_round_trip_is_identity_and_ccw_outward(tmp_path, vertices, faces, volume):
    from src.post_process.post_process import graph_to_cityjson

    g1 = write_and_parse(make_cityjson(vertices, faces), tmp_path)
    cj2 = graph_to_cityjson(*graph_to_dense(g1), building_id="b1")

    # parse o to_cityjson o parse == parse  (graph level)
    g2 = write_and_parse(cj2, tmp_path, name="roundtrip.city.json")
    assert canonical_graph(g2) == canonical_graph(g1)

    # to_cityjson o parse is the identity on its own output  (file level)
    cj3 = graph_to_cityjson(*graph_to_dense(g2), building_id="b1")
    assert cj3 == cj2

    # rings are CCW viewed from outside: positive enclosed volume, exact value
    assert solid_volume(cj2) == pytest.approx(volume)


def test_graph_to_cityjson_skips_malformed_faces():
    from src.post_process.post_process import graph_to_cityjson

    # a lone face node connected to only 2 vertices cannot form a ring
    coords = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]])
    labels = np.array([VERTEX, VERTEX, VERTEX, WALL])
    edge = np.zeros((4, 4), dtype=np.int64)
    edge[3, 0] = edge[0, 3] = EDGE_VF
    edge[3, 1] = edge[1, 3] = EDGE_VF
    assert graph_to_cityjson(coords, labels, edge) == {}
```

- [ ] **Step 2: Run tests, verify the new ones fail**

Run: `python -m pytest tests/test_levi_roundtrip.py -v`
Expected: Task 1 tests PASS; round-trip tests FAIL (old `graph_to_cityjson` signature takes `edge_probs`/`threshold`).

- [ ] **Step 3: Rewrite `src/post_process/post_process.py`**

Delete `find_cycles_dfs` (and its section header). Replace the exporter section:

```python
from src.dataset.dataset import EDGE_VF, EDGE_VV, GROUND, ROOF, VERTEX, WALL

# CityJSON semantic surface table used by graph_to_cityjson
_SEMANTIC_SURFACES = [
    {"type": "GroundSurface"}, {"type": "RoofSurface"}, {"type": "WallSurface"}
]
_CLASS_TO_SEMANTIC = {GROUND: 0, ROOF: 1, WALL: 2}


def _newell_normal(pts):
    """Area vector of a closed polygon (origin-independent), 2x the face normal."""
    n = np.zeros(3)
    for i in range(len(pts)):
        n += np.cross(pts[i], pts[(i + 1) % len(pts)])
    return n


def _order_ring(members, vv_adj, coords):
    """Recover the boundary order of a face's vertices.

    Walks the vertex-vertex cycle restricted to the member set. When that is not
    a clean cycle (malformed generated graph, or a chord contributed by another
    face), falls back to sorting by angle on the best-fit plane.
    """
    mset = set(members)
    nbrs = {v: [u for u in vv_adj.get(v, []) if u in mset] for v in members}
    if all(len(ns) == 2 for ns in nbrs.values()):
        ring = [members[0]]
        prev, cur = None, members[0]
        while len(ring) < len(members):
            a, b = nbrs[cur]
            nxt = b if a == prev else a
            if nxt == ring[0]:
                break
            ring.append(nxt)
            prev, cur = cur, nxt
        if len(ring) == len(members) and ring[0] in nbrs[ring[-1]]:
            return ring
    # ponytail: angle sort assumes a star-shaped ring; wrong for strongly
    # non-convex faces -- upgrade to a cycle search on the subgraph if those appear.
    pts = coords[members]
    centered = pts - pts.mean(axis=0)
    _, vecs = np.linalg.eigh(centered.T @ centered)
    a1 = vecs[:, 2]
    a2 = np.cross(vecs[:, 0], a1)
    ang = np.arctan2(centered @ a2, centered @ a1)
    return [m for _, m in sorted(zip(ang, members))]


def _orient_outward(ring, coords, centroid):
    """Flip the ring if its normal points toward the building centroid.

    ponytail: centroid heuristic; can misjudge faces of strongly concave
    buildings -- switch to ray casting if that shows up in generated output.
    """
    pts = coords[ring]
    if np.dot(_newell_normal(pts), pts.mean(axis=0) - centroid) < 0:
        ring = ring[::-1]
    return ring


def graph_to_cityjson(coords, node_classes, edge_classes, building_id="generated_building"):
    """Converts a Levi graph back into a CityJSON dictionary.

    Exact inverse of `parse_cityjson_file_to_graphs` (no regularization here;
    apply `regularize_building_geometry` separately to generated output).
    Face rings are ordered CCW viewed from outside, i.e. outward normals.

    Args:
        coords: [N, 3] float array, metres. Only vertex-node rows are read.
        node_classes: [N] int array of node class labels.
        edge_classes: [N, N] int array of edge class labels (symmetric).
    """
    coords = np.asarray(coords, dtype=float)
    node_classes = np.asarray(node_classes)
    edge_classes = np.asarray(edge_classes)

    vertex_ids = np.flatnonzero(node_classes == VERTEX)
    face_ids = np.flatnonzero(np.isin(node_classes, (GROUND, ROOF, WALL)))
    if len(vertex_ids) < 3 or len(face_ids) == 0:
        logger.warning("Graph has %d vertices and %d faces; cannot build a geometry.",
                       len(vertex_ids), len(face_ids))
        return {}

    vid_map = {int(g): i for i, g in enumerate(vertex_ids)}
    vv_adj = {
        int(v): [int(u) for u in vertex_ids if u != v and edge_classes[v, u] == EDGE_VV]
        for v in vertex_ids
    }
    centroid = coords[vertex_ids].mean(axis=0)

    cj_faces, sem_values = [], []
    for f in face_ids:
        members = [int(v) for v in vertex_ids if edge_classes[f, v] == EDGE_VF]
        if len(members) < 3:
            logger.warning("Face node %d has only %d member vertices, skipping.",
                           f, len(members))
            continue
        ring = _order_ring(members, vv_adj, coords)
        ring = _orient_outward(ring, coords, centroid)
        cj_faces.append([[vid_map[v] for v in ring]])
        sem_values.append(_CLASS_TO_SEMANTIC[int(node_classes[f])])

    if not cj_faces:
        logger.warning("No reconstructable faces in graph.")
        return {}

    return {
        "type": "CityJSON",
        "version": "1.1",
        "CityObjects": {
            building_id: {
                "type": "Building",
                "geometry": [
                    {
                        "type": "Solid",
                        "lod": "2",
                        "boundaries": [cj_faces],
                        "semantics": {
                            "surfaces": _SEMANTIC_SURFACES,
                            "values": [sem_values],
                        },
                    }
                ],
            }
        },
        "vertices": coords[vertex_ids].tolist(),
        "metadata": {"datasetLod": "2"},
    }
```

Update `src/post_process/__init__.py` to stop importing/exporting `find_cycles_dfs`.

- [ ] **Step 4: Run tests, verify they pass**

Run: `python -m pytest tests/test_levi_roundtrip.py -v`
Expected: all PASS (cube and house round trips exact, volumes 1.0 / 30.0).

- [ ] **Step 5: Commit**

```bash
git add src/post_process/post_process.py src/post_process/__init__.py tests/test_levi_roundtrip.py
git commit -m "feat: graph_to_cityjson as exact Levi inverse with CCW-outward rings"
```

---

### Task 3: Model/config wiring

**Files:**
- Modify: `src/models/diffusion.py` (ctor args, `_prepare`, `_shared_step`, `sample`, `generate_cityjson`, docstrings)
- Modify: `src/dataset/datamodule.py` (`compute_marginals` 3 edge classes)
- Modify: `src/utils/config.py` (`num_node_classes=5`, new `num_edge_classes=3`, `normalize_coords=False`, drop `edge_threshold`)
- Modify: `src/utils/setup_utils.py` (pass `num_edge_classes`)
- Modify: `src/inference.py` (drop threshold)
- Modify: `configs/default.yaml`, `configs/train.yaml`, `configs/inference.yaml`
- Modify: `tests/test_coord_scale.py`, `tests/test_numerical_stability.py` (fixtures to explicit class counts / new contracts)

**Interfaces:**
- Consumes: `graph_to_cityjson(coords, node_classes, edge_classes, building_id)` from Task 2; `VERTEX`, `NUM_EDGE_CLASSES` from Task 1.
- Produces: `CityJSONDiffusionModule(num_node_classes=5, ..., num_edge_classes=3, ...)`; `sample()` returns `(pos [B,N,3], node_labels [B,N] long, edge_labels [B,N,N] long)`; `generate_cityjson(batch_size=1)` (no threshold).

- [ ] **Step 1: Update the two existing test files (failing against current code is fine — they pin the new contract)**

`tests/test_coord_scale.py`:
- Replace the import line with `from src.models.diffusion import CityJSONDiffusionModule`.
- In `_batch`, replace `NUM_EDGE_CLASSES` with the literal `2`.
- In `_model`, pass `num_node_classes=2, num_edge_classes=2` explicitly (the fixtures are 2-class; the regression is about coordinate scaling, not classes).
- Replace `test_generate_cityjson_restores_metres` body:

```python
def test_generate_cityjson_restores_metres(monkeypatch):
    """The reverse chain runs in scaled units; CityJSON output must be in metres."""
    import src.post_process.post_process as pp

    scale = 5.0
    model = _model(coord_scale=scale)

    pos = torch.arange(4 * 3, dtype=torch.float32).reshape(1, 4, 3)
    node_labels = torch.zeros(1, 4, dtype=torch.long)      # all vertices
    edge_labels = torch.zeros(1, 4, 4, dtype=torch.long)
    monkeypatch.setattr(model, "sample",
                        lambda batch_size=1: (pos, node_labels, edge_labels))

    seen = {}

    def fake_graph_to_cityjson(coords, node_classes, edge_classes, building_id):
        seen["coords"] = coords
        return {"type": "CityJSON"}

    monkeypatch.setattr(pp, "graph_to_cityjson", fake_graph_to_cityjson)

    results = model.generate_cityjson(batch_size=1)

    assert len(results) == 1
    assert torch.allclose(torch.as_tensor(seen["coords"]), pos[0] * scale, atol=1e-6)
```

`tests/test_numerical_stability.py`: in `_drifted_model`, add `num_node_classes=2, num_edge_classes=2,` to the `CityJSONDiffusionModule(...)` call (fixtures stay 2-class).

- [ ] **Step 2: Edit `src/models/diffusion.py`**

- Delete `NUM_EDGE_CLASSES = 2  # 0 = no edge, 1 = edge`.
- Signature: `def __init__(self, num_node_classes=5, num_edge_classes=3, hidden_dim=64, ...)` and add `self.num_edge_classes = num_edge_classes` beside `self.num_node_classes`. Docstring: `e_marginals ... [num_edge_classes]`.
- Replace the three other uses: `e_marginals = torch.ones(num_edge_classes) / num_edge_classes`; `num_edge_classes=num_edge_classes` in the `rEGNNTransformer(...)` call; `F.one_hot(batch["y"].squeeze(-1).long(), self.num_edge_classes)` in `_prepare`; `E_pred[:, off_diag].reshape(-1, self.num_edge_classes)` in `_shared_step`.
- Module docstring bullet 1 becomes: nodes are vertex/ground/roof/wall/off classes of the Levi graph; edges are off / vertex-vertex / vertex-face. `node_mask` marks vertex (coordinate-carrying) nodes and is only used for coordinate centring and metrics.
- `_eval_step` comment: the `== 0` class is now the vertex class (metric names unchanged).
- `sample()` tail:

```python
        # The chain's final state is the sample; do not re-read the network head.
        node_labels = X_t.argmax(dim=-1)
        edge_labels = E_t.argmax(dim=-1)
        return pos, node_labels, edge_labels
```

with docstring: returns `(pos, node_labels [B,n_max] long, edge_labels [B,n_max,n_max] long)`.
- `generate_cityjson`:

```python
    @torch.no_grad()
    def generate_cityjson(self, batch_size=1):
        """
        Generates buildings via diffusion sampling and exports each as a CityJSON dict.
        """
        from src.dataset.dataset import VERTEX
        from src.post_process.post_process import graph_to_cityjson

        pos, node_labels, edge_labels = self.sample(batch_size=batch_size)

        results = []
        for i in range(batch_size):
            n_vertices = int((node_labels[i] == VERTEX).sum())
            if n_vertices < 3:
                logger.warning(f"Building {i} has only {n_vertices} vertex nodes, skipping.")
                continue

            # The chain runs in scaled units; CityJSON is metres.
            coords = (pos[i] * self.coord_scale).cpu().numpy()
            cj = graph_to_cityjson(
                coords,
                node_labels[i].cpu().numpy(),
                edge_labels[i].cpu().numpy(),
                building_id=f"generated_building_{i}",
            )
            if cj:
                results.append(cj)
        return results
```

- [ ] **Step 3: Edit the surrounding wiring**

- `src/dataset/datamodule.py` `compute_marginals`: import `NUM_EDGE_CLASSES` from `src.dataset.dataset`; `edge_counts = torch.zeros(NUM_EDGE_CLASSES, dtype=torch.float64)`; replace the two counting lines with

```python
            edges = adjacency[off_diag].long()
            edge_counts += torch.bincount(edges, minlength=NUM_EDGE_CLASSES).double()
```

  Docstring: node marginals over (vertex, ground, roof, wall, off), edge marginals `[3]` over (off, vertex-vertex, vertex-face).
- `src/utils/config.py`: `num_node_classes: int = 5`, add `num_edge_classes: int = 3` below it, `normalize_coords: bool = False`, delete `edge_threshold` from `InferenceConfig`.
- `src/utils/setup_utils.py` `create_model`: add `num_edge_classes=cfg.model.num_edge_classes,` after `num_node_classes`.
- `src/inference.py`: delete the `threshold = cfg.inference.edge_threshold` line; call `model.generate_cityjson(batch_size=batch_size)`.
- `configs/*.yaml` (all three): `normalize_coords: false`; where a `model:` section exists set `num_node_classes: 5` and add `num_edge_classes: 3`; remove `edge_threshold` lines.

- [ ] **Step 4: Run the whole suite**

Run: `python -m pytest tests/ -v`
Expected: all PASS (levi round-trip, coord scale, numerical stability).

- [ ] **Step 5: Commit**

```bash
git add src/models/diffusion.py src/dataset/datamodule.py src/utils/config.py src/utils/setup_utils.py src/inference.py configs/ tests/test_coord_scale.py tests/test_numerical_stability.py
git commit -m "feat: wire 5 node / 3 edge classes through model, config and marginals"
```

---

### Task 4: Plotly Levi visualization script

**Files:**
- Create: `src/visualize_levi.py`

**Interfaces:**
- Consumes: `parse_cityjson_file_to_graphs`, constants from Task 1.
- Exploratory/throwaway script — exempt from TDD and reproducibility requirements (per project instructions); says so in its docstring.

- [ ] **Step 1: Load the dataviz skill** (required before chart code), then write the script: walk a LOD folder, parse files until ~3 graphs with 30–50 vertex nodes are found, and emit one interactive HTML per graph to `outputs/levi_viz/`. Vertex nodes at their coordinates (hover: class + xyz), face nodes at the centroid of their member vertices (display position only; hover: class name), vertex-vertex and vertex-face edges as distinct line traces with invisible mid-point hover markers carrying the edge class. CLI: `python -m src.visualize_levi --dataset-dir "data/The Hague" --lod 2 --n 3 --min-v 30 --max-v 50`.

- [ ] **Step 2: Smoke-run it** on `data/The Hague` LOD2 and confirm HTML files exist and open.

- [ ] **Step 3: Commit**

```bash
git add src/visualize_levi.py
git commit -m "feat: interactive Plotly visualization of Levi graphs"
```

---

### Task 5: Verification & closeout

- [ ] Run `python -m pytest tests/ -v` — all green.
- [ ] Run gitnexus `detect_changes()` and confirm only expected symbols/flows changed.
- [ ] Reindex gitnexus (`node .gitnexus/run.cjs analyze`) since symbols were added/removed.
- [ ] Note session decisions for memsearch (5-class choice, node_mask semantic change, inverse-property test design).
