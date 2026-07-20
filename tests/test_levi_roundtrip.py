"""Levi representation: parsing and the parse <-> graph_to_cityjson inverse pair."""
import json
import shutil
from pathlib import Path

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
    # face nodes sit at their ring centroid
    expected_centroids = torch.tensor(
        np.array([np.mean([CUBE_VERTICES[v] for v in ring], axis=0)
                  for ring, _ in CUBE_FACES]),
        dtype=torch.float32,
    )
    assert torch.allclose(g["x"][8:], expected_centroids)
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


# ---------------------------------------------------------------- round trip

def graph_to_dense(g):
    n = g["x"].shape[0]
    edge = np.zeros((n, n), dtype=np.int64)
    ei, ea = g["edge_index"].numpy(), g["edge_attr"].numpy()
    edge[ei[0], ei[1]] = ea
    return g["x"].numpy().astype(float), g["node_labels"].numpy(), edge


def canonical_graph(g):
    """Index-permutation-invariant view: coord-keyed vertices, faces, vv edges."""
    coords, labels, edge = graph_to_dense(g)
    key = lambda i: tuple(round(c, 6) for c in coords[i])  # noqa: E731
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


# ---------------------------------------------------------------- real-fixture round trip

# tests/fixtures/real_lod2_building.city.json is HAND-AUTHORED: a rectangular
# 8x6m footprint, 3m eaves, gable roof to 5m ridge, at a realistic metric
# offset (tens of metres) -- not captured survey data. It stays skippable so
# a genuine dataset building can be dropped in at this path later without
# touching the test.
REAL = Path(__file__).parent / "fixtures" / "real_lod2_building.city.json"


@pytest.mark.skipif(not REAL.exists(), reason="real LoD2 fixture absent")
def test_real_lod2_round_trip_is_identity(tmp_path):
    from src.post_process.post_process import graph_to_cityjson
    g1 = write_and_parse(json.loads(REAL.read_text()), tmp_path, name="real_in.city.json")
    cj2 = graph_to_cityjson(*graph_to_dense(g1), building_id="b1")
    g2 = write_and_parse(cj2, tmp_path, name="real_rt.city.json")
    assert canonical_graph(g2) == canonical_graph(g1)


# ---------------------------------------------------------------- non-convex order recovery

# L-shaped (concave) floor face; convex angular sort would cross-link it.
L_VERTICES = [
    [0, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0],
    [0, 0, 1], [2, 0, 1], [2, 1, 1], [1, 1, 1], [1, 2, 1], [0, 2, 1],
]
L_FACES = [
    ([0, 5, 4, 3, 2, 1], "GroundSurface"),
    ([6, 7, 8, 9, 10, 11], "RoofSurface"),
    ([0, 1, 7, 6], "WallSurface"), ([1, 2, 8, 7], "WallSurface"),
    ([2, 3, 9, 8], "WallSurface"), ([3, 4, 10, 9], "WallSurface"),
    ([4, 5, 11, 10], "WallSurface"), ([5, 0, 6, 11], "WallSurface"),
]


def test_non_convex_face_order_recovered(tmp_path):
    from src.post_process.post_process import graph_to_cityjson
    g1 = write_and_parse(make_cityjson(L_VERTICES, L_FACES), tmp_path, name="L.city.json")
    cj2 = graph_to_cityjson(*graph_to_dense(g1), building_id="b1")
    g2 = write_and_parse(cj2, tmp_path, name="L_rt.city.json")
    assert canonical_graph(g2) == canonical_graph(g1)
    assert solid_volume(cj2) == pytest.approx(3.0)  # L-prism volume


# ---------------------------------------------------------------- adversarial generated topology

def test_broken_cycle_falls_back_without_crash():
    from src.post_process.post_process import graph_to_cityjson
    # 4 vertices in a face but vv edges form a path, not a cycle -> angle-sort fallback
    coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0]])
    labels = np.array([VERTEX, VERTEX, VERTEX, VERTEX, WALL])
    edge = np.zeros((5, 5), dtype=np.int64)
    for v in range(4):
        edge[4, v] = edge[v, 4] = EDGE_VF
    for a, b in [(0, 1), (1, 2), (2, 3)]:  # open path, no closing edge
        edge[a, b] = edge[b, a] = EDGE_VV
    cj = graph_to_cityjson(coords, labels, edge)
    assert cj["CityObjects"]  # produced a face via fallback, did not crash


def test_disconnected_graph_returns_empty():
    from src.post_process.post_process import graph_to_cityjson
    coords = np.array([[0, 0, 0], [1, 0, 0]])
    labels = np.array([VERTEX, VERTEX])          # <3 vertices, no faces
    edge = np.zeros((2, 2), dtype=np.int64)
    assert graph_to_cityjson(coords, labels, edge) == {}


# ---------------------------------------------------------------- gated val3dity check

@pytest.mark.skipif(shutil.which("val3dity") is None, reason="val3dity not on PATH")
def test_converted_cube_is_val3dity_valid(tmp_path):
    import subprocess
    from src.post_process.post_process import graph_to_cityjson, save_to_file
    g1 = write_and_parse(make_cityjson(CUBE_VERTICES, CUBE_FACES), tmp_path)
    cj2 = graph_to_cityjson(*graph_to_dense(g1), building_id="b1")
    out = tmp_path / "cube.city.json"
    save_to_file(cj2, out)
    proc = subprocess.run(["val3dity", str(out), "--report"], capture_output=True, text=True)
    assert '"validity": true' in proc.stdout or proc.returncode == 0
