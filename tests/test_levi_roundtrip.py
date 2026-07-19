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
