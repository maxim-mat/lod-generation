"""CityJSON -> Levi parser: semantic mapping and coordinate normalisation.

Pins the two places the parser used to lose information silently -- surface
classes it had no mapping for, and a base centre computed over rings the graph
never contains.
"""
import logging

import numpy as np
import pytest
import torch

from src.dataset.dataset import (
    GROUND, ROOF, WALL, SURFACE_TO_CLASS, parse_cityjson_file_to_graphs,
)

# A 10x10 footprint at z=100 with an off-centre 1..3 courtyard hole, walls and
# a roof. The hole is deliberately not concentric: a base centre that averages
# hole vertices lands at (3.5, 3.5) instead of the outer ring's (5, 5).
VERTS = [
    [0, 0, 100], [10, 0, 100], [10, 10, 100], [0, 10, 100],      # 0-3 footprint
    [1, 1, 100], [3, 1, 100], [3, 3, 100], [1, 3, 100],          # 4-7 hole
    [0, 0, 110], [10, 0, 110], [10, 10, 110], [0, 10, 110],      # 8-11 eaves
]


def _cj(surface_types, tmp_path, name="t.json"):
    """One Solid: ground (with hole), roof, and four walls."""
    faces = [
        [[0, 3, 2, 1], [4, 5, 6, 7]],        # ground, outer + hole
        [[8, 9, 10, 11]],                     # roof
        [[0, 1, 9, 8]], [[1, 2, 10, 9]],
        [[2, 3, 11, 10]], [[3, 0, 8, 11]],
    ]
    cj = {
        "type": "CityJSON", "version": "1.1", "vertices": VERTS,
        "CityObjects": {"b1": {"type": "Building", "geometry": [{
            "type": "Solid", "lod": "2.2", "boundaries": [faces],
            "semantics": {"surfaces": [{"type": t} for t in surface_types],
                          "values": [list(range(len(surface_types)))]},
        }]}},
    }
    import json
    p = tmp_path / name
    p.write_text(json.dumps(cj), encoding="utf-8")
    return p


DEFAULT = ("GroundSurface", "RoofSurface", "WallSurface",
           "WallSurface", "WallSurface", "WallSurface")


def _classes(path, normalize=False):
    graphs = parse_cityjson_file_to_graphs(str(path), normalize_coords=normalize)
    g = graphs["b1"]
    labels = g["node_labels"].tolist()
    n_vert = labels.count(0)
    return labels[n_vert:], g, n_vert


# --- semantic mapping -----------------------------------------------------

def test_outer_floor_surface_maps_to_roof(tmp_path):
    types = list(DEFAULT)
    types[2] = "OuterFloorSurface"
    face_labels, _, _ = _classes(_cj(types, tmp_path))
    assert face_labels[2] == ROOF


def test_outer_ceiling_surface_maps_to_wall_not_ground(tmp_path):
    """A downward overhang underside must not be labelled GROUND.

    The normal-based fallback returns GROUND for it, which puts a ground node
    at height and breaks the "ground is the base at z~0" invariant.
    """
    types = list(DEFAULT)
    types[0] = "OuterCeilingSurface"           # face 0 points straight down
    face_labels, _, _ = _classes(_cj(types, tmp_path))
    assert face_labels[0] == WALL
    assert face_labels[0] != GROUND            # what the normal fallback returns


def test_every_citygml_boundary_class_has_an_explicit_mapping():
    for name in ("GroundSurface", "RoofSurface", "WallSurface",
                 "OuterFloorSurface", "OuterCeilingSurface"):
        assert name in SURFACE_TO_CLASS


def test_unknown_label_still_falls_back_to_the_face_normal(tmp_path):
    types = list(DEFAULT)
    types[1] = "Nonsense"                      # the upward roof face
    face_labels, _, _ = _classes(_cj(types, tmp_path))
    assert face_labels[1] == ROOF


# --- normalisation --------------------------------------------------------

def test_normalize_levels_ground_to_zero_and_centres_the_outer_ring(tmp_path):
    _, g, n_vert = _classes(_cj(DEFAULT, tmp_path), normalize=True)
    coords = g["x"][:n_vert].numpy()
    # the graph holds outer rings only, so vertices 4-7 never appear
    assert n_vert == 8
    footprint = coords[coords[:, 2] < 5.0]      # the base ring, now near z=0
    assert np.allclose(footprint[:, 2], 0.0, atol=1e-6)
    assert np.allclose(footprint[:, :2].mean(axis=0), 0.0, atol=1e-6)


def test_normalize_is_a_no_op_without_the_flag(tmp_path):
    _, g, n_vert = _classes(_cj(DEFAULT, tmp_path), normalize=False)
    assert g["x"][:n_vert, 2].min() == pytest.approx(100.0)


# --- silent drops now announce themselves ---------------------------------

def test_unsupported_geometry_type_is_logged(tmp_path, caplog):
    import json
    cj = {
        "type": "CityJSON", "version": "1.1", "vertices": VERTS,
        "CityObjects": {"b1": {"type": "Building", "geometry": [
            {"type": "CompositeSolid", "lod": "2.2",
             "boundaries": [[[[[0, 1, 2, 3]]]]]}]}},
    }
    p = tmp_path / "u.json"
    p.write_text(json.dumps(cj), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        parse_cityjson_file_to_graphs(str(p))
    assert "CompositeSolid" in caplog.text


def test_discarded_inner_rings_are_reported(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        parse_cityjson_file_to_graphs(str(_cj(DEFAULT, tmp_path)))
    assert "inner ring" in caplog.text.lower()
