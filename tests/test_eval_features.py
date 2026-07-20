"""Core building geometric feature extraction on synthetic CityJSON fixtures."""
import numpy as np

from src.eval.building_features import building_features, feature_matrix, mesh_from_cityjson

# unit cube as CityJSON (6 quad faces, CCW-outward)
CUBE = {
    "type": "CityJSON", "version": "1.1",
    "CityObjects": {"b": {"type": "Building", "geometry": [{
        "type": "Solid", "lod": "2",
        "boundaries": [[
            [[0, 3, 2, 1]], [[4, 5, 6, 7]], [[0, 1, 5, 4]],
            [[1, 2, 6, 5]], [[2, 3, 7, 6]], [[3, 0, 4, 7]],
        ]],
    }]}},
    "vertices": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                 [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
}


def test_mesh_extraction():
    verts, faces = mesh_from_cityjson(CUBE)
    assert verts.shape == (8, 3)
    assert len(faces) == 6


def test_core_features_of_unit_cube():
    f = building_features(CUBE)
    assert f["num_vertices"] == 8
    assert f["num_faces"] == 6
    assert np.isclose(f["volume"], 1.0)
    assert np.isclose(f["area"], 6.0)
    assert np.isclose(f["height_diff"], 1.0)
    assert np.isclose(f["convex_hull_area"], 6.0)


def test_feature_matrix_stacks():
    X, names = feature_matrix([CUBE, CUBE])
    assert X.shape[0] == 2 and X.shape[1] == len(names)
    assert "volume" in names
