"""CityJSON -> mesh parser: geometry it cannot read must not vanish silently."""
import json
import logging

from src.dataset.mesh_dataset import parse_cityjson_file_to_meshes

VERTS = [
    [0, 0, 100], [10, 0, 100], [10, 10, 100], [0, 10, 100],
]


def test_unsupported_geometry_type_is_logged(tmp_path, caplog):
    cj = {
        "type": "CityJSON", "version": "1.1", "vertices": VERTS,
        "CityObjects": {"b1": {"type": "Building", "geometry": [
            {"type": "CompositeSolid", "lod": "2.2",
             "boundaries": [[[[[0, 1, 2, 3]]]]]}]}},
    }
    p = tmp_path / "u.json"
    p.write_text(json.dumps(cj), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        parse_cityjson_file_to_meshes(str(p))
    assert "CompositeSolid" in caplog.text
