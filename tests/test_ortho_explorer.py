"""Pure helpers of the orthophoto explorer (the Streamlit UI itself is not tested)."""
import numpy as np
import pytest
import torch

from src.ortho_explorer import (footprint_rings, map_deck, north_up, to_lonlat, to_uv,
                                window_corners)

QUADS = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
VERTS = np.array([[0, 0, 0], [10, 0, 0], [10, 8, 0], [0, 8, 0],
                  [0, 0, 6], [10, 0, 6], [10, 8, 6], [0, 8, 6]], float) + [81000, 455000, 1]


def _box(semantics=None):
    geom = {"type": "Solid", "lod": "2.2", "boundaries": [[[q] for q in QUADS]]}
    if semantics is not None:
        geom["semantics"] = semantics
    return {"type": "Building", "geometry": [geom]}


def test_footprint_is_the_semantic_ground_surface():
    sem = {"surfaces": [{"type": "GroundSurface"}, {"type": "RoofSurface"},
                        {"type": "WallSurface"}],
           "values": [[0, 1, 2, 2, 2, 2]]}
    rings = footprint_rings(_box(sem), VERTS)
    assert len(rings) == 1
    np.testing.assert_allclose(rings[0], VERTS[[0, 3, 2, 1], :2])


def test_footprint_falls_back_to_the_lowest_faces_without_semantics():
    rings = footprint_rings(_box(), VERTS)
    assert len(rings) == 1
    np.testing.assert_allclose(rings[0], VERTS[[0, 3, 2, 1], :2])


def test_uv_is_the_dataset_frame():
    uv = to_uv(np.array([[81005.0, 455004.0]]), center=np.array([81005.0, 455004.0, 4.0]),
               scale=np.array([10.0, 10.0, 11.0]))
    np.testing.assert_allclose(uv, [[0.0, 0.0]])


def test_lonlat_lands_in_the_hague():
    lon, lat = to_lonlat(np.array([[81500.0, 455500.0]]))[0]
    assert 4.30 < lon < 4.33 and 52.07 < lat < 52.10


def test_window_corners_are_bl_tl_tr_br():
    bl, tl, tr, br = window_corners((81000.0, 455000.0, 81100.0, 455100.0))
    assert tl[1] > bl[1] and tr[0] > tl[0] and br[1] < tr[1] and br[0] > bl[0]


def test_north_up_puts_the_largest_y_row_on_top():
    crop = torch.zeros((3, 4, 4), dtype=torch.uint8)
    crop[:, -1, :] = 255                        # mesh-order crop: last row is +y (north)
    img = north_up(crop)
    assert img.shape == (4, 4, 3)
    assert (img[0] == 255).all() and not img[1:].any()


def test_map_overlay_image_reaches_deckgl_as_a_literal_not_an_expression():
    """pydeck sends unquoted strings to deck.gl as JS expressions ("@@="); a data
    URL parsed that way fails in the browser with 'Unexpected ":"'."""
    import json

    ring = VERTS[[0, 3, 2, 1], :2]
    deck = map_deck([ring], (81000.0, 455000.0, 81010.0, 455010.0),
                    np.zeros((8, 8, 3), np.uint8), overlay=True, opacity=0.5)
    layers = {l["@@type"]: l for l in json.loads(deck.to_json())["layers"]}
    image = layers["BitmapLayer"]["image"]
    assert image.startswith("data:image/png;base64,")
    assert layers["PolygonLayer"]["getPolygon"] == "@@=polygon"     # accessor stays an accessor
