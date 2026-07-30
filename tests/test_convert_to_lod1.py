"""Geometric correctness tests for ``convert_to_lod1``.

The LOD1 solid must be a closed, consistently outward-oriented shell -- an
extrusion of the LOD2 footprint to the median roof height. Hand-built solids
only, never the real dataset.
"""
from collections import Counter

import numpy as np
import pytest

from src.geometry.geometry import convert_to_lod1


# --- helpers --------------------------------------------------------------

def outer_rings(solid):
    return [ring[0] for shell in solid["boundaries"] for ring in shell]


def all_rings(solid):
    return [ring for shell in solid["boundaries"] for face in shell for ring in face]


def semantic_types(solid):
    types = [s["type"] for s in solid["semantics"]["surfaces"]]
    return [types[i] if i is not None and i < len(types) else None
            for shell in solid["semantics"]["values"] for i in shell]


def signed_volume(solid, verts):
    """Divergence theorem; positive iff the shell is outward-oriented.

    Per face, Newell's area vector summed over the outer ring *and* its holes
    gives 2 * A_net * n, with oppositely wound holes subtracting themselves,
    so V = 1/3 * sum (p . n) A_net collapses to 1/6 * sum (p . N).
    """
    v = np.asarray(verts, dtype=float)
    total = 0.0
    for shell in solid["boundaries"]:
        for face in shell:
            area_vec = np.zeros(3)
            for ring in face:
                for i in range(len(ring)):
                    area_vec += np.cross(v[ring[i]], v[ring[(i + 1) % len(ring)]])
            total += np.dot(v[face[0][0]], area_vec) / 6.0
    return total


def newell_normal(ring, verts):
    v = np.asarray(verts, dtype=float)
    n = np.zeros(3)
    for i in range(len(ring)):
        n += np.cross(v[ring[i]], v[ring[(i + 1) % len(ring)]])
    return n / (np.linalg.norm(n) + 1e-12)


def edge_census(solid):
    """(directed edges used more than once, directed edges lacking their twin)."""
    used = Counter()
    for ring in all_rings(solid):
        for i in range(len(ring)):
            used[(ring[i], ring[(i + 1) % len(ring)])] += 1
    return (sum(1 for n in used.values() if n > 1),
            sum(1 for e in used if (e[1], e[0]) not in used))


def gabled(x0=0.0, y0=0.0, w=6.0, d=10.0, eaves=3.0, ridge=5.0):
    """A gabled LOD2 building: rectangular footprint, two pitched roof planes.

    Roof vertex heights are 3,3,5,5,3,3 -> median 3.0 is deliberately *not*
    the mean (3.67) or the max, so the height heuristic is pinned exactly.
    """
    verts = [
        [x0, y0, 0.0], [x0 + w, y0, 0.0], [x0 + w, y0 + d, 0.0], [x0, y0 + d, 0.0],       # 0-3 ground
        [x0, y0, eaves], [x0 + w, y0, eaves], [x0 + w, y0 + d, eaves], [x0, y0 + d, eaves],  # 4-7 eaves
        [x0 + w / 2, y0, ridge], [x0 + w / 2, y0 + d, ridge],                              # 8-9 ridge
    ]
    boundaries = [[
        [[0, 3, 2, 1]],                 # ground, CW from above -> normal down
        [[4, 5, 8]], [[7, 9, 6]],       # gable end triangles (treated as roof)
        [[4, 8, 9, 7]], [[8, 5, 6, 9]], # the two roof planes
        [[0, 1, 5, 4]], [[1, 2, 6, 5]], [[2, 3, 7, 6]], [[3, 0, 4, 7]],  # walls
    ]]
    semantics = {
        "surfaces": [{"type": "GroundSurface"}, {"type": "RoofSurface"}, {"type": "WallSurface"}],
        "values": [[0, 1, 1, 1, 1, 2, 2, 2, 2]],
    }
    return verts, {"type": "Solid", "lod": "2.2", "boundaries": boundaries, "semantics": semantics}


def cj_with(verts, geom):
    return {"type": "CityJSON", "version": "1.1",
            "CityObjects": {"b": {"type": "Building", "geometry": [geom]}},
            "vertices": verts}


def convert_one(verts, geom):
    out = convert_to_lod1(cj_with(verts, geom))
    return out["CityObjects"]["b"]["geometry"][0], out["vertices"]


# --- shell integrity ------------------------------------------------------

def test_lod1_shell_is_closed_and_orientable():
    solid, verts = convert_one(*gabled())
    reused, unpaired = edge_census(solid)
    assert (reused, unpaired) == (0, 0), (
        f"{reused} directed edges reused, {unpaired} missing their reverse twin")


def test_lod1_volume_is_positive_and_matches_the_extrusion():
    """Outward orientation makes the divergence-theorem volume positive."""
    solid, verts = convert_one(*gabled(w=6.0, d=10.0, eaves=3.0, ridge=5.0))
    # cap height is 5.0, footprint 6 x 10 -> 300 m^3
    assert signed_volume(solid, verts) == pytest.approx(300.0, rel=1e-9)


def test_ground_roof_and_wall_normals_all_point_outward():
    solid, verts = convert_one(*gabled())
    v = np.asarray(verts, dtype=float)
    centroid = v[sorted({i for r in all_rings(solid) for i in r})].mean(axis=0)

    seen = Counter()
    for ring, stype in zip(outer_rings(solid), semantic_types(solid)):
        n = newell_normal(ring, verts)
        assert np.dot(n, v[ring].mean(axis=0) - centroid) > 0, f"{stype} faces inward"
        seen[stype] += 1
        if stype == "GroundSurface":
            assert n[2] == pytest.approx(-1.0)
        elif stype == "RoofSurface":
            assert n[2] == pytest.approx(1.0)
        elif stype == "WallSurface":
            assert n[2] == pytest.approx(0.0, abs=1e-9)
    assert seen == {"GroundSurface": 1, "RoofSurface": 1, "WallSurface": 4}


def cap_height(solid, verts):
    v = np.asarray(verts, dtype=float)
    roof = [r for r, s in zip(outer_rings(solid), semantic_types(solid)) if s == "RoofSurface"][0]
    zs = v[roof][:, 2]
    assert np.ptp(zs) < 1e-9, "the LOD1 cap must be flat"
    return zs[0]


def test_roof_sits_at_the_area_weighted_percentile_of_vertex_heights():
    """Roof z is sampled per vertex, each carrying a share of its face's area.

    Eaves hold 44.06 of the 84.11 total weight and the ridge 40.06, so the
    0.64 quantile falls inside the ridge group.
    """
    solid, verts = convert_one(*gabled(eaves=3.0, ridge=5.0))
    assert cap_height(solid, verts) == pytest.approx(5.0)


def cluttered_roof():
    """One large high roof plane plus eight small low facets.

    Weighting by vertex count lets the clutter outvote the main roof 24:4;
    weighting by area does not, since the clutter is 4% of the roof.
    """
    verts = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],        # 0-3 footprint
             [0, 0, 6], [10, 0, 6], [10, 10, 6], [0, 10, 6]]        # 4-7 main roof at z=6
    roof_faces = [[[4, 5, 6, 7]]]
    for k in range(8):                                              # 8 triangles, ~0.5 m^2, at z=2
        base = len(verts)
        verts += [[k, 0, 2], [k + 1, 0, 2], [k, 1, 2]]
        roof_faces.append([[base, base + 1, base + 2]])
    boundaries = [[[[0, 3, 2, 1]]] + roof_faces]
    semantics = {"surfaces": [{"type": "GroundSurface"}, {"type": "RoofSurface"}],
                 "values": [[0] + [1] * len(roof_faces)]}
    return verts, {"type": "Solid", "lod": "2.2", "boundaries": boundaries, "semantics": semantics}


def test_small_low_facets_do_not_drag_the_cap_down():
    solid, verts = convert_one(*cluttered_roof())
    # 100 m^2 at z=6 against 4 m^2 at z=2; a per-vertex median would give 2.0
    assert cap_height(solid, verts) == pytest.approx(6.0)


def gabled_per_plane_semantics():
    """3DBAG shape: one RoofSurface entry per roof plane, two WallSurface kinds.

    The roof indices deliberately run backwards, so collecting only the last
    RoofSurface entry would find just a gable-end triangle, whose vertices span
    a different height mix than the full roof.
    """
    verts, geom = gabled(eaves=3.0, ridge=5.0)
    geom["semantics"] = {
        "surfaces": [
            {"type": "GroundSurface"},
            {"type": "RoofSurface", "b3_azimut": 0.0},      # 1 -> slant [8,5,6,9]
            {"type": "RoofSurface", "b3_azimut": 90.0},     # 2 -> slant [4,8,9,7]
            {"type": "RoofSurface", "b3_azimut": 180.0},    # 3 -> triangle [7,9,6]
            {"type": "RoofSurface", "b3_azimut": 270.0},    # 4 -> triangle [4,5,8]
            {"type": "WallSurface", "on_footprint_edge": True},
            {"type": "WallSurface", "on_footprint_edge": False},
        ],
        "values": [[0, 4, 3, 2, 1, 5, 6, 5, 6]],
    }
    return verts, geom


def test_every_roof_plane_counts_not_just_the_last_semantic_entry():
    solid, verts = convert_one(*gabled_per_plane_semantics())
    assert cap_height(solid, verts) == pytest.approx(5.0)


def test_all_ground_surfaces_are_kept_when_several_are_declared():
    """Two footprints, each with its own GroundSurface entry, both extrude."""
    verts = [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0],          # 0-3 footprint A, 16 m^2
             [10, 0, 0], [14, 0, 0], [14, 4, 0], [10, 4, 0],      # 4-7 footprint B, 16 m^2
             [0, 0, 5], [4, 0, 5], [4, 4, 5], [0, 4, 5]]          # 8-11 roof at z=5
    boundaries = [[[[0, 3, 2, 1]], [[4, 7, 6, 5]], [[8, 9, 10, 11]]]]
    semantics = {"surfaces": [{"type": "GroundSurface"}, {"type": "GroundSurface"},
                              {"type": "RoofSurface"}],
                 "values": [[0, 1, 2]]}
    solid, out = convert_one(verts, {"type": "Solid", "lod": "2.2",
                                     "boundaries": boundaries, "semantics": semantics})
    assert sum(1 for s in semantic_types(solid) if s == "GroundSurface") == 2
    assert signed_volume(solid, out) == pytest.approx(160.0, rel=1e-9)   # (16+16) * 5


def test_lod1_semantics_are_reduced_to_the_three_classes():
    """A flat cap has no use for per-plane azimuth or slope attributes."""
    solid, _ = convert_one(*gabled_per_plane_semantics())
    assert solid["semantics"]["surfaces"] == [
        {"type": "GroundSurface"}, {"type": "RoofSurface"}, {"type": "WallSurface"}]
    assert set(semantic_types(solid)) == {"GroundSurface", "RoofSurface", "WallSurface"}


def test_roof_percentile_is_tunable():
    verts, geom = gabled(eaves=3.0, ridge=5.0)
    low = convert_to_lod1(cj_with(verts, geom), roof_percentile=0.0)
    high = convert_to_lod1(cj_with(*gabled(eaves=3.0, ridge=5.0)), roof_percentile=1.0)
    lo = cap_height(low["CityObjects"]["b"]["geometry"][0], low["vertices"])
    hi = cap_height(high["CityObjects"]["b"]["geometry"][0], high["vertices"])
    assert lo == pytest.approx(3.0)        # the lowest roof vertex, at the eaves
    assert hi == pytest.approx(5.0)        # the highest roof vertex, at the ridge
    assert lo < hi


# --- footprints with holes ------------------------------------------------

def courtyard():
    """Square footprint with a square hole; roof at a uniform height of 4."""
    outer = [[0, 0], [10, 0], [10, 10], [0, 10]]
    hole = [[3, 3], [7, 3], [7, 7], [3, 7]]
    verts = ([p + [0.0] for p in outer] + [p + [0.0] for p in hole] +
             [p + [4.0] for p in outer] + [p + [4.0] for p in hole])
    ground = [[0, 3, 2, 1], [4, 5, 6, 7]]        # outer CW, hole CCW
    roof = [[8, 9, 10, 11], [15, 14, 13, 12]]    # outer CCW, hole CW
    boundaries = [[ground, roof]]
    semantics = {"surfaces": [{"type": "GroundSurface"}, {"type": "RoofSurface"}],
                 "values": [[0, 1]]}
    return verts, {"type": "Solid", "lod": "2.2", "boundaries": boundaries, "semantics": semantics}


def test_courtyard_hole_stays_closed_and_subtracts_from_the_volume():
    solid, verts = convert_one(*courtyard())
    reused, unpaired = edge_census(solid)
    assert (reused, unpaired) == (0, 0)
    # (10*10 - 4*4) * 4 = 336
    assert signed_volume(solid, verts) == pytest.approx(336.0, rel=1e-9)


# --- pass-through behaviour -----------------------------------------------

def test_multisurface_is_left_untouched():
    geom = {"type": "MultiSurface", "lod": "0", "boundaries": [[[0, 1, 2]]]}
    out, _ = convert_one([[0, 0, 0], [1, 0, 0], [0, 1, 0]], geom)
    assert out == geom


def test_solid_without_semantics_is_left_untouched():
    verts, geom = gabled()
    del geom["semantics"]
    out, _ = convert_one(verts, geom)
    assert out["lod"] == "2.2"


def test_solid_without_a_ground_surface_is_left_untouched():
    verts, geom = gabled()
    geom["semantics"]["surfaces"][0] = {"type": "RoofSurface"}
    out, _ = convert_one(verts, geom)
    assert out["lod"] == "2.2"


def test_empty_semantic_values_do_not_raise():
    verts, geom = gabled()
    geom["semantics"]["values"] = []
    out, _ = convert_one(verts, geom)
    assert out["lod"] == "2.2"
