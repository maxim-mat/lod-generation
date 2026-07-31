#!/usr/bin/env python3
from collections import Counter
from math import sqrt
from statistics import median

import numpy as np

# Calibrated so a converted LOD1 stands in the same relation to its LOD2 as
# real 3DBAG lod1.2 does to lod2.2 -- a volume ratio of 1.0725, measured over
# 4266 buildings. Matching that relation, rather than 3DBAG's published
# ``b3_h_dak_70p`` height directly, is what keeps synthetic pairs consistent
# with the real pairs they are trained alongside.
DEFAULT_ROOF_PERCENTILE = 0.64

# An LOD1 solid has exactly three kinds of face, so it gets a fresh minimal
# semantics block rather than the source's -- per-plane roof attributes like
# azimuth and slope describe LOD2 geometry that no longer exists here.
LOD1_GROUND, LOD1_ROOF, LOD1_WALL = 0, 1, 2
LOD1_SURFACES = ({"type": "GroundSurface"}, {"type": "RoofSurface"}, {"type": "WallSurface"})


def newell_vector(points):
    """Twice the area vector of a ring: ``2 * A * n``, translation-invariant."""
    p = np.asarray(points, dtype=float)
    return np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)


def face_normal(points):
    """Unit normal of a ring, or None when it encloses no area.

    None means the ring is degenerate -- collinear or collapsed -- which is a
    defect in its own right, not a case to paper over with a default normal.
    """
    n = newell_vector(points)
    mag = float(np.linalg.norm(n))
    return None if mag < 1e-12 else n / mag


def shell_edge_defects(faces):
    """``(unpaired, reused)`` directed edges of a shell.

    ``unpaired`` -- no oppositely-wound twin, so the surface is open there.
    ``reused``   -- traversed twice in the same direction, so two faces disagree
    about which side faces out.

    ``faces`` is a list of faces, each a list of rings of vertex indices. Kept
    separate from the copy in ``analysis.cityobject_analysis``, which stays
    independent of the pipeline on purpose so it can detect regressions in it.
    """
    used = Counter()
    for face in faces:
        for ring in face:
            for i in range(len(ring)):
                used[(ring[i], ring[(i + 1) % len(ring)])] += 1
    unpaired = [e for e in used if (e[1], e[0]) not in used]
    reused = [e for e, n in used.items() if n > 1]
    return unpaired, reused


def signed_area_2d(ring, _vertices):
    """
    Computes the signed area of a 2D ring.
    """
    area = 0.0
    n = len(ring)

    for i in range(n):
        x1, y1, _ = _vertices[ring[i]]
        x2, y2, _ = _vertices[ring[(i + 1) % n]]

        area += x1 * y2 - x2 * y1

    return area / 2.0


def ensure_orientation(ring, _vertices, ccw=True):
    """
    Ensures that a ring is oriented counter-clockwise (ccw=True) or clockwise (ccw=False).
    """
    area = signed_area_2d(ring, _vertices)

    if ccw and area < 0:
        return ring[::-1]

    if not ccw and area > 0:
        return ring[::-1]

    return ring


def _face_area(ring, _vertices, scale):
    """Area of a planar ring in world units, via the Newell area vector.

    Only ``scale`` is applied: the area vector of a closed ring is invariant
    under translation, so ``transform.translate`` cannot affect the result.
    """
    nx = ny = nz = 0.0
    n = len(ring)
    for i in range(n):
        x1, y1, z1 = (c * s for c, s in zip(_vertices[ring[i]], scale))
        x2, y2, z2 = (c * s for c, s in zip(_vertices[ring[(i + 1) % n]], scale))
        nx += y1 * z2 - z1 * y2
        ny += z1 * x2 - x1 * z2
        nz += x1 * y2 - y1 * x2
    return sqrt(nx * nx + ny * ny + nz * nz) / 2.0


def _weighted_percentile(values, weights, q):
    """Value at quantile ``q``, interpolated over midpoint cumulative weight."""
    pairs = sorted(zip(values, weights))
    total = sum(w for _, w in pairs)
    if total <= 0:
        return median([v for v, _ in pairs])

    target = q * total
    acc = 0.0
    prev_c = prev_v = None
    for v, w in pairs:
        c = acc + 0.5 * w
        if c >= target:
            if prev_c is None or c == prev_c:
                return v
            f = (target - prev_c) / (c - prev_c)
            return prev_v + f * (v - prev_v)
        prev_c, prev_v = c, v
        acc += w
    return pairs[-1][0]


def convert_to_lod1(_cj, roof_percentile=DEFAULT_ROOF_PERCENTILE):
    """
    Converts a single CityJSON dictionary from LOD2 representation to LOD1.

    LOD1 is not LOD2 with the roof deleted -- that would leave an open shell.
    The footprint is reused verbatim and extruded to a single flat cap, so the
    cap height is the only quantity this conversion actually decides. It is the
    ``roof_percentile`` quantile of roof-vertex height, each vertex weighted by
    its share of its face's area; weighting by vertex count instead lets a
    cluster of small facets outvote the main roof plane.
    """
    # Make a copy of the vertices to modify them safely
    _vertices = list(_cj["vertices"])
    scale = (_cj.get("transform") or {}).get("scale", (1.0, 1.0, 1.0))

    for obj_id, city_obj in _cj.get("CityObjects", {}).items():
        new_geometries = []

        for geom in city_obj.get("geometry", []):
            if geom.get("type") != "Solid":
                new_geometries.append(geom)
                continue

            semantics = geom.get("semantics")
            if not semantics:
                new_geometries.append(geom)
                continue

            surfaces = semantics.get("surfaces", [])
            values = semantics.get("values", [])

            # A type maps to a *set* of indices: 3DBAG declares one
            # RoofSurface per roof plane, carrying that plane's azimuth and
            # slope, and separate WallSurface entries per wall kind. Keeping
            # only the last index of each type would collect a fraction of the
            # roof and cap the building at the wrong height.
            ground_ids = {i for i, s in enumerate(surfaces)
                          if s and s.get("type") == "GroundSurface"}
            roof_ids = {i for i, s in enumerate(surfaces)
                        if s and s.get("type") == "RoofSurface"}

            if not ground_ids or not roof_ids:
                new_geometries.append(geom)
                continue

            # --------------------------------------------------
            # Collect ground faces and roof heights
            # --------------------------------------------------
            ground_faces = []
            roof_faces = []

            for shell_i, shell in enumerate(geom["boundaries"]):
                # Semantics may be ragged or absent for a shell; a face without
                # a value is simply unclassified, not a reason to crash.
                sem_shell = values[shell_i] if shell_i < len(values) else []
                for face_i, face in enumerate(shell):
                    sem = sem_shell[face_i] if face_i < len(sem_shell) else None
                    if sem in ground_ids:
                        ground_faces.append(face)
                    elif sem in roof_ids:
                        roof_faces.append(face)

            if not ground_faces or not roof_faces:
                new_geometries.append(geom)
                continue

            # Sample roof height per vertex, each carrying an equal share of
            # its face's area. Collapsing a face to its mean z instead caps the
            # reachable height at the highest face *average*, which cannot
            # reproduce the ridge-height end of the real LOD1 distribution.
            # Heights stay in the file's own vertex units so the new roof
            # vertices below can be appended as-is; only the area weights need
            # world units, and a quantile is invariant to their common factor.
            roof_z, roof_w = [], []
            for face in roof_faces:
                ring = face[0]
                share = _face_area(ring, _vertices, scale) / len(ring)
                for vid in ring:
                    roof_z.append(_vertices[vid][2])
                    roof_w.append(share)
            lod1_height = _weighted_percentile(roof_z, roof_w, roof_percentile)

            # --------------------------------------------------
            # Vertex cache
            # --------------------------------------------------
            roof_vertex_cache = {}

            def get_roof_vertex(base_vid):
                if base_vid in roof_vertex_cache:
                    return roof_vertex_cache[base_vid]

                x, y, _ = _vertices[base_vid]
                roof_vid = len(_vertices)
                _vertices.append([x, y, lod1_height])
                roof_vertex_cache[base_vid] = roof_vid
                return roof_vid

            # --------------------------------------------------
            # Build shell
            # --------------------------------------------------
            shell = []
            sem_values = []

            for face in ground_faces:
                outer_ring = ensure_orientation(
                    face[0],
                    _vertices,
                    ccw=False  # CW viewed from above → outward normal points down
                )

                hole_rings = [
                    ensure_orientation(r, _vertices, ccw=True)  # opposite of outer ring
                    for r in face[1:]
                ]

                ground_face = [outer_ring] + hole_rings
                shell.append(ground_face)
                sem_values.append(LOD1_GROUND)

                # The roof is the footprint lifted, so it inherits the ground's
                # CW winding and with it a downward normal. Reverse every ring
                # to turn it back up; holes follow so they stay opposed to the
                # outer ring.
                roof_face = [[get_roof_vertex(v) for v in reversed(outer_ring)]]

                for hole in hole_rings:
                    roof_face.append(
                        [get_roof_vertex(v) for v in reversed(hole)]
                    )

                shell.append(roof_face)
                sem_values.append(LOD1_ROOF)

                # walls from outer boundary
                def add_walls(ring):
                    n = len(ring)
                    for i in range(n):
                        b0 = ring[i]
                        b1 = ring[(i + 1) % n]

                        t0 = get_roof_vertex(b0)
                        t1 = get_roof_vertex(b1)

                        # Rise before running along the ring: b0->b1->t1->t0
                        # would wind the quad the other way and face inward.
                        wall = [[
                            b0,
                            t0,
                            t1,
                            b1
                        ]]

                        shell.append(wall)
                        sem_values.append(LOD1_WALL)

                add_walls(outer_ring)
                for hole in hole_rings:
                    add_walls(hole)

            new_geom = {
                "type": "Solid",
                "lod": "1",
                "boundaries": [shell],
                "semantics": {
                    "surfaces": [dict(s) for s in LOD1_SURFACES],
                    "values": [sem_values]
                }
            }
            new_geometries.append(new_geom)

        city_obj["geometry"] = new_geometries

    _cj["vertices"] = _vertices
    _cj.setdefault("metadata", {})
    _cj["metadata"]["datasetLod"] = "1"

    return _cj
