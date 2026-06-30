#!/usr/bin/env python3
from statistics import median

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


def convert_to_lod1(_cj):
    """
    Converts a single CityJSON dictionary from LOD2 representation to LOD1.
    """
    # Make a copy of the vertices to modify them safely
    _vertices = list(_cj["vertices"])

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

            ground_idx = None
            roof_idx = None
            wall_idx = None

            for i, s in enumerate(surfaces):
                stype = s.get("type")
                if stype == "GroundSurface":
                    ground_idx = i
                elif stype == "RoofSurface":
                    roof_idx = i
                elif stype == "WallSurface":
                    wall_idx = i

            if ground_idx is None or roof_idx is None:
                new_geometries.append(geom)
                continue

            if wall_idx is None:
                wall_idx = len(surfaces)
                surfaces = list(surfaces)
                surfaces.append({"type": "WallSurface"})

            # --------------------------------------------------
            # Collect ground faces and roof heights
            # --------------------------------------------------
            ground_faces = []
            roof_heights = []

            for shell_i, shell in enumerate(geom["boundaries"]):
                sem_shell = values[shell_i]
                for face_i, face in enumerate(shell):
                    sem = sem_shell[face_i]
                    if sem == ground_idx:
                        ground_faces.append(face)
                    elif sem == roof_idx:
                        for ring in face:
                            for vid in ring:
                                roof_heights.append(
                                    _vertices[vid][2]
                                )

            if not ground_faces or not roof_heights:
                new_geometries.append(geom)
                continue

            lod1_height = median(roof_heights)

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
                    ccw=True
                )

                hole_rings = [
                    ensure_orientation(r, _vertices, ccw=False)
                    for r in face[1:]
                ]

                ground_face = [outer_ring] + hole_rings
                shell.append(ground_face)
                sem_values.append(ground_idx)

                roof_face = []
                roof_face.append(
                    [get_roof_vertex(v) for v in outer_ring][::-1]
                )

                for hole in hole_rings:
                    roof_face.append(
                        [get_roof_vertex(v) for v in hole][::-1]
                    )

                shell.append(roof_face)
                sem_values.append(roof_idx)

                # walls from outer boundary
                def add_walls(ring):
                    n = len(ring)
                    for i in range(n):
                        b0 = ring[i]
                        b1 = ring[(i + 1) % n]

                        t0 = get_roof_vertex(b0)
                        t1 = get_roof_vertex(b1)

                        wall = [[
                            b0,
                            b1,
                            t1,
                            t0
                        ]]

                        shell.append(wall)
                        sem_values.append(wall_idx)

                add_walls(outer_ring)
                for hole in hole_rings:
                    add_walls(hole)

            new_geom = {
                "type": "Solid",
                "lod": "1",
                "boundaries": [shell],
                "semantics": {
                    "surfaces": surfaces,
                    "values": [sem_values]
                }
            }
            new_geometries.append(new_geom)

        city_obj["geometry"] = new_geometries

    _cj["vertices"] = _vertices
    _cj.setdefault("metadata", {})
    _cj["metadata"]["datasetLod"] = "1"

    return _cj
