import json
import logging
from pathlib import Path
import numpy as np

from src.dataset.dataset import EDGE_VF, EDGE_VV, GROUND, ROOF, VERTEX, WALL

logger = logging.getLogger(__name__)

# ==============================================================================
# SVD Plane Fitting & Node Projections
# ==============================================================================

def straighten_face(face_coords):
    """
    Fits a plane to a 3D polygon face using Eigendecomposition (equivalent to SVD),
    snaps the normal to standard axes (roof/ground vs wall), and projects vertices.
    """
    K = len(face_coords)
    center = np.mean(face_coords, axis=0)
    centered = face_coords - center
    
    cov = np.dot(centered.T, centered)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    normal = eigenvectors[:, 0]
    
    # Snap normal to standard axes for structural rigidity
    if abs(normal[2]) > 0.9:
        # Snap to horizontal plane (Ground or Roof)
        normal = np.array([0.0, 0.0, 1.0]) if normal[2] > 0 else np.array([0.0, 0.0, -1.0])
        surface_type = "RoofSurface" if normal[2] > 0 else "GroundSurface"
    elif abs(normal[2]) < 0.1:
        # Snap to vertical plane (Wall)
        normal[2] = 0.0
        norm_xy = np.linalg.norm(normal)
        if norm_xy > 1e-6:
            normal = normal / norm_xy
        surface_type = "WallSurface"
    else:
        # Keep general sloped plane
        surface_type = "RoofSurface" if normal[2] > 0 else "WallSurface"
        
    projected = []
    for p in face_coords:
        dist_to_plane = np.dot(p - center, normal)
        p_proj = p - dist_to_plane * normal
        projected.append(p_proj)
        
    return np.array(projected), surface_type

# ==============================================================================
# Global Rigidity Regularization
# ==============================================================================

def regularize_building_geometry(nodes, faces):
    """
    Fits planes and projects faces. Averages coordinates of shared vertices
    across projected planes to maintain a closed, water-tight building geometry.
    """
    num_nodes = len(nodes)
    vertex_projections = {i: [] for i in range(num_nodes)}
    face_surface_types = []
    
    for face in faces:
        face_coords = nodes[face]
        proj_coords, surf_type = straighten_face(face_coords)
        face_surface_types.append(surf_type)
        
        for idx_in_face, global_node_idx in enumerate(face):
            vertex_projections[global_node_idx].append(proj_coords[idx_in_face])
            
    new_nodes = np.zeros_like(nodes)
    for i in range(num_nodes):
        projs = vertex_projections[i]
        if projs:
            new_nodes[i] = np.mean(projs, axis=0)
        else:
            new_nodes[i] = nodes[i]
            
    return new_nodes, face_surface_types

# ==============================================================================
# CityJSON Format Exporter
# ==============================================================================

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


def save_to_file(cityjson_dict, output_path):
    """
    Writes CityJSON dictionary to file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cityjson_dict, f)
    logger.info(f"CityJSON saved successfully to {output_path}")
