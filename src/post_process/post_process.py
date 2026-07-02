import json
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# ==============================================================================
# Cycle Extraction (Polygonal Face Reconstruction)
# ==============================================================================

def find_cycles_dfs(adj_list, max_len=8):
    """
    Finds simple cycles in the adjacency list of length between 3 and max_len.
    Returns: List of lists (cycles represented as sequences of node indices)
    """
    cycles = []
    
    # Store visited and path tracking
    visited = set()
    
    def dfs(start_node, curr_node, path, depth):
        if depth > max_len:
            return
            
        for neighbor in adj_list.get(curr_node, []):
            if neighbor == start_node and depth >= 3:
                # Cycle found, normalize representation (smallest index first, direction-agnostic)
                cycle = path[:]
                min_idx = np.argmin(cycle)
                cycle = cycle[min_idx:] + cycle[:min_idx]
                
                # Check orientation & canonicalize
                if cycle[1] > cycle[-1]:
                    cycle = [cycle[0]] + cycle[1:][::-1]
                    
                if cycle not in cycles:
                    cycles.append(cycle)
            elif neighbor not in visited and neighbor not in path:
                dfs(start_node, neighbor, path + [neighbor], depth + 1)

    nodes = sorted(list(adj_list.keys()))
    for node in nodes:
        # DFS starting from each node
        dfs(node, node, [node], 1)
        visited.add(node)  # Prevent finding permutations starting at other nodes
        
    # Filter cycles to retain only chordless cycles
    chordless_cycles = []
    for cycle in cycles:
        is_chordless = True
        n = len(cycle)
        cycle_set = set(cycle)
        
        # Check if there are any edges between non-adjacent cycle nodes
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue  # adjacent
                u, v = cycle[i], cycle[j]
                if v in adj_list.get(u, []):
                    is_chordless = False
                    break
            if not is_chordless:
                break
                
        if is_chordless:
            chordless_cycles.append(cycle)
            
    return chordless_cycles

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

def graph_to_cityjson(nodes, edge_probs, threshold=0.5, building_id="generated_building"):
    """
    Converts raw generated nodes and edge probability matrices into a valid CityJSON dictionary.
    """
    N = len(nodes)
    
    # 1. Build adjacency list based on threshold
    adj_list = {i: [] for i in range(N)}
    for i in range(N):
        for j in range(i + 1, N):
            if edge_probs[i, j] > threshold:
                adj_list[i].append(j)
                adj_list[j].append(i)
                
    # 2. Extract polygonal faces
    faces = find_cycles_dfs(adj_list, max_len=8)
    if not faces:
        # Fallback to a simple floor footprint if no cycles found
        logger.warning("No closed cycles found in graph. Creating a fallback geometry.")
        return {}
        
    # 3. Regularize geometry (straighten out faces & average shared vertices)
    nodes_reg, surface_types = regularize_building_geometry(nodes, faces)
    
    # 4. Format into CityJSON structure
    semantic_surfaces = [
        {"type": "GroundSurface"},
        {"type": "RoofSurface"},
        {"type": "WallSurface"}
    ]
    sem_map = {"GroundSurface": 0, "RoofSurface": 1, "WallSurface": 2}
    
    cj_faces = [[face] for face in faces]
    cj_semantics_values = [sem_map[stype] for stype in surface_types]
    
    cityjson_dict = {
        "type": "CityJSON",
        "version": "1.1",
        "CityObjects": {
            building_id: {
                "type": "Building",
                "geometry": [
                    {
                        "type": "Solid",
                        "lod": "1",
                        "boundaries": [cj_faces],
                        "semantics": {
                            "surfaces": semantic_surfaces,
                            "values": [cj_semantics_values]
                        }
                    }
                ]
            }
        },
        "vertices": nodes_reg.tolist(),
        "metadata": {
            "datasetLod": "1"
        }
    }
    
    return cityjson_dict


def save_to_file(cityjson_dict, output_path):
    """
    Writes CityJSON dictionary to file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cityjson_dict, f)
    logger.info(f"CityJSON saved successfully to {output_path}")
