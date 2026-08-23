import json
import os
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ==============================================================================
# Levi graph classes
# ==============================================================================
# Nodes: vertices carry coordinates; each face of the building becomes its own
# node whose class is its surface semantic; OFF pads to n_max.
VERTEX, GROUND, ROOF, WALL, OFF = 0, 1, 2, 3, 4
NUM_NODE_CLASSES = 5
NODE_CLASS_NAMES = ("Vertex", "GroundSurface", "RoofSurface", "WallSurface", "Off")
# Every CityGML boundary class the corpus uses maps explicitly. Left implicit,
# OuterFloorSurface and OuterCeilingSurface fell through to the normal-based
# fallback, which sent ~5.5k overhang undersides to GROUND -- ground nodes
# floating at height, breaking the "ground is the base at z~0" invariant that
# normalize_coords and the ground-level statistics both rely on. Balcony tops
# join ROOF (where the normal already put them); undersides become WALL, i.e.
# envelope that is not the footprint.
SURFACE_TO_CLASS = {
    "GroundSurface": GROUND,
    "RoofSurface": ROOF,
    "WallSurface": WALL,
    "OuterFloorSurface": ROOF,
    "OuterCeilingSurface": WALL,
}
# Edges: ring adjacency between vertices, membership between a face and its vertices.
EDGE_OFF, EDGE_VV, EDGE_VF = 0, 1, 2
NUM_EDGE_CLASSES = 3

# ==============================================================================
# Helper functions for base footprint extraction & normalization
# ==============================================================================

def get_base_center(geom_list, v_raw, active_vertex_indices):
    """
    Computes the center of the base (GroundSurface footprint) of a building.
    If no semantic GroundSurface is found, falls back to the vertices matching
    the minimum Z coordinate with a 10cm tolerance.
    """
    ground_vertex_indices = set()
    
    # 1. Try to find vertices belonging to GroundSurface semantics
    for geom in geom_list:
        geom_type = geom.get("type")
        semantics = geom.get("semantics", {})
        if not semantics:
            continue
            
        surfaces = semantics.get("surfaces", [])
        values = semantics.get("values", [])
        if not surfaces or not values:
            continue
            
        # Get indices of all GroundSurface types
        ground_surface_indices = {
            i for i, s in enumerate(surfaces)
            if s and s.get("type") == "GroundSurface"
        }
        if not ground_surface_indices:
            continue
            
        boundaries = geom.get("boundaries", [])
        
        if geom_type == "Solid":
            # boundaries structure: [shell][face][ring][vertex]
            # values structure: [shell][face]
            for shell_i, shell in enumerate(boundaries):
                if shell_i >= len(values):
                    continue
                sem_shell = values[shell_i]
                for face_i, face in enumerate(shell):
                    if face_i >= len(sem_shell):
                        continue
                    sem_idx = sem_shell[face_i]
                    if sem_idx in ground_surface_indices:
                        # Outer ring only: the graph never contains hole
                        # vertices, so averaging them centres the building on
                        # a point the model cannot see. An off-centre
                        # courtyard shifted the footprint by up to 4.8 m.
                        for vid in face[0]:
                            ground_vertex_indices.add(vid)

        elif geom_type in ["MultiSurface", "CompositeSurface"]:
            # boundaries structure: [face][ring][vertex]
            # values structure: [face]
            for face_i, face in enumerate(boundaries):
                if face_i >= len(values):
                    continue
                sem_idx = values[face_i]
                if sem_idx in ground_surface_indices:
                    for vid in face[0]:              # outer ring only, as above
                        ground_vertex_indices.add(vid)

    # 2. Geometric fallback: Vertices close to the minimum Z elevation
    if not ground_vertex_indices and active_vertex_indices:
        all_vids = list(active_vertex_indices)
        min_z = min(v_raw[vid][2] for vid in all_vids)
        # Select vertices within 10cm of the base elevation
        ground_vertex_indices = {vid for vid in all_vids if abs(v_raw[vid][2] - min_z) < 0.1}
        
    if not ground_vertex_indices:
        return np.zeros(3)
        
    # 3. Compute base center coordinates (mean X, Y, Z)
    ground_coords = [v_raw[vid] for vid in ground_vertex_indices]
    cx = sum(c[0] for c in ground_coords) / len(ground_coords)
    cy = sum(c[1] for c in ground_coords) / len(ground_coords)
    cz = sum(c[2] for c in ground_coords) / len(ground_coords)
    
    return np.array([cx, cy, cz], dtype=float)


def surface_class_from_normal(ring_coords):
    """Fallback face class when the file has no semantics.

    Same snap thresholds as `straighten_face`: |nz| > 0.9 horizontal
    (roof up / ground down), < 0.1 wall, sloped otherwise (roof if up).
    Relies on the CityJSON convention that exterior rings wind CCW seen
    from outside, i.e. outward normals.
    """
    n = np.zeros(3)
    for i in range(len(ring_coords)):
        n += np.cross(ring_coords[i], ring_coords[(i + 1) % len(ring_coords)])
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return WALL
    nz = n[2] / norm
    if abs(nz) > 0.9:
        return ROOF if nz > 0 else GROUND
    if abs(nz) < 0.1:
        return WALL
    return ROOF if nz > 0 else WALL


def _iter_faces(geom):
    """Yield (outer_ring, semantic_type_or_None) for each surface of a geometry.

    The *outer* ring only. Kept deliberately: a Levi face node is one ring --
    it carries that ring's centroid and an EDGE_VF to each of its vertices --
    so a surface with holes has no representation here without deciding what a
    hole node is. `parse_cityjson_file_to_graphs` counts the interior rings it
    skips into `inner_rings` so the omission stays visible.

    Anything that triangulates wants `_iter_surfaces` instead, which yields the
    whole surface: dropping the holes there fills in real courtyards.
    """
    for rings, stype in _iter_surfaces(geom):
        yield rings[0], stype


def _iter_surfaces(geom):
    """Yield (rings, semantic_type_or_None) for each surface of a geometry.

    ``rings`` is the surface as CityJSON stores it -- exterior first, interior
    rings after (spec 2.0.1). Only Solid / MultiSurface / CompositeSurface
    contribute; anything else warns rather than yielding nothing quietly.
    """
    boundaries = geom.get("boundaries", [])
    gtype = geom.get("type")
    sem = geom.get("semantics") or {}
    surfaces = sem.get("surfaces") or []
    values = sem.get("values") or []
    if gtype == "Solid":
        faces = [f for shell in boundaries for f in shell]
        vals = [v for shell_vals in values for v in shell_vals] if values else []
    elif gtype in ("MultiSurface", "CompositeSurface"):
        faces, vals = boundaries, values
    else:
        # Silently yielding nothing here once hid 265k lod-0 footprints; a
        # geometry the parser cannot read must say so.
        logger.warning("unsupported geometry type %r (lod %s): %d boundaries ignored",
                       gtype, geom.get("lod"), len(boundaries))
        return
    for i, face in enumerate(faces):
        if not face or len(face[0]) < 3:
            continue
        stype = None
        if i < len(vals) and vals[i] is not None and vals[i] < len(surfaces):
            s = surfaces[vals[i]]
            stype = s.get("type") if s else None
        yield face, stype


def parse_cityjson_file_to_graphs(filepath, normalize_coords=False):
    """Parses a single CityJSON file into a dict of Levi building graphs.

    Node order: the building's vertices (class VERTEX, 3D coords) followed by
    one node per face outer ring (classes GROUND/ROOF/WALL, positioned at
    their ring centroid).
    Edges: EDGE_VV ring adjacency, EDGE_VF face membership; both directions.
    Face semantics come from the file, falling back to the face normal.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        cj = json.load(f)

    v_raw = np.array(cj["vertices"], dtype=float)
    if "transform" in cj:
        scale = np.array(cj["transform"]["scale"])
        translate = np.array(cj["transform"]["translate"])
        v_raw = v_raw * scale + translate

    graphs = {}
    inner_rings = 0        # holes the Levi graph cannot represent
    by_normal = 0          # faces with no usable semantic label

    for obj_id, city_obj in cj.get("CityObjects", {}).items():
        geom_list = city_obj.get("geometry", [])
        if not geom_list:
            continue

        for geom in geom_list:
            bounds = geom.get("boundaries") or []
            if geom.get("type") == "Solid":
                bounds = [f for shell in bounds for f in shell]
            elif geom.get("type") not in ("MultiSurface", "CompositeSurface"):
                bounds = []
            inner_rings += sum(max(len(face) - 1, 0)
                               for face in bounds if isinstance(face, list))

        faces = []  # (original-index ring, node class)
        for geom in geom_list:
            for ring, stype in _iter_faces(geom):
                cls = SURFACE_TO_CLASS.get(stype)
                if cls is None:
                    by_normal += 1
                    cls = surface_class_from_normal(v_raw[ring])
                faces.append((ring, cls))
        if not faces:
            continue

        active = sorted({vid for ring, _ in faces for vid in ring})
        idx_map = {old: new for new, old in enumerate(active)}
        n_vertices = len(active)

        coords = v_raw[active]
        # Normalize coordinates relative to base footprint center
        if normalize_coords:
            coords = coords - get_base_center(geom_list, v_raw, set(active))

        x = np.zeros((n_vertices + len(faces), 3))
        x[:n_vertices] = coords
        for f_i, (ring, _) in enumerate(faces):
            x[n_vertices + f_i] = coords[[idx_map[v] for v in ring]].mean(axis=0)
        node_labels = [VERTEX] * n_vertices + [cls for _, cls in faces]

        edges = {}
        for f_i, (ring, _) in enumerate(faces):
            f_node = n_vertices + f_i
            for k in range(len(ring)):
                u, v = idx_map[ring[k]], idx_map[ring[(k + 1) % len(ring)]]
                edges[(u, v)] = EDGE_VV
                edges[(v, u)] = EDGE_VV
                edges[(f_node, u)] = EDGE_VF
                edges[(u, f_node)] = EDGE_VF

        edge_index = np.array(list(edges.keys()), dtype=np.int64).T
        edge_attr = np.array(list(edges.values()), dtype=np.int64)

        graphs[obj_id] = {
            "id": obj_id,
            "x": torch.tensor(x, dtype=torch.float32),
            "node_labels": torch.tensor(node_labels, dtype=torch.long),
            "edge_index": torch.tensor(edge_index, dtype=torch.long),
            "edge_attr": torch.tensor(edge_attr, dtype=torch.long),
            "type": city_obj.get("type", "Unknown"),
        }

    if inner_rings or by_normal:
        # A Levi face node carries one ring, so a courtyard's inner ring cannot
        # be represented and its adjacency is lost. Reported, not silent.
        logger.info("%s: %d graphs, %d inner ring(s) discarded, "
                    "%d face(s) classified by normal (no semantic label)",
                    Path(filepath).name, len(graphs), inner_rings, by_normal)
    return graphs

# ==============================================================================
# PyTorch Dataset Object
# ==============================================================================

class CityJSONDataset(Dataset):
    def __init__(self, dataset_dir, lods, normalize_coords=False, transform=None,
                 n_max=None, upper_limit_nodes=None):
        """
        Args:
            dataset_dir (str or Path): Folder containing dataset (e.g. data/The Hague).
            lods (int, str, list, tuple): Single LOD or a list of LODs to match.
            normalize_coords (bool): If True, shifts nodes such that center of building base is at (0, 0, 0).
            transform (callable, optional): Optional transform to apply on graph items.
            n_max (int, optional): Maximum number of nodes per graph. All graphs are padded
                to this size with Virtual nodes. If None, auto-detected from the dataset
                as the maximum observed node count.
            upper_limit_nodes (int, optional): Upper limit on the number of nodes per graph. 
                Graphs exceeding this limit are excluded from the dataset.
        """
        self.dataset_dir = Path(dataset_dir)
        self.normalize_coords = normalize_coords
        self.transform = transform
        
        # Parse LOD input type
        if isinstance(lods, (list, tuple, set, np.ndarray)):
            self.lods = [int(l) for l in lods]
        else:
            self.lods = [int(lods)]
        self.is_single_lod = len(self.lods) == 1
            
        self.lod_data = {}
        
        # Load and parse folders for each specified LOD
        for lod in self.lods:
            self.lod_data[lod] = {}
            
            # Find the LOD folder case-insensitively
            target_lod_name = f"lod{lod}"
            lod_dir = None
            for d in self.dataset_dir.iterdir():
                if d.is_dir() and d.name.lower() == target_lod_name:
                    lod_dir = d
                    break
                    
            if not lod_dir:
                raise FileNotFoundError(
                    f"Could not find folder for LOD '{lod}' under dataset '{self.dataset_dir.name}'"
                )
                
            # Scan directories recursively for JSON files
            for root, _, files in os.walk(lod_dir):
                root_path = Path(root)
                for f in files:
                    if f.lower().endswith(('.json', '.city.json')) and f != "description.txt":
                        filepath = root_path / f
                        graphs = parse_cityjson_file_to_graphs(filepath, self.normalize_coords)
                        self.lod_data[lod].update(graphs)
                        
        # Intersect object IDs to align buildings across all requested LODs
        if len(self.lods) > 1:
            common_ids = set(self.lod_data[self.lods[0]].keys())
            for lod in self.lods[1:]:
                common_ids.intersection_update(self.lod_data[lod].keys())
            self.ids = sorted(list(common_ids))
        else:
            self.ids = sorted(list(self.lod_data[self.lods[0]].keys()))

        # Filter out building graphs exceeding upper_limit_nodes
        if upper_limit_nodes is not None:
            filtered_ids = []
            removed_ids = []
            for obj_id in self.ids:
                keep = True
                for lod in self.lods:
                    graph = self.lod_data[lod].get(obj_id)
                    if graph is not None and graph["x"].size(0) > upper_limit_nodes:
                        keep = False
                        break
                if keep:
                    filtered_ids.append(obj_id)
                else:
                    removed_ids.append(obj_id)
            
            # Remove from memory (lod_data)
            for obj_id in removed_ids:
                for lod in self.lods:
                    if obj_id in self.lod_data[lod]:
                        del self.lod_data[lod][obj_id]
            
            self.ids = filtered_ids
            logger.info(
                f"Filtered out {len(removed_ids)} buildings exceeding upper_limit_nodes={upper_limit_nodes}. "
                f"{len(self.ids)} remaining."
            )
            
        if not self.ids:
            logger.warning(f"No matching CityObjects found across requested LODs: {self.lods}")

        # ------------------------------------------------------------------
        # Compute N_max: the fixed graph size for padding
        # ------------------------------------------------------------------
        observed_max = 0
        for lod in self.lods:
            for obj_id in self.ids:
                graph = self.lod_data[lod].get(obj_id)
                if graph is not None:
                    num_nodes = graph["x"].size(0)
                    observed_max = max(observed_max, num_nodes)
        
        if n_max is not None:
            if n_max < observed_max:
                raise ValueError(
                    f"Provided n_max={n_max} is smaller than the largest graph "
                    f"in the dataset ({observed_max} nodes). Use n_max >= {observed_max}."
                )
            self.n_max = n_max
        else:
            self.n_max = observed_max
            
        logger.info(f"N_max = {self.n_max} (dataset max: {observed_max})")

    def __len__(self):
        return len(self.ids)

    def _pad_graph(self, graph):
        """
        Pads a variable-size Levi graph to fixed N_max size as dense tensors.

        Returns a dict with:
            "x":               [N_max, 3]        — coords (zero for face/off nodes)
            "node_categories": [N_max, 5]        — one-hot over
                                                   (vertex, ground, roof, wall, off)
            "y":               [N_max, N_max, 1] — edge class labels
                                                   (0=off, 1=vertex-vertex, 2=vertex-face)
            "node_mask":       [N_max]           — 1 for vertex (coordinate-carrying)
                                                   nodes; consumed by coordinate
                                                   centring/metrics/scale only
            "id":              str
            "type":            str
        """
        x = graph["x"]                  # [N, 3]
        edge_index = graph["edge_index"]  # [2, E]
        N = x.size(0)
        N_max = self.n_max

        # 1. Pad coordinates: vertex nodes keep their coords, the rest are zeros
        x_padded = torch.zeros((N_max, 3), dtype=torch.float32)
        x_padded[:N] = x

        # 2. Node categories: one-hot over the 5 Levi classes, OFF pads
        labels_padded = torch.full((N_max,), OFF, dtype=torch.long)
        labels_padded[:N] = graph["node_labels"]
        node_categories = F.one_hot(labels_padded, NUM_NODE_CLASSES).float()

        # 3. Node mask: 1 for coordinate-carrying (vertex) nodes
        node_mask = (labels_padded == VERTEX).float()

        # 4. Dense edge-class matrix from sparse labeled edges
        y = torch.zeros((N_max, N_max, 1), dtype=torch.float32)
        if edge_index.numel() > 0:
            y[edge_index[0], edge_index[1], 0] = graph["edge_attr"].float()

        return {
            "x": x_padded,
            "node_categories": node_categories,
            "y": y,
            "node_mask": node_mask,
            "id": graph["id"],
            "type": graph["type"],
        }

    def __getitem__(self, index):
        obj_id = self.ids[index]
        
        if self.is_single_lod:
            item = self._pad_graph(self.lod_data[self.lods[0]][obj_id])
            if self.transform:
                item = self.transform(item)
            return item
        else:
            items = tuple(
                self._pad_graph(self.lod_data[lod][obj_id]) for lod in self.lods
            )
            if self.transform:
                items = tuple(self.transform(item) for item in items)
            return items

# ==============================================================================
# PyTorch DataLoader Collate Function
# ==============================================================================

def collate_single_lod(graphs):
    """
    Collates a list of padded graph dictionaries into a single batch via torch.stack.
    All graphs are already padded to the same N_max, so simple stacking works.
    """
    ids = [g["id"] for g in graphs]
    types = [g["type"] for g in graphs]
    
    return {
        "x": torch.stack([g["x"] for g in graphs], dim=0),                  # [B, N_max, 3]
        "node_categories": torch.stack([g["node_categories"] for g in graphs], dim=0),  # [B, N_max, 2]
        "y": torch.stack([g["y"] for g in graphs], dim=0),                  # [B, N_max, N_max, 1]
        "node_mask": torch.stack([g["node_mask"] for g in graphs], dim=0),  # [B, N_max]
        "ids": ids,
        "types": types,
    }


def graph_collate_fn(batch):
    """
    Custom collate function for PyTorch DataLoader. Supports batching single graphs
    or aligned pairs/triplets of graphs across multiple LODs.
    """
    if not batch:
        return {}
        
    # Check if we have pairs/triplets
    if isinstance(batch[0], tuple):
        # Transpose list of tuples to tuple of lists
        lods_batch = list(zip(*batch))
        return tuple(collate_single_lod(lod_graphs) for lod_graphs in lods_batch)
    else:
        return collate_single_lod(batch)
