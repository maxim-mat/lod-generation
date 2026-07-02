#!/usr/bin/env python3
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

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
                        for ring in face:
                            for vid in ring:
                                ground_vertex_indices.add(vid)
                                
        elif geom_type in ["MultiSurface", "CompositeSurface"]:
            # boundaries structure: [face][ring][vertex]
            # values structure: [face]
            for face_i, face in enumerate(boundaries):
                if face_i >= len(values):
                    continue
                sem_idx = values[face_i]
                if sem_idx in ground_surface_indices:
                    for ring in face:
                        for vid in ring:
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


def parse_cityjson_file_to_graphs(filepath, normalize_coords=False):
    """
    Parses a single CityJSON file into a list of building graph dicts.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        cj = json.load(f)

    v_raw = np.array(cj["vertices"], dtype=float)
    if "transform" in cj:
        scale = np.array(cj["transform"]["scale"])
        translate = np.array(cj["transform"]["translate"])
        v_raw = v_raw * scale + translate

    graphs = {}

    for obj_id, city_obj in cj.get("CityObjects", {}).items():
        geom_list = city_obj.get("geometry", [])
        if not geom_list:
            continue

        edges = set()
        active_vertex_indices = set()

        # Traverse geometry elements to retrieve active vertices and edges
        for geom in geom_list:
            boundaries = geom.get("boundaries", [])
            if geom["type"] == "Solid":
                boundaries = [surface for shell in boundaries for surface in shell]
            elif geom["type"] not in ["MultiSurface", "CompositeSurface"]:
                continue

            for surface in boundaries:
                if not surface:
                    continue
                # The outer boundary ring is surface[0]
                ring = surface[0]
                for i in range(len(ring)):
                    u = ring[i]
                    v = ring[(i + 1) % len(ring)]
                    active_vertex_indices.add(u)
                    active_vertex_indices.add(v)
                    edges.add((u, v))
                    edges.add((v, u))

        if not active_vertex_indices:
            continue

        # Map original vertex indices to local indices 0..N-1
        sorted_active_vids = sorted(active_vertex_indices)
        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_active_vids)}

        node_features = np.array([v_raw[old_idx] for old_idx in sorted_active_vids])

        # Normalize coordinates relative to base footprint center
        if normalize_coords:
            base_center = get_base_center(geom_list, v_raw, active_vertex_indices)
            node_features = node_features - base_center

        edge_index = np.array([[idx_map[u], idx_map[v]] for u, v in edges]).T
        if edge_index.size == 0:
            edge_index = np.empty((2, 0), dtype=np.int64)

        graphs[obj_id] = {
            "id": obj_id,
            "x": torch.tensor(node_features, dtype=torch.float32),
            "edge_index": torch.tensor(edge_index, dtype=torch.long),
            "type": city_obj.get("type", "Unknown")
        }

    return graphs

# ==============================================================================
# PyTorch Dataset Object
# ==============================================================================

NUM_NODE_CLASSES = 2  # 0 = Active, 1 = Virtual

class CityJSONDataset(Dataset):
    def __init__(self, dataset_dir, lods, normalize_coords=False, transform=None,
                 n_max=None):
        """
        Args:
            dataset_dir (str or Path): Folder containing dataset (e.g. data/The Hague).
            lods (int, str, list, tuple): Single LOD or a list of LODs to match.
            normalize_coords (bool): If True, shifts nodes such that center of building base is at (0, 0, 0).
            transform (callable, optional): Optional transform to apply on graph items.
            n_max (int, optional): Maximum number of nodes per graph. All graphs are padded
                to this size with Virtual nodes. If None, auto-detected from the dataset
                as the maximum observed node count.
        """
        self.dataset_dir = Path(dataset_dir)
        self.normalize_coords = normalize_coords
        self.transform = transform
        
        # Parse LOD input type
        if isinstance(lods, (list, tuple, set, np.ndarray)):
            self.lods = [int(l) for l in lods]
            self.is_single_lod = False
        else:
            self.lods = [int(lods)]
            self.is_single_lod = True
            
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
            
        if not self.ids:
            print(f"Warning: No matching CityObjects found across requested LODs: {self.lods}")

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
            
        print(f"N_max = {self.n_max} (dataset max: {observed_max})")

    def __len__(self):
        return len(self.ids)

    def _pad_graph(self, graph):
        """
        Pads a variable-size graph to fixed N_max size and produces dense tensors
        with Active/Virtual node categories.
        
        Returns a dict with:
            "x":               [N_max, 3]        — coordinates (zero for virtual)
            "node_categories": [N_max, 2]        — one-hot: [1,0]=Active, [0,1]=Virtual
            "y":               [N_max, N_max, 1]  — dense adjacency (0 for virtual-involving)
            "node_mask":       [N_max]            — 1=Active, 0=Virtual
            "id":              str
            "type":            str
        """
        x = graph["x"]                  # [N, 3]
        edge_index = graph["edge_index"]  # [2, E]
        N = x.size(0)
        N_max = self.n_max
        
        # 1. Pad coordinates: active nodes keep their coords, virtual nodes get zeros
        x_padded = torch.zeros((N_max, 3), dtype=torch.float32)
        x_padded[:N] = x
        
        # 2. Node categories: one-hot [Active, Virtual]
        #    Active = [1, 0], Virtual = [0, 1]
        node_categories = torch.zeros((N_max, NUM_NODE_CLASSES), dtype=torch.float32)
        node_categories[:N, 0] = 1.0   # Active
        node_categories[N:, 1] = 1.0   # Virtual
        
        # 3. Node mask: 1 for active, 0 for virtual
        node_mask = torch.zeros(N_max, dtype=torch.float32)
        node_mask[:N] = 1.0
        
        # 4. Dense adjacency matrix from sparse edge_index
        y = torch.zeros((N_max, N_max, 1), dtype=torch.float32)
        if edge_index.numel() > 0:
            src = edge_index[0]  # [E]
            dst = edge_index[1]  # [E]
            y[src, dst, 0] = 1.0
        
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
