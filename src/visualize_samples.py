#!/usr/bin/env python3
import argparse
import json
import os
import sys
import random
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

# ==============================================================================
# 3D Side-by-Side Visualization Function
# ==============================================================================

def visualize_single_building_side_by_side(obj_id, lod1_obj, lod1_vertices, lod2_obj, lod2_vertices, title):
    """
    Generates a Plotly Figure showing a single building's LOD1 (left) and LOD2 (right) 
    geometries side-by-side.
    """
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("LOD1 (Simplified)", "LOD2 (Detailed)")
    )
    
    palette = qualitative.Alphabet
    obj_color = palette[0]  # Use a consistent, prominent color for the building
    
    def add_geom_to_subplot(city_obj, vertices, col):
        show_in_legend = True
        obj_name = city_obj.get("type", "Building")
        
        for geom in city_obj.get("geometry", []):
            boundaries = geom.get("boundaries", [])
            if geom["type"] == "Solid":
                boundaries = [surface for shell in boundaries for surface in shell]
            elif geom["type"] not in ["MultiSurface", "CompositeSurface"]:
                continue
            
            for surface in boundaries:
                outer_ring = surface[0]
                try:
                    poly_pts = vertices[outer_ring]
                except IndexError:
                    continue
                
                # Append first vertex to close the line loop
                x = np.append(poly_pts[:, 0], poly_pts[0, 0])
                y = np.append(poly_pts[:, 1], poly_pts[0, 1])
                z = np.append(poly_pts[:, 2], poly_pts[0, 2])
                
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color=obj_color, width=3),
                    name=f"{obj_id} ({obj_name})",
                    legendgroup=obj_id,
                    showlegend=show_in_legend
                ), row=1, col=col)
                show_in_legend = False

    add_geom_to_subplot(lod1_obj, lod1_vertices, col=1)
    add_geom_to_subplot(lod2_obj, lod2_vertices, col=2)
    
    # Premium Dark-Mode Styling
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111216",
        plot_bgcolor="#111216",
        font=dict(family="Outfit, Inter, sans-serif", color="#e2e8f0"),
        title=dict(
            text=title,
            x=0.5,
            y=0.95,
            xanchor="center",
            yanchor="top",
            font=dict(size=20, color="#f8fafc", weight="bold")
        ),
        scene=dict(
            aspectmode='data',
            xaxis=dict(gridcolor="#334155", showbackground=False, zerolinecolor="#475569"),
            yaxis=dict(gridcolor="#334155", showbackground=False, zerolinecolor="#475569"),
            zaxis=dict(gridcolor="#334155", showbackground=False, zerolinecolor="#475569")
        ),
        scene2=dict(
            aspectmode='data',
            xaxis=dict(gridcolor="#334155", showbackground=False, zerolinecolor="#475569"),
            yaxis=dict(gridcolor="#334155", showbackground=False, zerolinecolor="#475569"),
            zaxis=dict(gridcolor="#334155", showbackground=False, zerolinecolor="#475569")
        ),
        margin=dict(l=20, r=20, b=20, t=100),
        showlegend=False
    )
    return fig

# ==============================================================================
# Helper to Load & Transform Vertices
# ==============================================================================

def load_cityjson_vertices(cj_data):
    """
    Extracts and scales/translates vertices from CityJSON dictionary.
    """
    vertices = np.array(cj_data["vertices"], dtype=float)
    if "transform" in cj_data:
        scale = np.array(cj_data["transform"]["scale"])
        translate = np.array(cj_data["transform"]["translate"])
        vertices = vertices * scale + translate
    return vertices

# ==============================================================================
# Main Execution Logic
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sample and visualize individual matching LOD1 and LOD2 building geometries side-by-side."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Name of the dataset folder under the data folder (e.g., 'The Hague')."
    )
    parser.add_argument(
        "--num-objects",
        type=int,
        default=30,
        help="Number of matching building geometries to sample and visualize (default: 30)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42)."
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Locate the data directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    possible_data_paths = [
        project_root / "data",
        Path.cwd() / "data",
        Path.cwd() / ".." / "data",
    ]

    data_dir = None
    for p in possible_data_paths:
        if p.is_dir():
            data_dir = p.resolve()
            break

    if not data_dir:
        print("Error: Could not locate 'data' directory.", file=sys.stderr)
        sys.exit(1)

    datasets = [d.name for d in data_dir.iterdir() if d.is_dir()]

    if not args.dataset:
        print("Error: No dataset specified.", file=sys.stderr)
        if datasets:
            print("\nAvailable datasets in data folder:")
            for d in datasets:
                print(f"  - {d}")
            print(f"\nUsage: python {sys.argv[0]} \"<dataset_name>\"")
        else:
            print("\nNo datasets found under data directory.")
        sys.exit(1)

    # Find the dataset folder
    dataset_name = args.dataset
    dataset_dir = None
    for d in data_dir.iterdir():
        if d.is_dir() and d.name.lower() == dataset_name.lower():
            dataset_dir = d
            dataset_name = d.name
            break

    if not dataset_dir:
        print(f"Error: Dataset '{args.dataset}' not found under data directory.", file=sys.stderr)
        sys.exit(1)

    # Find LOD1 and LOD2 subfolders
    lod2_dir = None
    for d in dataset_dir.iterdir():
        if d.is_dir() and d.name.lower() == "lod2":
            lod2_dir = d
            break

    if not lod2_dir:
        print(f"Error: No 'LOD2' subfolder found in dataset '{dataset_name}'.", file=sys.stderr)
        sys.exit(1)

    lod1_dir = dataset_dir / ("LOD1" if lod2_dir.name.isupper() else "lod1")
    if not lod1_dir.is_dir():
        print(f"Error: Converted LOD1 directory '{lod1_dir.name}' does not exist.", file=sys.stderr)
        print("Please run the conversion script first, e.g.:")
        print(f"  python src/convert_lod2_to_lod1.py \"{dataset_name}\"")
        sys.exit(1)

    # Create target sample directory under the dataset directory
    sample_dir = dataset_dir / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Match LOD2 and LOD1 files, and build mapping of all common buildings
    matched_pairs = []
    building_map = {} # obj_id -> (lod2_file_path, lod1_file_path, relative_path)

    print("Scanning dataset files...")
    for root, _, files in os.walk(lod2_dir):
        root_path = Path(root)
        for f in files:
            if f.lower().endswith('.json'):
                lod2_file = root_path / f
                rel_path = lod2_file.relative_to(lod2_dir)
                lod1_file = lod1_dir / rel_path
                if lod1_file.exists():
                    matched_pairs.append((lod2_file, lod1_file, rel_path))
                    
                    # Quickly check building keys without parsing huge vertices yet
                    with open(lod2_file, "r", encoding="utf-8") as f_in:
                        try:
                            lod2_data = json.load(f_in)
                            lod2_keys = set(lod2_data.get("CityObjects", {}).keys())
                        except Exception:
                            continue
                    with open(lod1_file, "r", encoding="utf-8") as f_in:
                        try:
                            lod1_data = json.load(f_in)
                            lod1_keys = set(lod1_data.get("CityObjects", {}).keys())
                        except Exception:
                            continue
                            
                    common_keys = lod2_keys.intersection(lod1_keys)
                    for key in common_keys:
                        building_map[key] = (lod2_file, lod1_file, rel_path)

    if not building_map:
        print("Error: No matching building geometries found in the matched files.")
        sys.exit(1)

    total_buildings = len(building_map)
    print("=" * 60)
    print(f"Dataset:                  {dataset_name}")
    print(f"Total Matched File Pairs: {len(matched_pairs)}")
    print(f"Total Matching Buildings: {total_buildings}")
    print(f"Output Directory:         {sample_dir}")
    print("=" * 60)

    # Sample a small number of building geometries (few dozen)
    sample_size = min(args.num_objects, total_buildings)
    sampled_keys = random.sample(sorted(list(building_map.keys())), sample_size)

    print(f"Sampling {sample_size} building geometries...")

    # Group sampled keys by file pair to minimize file loading operations
    file_to_keys = {}
    for key in sampled_keys:
        paths = building_map[key]
        file_to_keys.setdefault(paths, []).append(key)

    processed_count = 0
    for (lod2_path, lod1_path, rel_path), keys in file_to_keys.items():
        # Load the file pair once
        with open(lod2_path, "r", encoding="utf-8") as f:
            lod2_data = json.load(f)
        with open(lod1_path, "r", encoding="utf-8") as f:
            lod1_data = json.load(f)

        lod2_verts = load_cityjson_vertices(lod2_data)
        lod1_verts = load_cityjson_vertices(lod1_data)

        lod2_objs = lod2_data.get("CityObjects", {})
        lod1_objs = lod1_data.get("CityObjects", {})

        for obj_key in keys:
            processed_count += 1
            print(f"[{processed_count}/{sample_size}] Visualizing geometry: {obj_key}")

            obj_l1 = lod1_objs[obj_key]
            obj_l2 = lod2_objs[obj_key]

            # Create clean, safe filename
            safe_rel_name = str(rel_path.parent).replace("/", "_").replace("\\", "_").replace(" ", "_")
            if safe_rel_name and safe_rel_name != ".":
                filename = f"{safe_rel_name}_{rel_path.stem}_{obj_key}_sxs.html"
            else:
                filename = f"{rel_path.stem}_{obj_key}_sxs.html"
                
            save_path = sample_dir / filename

            title = f"Building Geometry: {obj_key} (LOD1 vs LOD2)"
            fig = visualize_single_building_side_by_side(
                obj_id=obj_key,
                lod1_obj=obj_l1,
                lod1_vertices=lod1_verts,
                lod2_obj=obj_l2,
                lod2_vertices=lod2_verts,
                title=title
            )
            fig.write_html(str(save_path))

    print("\n" + "=" * 60)
    print("Sampling & Visualization Complete!")
    print(f"Saved {sample_size} HTML building pair visualizations to: {sample_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
