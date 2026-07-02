#!/usr/bin/env python3
import argparse
import json
import os
import sys
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import median

# ==============================================================================
# Geometric Conversion Functions
# ==============================================================================
from src.geometry.geometry import convert_to_lod1


# ==============================================================================
# Worker Function
# ==============================================================================

def process_single_file(src_path: Path, dst_path: Path) -> tuple[str, Path, str]:
    """
    Processes a single file. Converts it if it's a CityJSON file, otherwise copies as-is.
    Returns: (status, relative_path, message)
      status: 'converted', 'copied', or 'error'
    """
    rel_path = src_path.name
    # Include parent subfolder name in key for logging readability
    if src_path.parent.name:
        rel_path = f"{src_path.parent.name}/{rel_path}"

    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if JSON
        if src_path.suffix.lower() == '.json':
            with open(src_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, dict) and data.get("type") == "CityJSON":
                converted = convert_to_lod1(data)
                with open(dst_path, 'w', encoding='utf-8') as f:
                    json.dump(converted, f)
                return 'converted', Path(rel_path), "Successfully converted CityJSON to LOD1"

        # If it reaches here, it's either not JSON or not a CityJSON format.
        # Copy as-is to preserve subfolder structures and non-JSON assets (e.g. description.txt)
        shutil.copy2(src_path, dst_path)
        return 'copied', Path(rel_path), "Copied file as-is"

    except Exception as e:
        return 'error', Path(rel_path), str(e)

# ==============================================================================
# Main Runner
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert CityJSON folders from LOD2 to LOD1, maintaining source structures."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Name of the dataset folder (e.g., 'The Hague')."
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run sequentially instead of in parallel."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes for parallel conversion. Defaults to CPU count."
    )
    args = parser.parse_args()

    # Find the data directory
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
        print("Error: Could not locate 'data' directory in project root or relative paths.", file=sys.stderr)
        sys.exit(1)

    # Get available datasets
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

    # Find the dataset directory case-insensitively
    dataset_name = args.dataset
    dataset_dir = None
    for d in data_dir.iterdir():
        if d.is_dir() and d.name.lower() == dataset_name.lower():
            dataset_dir = d
            dataset_name = d.name  # Keep actual casing
            break

    if not dataset_dir:
        print(f"Error: Dataset '{args.dataset}' not found under data directory.", file=sys.stderr)
        if datasets:
            print("\nAvailable datasets:")
            for d in datasets:
                print(f"  - {d}")
        sys.exit(1)

    # Find LOD2 folder (case-insensitive)
    lod2_dir = None
    lod2_casing = "LOD2"
    for d in dataset_dir.iterdir():
        if d.is_dir() and d.name.lower() == "lod2":
            lod2_dir = d
            lod2_casing = d.name  # Keep actual casing
            break

    if not lod2_dir:
        print(f"Error: No LOD2/lod2 folder found in dataset '{dataset_name}'.", file=sys.stderr)
        sys.exit(1)

    # Define LOD1 folder matching casing
    lod1_casing = "LOD1" if lod2_casing.isupper() else "lod1"
    lod1_dir = dataset_dir / lod1_casing

    print("=" * 60)
    print(f"Dataset:       {dataset_name}")
    print(f"Source path:   {lod2_dir}")
    print(f"Target path:   {lod1_dir}")
    print("=" * 60)

    # Collect files recursively
    tasks = []
    for root, dirs, files in os.walk(lod2_dir):
        root_path = Path(root)
        for f in files:
            src_file = root_path / f
            # Construct relative path from lod2 root
            rel_parts = src_file.relative_to(lod2_dir)
            dst_file = lod1_dir / rel_parts
            tasks.append((src_file, dst_file))

    if not tasks:
        print("No files found to process.")
        sys.exit(0)

    print(f"Found {len(tasks)} files to process. Starting conversion...")

    converted_count = 0
    copied_count = 0
    error_count = 0
    start_time = time.time()

    if args.sequential:
        print("Running in sequential mode...")
        for src, dst in tasks:
            status, rel_p, msg = process_single_file(src, dst)
            if status == 'converted':
                converted_count += 1
                print(f"[CONVERTED] {rel_p}")
            elif status == 'copied':
                copied_count += 1
                print(f"[COPIED]    {rel_p} - {msg}")
            else:
                error_count += 1
                print(f"[ERROR]     {rel_p} - {msg}", file=sys.stderr)
    else:
        max_workers = args.workers or os.cpu_count()
        print(f"Running in parallel mode with {max_workers} worker processes...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_file, src, dst): (src, dst) for src, dst in tasks}
            
            for future in as_completed(futures):
                status, rel_p, msg = future.result()
                if status == 'converted':
                    converted_count += 1
                    print(f"[CONVERTED] {rel_p}")
                elif status == 'copied':
                    copied_count += 1
                    print(f"[COPIED]    {rel_p} - {msg}")
                else:
                    error_count += 1
                    print(f"[ERROR]     {rel_p} - {msg}", file=sys.stderr)

    elapsed = time.time() - start_time
    print("=" * 60)
    print("Conversion Summary:")
    print(f"  Total processed: {len(tasks)}")
    print(f"  Converted LOD1:  {converted_count}")
    print(f"  Copied as-is:    {copied_count}")
    print(f"  Errors:          {error_count}")
    print(f"  Time taken:      {elapsed:.2f} seconds")
    print("=" * 60)

    if error_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
