import json
from pathlib import Path

import numpy as np


def cityjson_to_obj(cj):
    """Minimal OBJ (vertices + faces) for wandb.Object3D / local inspection."""
    verts = cj["vertices"]
    obj = next(iter(cj["CityObjects"].values()))
    lines = [f"v {x} {y} {z}" for x, y, z in verts]
    for ring in obj["geometry"][0]["boundaries"][0]:
        idx = ring[0]
        lines.append("f " + " ".join(str(i + 1) for i in idx))  # OBJ is 1-indexed
    return "\n".join(lines) + "\n"


def save_records(records, out_dir, n):
    """Persist the first n records: graph .npz + CityJSON + OBJ. Returns saved paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, rec in enumerate(records[:n]):
        stem = out_dir / f"gen_{i}"
        np.savez(stem.with_suffix(".npz"),
                 coords=rec["coords"], node_labels=rec["node_labels"],
                 edge_labels=rec["edge_labels"])
        stem.with_suffix(".city.json").write_text(json.dumps(rec["cityjson"]), encoding="utf-8")
        stem.with_suffix(".obj").write_text(cityjson_to_obj(rec["cityjson"]), encoding="utf-8")
        paths.append(stem)
    return paths
