"""Per-building geometric feature vectors.

Property set follows 3dSAGER (Genossar et al., arXiv:2511.06300, Table 1).
This module computes the analytically-defined core here; shape descriptors
ported from tudelft3d/3d-building-metrics are added alongside (see Task 5).
numpy + scipy only.
"""
import numpy as np
from scipy.spatial import ConvexHull

CORE_FEATURES = [
    "area", "volume", "height_diff", "num_vertices", "num_faces",
    "convex_hull_area", "ave_centroid_distance", "bbox_diagonal",
]


def mesh_from_cityjson(cj):
    """(verts[V,3], faces) from the first Solid; faces are outer-ring index lists."""
    verts = np.asarray(cj["vertices"], dtype=float)
    obj = next(iter(cj["CityObjects"].values()))
    faces = [ring[0] for ring in obj["geometry"][0]["boundaries"][0]]
    return verts, faces


def _triangulate(face):
    """Fan-triangulate a polygon ring into (i0,i1,i2) index triples."""
    return [(face[0], face[k], face[k + 1]) for k in range(1, len(face) - 1)]


def surface_area(verts, faces):
    total = 0.0
    for face in faces:
        for a, b, c in _triangulate(face):
            total += 0.5 * np.linalg.norm(np.cross(verts[b] - verts[a], verts[c] - verts[a]))
    return float(total)


def signed_volume(verts, faces):
    """Divergence theorem; assumes CCW-outward rings (our converter guarantees this)."""
    vol = 0.0
    for face in faces:
        for a, b, c in _triangulate(face):
            vol += np.dot(verts[a], np.cross(verts[b], verts[c]))
    return abs(vol) / 6.0


def building_features(cj, feature_set="full"):
    verts, faces = mesh_from_cityjson(cj)
    centroid = verts.mean(axis=0)
    try:
        hull_area = float(ConvexHull(verts).area)
    except Exception:
        hull_area = float("nan")  # coplanar/degenerate hull
    f = {
        "area": surface_area(verts, faces),
        "volume": signed_volume(verts, faces),
        "height_diff": float(verts[:, 2].max() - verts[:, 2].min()),
        "num_vertices": float(len(verts)),
        "num_faces": float(len(faces)),
        "convex_hull_area": hull_area,
        "ave_centroid_distance": float(np.linalg.norm(verts - centroid, axis=1).mean()),
        "bbox_diagonal": float(np.linalg.norm(verts.max(0) - verts.min(0))),
    }
    return f


def feature_matrix(cjs, feature_set="full"):
    names = list(building_features(cjs[0], feature_set).keys()) if cjs else CORE_FEATURES
    rows = [[building_features(cj, feature_set)[n] for n in names] for cj in cjs]
    return np.asarray(rows, dtype=float), names
