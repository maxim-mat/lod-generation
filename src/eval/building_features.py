"""Per-building geometric feature vectors.

Property set follows 3dSAGER (Genossar et al., arXiv:2511.06300, Table 1).
The analytically-defined core is computed here; the Table-1 shape descriptors
are ported one-to-one from 3dSAGER's own `object_properties.py`
(github.com/BarGenossar/3dSAGER, ObjectPropertiesProcessor) — attributed inline.
numpy + scipy only.
"""
import math

import numpy as np
from scipy.spatial import ConvexHull

CORE_FEATURES = [
    "area", "volume", "height_diff", "num_vertices", "num_faces",
    "convex_hull_area", "ave_centroid_distance", "bbox_diagonal",
]

# 3dSAGER Table-1 shape descriptors, added on top of the core.
SHAPE_FEATURES = [
    "perimeter", "perimeter_index", "shape_index", "circumference",
    "fractality", "elongation", "hemisphericality", "cubeness",
    "axes_symmetry", "density", "num_floors",
]

FULL_FEATURES = CORE_FEATURES + SHAPE_FEATURES
# "welldefined" drops the LoD-ambiguous / ill-defined descriptors: num_floors
# (distinct-z count is meaningless once vertices diffuse), fractality (log-ratio
# blows up on near-flat meshes), circumference (compactness proxy already
# covered by cubeness/hemisphericality).
WELLDEFINED_FEATURES = [f for f in FULL_FEATURES
                        if f not in ("num_floors", "fractality", "circumference")]

_FEATURE_SETS = {"full": FULL_FEATURES, "welldefined": WELLDEFINED_FEATURES}


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


def ring_perimeter(verts, faces):
    """Length of the footprint ring: the face whose vertices all sit at z_min,
    falling back to the z_max ring. Per 3dSAGER `_get_perimeter`; clamped to >=1
    so downstream ratios (shape_index, density) stay finite on degenerate meshes."""
    for ref in (verts[:, 2].min(), verts[:, 2].max()):
        for face in faces:
            ring = verts[face]
            if len(ring) >= 2 and np.allclose(ring[:, 2], ref):
                p = sum(np.linalg.norm(ring[k] - ring[(k + 1) % len(ring)])
                        for k in range(len(ring)))
                if p > 0:
                    return max(float(p), 1.0)
    return 1.0


def _elongation(verts):
    """sqrt(lambda_max / lambda_min) of the vertex-covariance PCA. 3dSAGER
    `_get_elongation`. nan on a degenerate (flat/collinear) point cloud."""
    if len(verts) < 3:
        return float("nan")
    w = np.linalg.eigvalsh(np.cov(verts, rowvar=False))
    return float(np.sqrt(w.max() / w.min())) if w.min() > 0 else float("nan")


def _axes_symmetry(verts):
    """Mean per-axis std over the unique coordinate values. 3dSAGER
    `_get_axes_symmetry` (its coord arrays are de-duplicated per axis)."""
    return float(np.mean([np.std(np.unique(verts[:, k])) for k in range(3)]))


def building_features(cj, feature_set="full"):
    if feature_set not in _FEATURE_SETS:
        raise ValueError(f"unknown feature_set {feature_set!r}; "
                         f"expected one of {sorted(_FEATURE_SETS)}")
    verts, faces = mesh_from_cityjson(cj)
    centroid = verts.mean(axis=0)
    try:
        hull_area = float(ConvexHull(verts).area)
    except Exception:
        hull_area = float("nan")  # coplanar/degenerate hull

    area = surface_area(verts, faces)
    volume = signed_volume(verts, faces)
    perimeter = ring_perimeter(verts, faces)
    # 3dSAGER clamps area to >=1 before the ratio/log descriptors.
    a = max(area, 1.0)

    f = {
        # --- core (Task 4) ---
        "area": area,
        "volume": volume,
        "height_diff": float(verts[:, 2].max() - verts[:, 2].min()),
        "num_vertices": float(len(verts)),
        "num_faces": float(len(faces)),
        "convex_hull_area": hull_area,
        "ave_centroid_distance": float(np.linalg.norm(verts - centroid, axis=1).mean()),
        "bbox_diagonal": float(np.linalg.norm(verts.max(0) - verts.min(0))),
        # --- 3dSAGER Table-1 descriptors (object_properties.py) ---
        "perimeter": perimeter,
        "perimeter_index": 2 * math.sqrt(math.pi * a) / perimeter,   # _get_perimeter_ind
        "shape_index": perimeter / math.sqrt(4 * math.pi * a),       # _get_shape_ind
        "circumference": 4 * math.pi * math.pow(3 * volume / (4 * math.pi), 2 / 3) / a
        if volume > 0 else float("nan"),                             # _get_circumference
        "fractality": 1 - math.log(volume) / (1.5 * math.log(a))
        if volume > 0 and a > 1 else float("nan"),                   # _get_fractality
        "elongation": _elongation(verts),                            # _get_elongation
        "hemisphericality": 3 * math.sqrt(2) * math.sqrt(math.pi) * volume / math.pow(a, 1.5),
        "cubeness": 6 * math.pow(volume, 2 / 3) / a if volume > 0 else 0.0,  # _get_cubeness
        "axes_symmetry": _axes_symmetry(verts),                      # _get_axes_symmetry
        "density": area / perimeter,                                 # _get_density
        "num_floors": float(len(np.unique(verts[:, 2]))),            # _get_num_floors
    }
    return {name: f[name] for name in _FEATURE_SETS[feature_set]}


def feature_matrix(cjs, feature_set="full"):
    names = _FEATURE_SETS.get(feature_set, FULL_FEATURES) if not cjs \
        else list(building_features(cjs[0], feature_set).keys())
    rows = [[building_features(cj, feature_set)[n] for n in names] for cj in cjs]
    return np.asarray(rows, dtype=float), names
