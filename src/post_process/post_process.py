import json
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# ==============================================================================
# CityJSON Format Exporter
# ==============================================================================

# CityJSON semantic surface table used by mesh_to_cityjson
_SEMANTIC_SURFACES = [
    {"type": "GroundSurface"}, {"type": "RoofSurface"}, {"type": "WallSurface"}
]


def _newell_normal(pts):
    """Area vector of a closed polygon (origin-independent), 2x the face normal."""
    n = np.zeros(3)
    for i in range(len(pts)):
        n += np.cross(pts[i], pts[(i + 1) % len(pts)])
    return n


def _plane_patches(verts, faces, normals, offsets, normal_tol, offset_tol):
    """Union-find triangles into coplanar *connected* patches.

    Connectivity is required, not just coplanarity: two roof planes at the same
    height on opposite ends of a building are coplanar but are separate
    surfaces. Sharing an edge is the test, and since disjoint patches share no
    edge the distinction costs nothing extra.

    Returns:
        dict: root triangle index -> list of triangle indices.
    """
    parent = list(range(len(faces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edge_owners = {}
    for t, tri in enumerate(faces):
        for k in range(3):
            key = (min(tri[k], tri[(k + 1) % 3]), max(tri[k], tri[(k + 1) % 3]))
            edge_owners.setdefault(key, []).append(t)

    for owners in edge_owners.values():
        # All pairs, not just the first two: a non-manifold edge with three
        # incident triangles still merges whichever of them are coplanar.
        for a in range(len(owners)):
            for b in range(a + 1, len(owners)):
                t1, t2 = owners[a], owners[b]
                if (float(normals[t1] @ normals[t2]) > 1.0 - normal_tol
                        and abs(offsets[t1] - offsets[t2]) <= offset_tol):
                    r1, r2 = find(t1), find(t2)
                    if r1 != r2:
                        parent[r1] = r2

    patches = {}
    for t in range(len(faces)):
        patches.setdefault(find(t), []).append(t)
    return patches


def _boundary_loops(tris):
    """Closed vertex loops bounding a set of triangles.

    An edge on the patch boundary is traversed once; an interior edge is
    traversed once in each direction. So the boundary is exactly the directed
    edges whose reverse is absent.

    Returns:
        list: closed loops, each a list of vertex ids. Open walks are dropped
        individually -- a malformed patch loses that loop, not the whole mesh.
    """
    directed = set()
    for tri in tris:
        for k in range(3):
            directed.add((int(tri[k]), int(tri[(k + 1) % 3])))

    succ = {}
    for u, v in directed:
        if (v, u) not in directed:
            succ.setdefault(u, []).append(v)

    loops, budget = [], sum(len(v) for v in succ.values())
    while succ:
        start = next(iter(succ))
        loop, cur = [start], start
        while True:
            if not succ.get(cur):
                loop = None                      # dead end: drop this walk
                break
            nxt = succ[cur].pop()
            if not succ[cur]:
                del succ[cur]
            if nxt == start:
                break
            loop.append(nxt)
            cur = nxt
            if len(loop) > budget:               # cycle that never returns
                loop = None
                break
        if loop and len(loop) >= 3:
            loops.append(loop)
    return loops


def _ring_corners(ring, coords):
    """Vertices of a closed ring that actually turn a corner.

    Straight-through vertices are fan-triangulation debris and should go, but
    only if *no* ring needs them -- see `_prune_collinear`.
    """
    out = [v for i, v in enumerate(ring) if v != ring[i - 1]]
    if len(out) < 3:
        return set()

    corners, n = set(), len(out)
    for i in range(n):
        prev, cur, nxt = coords[out[i - 1]], coords[out[i]], coords[out[(i + 1) % n]]
        e1, e2 = cur - prev, nxt - cur
        n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
        if n1 < 1e-12 or n2 < 1e-12:
            continue
        if np.linalg.norm(np.cross(e1, e2)) > 1e-9 * n1 * n2:
            corners.add(out[i])
    return corners


def _prune_collinear(rings, coords):
    """Drop straight-through vertices, but only where every ring agrees.

    A vertex mid-edge on one surface is often a genuine corner of the surface
    next to it. Removing it from the first and not the second leaves a
    T-junction, whose edges no longer pair -- so a merge that prunes ring by
    ring silently opens solids that were closed on input. Measured on mini,
    that alone cost ~8 points of watertightness, and `volumetric_iou` is
    undefined without it.

    Args:
        rings: every ring of the whole mesh, as vertex-id lists.

    Returns:
        list: the same rings with globally-collinear vertices removed; rings
        left with fewer than 3 vertices become empty lists.
    """
    corners = set()
    for ring in rings:
        corners |= _ring_corners(ring, coords)

    pruned = []
    for ring in rings:
        keep = [v for i, v in enumerate(ring) if v != ring[i - 1] and v in corners]
        pruned.append(keep if len(keep) >= 3 else [])
    return pruned


def mesh_to_cityjson(verts, faces, building_id="generated_building", lod="2",
                     normal_tol=1e-2, offset_tol=None):
    """Triangle mesh to CityJSON, merging coplanar triangles back into polygons.

    The inverse of the tokenizer's forward path: `parse_cityjson_file_to_meshes`
    fan-triangulates every surface, so a bare writeback would emit a triangle
    soup wearing a CityJSON hat -- hundreds of 3-vertex "surfaces" where the
    source had a few dozen planar ones. This merges them back, which is what
    makes the output comparable to a real LOD2 file on face counts, semantics
    and val3dity.

    Returns ``{}`` on failure and never raises, so it is safe to call on
    garbage from an untrained model.

    Args:
        verts: [V, 3] coordinates in metres.
        faces: [F, 3] triangle vertex indices, wound outward.
        normal_tol: two triangles are coplanar when ``n1 . n2 > 1 - this``.
        offset_tol: allowed plane-offset gap, metres. None -> ``1e-3 * diagonal``.
            Relative by default on purpose: detokenized vertices sit on a
            128-bin grid whose spacing is ~0.17 m on a 20 m building, so a fixed
            metre tolerance either shatters every sloped plane or fuses
            unrelated ones.

    Returns:
        dict: CityJSON 1.1 with one Building, a single-shell Solid, and
        Ground/Roof/Wall semantics derived from each patch's normal. ``{}`` if
        nothing reconstructable came out.

    Known limitation: a patch joined only at a single vertex can yield two
    genuine outer loops, and largest-area-wins demotes one to a hole.
    """
    try:
        v = np.asarray(verts, dtype=float)
        f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        if len(v) < 3 or len(f) == 0:
            return {}

        diag = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)))
        if not np.isfinite(diag) or diag <= 0:
            return {}
        if offset_tol is None:
            offset_tol = 1e-3 * diag

        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        raw = np.cross(b - a, c - a)
        mag = np.linalg.norm(raw, axis=1)
        alive = mag > 1e-12 * diag * diag
        if not alive.any():
            return {}
        f, a, raw, mag = f[alive], a[alive], raw[alive], mag[alive]

        normals = raw / mag[:, None]
        offsets = np.einsum("ij,ij->i", normals, a)

        # Two passes: every ring first, so collinear pruning can consult the
        # whole mesh rather than one surface at a time.
        patch_rings, patch_normals = [], []
        for tris in _plane_patches(v, f, normals, offsets, normal_tol, offset_tol).values():
            loops = _boundary_loops(f[tris])
            if loops:
                patch_rings.append(loops)
                patch_normals.append(normals[tris[0]])

        flat = _prune_collinear([r for loops in patch_rings for r in loops], v)
        it = iter(flat)
        patch_rings = [[next(it) for _ in loops] for loops in patch_rings]

        cj_faces, sem_values = [], []
        for rings, n_patch in zip(patch_rings, patch_normals):
            rings = [r for r in rings if r]
            if not rings:
                continue

            # Largest ring is the outer boundary; the rest are courtyards.
            areas = [np.linalg.norm(_newell_normal(v[r])) / 2.0 for r in rings]
            order = np.argsort(areas)[::-1]
            rings = [rings[i] for i in order]

            # The input winding already encodes outward -- `canonicalize` rotates
            # faces but never reverses them -- so the patch normal decides, with
            # no centroid heuristic to misjudge a concave building.
            out_rings = []
            for i, ring in enumerate(rings):
                aligned = float(_newell_normal(v[ring]) @ n_patch) > 0
                want = aligned if i == 0 else not aligned
                out_rings.append(ring if want else ring[::-1])

            cj_faces.append(out_rings)
            # Winding-derived nz, never an eigh plane fit: its normal has an
            # arbitrary sign and would swap roof and ground at random. The 0.1
            # cut keeps sloped roof planes as roofs rather than walls.
            nz = float(n_patch[2])
            sem_values.append(1 if nz > 0.1 else (0 if nz < -0.1 else 2))

        if not cj_faces:
            return {}

        used = np.unique(np.concatenate([np.concatenate(rs) for rs in cj_faces]))
        remap = {int(old): i for i, old in enumerate(used)}
        cj_faces = [[[remap[int(x)] for x in ring] for ring in rings] for rings in cj_faces]

        return {
            "type": "CityJSON",
            "version": "1.1",
            "CityObjects": {
                building_id: {
                    "type": "Building",
                    "geometry": [
                        {
                            "type": "Solid",
                            "lod": lod,
                            "boundaries": [cj_faces],
                            "semantics": {
                                "surfaces": _SEMANTIC_SURFACES,
                                "values": [sem_values],
                            },
                        }
                    ],
                }
            },
            "vertices": v[used].tolist(),
            "metadata": {"datasetLod": lod},
        }
    except Exception:
        logger.exception("mesh_to_cityjson failed on a %s-triangle mesh",
                         len(faces) if faces is not None else "?")
        return {}


def save_to_file(cityjson_dict, output_path):
    """
    Writes CityJSON dictionary to file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cityjson_dict, f)
    logger.info(f"CityJSON saved successfully to {output_path}")
