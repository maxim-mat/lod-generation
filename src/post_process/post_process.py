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

    Mirrors `graph_to_cityjson`'s contract: returns ``{}`` on failure and never
    raises, so it is safe to call on garbage from an untrained model.

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
            # Winding-derived nz, never `straighten_face`: its eigh normal has an
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
