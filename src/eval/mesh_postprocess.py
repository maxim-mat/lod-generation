#!/usr/bin/env python3
"""MeshAnything's post-decode mesh cleanup, as an evaluation-only diagnostic.

The reference runs three trimesh calls on the decoded triangle soup before it
writes the .obj (`buaacyw/MeshAnything`, `main.py`, the generation loop):

    scene_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, ...)
    scene_mesh.merge_vertices()
    scene_mesh.update_faces(scene_mesh.unique_faces())
    scene_mesh.fix_normals()

This reproduces those three steps on numpy arrays. trimesh is not a dependency
of this project and is not added for a notebook diagnostic; `canonicalize`
already implements the weld, and the other two are a face dedupe and a winding
propagation over the face-adjacency graph.

**Nothing in the training, evaluation or inference pipeline calls this.** It
exists to measure how much of the `watertight_rate` gap is bookkeeping rather
than geometry -- see `docs/2026-08-12-mesh-tokenizer-path-issues.md`, Issue 4.
Wire it into `mesh_eval` only after that measurement says it is worth it.

Faithfulness notes, both deliberate:
  * Degenerate faces (two vertices collapsed onto one grid point by `quantize`)
    are **not** removed. The reference does not remove them either -- it calls
    `unique_faces`, not `nondegenerate_faces`. They are counted and reported
    instead, because on this corpus they come from the 128-bin grid rather than
    from the model and that distinction matters when reading the numbers.
  * `fix_winding` orients each connected component consistently and then flips
    the whole mesh once if its signed volume is negative. That matches
    trimesh's default `fix_normals(multibody=False)`, which is what `main.py`
    calls.
"""
import logging

import numpy as np

# `fix_winding` / `signed_volume` live with the other geometry primitives rather
# than here: `amt_detokenize` needs winding repair as part of its inverse, and
# importing this module from the dataset would be a cycle. Re-exported so this
# module still reads as the reference's three steps in one place.
from src.dataset.mesh_dataset import canonicalize, fix_winding, signed_volume

logger = logging.getLogger(__name__)

_signed_volume = signed_volume        # back-compat for the self-check below

__all__ = ["weld", "drop_duplicate_faces", "fix_winding", "postprocess",
           "n_degenerate", "report", "signed_volume"]


def weld(verts, faces):
    """`merge_vertices`: merge coincident vertices and reindex the faces.

    Delegates to `canonicalize`, which merges by exact equality on `np.unique`.
    That is the same rule trimesh applies in practice here: its default
    `tol.merge` is 1e-8 while decoded coordinates sit on a 1/128 grid, so no two
    distinct grid points ever fall inside the tolerance.

    Args:
        verts: [V, 3] coordinates.
        faces: [F, 3] vertex indices.

    Returns:
        tuple: (verts [V', 3], faces [F, 3] int64). Face count is preserved.
    """
    return canonicalize(verts, faces)


def drop_duplicate_faces(faces):
    """`update_faces(unique_faces())`: keep the first copy of each face.

    Faces are compared as *unordered* vertex triples, so a face and its
    reversal count as duplicates -- trimesh's `unique_faces` groups on sorted
    rows the same way. Original face order is preserved among the survivors,
    which keeps the canonical ordering `canonicalize` established.

    Args:
        faces: [F, 3] vertex indices.

    Returns:
        np.ndarray: [F', 3] int64.
    """
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        return faces
    _, keep = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    return faces[np.sort(keep)]


def postprocess(verts, faces):
    """All three reference steps, in the reference's order.

    Args:
        verts: [V, 3] coordinates, in any frame (metres or the unit box).
        faces: [F, 3] vertex indices.

    Returns:
        tuple: (verts [V', 3], faces [F', 3] int64).
    """
    verts, faces = weld(verts, faces)
    faces = drop_duplicate_faces(faces)
    return verts, fix_winding(verts, faces)


def n_degenerate(faces):
    """Faces with a repeated vertex index. Reported, never removed -- see module docstring."""
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        return 0
    return int(((faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2])
                | (faces[:, 0] == faces[:, 2])).sum())


def report(mesh):
    """Before/after counts for one ``(verts, faces)`` mesh.

    Returns:
        dict: ``verts``/``faces``/``degenerate``/``watertight`` before and after,
        plus the cleaned mesh under ``mesh``. `watertight` is None if the metric
        helper is unavailable, so a notebook panel never dies on an import.
    """
    verts, faces = mesh
    clean = postprocess(verts, faces)

    try:
        from src.eval.mesh_metrics import is_watertight_mesh
        wt_before = bool(is_watertight_mesh(faces))
        wt_after = bool(is_watertight_mesh(clean[1]))
    except Exception as exc:                      # noqa: BLE001 - diagnostic only
        logger.warning("watertight check unavailable: %s", exc)
        wt_before = wt_after = None

    return {
        "verts_before": len(verts), "verts_after": len(clean[0]),
        "faces_before": len(faces), "faces_after": len(clean[1]),
        "degenerate_before": n_degenerate(faces),
        "degenerate_after": n_degenerate(clean[1]),
        "watertight_before": wt_before, "watertight_after": wt_after,
        "mesh": clean,
    }


if __name__ == "__main__":
    # Self-check: a unit cube, deliberately broken in the three ways the three
    # steps are meant to repair. Run: python -m src.eval.mesh_postprocess
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    f = np.array([[0, 2, 1], [0, 3, 2],      # bottom (-z)
                  [4, 5, 6], [4, 6, 7],      # top    (+z)
                  [0, 1, 5], [0, 5, 4],      # -y
                  [1, 2, 6], [1, 6, 5],      # +x
                  [2, 3, 7], [2, 7, 6],      # +y
                  [3, 0, 4], [3, 4, 7]],     # -x
                 dtype=np.int64)
    assert abs(_signed_volume(v, f) - 1.0) < 1e-9, "fixture cube is not unit/outward"

    # 1. duplicated vertex -> weld must merge it and keep the face count
    v_dup = np.vstack([v, v[6]])
    f_dup = f.copy()
    f_dup[f_dup == 6] = len(v)                     # last face row now uses the copy
    wv, wf = weld(v_dup, f_dup)
    assert len(wv) == 8, f"weld left {len(wv)} vertices, expected 8"
    assert len(wf) == len(f), "weld changed the face count"

    # 2. duplicated face (and its reversal) -> dedupe must drop both copies
    f_extra = np.vstack([f, f[3:4], f[5:6][:, ::-1]])
    assert len(drop_duplicate_faces(f_extra)) == len(f), "duplicate faces survived"

    # 3. scrambled winding -> fix_winding must restore consistency and outwardness
    f_bad = f.copy()
    f_bad[[1, 4, 7, 10]] = f_bad[[1, 4, 7, 10]][:, ::-1]
    assert _signed_volume(v, f_bad) != _signed_volume(v, f), "fixture not actually broken"
    assert abs(_signed_volume(v, fix_winding(v, f_bad)) - 1.0) < 1e-9, "winding not repaired"

    # inward-facing input must come back outward
    assert abs(_signed_volume(v, fix_winding(v, f[:, ::-1])) - 1.0) < 1e-9, "not reoriented"

    # 4. all three together, on a mesh broken every way at once
    pv, pf = postprocess(v_dup, np.vstack([f_dup, f_dup[2:3]]))
    assert len(pv) == 8 and len(pf) == len(f), f"postprocess gave {len(pv)}v {len(pf)}f"
    assert abs(_signed_volume(pv, pf) - 1.0) < 1e-9, "postprocess left a bad orientation"

    # empty in, empty out -- a generation that decoded to nothing must not raise
    ev, ef = postprocess(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    assert len(ev) == 0 and len(ef) == 0

    print("mesh_postprocess self-check OK")
