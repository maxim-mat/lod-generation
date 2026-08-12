"""(LOD1, LOD2) triangle-mesh pairs and the MeshAnything-style mesh tokenizer.

Tokenization follows MeshAnything: Artist-Created Mesh Generation with
Autoregressive Transformers (Chen et al., arXiv:2406.10163), which in turn
inherits the sequence ordering of MeshGPT / PolyGen: vertices sorted ascending
by z-y-x (z vertical), each face rotated so its lowest vertex index comes first,
faces ordered by their vertex indices, and every mesh rescaled into the unit box
[-0.5, 0.5] before its coordinates are discretized.

Deliberate departure from the paper: MeshAnything feeds the transformer VQ-VAE
codebook indices, learned over face embeddings. We emit the discretized
coordinates directly (9 tokens per triangle), so the tokenizer is an exact,
trainable-parameter-free inverse and there is no second model to train first.
The VQ-VAE is a compression step, not a correctness one -- worth adding only
once sequence length is the binding constraint.

CityJSON parsing and fan triangulation are reused from the existing pipeline
(`src.dataset.dataset._iter_faces`, `src.eval.building_features._triangulate`)
rather than reimplemented.
"""
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset.dataset import _iter_faces
from src.eval.building_features import _triangulate

logger = logging.getLogger(__name__)

# Vocabulary layout: [0, num_bins) are coordinate values, then three specials.
# Kept as module constants for the default bin count; `vocab_size` is the only
# thing the model needs, and the specials keep their offsets relative to it.
NUM_BINS = 128
BOS, EOS, PAD = NUM_BINS, NUM_BINS + 1, NUM_BINS + 2
# Adjacent Mesh Tokenization's interruption token, the paper's '&'. Sits *above*
# the existing three so the coordinate vocabulary keeps its exact layout and a
# checkpoint trained before AMT still has the right number of output rows.
BREAK = NUM_BINS + 3


def vocab_size(num_bins=NUM_BINS, tokenization="coord"):
    """Coordinate bins plus BOS / EOS / PAD, and BREAK under AMT.

    Args:
        num_bins (int): coordinate discretization.
        tokenization (str): ``"coord"`` (default) or ``"amt"``. Only AMT pays
            for the break token, so existing runs are bit-identical.
    """
    return num_bins + (4 if tokenization == "amt" else 3)


def specials(num_bins=NUM_BINS):
    """(bos, eos, pad) token ids for a given bin count.

    Deliberately still a 3-tuple: several call sites unpack it as one, and BREAK
    is not a sequence delimiter in the same sense -- it appears *inside* the
    mesh body. Use `break_token` for that.
    """
    return num_bins, num_bins + 1, num_bins + 2


def break_token(num_bins=NUM_BINS):
    """AMT's '&' interruption token for a given bin count."""
    return num_bins + 3


# ----------------------------------------------------------------------
# Normalization and quantization
# ----------------------------------------------------------------------

def normalize_to_unit_box(verts, ref=None, margin_lo=0.0, margin_hi=0.0):
    """Map ``verts`` into [-0.5, 0.5] and return ``(verts_n, center, scale)``.

    The longest axis of the bounding box spans the full unit interval; the
    others stay proportional, so building shape is not distorted.

    Args:
        verts: [V, 3] coordinates in metres.
        ref: optional [*, 3] whose bounding box defines the frame instead. The
            LOD1/LOD2 pair must share one frame -- rescaling each mesh by its own
            box would erase exactly the height difference the model has to learn,
            and at sampling time only the LOD1 box is known.
        margin_lo, margin_hi: scalar or [3] headroom added below / above the
            reference box on each axis, as a fraction of its scale. LOD2 is not
            contained in the LOD1 box -- the ridge rises above the LOD1 height,
            which is only the *median* roof height -- and without headroom
            `quantize` clips it flat onto the box face. Split by side because
            the overflow is one-directional: measured on mini, +z needs ~0.09
            while the other five sides need nothing at p99. A symmetric margin
            would buy the same ridge clearance for twice the range, coarsening
            every z bin to reserve space under the ground plane that no vertex
            ever reaches. Both depend only on the LOD1 box, so sampling
            reproduces the frame exactly.

    Returns:
        tuple: (verts_n [V, 3], center [3], scale [3]). ``center`` is the box
        centre shifted by the margin imbalance and ``scale`` already includes
        the margins, so the inverse stays ``verts_n * scale + center``. Scale is
        per axis, so uneven margins scale the axes differently -- by the same
        ratio for every building, which keeps the frame consistent across the
        corpus and exactly invertible.
    """
    box = np.asarray(verts if ref is None else ref, dtype=float)
    lo, hi = box.min(axis=0), box.max(axis=0)
    base = float((hi - lo).max())
    if not np.isfinite(base) or base <= 0:
        base = 1.0  # a degenerate (single-point) building must not divide by 0

    m_lo = np.broadcast_to(np.asarray(margin_lo, dtype=float), (3,))
    m_hi = np.broadcast_to(np.asarray(margin_hi, dtype=float), (3,))
    center = (lo + hi) / 2.0 + base * (m_hi - m_lo) / 2.0
    scale = base * (1.0 + m_lo + m_hi)
    return (np.asarray(verts, dtype=float) - center) / scale, center, scale


def quantize(verts, num_bins=NUM_BINS):
    """[-0.5, 0.5] coordinates to integer bins in [0, num_bins)."""
    q = np.rint((np.asarray(verts, dtype=float) + 0.5) * (num_bins - 1))
    return np.clip(q, 0, num_bins - 1).astype(np.int64)


def dequantize(q, num_bins=NUM_BINS):
    """Exact inverse of `quantize` on the grid: bin centres back to [-0.5, 0.5]."""
    return np.asarray(q, dtype=float) / (num_bins - 1) - 0.5


# ----------------------------------------------------------------------
# Canonical ordering
# ----------------------------------------------------------------------

def canonicalize(verts, faces):
    """Put a triangle mesh in the paper's canonical order.

    Duplicate vertices are merged, vertices sorted ascending by (z, y, x), each
    face cyclically rotated so its lowest index leads, and faces sorted by their
    index triple. Rotation rather than reversal: reversing would flip the face
    normal, and these meshes are closed solids whose winding carries the
    inside/outside distinction.

    Deterministic and idempotent, which is what makes tokenize/detokenize an
    exact pair.

    Args:
        verts: [V, 3] float or integer coordinates.
        faces: [F, 3] vertex indices.

    Returns:
        tuple: (verts [V', 3] same dtype, faces [F, 3] int64).
    """
    verts = np.asarray(verts)
    faces = np.asarray(faces, dtype=np.int64)

    # Merge exact duplicates first: quantization can collapse two vertices onto
    # one grid point, and a second pass over the same mesh must then agree.
    uniq, inverse = np.unique(verts, axis=0, return_inverse=True)

    # lexsort's last key is primary, so this is a z-then-y-then-x sort.
    order = np.lexsort((uniq[:, 0], uniq[:, 1], uniq[:, 2]))
    rank = np.empty(len(uniq), dtype=np.int64)
    rank[order] = np.arange(len(uniq))

    faces = rank[inverse.reshape(-1)[faces]]
    roll = faces.argmin(axis=1)
    cols = (roll[:, None] + np.arange(3)[None, :]) % 3
    faces = np.take_along_axis(faces, cols, axis=1)
    faces = faces[np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))]

    return uniq[order], faces


def signed_volume(verts, faces):
    """Divergence-theorem volume. Meaningful only for a closed surface.

    Positive when the winding puts normals outward, which is the convention
    `canonicalize` preserves and every downstream normal metric assumes.
    """
    if len(faces) == 0:
        return 0.0
    tri = np.asarray(verts, dtype=float)[np.asarray(faces, dtype=np.int64)]
    return float(np.einsum("ij,ij->i",
                           tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def fix_winding(verts, faces):
    """Make winding consistent across shared edges, then orient outward.

    Two triangles sharing an edge are consistently wound exactly when they
    traverse that edge in *opposite* directions. This walks the face-adjacency
    graph from an arbitrary seed per connected component, flipping any face that
    disagrees with the neighbour it was reached from, then flips the whole mesh
    if the resulting signed volume is negative.

    trimesh's `fix_normals(multibody=False)`, which is what MeshAnything's
    `main.py` calls after decoding (`buaacyw/MeshAnything`).

    Lives here rather than in `src.eval` because `amt_detokenize` *needs* it:
    AMT stores a face as "the previous two vertices plus one new one", which
    fixes the triangle but not which way round it is wound. See that function.

    Args:
        verts: [V, 3] coordinates, used only for the final outward test.
        faces: [F, 3] vertex indices.

    Returns:
        np.ndarray: [F, 3] int64, some rows reversed.
    """
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        return faces

    edge_faces = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            if u != v:                      # a degenerate edge joins nothing
                edge_faces[(min(u, v), max(u, v))].append(fi)

    flip = np.zeros(len(faces), dtype=bool)
    seen = np.zeros(len(faces), dtype=bool)

    for seed in range(len(faces)):
        if seen[seed]:
            continue
        seen[seed] = True
        stack = [seed]
        while stack:
            fi = stack.pop()
            tri = faces[fi][::-1] if flip[fi] else faces[fi]
            for i in range(3):
                u, v = int(tri[i]), int(tri[(i + 1) % 3])
                if u == v:
                    continue
                for fj in edge_faces[(min(u, v), max(u, v))]:
                    if seen[fj]:
                        continue
                    seen[fj] = True
                    g = faces[fj]
                    # fj is unvisited so its flip flag is still False; if it
                    # walks (u, v) the same way round, the two disagree.
                    flip[fj] = any(int(g[k]) == u and int(g[(k + 1) % 3]) == v
                                   for k in range(3))
                    stack.append(fj)

    out = faces.copy()
    out[flip] = out[flip][:, ::-1]
    if signed_volume(verts, out) < 0:
        out = out[:, ::-1]
    return out


# ----------------------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------------------

def tokenize(verts, faces, num_bins=NUM_BINS):
    """Triangle mesh in [-0.5, 0.5] to a flat token sequence, 9 tokens per face.

    Quantization happens before canonicalization so the ordering is decided on
    the same grid the tokens live on; ordering the floats first lets two nearly
    equal coordinates land in one bin *after* being ranked apart, which breaks
    the round trip.

    Returns:
        np.ndarray: [9 * F] int64 coordinate tokens, no BOS/EOS (the dataset
        adds those, matching the paper's placement around the mesh tokens only).
    """
    verts, faces = canonicalize(quantize(verts, num_bins), faces)
    return verts[faces].reshape(-1).astype(np.int64)


def detokenize(tokens, num_bins=NUM_BINS):
    """Exact inverse of `tokenize`, up to the quantization step.

    Special tokens and any trailing partial face are dropped, so a truncated or
    padded generated sequence still decodes to the mesh it managed to emit.

    Returns:
        tuple: (verts [V, 3] float, faces [F, 3] int64) in [-0.5, 0.5].
    """
    tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
    tokens = tokens[(tokens >= 0) & (tokens < num_bins)]
    tokens = tokens[: len(tokens) - len(tokens) % 9]
    if len(tokens) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

    q = tokens.reshape(-1, 3)                       # one row per face vertex
    faces = np.arange(len(q), dtype=np.int64).reshape(-1, 3)
    q, faces = canonicalize(q, faces)
    return dequantize(q, num_bins), faces


# ----------------------------------------------------------------------
# Adjacent Mesh Tokenization (MeshAnything V2, arXiv:2408.02555, Algorithm 1)
# ----------------------------------------------------------------------

def _amt_strip(faces):
    """Algorithm 1: faces to a vertex-index walk, ``None`` marking a break.

    A face adjacent to the previous one costs a *single* new vertex, because its
    other two are the last two already in the sequence. When no unvisited
    adjacent face exists the walk emits a break and restarts from the
    lowest-indexed unvisited face -- the paper's '&'.

    Ties are broken by lowest face index, which is what keeps the output unique
    per mesh: `canonicalize` has already fixed that order, and the paper relies
    on the same ("we consistently choose the face with the earlier index in the
    sorted list whenever there are multiple choices").

    Deliberately no vertex swap ('$', V2 section 3.1). It buys a further ~8%
    compression at slightly worse accuracy in the paper's own Table 2, and costs
    a second special token plus a case in the inverse.

    Args:
        faces: [F, 3] vertex indices, already canonical.

    Returns:
        list: vertex indices, with ``None`` where a break belongs.
    """
    n = len(faces)
    if n == 0:
        return []

    edge_faces = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            if u != v:
                edge_faces[(min(u, v), max(u, v))].append(fi)

    unvisited = set(range(n))
    seq, cursor = [], 0

    def take_lowest_unvisited():
        nonlocal cursor
        while cursor < n and cursor not in unvisited:
            cursor += 1
        return cursor if cursor < n else None

    start = take_lowest_unvisited()
    unvisited.discard(start)
    seq.extend(int(v) for v in faces[start])

    while unvisited:
        u, v = seq[-2], seq[-1]
        third = None
        for fj in sorted(edge_faces.get((min(u, v), max(u, v)), [])):
            if fj not in unvisited:
                continue
            rest = [int(w) for w in faces[fj] if w != u and w != v]
            if rest:                       # a degenerate face has no third vertex
                third, _ = rest[0], unvisited.discard(fj)
                break
        if third is not None:
            seq.append(third)
            continue

        nxt = take_lowest_unvisited()
        if nxt is None:
            break
        unvisited.discard(nxt)
        seq.append(None)
        seq.extend(int(w) for w in faces[nxt])

    return seq


def amt_tokenize(verts, faces, num_bins=NUM_BINS):
    """Triangle mesh in [-0.5, 0.5] to an AMT token sequence.

    Same canonical order and same quantization as `tokenize` -- AMT changes only
    how the face list is spelled out, not the geometry or the sort. A vertex
    costs 3 coordinate tokens; a break costs 1.

    Measured on `data/The Hague/mini` LOD2: 0.55x the coordinate tokenizer's
    length, against ~0.49x reported on Objaverse.

    Returns:
        np.ndarray: [L] int64, no BOS/EOS (the dataset adds those).
    """
    q, f = canonicalize(quantize(verts, num_bins), faces)
    brk = break_token(num_bins)

    out = []
    for entry in _amt_strip(f):
        if entry is None:
            out.append(brk)
        else:
            out.extend(int(c) for c in q[entry])
    return np.asarray(out, dtype=np.int64)


def amt_detokenize(tokens, num_bins=NUM_BINS):
    """Inverse of `amt_tokenize`: reverse Algorithm 1.

    Within a strip the first three vertices are a face and every vertex after
    that closes one with the preceding two -- a triangle strip. A break resets
    the strip. BOS/EOS/PAD and any trailing partial vertex are dropped, so a
    truncated generation still decodes the faces it managed to close.

    **Winding is recovered, not stored.** AMT records that three vertices form a
    triangle but not which way round it is wound: the reconstruction always
    reads `(previous two, new)`, which for a consistently wound closed mesh is
    the *reverse* of the original face. `fix_winding` repairs that -- consistent
    across shared edges, then outward by the volume test -- which is exactly why
    the reference runs `fix_normals()` after decoding. On an open or badly
    broken mesh the global flip is a guess, so AMT is a poorer fit there than
    the coordinate tokenizer, whose inverse is exact.

    Returns:
        tuple: (verts [V, 3] float, faces [F, 3] int64) in [-0.5, 0.5].
    """
    brk = break_token(num_bins)
    tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
    tokens = tokens[(tokens >= 0) & ((tokens < num_bins) | (tokens == brk))]

    tris, strip, buf = [], [], []
    for t in tokens:
        if t == brk:
            strip, buf = [], []
            continue
        buf.append(int(t))
        if len(buf) < 3:
            continue
        strip.append(buf)
        buf = []
        if len(strip) >= 3:
            tris.append(strip[-3] + strip[-2] + strip[-1])

    if not tris:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

    q = np.asarray(tris, dtype=np.int64).reshape(-1, 3)   # one row per face vertex
    faces = np.arange(len(q), dtype=np.int64).reshape(-1, 3)
    q, faces = canonicalize(q, faces)
    verts = dequantize(q, num_bins)
    return verts, fix_winding(verts, faces)


# ----------------------------------------------------------------------
# Mesh IO
# ----------------------------------------------------------------------

def write_obj(path, verts, faces):
    """Write a triangle mesh as Wavefront .obj (1-based indices, no normals)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in np.asarray(verts, dtype=float)]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in np.asarray(faces, dtype=np.int64)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_cityjson_file_to_meshes(filepath):
    """Parse a CityJSON file into ``{object_id: (verts [V, 3], faces [F, 3])}``.

    Every surface's outer ring is fan-triangulated; inner rings (courtyards) are
    dropped, exactly as `_iter_faces` already dropped them for the Levi graphs.
    Coordinates come out in metres, with the file's transform applied.

    Raises:
        OSError: the file cannot be read.
        ValueError: the file is not parseable CityJSON.
    """
    filepath = Path(filepath)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            cj = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{filepath} is not valid JSON: {exc}") from exc

    try:
        v_raw = np.array(cj["vertices"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{filepath} has no usable 'vertices' array: {exc}") from exc

    if "transform" in cj:
        v_raw = v_raw * np.array(cj["transform"]["scale"]) + np.array(cj["transform"]["translate"])

    meshes = {}
    for obj_id, city_obj in cj.get("CityObjects", {}).items():
        tris = []
        for geom in city_obj.get("geometry", []):
            for ring, _ in _iter_faces(geom):
                tris.extend(_triangulate(ring))
        if not tris:
            continue

        tris = np.asarray(tris, dtype=np.int64)
        active, faces = np.unique(tris, return_inverse=True)
        meshes[obj_id] = (v_raw[active], faces.reshape(-1, 3).astype(np.int64))

    return meshes


def _scan_lod_dir(dataset_dir, lod_dir, max_files=None):
    """Load every CityJSON under ``<dataset_dir>/<lod_dir>`` into meshes."""
    root = Path(dataset_dir) / lod_dir
    if not root.is_dir():
        raise FileNotFoundError(f"No LOD folder {root!s} under {dataset_dir!r}")

    meshes, seen = {}, 0
    for parent, _, files in sorted(os.walk(root)):
        for name in sorted(files):
            if not name.lower().endswith((".json", ".city.json")):
                continue
            if max_files is not None and seen >= max_files:
                return meshes
            seen += 1
            try:
                meshes.update(parse_cityjson_file_to_meshes(Path(parent) / name))
            except (OSError, ValueError) as exc:
                # One corrupt tile must not kill an epoch's worth of loading.
                logger.warning("skipping %s: %s", name, exc)
    return meshes


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

class MeshDataset(Dataset):
    """(LOD1, LOD2) token-sequence pairs, one item per building.

    Both meshes of a pair are normalized in the *LOD1* frame, which is the only
    frame available at sampling time.

    Args:
        dataset_dir (str | Path): e.g. ``data/The Hague/mini``.
        lod_in (str): sub-directory holding the conditioning meshes. Defaults to
            ``LOD1_synth``: the real ``LOD1`` folder only covers Source B,
            while the synthesized LOD1 exists for every LOD2 building.
        lod_out (str): sub-directory holding the target meshes.
        num_bins (int): coordinate discretization, 128 in the paper.
        margin_lo, margin_hi (float | sequence): per-axis (x, y, z) headroom
            below / above the LOD1 box, as a fraction of its scale. See
            :func:`normalize_to_unit_box`; the measured requirement for all six
            sides is logged at construction so these can be calibrated rather
            than guessed. Only +z is non-zero by default.
        max_faces (int, optional): drop buildings whose LOD2 exceeds this;
            sequence length is 9 tokens per face and attention is quadratic.
        max_files (int, optional): stop after this many files per LOD. For
            smoke tests and sample dumps -- not a sampling strategy.
    """

    def __init__(self, dataset_dir, lod_in="LOD1_synth", lod_out="LOD2",
                 num_bins=NUM_BINS, margin_lo=(0.0, 0.0, 0.0),
                 margin_hi=(0.0, 0.0, 0.1), max_faces=None, max_files=None,
                 tokenization="coord"):
        self.dataset_dir = Path(dataset_dir)
        self.num_bins = num_bins
        if tokenization not in ("coord", "amt"):
            raise ValueError(f"Unknown tokenization: {tokenization!r}. "
                             "Expected 'coord' or 'amt'.")
        self.tokenization = tokenization
        self.margin_lo = np.broadcast_to(np.asarray(margin_lo, dtype=float), (3,))
        self.margin_hi = np.broadcast_to(np.asarray(margin_hi, dtype=float), (3,))
        self.bos, self.eos, self.pad = specials(num_bins)

        meshes_in = _scan_lod_dir(self.dataset_dir, lod_in, max_files)
        meshes_out = _scan_lod_dir(self.dataset_dir, lod_out, max_files)

        ids = sorted(set(meshes_in) & set(meshes_out))
        if max_faces is not None:
            kept = [i for i in ids
                    if len(meshes_out[i][1]) <= max_faces and len(meshes_in[i][1]) <= max_faces]
            logger.info("Dropped %d/%d buildings over max_faces=%d",
                        len(ids) - len(kept), len(ids), max_faces)
            ids = kept

        self.ids = ids
        self.pairs = [(meshes_in[i], meshes_out[i]) for i in ids]
        if not self.ids:
            logger.warning("No buildings shared between %s and %s under %s",
                           lod_in, lod_out, self.dataset_dir)

        # Longest *segment* the model must position-embed. Positions restart in
        # the target segment (see `MeshTransformer.forward`), so this is a max
        # over the two halves, not their sum -- and unlike a sum it cannot be
        # exceeded by a batch that pairs the longest condition with the longest
        # target. The condition contributes 9 tokens per face, the target those
        # plus BOS (the trailing EOS is never fed back in).
        if self.tokenization == "amt":
            # AMT's length depends on face *adjacency*, not just face count, so
            # unlike the 9-per-face formula it cannot be derived -- it has to be
            # measured. One tokenizing pass over the corpus at load (~1 ms per
            # building); the alternative is a 10-per-face upper bound that would
            # oversize the positional embedding by ~2x and throw away the
            # compression this tokenizer exists for.
            self.max_seq_len = max((max(len(item["cond"]), len(item["tgt"]) - 1)
                                    for item in (self[i] for i in range(len(self.ids)))),
                                   default=1)
        else:
            self.max_seq_len = max((max(9 * len(a[1]), 9 * len(b[1]) + 1)
                                    for a, b in self.pairs), default=1)
        logger.info("MeshDataset: %d buildings, %s tokenizer, longest segment %d tokens",
                    len(self.ids), self.tokenization, self.max_seq_len)
        self._log_overflow()

    def _log_overflow(self):
        """Report the margin LOD2 actually needs on each of the six box sides.

        The margins must cover this or `quantize` clips those vertices flat onto
        a box face. Reported as percentiles, not a max: on the mini corpus five
        of the six sides need nothing at all and the extreme tail is source
        defects (one LOD2 sits 9 m outside a 4 m LOD1 box), so a max-based
        target would trade everyone's coordinate resolution for a handful of
        broken pairs. Measured at load so the config stays calibrated to
        whatever corpus is in use -- mini is explicitly not
        distribution-faithful and under-represents the tallest roofs.
        """
        if not self.pairs:
            return

        # With no margin the normalized box is exactly [-0.5, 0.5] on its
        # longest axis, so `n.max - 0.5` and `-n.min - 0.5` are precisely what
        # margin_hi and margin_lo have to match, side by side.
        need_hi, need_lo = [], []
        for (v_in, _), (v_out, _) in self.pairs:
            n = normalize_to_unit_box(v_out, ref=v_in)[0]
            need_hi.append(n.max(axis=0) - 0.5)
            need_lo.append(-n.min(axis=0) - 0.5)
        need_hi, need_lo = np.array(need_hi), np.array(need_lo)

        # Half a bin of slack: `quantize` rounds to nearest, so an excess under
        # that lands on the outermost bin regardless -- which is where it
        # belongs. Without it this counts the ~1e-16 float noise from LOD1_synth
        # sharing the LOD2 footprint exactly, and reports half the corpus.
        tol = 0.5 / (self.num_bins - 1)
        clipped = ((need_hi > self.margin_hi + tol) |
                   (need_lo > self.margin_lo + tol)).any(axis=1).mean()

        log = logger.warning if clipped > 0.02 else logger.info
        for need, margin, sign in ((need_hi, self.margin_hi, "+"),
                                   (need_lo, self.margin_lo, "-")):
            for k, axis in enumerate("xyz"):
                log("margin needed on %s%s: p50 %+.3f p95 %+.3f p99 %+.3f "
                    "max %+.3f (configured %.3f)", sign, axis,
                    *np.percentile(need[:, k], [50, 95, 99]), need[:, k].max(),
                    margin[k])
        log("margins lo=%s hi=%s clip %.1f%% of buildings",
            np.round(self.margin_lo, 3).tolist(),
            np.round(self.margin_hi, 3).tolist(), 100 * clipped)

    def __len__(self):
        return len(self.ids)

    def mesh_pair(self, index):
        """Raw ``((verts, faces), (verts, faces))`` in metres, for .obj dumps."""
        return self.pairs[index]

    def __getitem__(self, index):
        (v_in, f_in), (v_out, f_out) = self.pairs[index]
        v_in_n, center, scale = normalize_to_unit_box(
            v_in, margin_lo=self.margin_lo, margin_hi=self.margin_hi)
        v_out_n, _, _ = normalize_to_unit_box(
            v_out, ref=v_in, margin_lo=self.margin_lo, margin_hi=self.margin_hi)

        tok = amt_tokenize if self.tokenization == "amt" else tokenize
        cond = tok(v_in_n, f_in, self.num_bins)
        body = tok(v_out_n, f_out, self.num_bins)
        tgt = np.concatenate([[self.bos], body, [self.eos]])

        return {
            "cond": torch.from_numpy(cond),
            "tgt": torch.from_numpy(tgt),
            "id": self.ids[index],
            # Kept so a generated sequence can be put back in metres.
            "center": torch.tensor(center, dtype=torch.float32),
            "scale": torch.tensor(scale, dtype=torch.float32),
        }


def mesh_collate_fn(batch, pad=PAD):
    """Right-pad variable-length sequences and report where the padding is.

    ``*_pad_mask`` is True at padded positions, matching the
    ``key_padding_mask`` convention of ``torch.nn.MultiheadAttention``.
    """
    def stack(key):
        seqs = [item[key] for item in batch]
        width = max(len(s) for s in seqs)
        out = torch.full((len(seqs), width), pad, dtype=torch.long)
        mask = torch.ones((len(seqs), width), dtype=torch.bool)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = s
            mask[i, : len(s)] = False
        return out, mask

    cond, cond_mask = stack("cond")
    tgt, tgt_mask = stack("tgt")
    out = {"cond": cond, "cond_pad_mask": cond_mask,
           "tgt": tgt, "tgt_pad_mask": tgt_mask,
           "ids": [item["id"] for item in batch]}
    for key in ("center", "scale"):
        if key in batch[0]:
            out[key] = torch.stack([item[key] for item in batch])
    return out


if __name__ == "__main__":
    # Dump a few real (LOD1, LOD2) pairs as .obj so the inputs can be eyeballed.
    # Reads a handful of mini-dataset tiles only.
    import sys

    logging.basicConfig(level=logging.INFO)
    root = sys.argv[1] if len(sys.argv) > 1 else "data/The Hague/mini"
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "outputs/mesh_samples")

    # Three buildings x (lod1, lod2, lod2-round-tripped) = 9 small files.
    ds = MeshDataset(root, max_files=1)
    for k in range(min(3, len(ds))):
        (v1, f1), (v2, f2) = ds.mesh_pair(k)
        name = ds.ids[k].replace("/", "_")
        write_obj(out_dir / f"{name}_lod1.obj", v1, f1)
        write_obj(out_dir / f"{name}_lod2.obj", v2, f2)
        # Round-tripped LOD2: what the model can at best reproduce at this
        # bin count. Eyeballing this separates tokenizer loss from model loss.
        item = ds[k]
        v_rt, f_rt = detokenize(item["tgt"].numpy(), ds.num_bins)
        write_obj(out_dir / f"{name}_lod2_roundtrip.obj",
                  v_rt * item["scale"].numpy() + item["center"].numpy(), f_rt)
        print(f"{ds.ids[k]}: lod1 {len(f1)} tris, lod2 {len(f2)} tris, "
              f"seq {len(item['cond'])}+{len(item['tgt'])}")
    print(f"wrote {out_dir}")
