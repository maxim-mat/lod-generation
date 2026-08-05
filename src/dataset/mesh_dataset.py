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


def vocab_size(num_bins=NUM_BINS):
    """Coordinate bins plus BOS / EOS / PAD."""
    return num_bins + 3


def specials(num_bins=NUM_BINS):
    """(bos, eos, pad) token ids for a given bin count."""
    return num_bins, num_bins + 1, num_bins + 2


# ----------------------------------------------------------------------
# Normalization and quantization
# ----------------------------------------------------------------------

def normalize_to_unit_box(verts, ref=None):
    """Map ``verts`` into [-0.5, 0.5] and return ``(verts_n, center, scale)``.

    The longest axis of the bounding box spans the full unit interval; the
    others stay proportional, so building shape is not distorted.

    Args:
        verts: [V, 3] coordinates in metres.
        ref: optional [*, 3] whose bounding box defines the frame instead. The
            LOD1/LOD2 pair must share one frame -- rescaling each mesh by its own
            box would erase exactly the height difference the model has to learn,
            and at sampling time only the LOD1 box is known.

    Returns:
        tuple: (verts_n [V, 3], center [3], scale float). Inverse is
        ``verts_n * scale + center``.
    """
    box = np.asarray(verts if ref is None else ref, dtype=float)
    lo, hi = box.min(axis=0), box.max(axis=0)
    center = (lo + hi) / 2.0
    scale = float((hi - lo).max())
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0  # a degenerate (single-point) building must not divide by 0
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
        max_faces (int, optional): drop buildings whose LOD2 exceeds this;
            sequence length is 9 tokens per face and attention is quadratic.
        max_files (int, optional): stop after this many files per LOD. For
            smoke tests and sample dumps -- not a sampling strategy.
    """

    def __init__(self, dataset_dir, lod_in="LOD1_synth", lod_out="LOD2",
                 num_bins=NUM_BINS, max_faces=None, max_files=None):
        self.dataset_dir = Path(dataset_dir)
        self.num_bins = num_bins
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

        # Longest sequence the model must position-embed: cond + BOS + tgt + EOS.
        self.max_seq_len = max((9 * (len(a[1]) + len(b[1])) + 2
                                for a, b in self.pairs), default=2)
        logger.info("MeshDataset: %d buildings, max sequence length %d",
                    len(self.ids), self.max_seq_len)

    def __len__(self):
        return len(self.ids)

    def mesh_pair(self, index):
        """Raw ``((verts, faces), (verts, faces))`` in metres, for .obj dumps."""
        return self.pairs[index]

    def __getitem__(self, index):
        (v_in, f_in), (v_out, f_out) = self.pairs[index]
        v_in_n, center, scale = normalize_to_unit_box(v_in)
        v_out_n, _, _ = normalize_to_unit_box(v_out, ref=v_in)

        cond = tokenize(v_in_n, f_in, self.num_bins)
        body = tokenize(v_out_n, f_out, self.num_bins)
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
                  v_rt * float(item["scale"]) + item["center"].numpy(), f_rt)
        print(f"{ds.ids[k]}: lod1 {len(f1)} tris, lod2 {len(f2)} tris, "
              f"seq {len(item['cond'])}+{len(item['tgt'])}")
    print(f"wrote {out_dir}")
