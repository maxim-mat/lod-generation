"""Face-level matching: how much of a mesh is a verbatim copy of another.

The copying question is about whole triangles, not vertices -- a shared vertex
says little, a shared face is something the model could have echoed from its
condition. A face here is its canonical 9-token row: the three quantized corners,
rotated cyclically so the (z, y, x)-smallest leads. That is exactly the row
`tokenize` emits for it, so two meshes share a face iff they share a token row.
Matching is set-based: face order, vertex order and the face's position in the
sequence are all irrelevant. Reversed winding is a *different* face -- a
different token sequence, and a flipped surface.

Three meshes per building:
    gen   -- the model's output,
    gt    -- the LOD2 target,
    lod1  -- the LOD1 condition.
`gt` vs `lod1` is the reference: how much copying the data itself rewards.
"""
import math

import numpy as np
import pandas as pd

from src.dataset.mesh_dataset import (NUM_BINS, amt_detokenize, detokenize,
                                      mesh_collate_fn, quantize, specials)

COUNT_COLUMNS = (
    "n_gen", "n_gt", "n_lod1",
    "gen_in_gt", "gen_in_lod1", "gen_in_lod1_not_gt", "gen_in_gt_not_lod1",
    "gt_in_lod1", "gt_not_lod1",
)
RATE_COLUMNS = (
    "gen_copy",        # generated faces found in LOD1: the model's copy rate
    "gen_precision",   # generated faces found in GT
    "gen_recall",      # GT faces the model produced
    "spurious_copy",   # generated faces copied from LOD1 that GT does not have
    "novel_recall",    # GT faces absent from LOD1 that the model produced
    "copy_recall",     # GT faces present in LOD1 that the model produced
    "gt_copy",         # reference: GT faces found in LOD1 (the data's copy rate)
    "lod1_precision",  # reference: LOD1 faces found in GT (a pure copier's precision)
)


def face_set(verts, faces, num_bins=NUM_BINS):
    """A mesh in the unit box as its set of canonical face rows.

    Args:
        verts: [V, 3] coordinates in [-0.5, 0.5] -- a decoded token sequence or
            a normalized mesh; quantized here, losslessly for grid points.
        faces: [F, 3] vertex indices.
        num_bins: the tokenizer's grid.

    Returns:
        set of 9-tuples of ints, one per non-degenerate face. Faces whose corners
        collapse into one bin are dropped, as `canonicalize` drops them.
    """
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(faces) == 0:
        return set()
    tri = quantize(verts, num_bins)[faces]                    # [F, 3, 3] bins
    key = (tri[..., 2] * num_bins + tri[..., 1]) * num_bins + tri[..., 0]
    ok = ((key[:, 0] != key[:, 1]) & (key[:, 1] != key[:, 2])
          & (key[:, 0] != key[:, 2]))
    tri, key = tri[ok], key[ok]
    # Rotate, never reorder: the (z, y, x)-smallest corner first, winding kept.
    # It is the same rule `canonicalize` applies through vertex ranks, so the row
    # depends on the triangle alone and not on the rest of the mesh.
    cols = (key.argmin(axis=1)[:, None] + np.arange(3)[None, :]) % 3
    tri = np.take_along_axis(tri, cols[:, :, None], axis=1)
    return set(map(tuple, tri.reshape(-1, 9).tolist()))


def face_match(gen, gt, lod1):
    """Overlap counts between three face sets of one building.

    Args:
        gen, gt, lod1: sets from `face_set`.

    Returns:
        dict keyed by `COUNT_COLUMNS`.
    """
    gen_gt, gen_lod1 = gen & gt, gen & lod1
    return {
        "n_gen": len(gen), "n_gt": len(gt), "n_lod1": len(lod1),
        "gen_in_gt": len(gen_gt),
        "gen_in_lod1": len(gen_lod1),
        "gen_in_lod1_not_gt": len(gen_lod1 - gt),
        "gen_in_gt_not_lod1": len(gen_gt - lod1),
        "gt_in_lod1": len(gt & lod1),
        "gt_not_lod1": len(gt - lod1),
    }


def _ratio(num, den):
    return float(num) / float(den) if den else math.nan


def match_rates(counts):
    """Rates from one building's counts, or from counts summed over many.

    Summed counts give the micro (face-pooled) average; `summarize` does both.
    A rate whose denominator is empty is NaN, not 0: a building with no
    generated faces has no copy rate, it does not have a copy rate of zero.

    Returns:
        dict keyed by `RATE_COLUMNS`.
    """
    c = counts
    return {
        "gen_copy": _ratio(c["gen_in_lod1"], c["n_gen"]),
        "gen_precision": _ratio(c["gen_in_gt"], c["n_gen"]),
        "gen_recall": _ratio(c["gen_in_gt"], c["n_gt"]),
        "spurious_copy": _ratio(c["gen_in_lod1_not_gt"], c["n_gen"]),
        "novel_recall": _ratio(c["gen_in_gt_not_lod1"], c["gt_not_lod1"]),
        "copy_recall": _ratio(c["gen_in_gt"] - c["gen_in_gt_not_lod1"], c["gt_in_lod1"]),
        "gt_copy": _ratio(c["gt_in_lod1"], c["n_gt"]),
        "lod1_precision": _ratio(c["gt_in_lod1"], c["n_lod1"]),
    }


def summarize(df, by=None):
    """Micro and macro averages of the rates over the rows of `df`.

    micro pools every face of every building (rates of the summed counts), so
    large buildings weigh more; macro is the mean per-building rate, NaN
    buildings skipped.

    Args:
        df: one row per building with `COUNT_COLUMNS` (e.g. `split_face_match`).
        by: optional column(s) to group by, e.g. "split" or ["split", "mode"].

    Returns:
        DataFrame indexed by (*groups, "micro" | "macro") with a `buildings`
        column and `RATE_COLUMNS`.
    """
    def one(group):
        counts = group[list(COUNT_COLUMNS)]
        micro = match_rates(counts.sum().to_dict())
        per = pd.DataFrame([match_rates(r) for r in counts.to_dict("records")],
                           columns=list(RATE_COLUMNS))
        macro = per.mean(skipna=True).to_dict()
        return pd.DataFrame([{"buildings": len(group), **micro},
                             {"buildings": len(group), **macro}],
                            index=pd.Index(["micro", "macro"], name="average"))

    if by is None:
        return one(df)
    parts = {key: one(group) for key, group in df.groupby(by, sort=False)}
    names = [by] if isinstance(by, str) else list(by)
    return pd.concat(parts, names=names)


def split_face_match(model, dataset, generate, batch_size=16, limit=None, device=None,
                     progress=None):
    """Generate over a dataset and score face matching one batch at a time.

    Only the counts are kept: each batch's meshes are reduced to face sets,
    matched and dropped, so memory stays flat however large the split is.

    Args:
        model: provides `num_bins`, `tokenization` and `decode_tokens`. Coordinate
            and AMT runs only -- a conditioned VQ-VAE decoder needs a condition
            this does not pass.
        dataset: items as `MeshDataset` yields them (a split `Subset` is fine).
        generate: ``generate(batch, positions) -> (tgt, out)``, both [B, L]
            token tensors -- the target as the model was fit on it and the
            generated sequence. `positions` are the dataset indices in the
            batch, for callers that need other items (e.g. a shuffled image).
        batch_size: buildings per `generate` call.
        limit: score only the first `limit` buildings.
        device: if given, batch tensors are moved there before `generate`.
        progress: optional ``progress(done, total)`` callback, once per batch.

    Returns:
        DataFrame, one row per building: ``position``, ``id`` and `COUNT_COLUMNS`.
    """
    n = len(dataset) if limit is None else min(limit, len(dataset))
    pad = specials(model.num_bins)[2]
    cond_inverse = amt_detokenize if model.tokenization == "amt" else detokenize
    rows = []
    for start in range(0, n, batch_size):
        positions = list(range(start, min(start + batch_size, n)))
        batch = mesh_collate_fn([dataset[i] for i in positions], pad=pad)
        if device is not None:
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        tgt, out = generate(batch, positions)
        for k, pos in enumerate(positions):
            lod1 = face_set(*cond_inverse(batch["cond"][k].cpu().numpy(), model.num_bins),
                            model.num_bins)
            gt = face_set(*model.decode_tokens(tgt[k]), model.num_bins)
            gen = face_set(*model.decode_tokens(out[k]), model.num_bins)
            rows.append({"position": pos, "id": batch["ids"][k], **face_match(gen, gt, lod1)})
        if progress is not None:
            progress(positions[-1] + 1, n)
    return pd.DataFrame(rows, columns=["position", "id", *COUNT_COLUMNS])
