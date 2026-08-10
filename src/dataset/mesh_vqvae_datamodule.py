#!/usr/bin/env python3
"""Per-face batches for stage-1 VQ-VAE training.

The transformer datamodule serves one item per *building*, as a pair of token
sequences. The VQ-VAE does not care about the pairing -- it learns a vocabulary
of faces -- so this re-views the same dataset as one item per *mesh*, LOD1 and
LOD2 alike.

Both LODs on purpose: LOD1 contributes large wall quads and a ground plane,
LOD2 the small roof triangles. A codebook trained on LOD2 only would see LOD1's
faces as out-of-distribution at exactly the moment stage 2 asks it to encode the
condition.

The split comes from `MeshDataModule` with the same seed, so a building in the
transformer's test split is never in the VQ-VAE's training set. Without that,
every stage-2 test number would be contaminated by a tokenizer that had already
seen the answer.
"""
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from src.dataset.mesh_datamodule import MeshDataModule


class _FaceView(Dataset):
    """One item per mesh: [F, 9] discretized face coordinates.

    Wraps a split of `MeshDataset`; index 2i is item i's LOD1 mesh, 2i+1 its
    LOD2 mesh.
    """

    def __init__(self, split):
        self.split = split

    def __len__(self):
        return 2 * len(self.split)

    def __getitem__(self, index):
        item = self.split[index // 2]
        if index % 2 == 0:
            tokens = item["cond"]                 # no BOS/EOS on the condition
        else:
            tokens = item["tgt"][1:-1]            # strip BOS and EOS
        return tokens.reshape(-1, 9)


class _PairView(Dataset):
    """One item per building: its LOD2 mesh, with its LOD1 mesh as condition.

    For the noise-resistant decoder fine-tune (arXiv:2406.10163 section 4.2),
    which needs the shape condition alongside the mesh being reconstructed.

    LOD2 only, unlike `_FaceView`, because that is all stage 2 ever decodes --
    the LOD1 condition is encoded, never quantized. Serving LOD1 items here too
    would hand the decoder a batch where condition and target are the same mesh,
    and it would learn to copy the condition instead of correcting bad codes.
    """

    def __init__(self, split):
        self.split = split

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index):
        item = self.split[index]
        return {"coords": item["tgt"][1:-1].reshape(-1, 9),   # strip BOS/EOS
                "cond": item["cond"].reshape(-1, 9)}


def _pad_faces(seqs):
    """Right-pad ``[F, 9]`` meshes to the longest; mask is True at padding."""
    width = max(len(faces) for faces in seqs)
    coords = torch.zeros((len(seqs), width, 9), dtype=torch.long)
    mask = torch.ones((len(seqs), width), dtype=torch.bool)
    for i, faces in enumerate(seqs):
        coords[i, : len(faces)] = faces
        mask[i, : len(faces)] = False
    return coords, mask


def face_collate_fn(batch):
    """Right-pad to the longest mesh; ``pad_mask`` is True at padded faces."""
    coords, mask = _pad_faces(batch)
    return {"coords": coords, "pad_mask": mask}


def pair_collate_fn(batch):
    """`face_collate_fn` for the mesh and its condition, padded independently."""
    coords, mask = _pad_faces([item["coords"] for item in batch])
    cond, cond_mask = _pad_faces([item["cond"] for item in batch])
    return {"coords": coords, "pad_mask": mask,
            "cond": cond, "cond_pad_mask": cond_mask}


class MeshVQVAEDataModule(L.LightningDataModule):
    """`MeshDataModule`'s splits, served as face batches.

    Takes the same arguments as `MeshDataModule` and delegates to it rather than
    re-reading the corpus, so the two stages cannot drift apart on split ratios,
    seed, margins or `max_faces`.

    Args:
        condition (bool): serve `(LOD2 mesh, LOD1 condition)` pairs instead of
            single meshes. What the noise-resistant decoder fine-tune needs;
            plain stage-1 training does not use a condition.
    """

    def __init__(self, batch_size=16, condition=False, **mesh_datamodule_kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.condition = condition
        self.inner = MeshDataModule(**mesh_datamodule_kwargs)

    def setup(self, stage=None):
        self.inner.setup(stage)

    @property
    def max_faces_seen(self):
        """Longest mesh in the corpus, in faces; sizes the positional embedding."""
        # max_seq_len counts tokens of the longer segment, target-side including
        # its BOS, so this is the ceiling on faces either LOD can contribute.
        return (self.inner.max_seq_len + 8) // 9

    def _loader(self, split, shuffle):
        view, collate = ((_PairView, pair_collate_fn) if self.condition
                         else (_FaceView, face_collate_fn))
        return DataLoader(
            view(split), batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.inner.num_workers,
            persistent_workers=self.inner.persistent_workers,
            collate_fn=collate, pin_memory=True,
        )

    def train_dataloader(self):
        return self._loader(self.inner.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.inner.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.inner.test_dataset, shuffle=False)
