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


def face_collate_fn(batch):
    """Right-pad to the longest mesh; ``pad_mask`` is True at padded faces."""
    width = max(len(faces) for faces in batch)
    coords = torch.zeros((len(batch), width, 9), dtype=torch.long)
    mask = torch.ones((len(batch), width), dtype=torch.bool)
    for i, faces in enumerate(batch):
        coords[i, : len(faces)] = faces
        mask[i, : len(faces)] = False
    return {"coords": coords, "pad_mask": mask}


class MeshVQVAEDataModule(L.LightningDataModule):
    """`MeshDataModule`'s splits, served as face batches.

    Takes the same arguments as `MeshDataModule` and delegates to it rather than
    re-reading the corpus, so the two stages cannot drift apart on split ratios,
    seed, margins or `max_faces`.
    """

    def __init__(self, batch_size=16, **mesh_datamodule_kwargs):
        super().__init__()
        self.batch_size = batch_size
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
        return DataLoader(
            _FaceView(split), batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.inner.num_workers,
            persistent_workers=self.inner.persistent_workers,
            collate_fn=face_collate_fn, pin_memory=True,
        )

    def train_dataloader(self):
        return self._loader(self.inner.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.inner.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.inner.test_dataset, shuffle=False)
