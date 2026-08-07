#!/usr/bin/env python3
"""LightningDataModule for (LOD1, LOD2) mesh-token pairs."""
from functools import partial

import lightning as L
import torch
from torch.utils.data import DataLoader, random_split

from src.dataset.mesh_dataset import NUM_BINS, MeshDataset, mesh_collate_fn, specials


class MeshDataModule(L.LightningDataModule):
    """Splits `MeshDataset` and serves padded token batches.

    Mirrors `CityJSONDataModule`: same split ratios, same seeded `random_split`,
    same worker plumbing. It has no marginals/coord-scale statistics to compute --
    the tokenizer's per-building unit-box normalization already does that job.

    Args:
        dataset_dir (str | Path): e.g. ``data/The Hague/mini``.
        lod_in, lod_out (str): sub-directory names of the two LODs.
        num_bins (int): coordinate discretization; must match the model's.
        margin_lo, margin_hi (float | sequence): per-axis (x, y, z) headroom
            below / above the LOD1 normalization box, so LOD2 geometry that
            leaves that box is not clipped. Overflow is one-directional --
            only +z is non-zero by default -- hence the split by side.
        max_faces (int, optional): drop buildings above this triangle count.
        max_files (int, optional): read only the first N files per LOD.
        seed (int): seeds the train/val/test split, for reproducibility.
    """

    def __init__(self, dataset_dir, lod_in="LOD1_synth", lod_out="LOD2",
                 num_bins=NUM_BINS, margin_lo=(0.0, 0.0, 0.0),
                 margin_hi=(0.0, 0.0, 0.1), max_faces=None, max_files=None,
                 batch_size=8, train_val_test_split=(0.8, 0.1, 0.1),
                 num_workers=0, persistent_workers=False, seed=42):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.lod_in = lod_in
        self.lod_out = lod_out
        self.num_bins = num_bins
        self.margin_lo = margin_lo
        self.margin_hi = margin_hi
        self.max_faces = max_faces
        self.max_files = max_files
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.num_workers = num_workers
        # torch raises if persistent_workers is set with num_workers=0.
        self.persistent_workers = persistent_workers and num_workers > 0
        self.seed = seed

        self.full_dataset = None
        self.train_dataset = self.val_dataset = self.test_dataset = None

        assert abs(sum(train_val_test_split) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    def setup(self, stage=None):
        if self.full_dataset is not None:
            return

        self.full_dataset = MeshDataset(
            dataset_dir=self.dataset_dir, lod_in=self.lod_in, lod_out=self.lod_out,
            num_bins=self.num_bins, margin_lo=self.margin_lo,
            margin_hi=self.margin_hi,
            max_faces=self.max_faces, max_files=self.max_files,
        )
        total = len(self.full_dataset)
        if total == 0:
            raise ValueError(f"No LOD pairs found under {self.dataset_dir}")

        r_train, r_val, _ = self.train_val_test_split
        train_size = int(r_train * total)
        val_size = int(r_val * total)
        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.full_dataset,
            [train_size, val_size, total - train_size - val_size],
            generator=generator,
        )

    @property
    def max_seq_len(self):
        """Longest single segment in the corpus; sizes the positional embedding."""
        if self.full_dataset is None:
            raise RuntimeError("max_seq_len requires setup() to have run first.")
        return self.full_dataset.max_seq_len

    def _loader(self, dataset, shuffle):
        # partial, not a lambda: worker processes spawn on Windows and have to
        # pickle the collate_fn.
        collate = partial(mesh_collate_fn, pad=specials(self.num_bins)[2])
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            collate_fn=collate,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)
