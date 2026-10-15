#!/usr/bin/env python3
"""LightningDataModule for (LOD1, LOD2) mesh-token pairs."""
import logging
from functools import partial

import lightning as L
from torch.utils.data import DataLoader, Subset

from src.dataset.mesh_dataset import NUM_BINS, MeshDataset, mesh_collate_fn, specials
from src.dataset.splits import (buffer_train, grid_labels, random_split_indices,
                                 tile_split_indices)

logger = logging.getLogger(__name__)


class MeshDataModule(L.LightningDataModule):
    """Splits `MeshDataset` and serves padded token batches.

    The default ``split="random"`` is the seeded per-building `random_split`
    every run up to v5 used; ``split="tile"`` keeps each 3DBAG tile (CityJSON
    file) whole within one split (see `src.dataset.splits`). It has no marginals/coord-scale statistics to compute --
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
        split (str): ``"random"`` (per building), ``"tile"`` (whole files) or
            ``"grid"`` (whole square blocks of `split_block_m` metres).
        split_block_m (float): grid block side; required for ``"grid"``.
        split_buffer_m (float): for ``"tile"``/``"grid"``, drop training
            buildings closer than this to any val/test building.
        ortho_dir, image_size, image_margin: optional orthophoto crops, passed
            straight to `MeshDataset`; ``ortho_dir=None`` disables them.
        image_min_coverage (float): with `ortho_dir`, buildings whose mosaic
            covers less than this are dropped from each split *after*
            splitting, so the split itself is unchanged.
    """

    def __init__(self, dataset_dir, lod_in="LOD1_synth", lod_out="LOD2",
                 num_bins=NUM_BINS, margin_lo=(0.0, 0.0, 0.0),
                 margin_hi=(0.0, 0.0, 0.1), max_faces=None, max_files=None,
                 batch_size=8, train_val_test_split=(0.8, 0.1, 0.1),
                 num_workers=0, persistent_workers=False, seed=42,
                 tokenization="coord", ortho_dir=None, image_size=256, image_margin=0.1,
                 image_min_coverage=1.0, split="random", split_block_m=None,
                 split_buffer_m=0.0):
        super().__init__()
        if split not in ("random", "tile", "grid"):
            raise ValueError(f"Unknown split {split!r}; expected 'random', 'tile' or 'grid'.")
        if split == "grid" and not split_block_m:
            raise ValueError("split='grid' needs split_block_m (block side in metres).")
        if split == "random" and split_buffer_m:
            raise ValueError("split_buffer_m needs a 'tile' or 'grid' split: a buffer around "
                             "randomly scattered val/test buildings would empty train.")
        self.split = split
        self.split_block_m = split_block_m
        self.split_buffer_m = split_buffer_m
        self.tokenization = tokenization
        self.ortho_dir = ortho_dir
        self.image_size = image_size
        self.image_margin = image_margin
        self.image_min_coverage = image_min_coverage
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
        self.skipped_no_image = {}
        self.split_groups = {}
        self.buffer_dropped = 0

        assert abs(sum(train_val_test_split) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    def setup(self, stage=None):
        if self.full_dataset is not None:
            return

        self.full_dataset = MeshDataset(
            dataset_dir=self.dataset_dir, lod_in=self.lod_in, lod_out=self.lod_out,
            num_bins=self.num_bins, margin_lo=self.margin_lo,
            margin_hi=self.margin_hi,
            max_faces=self.max_faces, max_files=self.max_files,
            tokenization=self.tokenization, ortho_dir=self.ortho_dir,
            image_size=self.image_size, image_margin=self.image_margin,
            image_min_coverage=self.image_min_coverage,
        )
        total = len(self.full_dataset)
        if total == 0:
            raise ValueError(f"No LOD pairs found under {self.dataset_dir}")

        ratios = self.train_val_test_split
        if self.split == "random":
            parts = random_split_indices(total, ratios, self.seed)
            self.split_groups = {}
        else:
            groups = (self.full_dataset.files if self.split == "tile"
                      else grid_labels(self.full_dataset.xy, self.split_block_m))
            parts, assign = tile_split_indices(groups, ratios, self.seed)
            names = ("train", "val", "test")
            self.split_groups = {g: names[k] for g, k in assign.items()}
            parts[0], self.buffer_dropped = buffer_train(
                self.full_dataset.xy, parts[0], parts[1] + parts[2], self.split_buffer_m)
            if self.buffer_dropped:
                logger.info("Buffer of %g m dropped %d/%d training buildings",
                            self.split_buffer_m, self.buffer_dropped,
                            self.buffer_dropped + len(parts[0]))
        self.train_dataset, self.val_dataset, self.test_dataset = (
            Subset(self.full_dataset, p) for p in parts)

        # Split first, then drop buildings without an orthophoto: the split
        # stays exactly v5's, so an image run initialised from v5 never
        # validates on buildings v5 trained on.
        if self.full_dataset.ortho is not None:
            keep = self.full_dataset.has_image
            for name in ("train", "val", "test"):
                subset = getattr(self, f"{name}_dataset")
                kept = [i for i in subset.indices if keep[i]]
                self.skipped_no_image[name] = len(subset.indices) - len(kept)
                setattr(self, f"{name}_dataset", Subset(self.full_dataset, kept))
            logger.info("Skipped buildings without an orthophoto: %s", self.skipped_no_image)

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
