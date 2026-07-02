#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader, random_split

import lightning as L

from src.dataset.dataset import CityJSONDataset, graph_collate_fn


class CityJSONDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_dir,
        lods,
        batch_size=32,
        train_val_test_split=(0.8, 0.1, 0.1),
        normalize_coords=False,
        num_workers=0,
        seed=42,
        n_max=None,
    ):
        """
        PyTorch Lightning DataModule for CityJSON graph datasets.
        
        Args:
            dataset_dir (str or Path): Path to the dataset directory.
            lods (int, str, list): LOD(s) to load.
            batch_size (int): Size of batches returned by the DataLoaders.
            train_val_test_split (tuple of 3 floats): Split ratios for train, val, and test sets. Sum must be 1.0.
            normalize_coords (bool): If True, shifts nodes so base center is at (0, 0, 0).
            num_workers (int): Number of subprocesses to use for data loading.
            seed (int): Random seed for reproducibility of splits.
            n_max (int, optional): Maximum number of nodes per graph. If None, auto-detected
                from the dataset as the maximum observed node count.
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.lods = lods
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.normalize_coords = normalize_coords
        self.num_workers = num_workers
        self.seed = seed
        self.n_max = n_max
        
        # Datasets placeholders
        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        assert abs(sum(train_val_test_split) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    def prepare_data(self):
        # CityJSON data is already local, so no download is needed here.
        pass

    def setup(self, stage=None):
        """
        Preloads the full CityJSONDataset and splits it into train/val/test partitions.
        """
        if self.full_dataset is None:
            self.full_dataset = CityJSONDataset(
                dataset_dir=self.dataset_dir,
                lods=self.lods,
                normalize_coords=self.normalize_coords,
                n_max=self.n_max,
            )
            
            # Expose the resolved n_max for downstream consumers (e.g. model)
            self.n_max = self.full_dataset.n_max
            
            # Split dataset using seed
            total = len(self.full_dataset)
            if total == 0:
                raise ValueError(f"Cannot setup split on empty dataset under {self.dataset_dir}")
                
            r_train, r_val, r_test = self.train_val_test_split
            
            # Compute exact split sizes
            train_size = int(r_train * total)
            val_size = int(r_val * total)
            test_size = total - train_size - val_size  # Ensure remainder matches exactly
            
            generator = torch.Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.full_dataset,
                [train_size, val_size, test_size],
                generator=generator,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=graph_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=graph_collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=graph_collate_fn,
            pin_memory=True,
        )
