#!/usr/bin/env python3
import math

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
        persistent_workers=False,
        seed=42,
        n_max=None,
        upper_limit_nodes=None,
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
            persistent_workers (bool): Keep DataLoader workers alive across epochs.
                Ignored when num_workers=0.
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
        # torch raises if persistent_workers is set with num_workers=0.
        self.persistent_workers = persistent_workers and num_workers > 0
        self.seed = seed
        self.n_max = n_max
        self.upper_limit_nodes = upper_limit_nodes
        
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
                upper_limit_nodes=self.upper_limit_nodes,
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

    def compute_marginals(self):
        """Empirical node/edge class frequencies over the training split.

        These define the limit distribution of the discrete diffusion when
        `discrete_noise_type='marginal'` (MiDi's default). Computed on the train
        split only, so the val/test splits do not leak into the noise schedule.

        Returns:
            tuple[Tensor, Tensor]: node marginals [num_node_classes] over
            (vertex, ground, roof, wall, off), and edge marginals
            [num_edge_classes] over (off, vertex-vertex, vertex-face) counted
            across off-diagonal entries of the padded edge-class matrix.
        """
        from src.dataset.dataset import NUM_EDGE_CLASSES

        if self.train_dataset is None:
            raise RuntimeError("compute_marginals() requires setup() to have run first.")

        node_counts = None
        edge_counts = torch.zeros(NUM_EDGE_CLASSES, dtype=torch.float64)

        for item in self.train_dataset:
            # Multi-LOD datasets yield a tuple; the model trains on the first LOD.
            if isinstance(item, tuple):
                item = item[0]

            categories = item["node_categories"]
            if node_counts is None:
                node_counts = torch.zeros(categories.shape[-1], dtype=torch.float64)
            node_counts += categories.sum(dim=0).double()

            adjacency = item["y"].squeeze(-1)
            n = adjacency.shape[0]
            off_diag = ~torch.eye(n, dtype=torch.bool)
            edges = adjacency[off_diag].long()
            edge_counts += torch.bincount(edges, minlength=NUM_EDGE_CLASSES).double()

        x_marginals = (node_counts / node_counts.sum()).float()
        e_marginals = (edge_counts / edge_counts.sum()).float()
        return x_marginals, e_marginals

    def compute_z_shift(self):
        """Pooled mean of vertex-node z over the train split (se2 only).

        The se2 mode keeps absolute heights; subtracting the train-split mean
        makes the z channel zero-mean so the N(0, 1) position prior matches
        the data through the whole chain. Stored in checkpoint hparams like
        `coord_scale`.
        """
        if self.train_dataset is None:
            raise RuntimeError("compute_z_shift() requires setup() to have run first.")

        total, count = 0.0, 0.0
        for item in self.train_dataset:
            if isinstance(item, tuple):
                item = item[0]
            mask = item["node_mask"].bool()
            total += float(item["x"][mask][:, 2].double().sum())
            count += float(mask.sum())
        return total / max(count, 1.0)

    def compute_coord_scale(self):
        """Pooled standard deviation of the active-node coordinates, train split only.

        Dividing the targets by this makes the position diffusion's unit-variance
        Gaussian noise and its N(0, I) limit distribution match the data. Raw
        metres leave the schedule badly skewed: with a std of ~4.3 m the SNR at
        t = T/2 is 16.3 rather than 2.3, so the low-SNR regime the reverse chain
        starts from is barely trained.

        A single scalar, not per-building (that would erase building size and is
        not invertible at sampling time) and not per-axis (that would break the
        yaw-equivariance of the 'so2' network). Coordinates are centred exactly
        as `CityJSONDiffusionModule._centre_positions` centres them, so the
        statistic describes the tensor the model actually regresses.

        Computed on the train split only, so val/test do not leak into it.

        Returns:
            float: the scale in metres, strictly positive.
        """
        if self.train_dataset is None:
            raise RuntimeError("compute_coord_scale() requires setup() to have run first.")

        count = 0
        total = 0.0
        total_sq = 0.0

        for item in self.train_dataset:
            # Multi-LOD datasets yield a tuple; the model trains on the first LOD.
            if isinstance(item, tuple):
                item = item[0]

            # Real nodes = vertices *and* face nodes, matching
            # `_centre_positions`, which centres over 1 - P(Off). node_mask
            # marks vertex nodes only; using it measured a different tensor
            # from the one the model divides (7% high on The Hague LOD2).
            real = item["node_categories"][..., -1] == 0
            active = item["x"][real]
            if active.numel() == 0:
                continue

            centred = (active - active.mean(dim=0, keepdim=True)).double()
            count += centred.numel()
            total += centred.sum().item()
            total_sq += (centred ** 2).sum().item()

        if count == 0:
            raise ValueError("No active nodes in the training split; cannot compute a coordinate scale.")

        mean = total / count
        variance = max(total_sq / count - mean * mean, 0.0)
        # A degenerate split (every building a single point) would give 0.
        return max(math.sqrt(variance), 1e-6)

    def compute_dist_r_max(self, coord_scale, quantile=0.999):
        """Cutoff radius r_max for the Bessel distance basis, in the model's
        normalised coordinate units, from the train split.

        The DimeNet Bessel basis needs a length scale to place its zeros across.
        Rather than a preset molecular default, take a high quantile of the
        per-building maximum pairwise distance (robust to a few oversized
        footprints) and divide by `coord_scale` to match the normalised
        coordinates the network sees. Pairwise distances are translation- and
        z-shift-invariant, so no centring is needed.

        Efficient on the fly: one pass over the (already in-memory) train split,
        a small `cdist` per building -- the same order of work as
        `compute_coord_scale`, not a new heavyweight traversal. Computed on the
        train split only, so val/test do not leak in.

        Args:
            coord_scale (float): metres per normalised unit, from
                `compute_coord_scale`.
            quantile (float): distance quantile used as the cutoff.

        Returns:
            float: r_max in normalised units, strictly positive.
        """
        if self.train_dataset is None:
            raise RuntimeError("compute_dist_r_max() requires setup() to have run first.")

        per_building = []
        for item in self.train_dataset:
            if isinstance(item, tuple):
                item = item[0]
            active = item["x"][item["node_mask"].bool()]
            if active.shape[0] < 2:
                continue
            per_building.append(float(torch.cdist(active, active).max()))

        if not per_building:
            raise ValueError("No multi-node buildings in the train split; cannot compute r_max.")

        r_max_metres = float(torch.tensor(per_building).quantile(quantile))
        return max(r_max_metres / coord_scale, 1e-6)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            collate_fn=graph_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            collate_fn=graph_collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            collate_fn=graph_collate_fn,
            pin_memory=True,
        )
