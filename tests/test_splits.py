"""Train/val/test splitting: random (v5's exact split) and geographic by tile."""
import collections

import pytest
import torch
from torch.utils.data import random_split

from src.dataset.splits import random_split_indices, tile_split_indices

RATIOS = (0.8, 0.1, 0.1)


def test_random_split_is_index_identical_to_torch_random_split():
    n, seed = 1037, 42
    sizes = [int(0.8 * n), int(0.1 * n)]
    sizes.append(n - sum(sizes))
    ref = random_split(range(n), sizes, generator=torch.Generator().manual_seed(seed))
    assert random_split_indices(n, RATIOS, seed) == [list(s.indices) for s in ref]


def _groups():
    # 120 tiles of uneven size, like 3DBAG tiles.
    sizes = [5 + (37 * t) % 90 for t in range(120)]
    return [f"tile-{t}" for t, k in enumerate(sizes) for _ in range(k)]


def test_tile_split_never_puts_one_tile_in_two_splits_and_covers_everything():
    groups = _groups()
    splits, assign = tile_split_indices(groups, RATIOS, seed=3)
    seen = sorted(i for s in splits for i in s)
    assert seen == list(range(len(groups)))
    for k, idx in enumerate(splits):
        assert {groups[i] for i in idx} == {g for g, s in assign.items() if s == k}


def test_tile_split_hits_the_target_shares_within_one_tile():
    groups = _groups()
    splits, _ = tile_split_indices(groups, RATIOS, seed=3)
    biggest = max(collections.Counter(groups).values())
    for share, idx in zip(RATIOS, splits):
        assert abs(len(idx) - share * len(groups)) <= biggest


def test_tile_split_is_deterministic_per_seed():
    groups = _groups()
    assert tile_split_indices(groups, RATIOS, 3) == tile_split_indices(groups, RATIOS, 3)
    assert tile_split_indices(groups, RATIOS, 3)[1] != tile_split_indices(groups, RATIOS, 4)[1]


def test_datamodule_tile_split_keeps_each_file_in_one_split(tmp_path):
    from src.dataset.mesh_datamodule import MeshDataModule
    from tests.test_ortho_tiles import write_tile

    root = tmp_path / "The Hague" / "mini_cleaner"
    for t in range(6):
        boxes = {f"t{t}-b{b}": (100 * t + 20 * b, 0, 100 * t + 20 * b + 10, 10, 5) for b in range(4)}
        for lod in ("LOD1", "LOD2"):
            write_tile(root / lod / "Source B" / f"10-{t}-0.city.json", boxes)
    dm = MeshDataModule(root, lod_in="LOD1", lod_out="LOD2", batch_size=1, seed=1,
                        train_val_test_split=(0.5, 0.25, 0.25), split="tile")
    dm.setup()
    files = dm.full_dataset.files
    per_split = {name: {files[i] for i in getattr(dm, f"{name}_dataset").indices}
                 for name in ("train", "val", "test")}
    assert not (per_split["train"] & per_split["val"] or per_split["train"] & per_split["test"]
                or per_split["val"] & per_split["test"])
    assert set(dm.split_groups.values()) <= {"train", "val", "test"}


def test_split_defaults_keep_v5_random():
    from pathlib import Path

    from src.utils.config import Config
    from src.utils.initialization import load_config
    assert Config().training.split == "random"
    assert load_config(Path("configs/mesh-v5-full-opt-coord.yaml"), []).training.split == "random"


def test_unknown_split_is_refused(tmp_path):
    from src.dataset.mesh_datamodule import MeshDataModule
    with pytest.raises(ValueError, match="split"):
        MeshDataModule(tmp_path, split="spatial")


def test_module_records_the_split_scheme_it_was_trained_on():
    from src.dataset.mesh_dataset import NUM_BINS
    from src.models.mesh_transformer import MeshTransformerModule
    kw = dict(num_bins=NUM_BINS, d_model=16, n_head=2, num_layers=1, dropout=0.0, max_seq_len=64)
    assert MeshTransformerModule(**kw).hparams.data_split == "random"
    assert MeshTransformerModule(data_split="tile", **kw).hparams.data_split == "tile"


def test_train_mesh_passes_the_split_and_writes_split_json():
    from pathlib import Path
    src = Path("src/train_mesh.py").read_text(encoding="utf-8")
    assert "split=cfg.training.split" in src
    assert "data_split=cfg.training.split" in src
    assert '"split.json"' in src
    assert "split=ref_cfg.training.split" in Path("src/eval/posthoc_eval.py").read_text(
        encoding="utf-8")
