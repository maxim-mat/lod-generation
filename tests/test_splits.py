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
    assert "split_block_m=cfg.training.split_block_m" in src
    assert "split_buffer_m=cfg.training.split_buffer_m" in src
    assert "data_split=split_name(" in src
    assert '"split.json"' in src
    posthoc = Path("src/eval/posthoc_eval.py").read_text(encoding="utf-8")
    assert "split=ref_cfg.training.split" in posthoc
    assert "split_block_m=ref_cfg.training.split_block_m" in posthoc
    assert "split_buffer_m=ref_cfg.training.split_buffer_m" in posthoc


def _box_mesh(tmp_path, box):
    from src.dataset.mesh_dataset import parse_cityjson_file_to_meshes
    from tests.test_ortho_tiles import write_tile
    path = write_tile(tmp_path / "t.city.json", {"a": box})
    return parse_cityjson_file_to_meshes(path)["a"]


def test_footprint_is_the_base_area_and_centroid(tmp_path):
    from src.dataset.splits import footprint
    area, (x, y) = footprint(*_box_mesh(tmp_path, (100, 200, 110, 208, 6)))
    assert area == pytest.approx(80.0) and (x, y) == pytest.approx((105.0, 204.0))


def test_footprint_falls_back_to_the_bbox_centre_without_a_base(tmp_path):
    import numpy as np

    from src.dataset.splits import footprint
    verts, faces = _box_mesh(tmp_path, (100, 200, 110, 208, 6))
    roofless_floorless = faces[~(verts[faces][:, :, 2] == 0).all(axis=1)]
    area, (x, y) = footprint(verts, roofless_floorless)
    assert area == 0.0 and (x, y) == pytest.approx((105.0, 204.0))
    assert np.isfinite([x, y]).all()


def test_grid_labels_bin_by_block_including_negative_coordinates():
    import numpy as np

    from src.dataset.splits import grid_labels
    xy = np.array([[10.0, 10.0], [-10.0, 5.0], [250.0, 260.0], [99.9, 0.0]])
    assert grid_labels(xy, 100) == ["0_0", "-1_0", "2_2", "0_0"]


def test_buffer_drops_exactly_the_train_buildings_near_eval_ones():
    import numpy as np

    from src.dataset.splits import buffer_train
    xy = np.array([[10.0 * i, 0.0] for i in range(6)])        # 0, 10, ..., 50 m
    kept, dropped = buffer_train(xy, [0, 1, 2, 5], [3], buffer_m=15)
    assert kept == [0, 1, 5] and dropped == 1                    # 2 is 10 m away; 1 and 5 are 20 m
    assert buffer_train(xy, [0, 1, 2, 5], [3], buffer_m=0) == ([0, 1, 2, 5], 0)


def test_split_name_encodes_scheme_block_and_buffer():
    from src.dataset.splits import split_name
    assert split_name("random") == "random"
    assert split_name("tile") == "tile"
    assert split_name("tile", buffer_m=50) == "tile-buffer50m"
    assert split_name("grid", 250) == "grid250m"
    assert split_name("grid", 250.0, 50.0) == "grid250m-buffer50m"


def _line_subset(tmp_path):
    """12 buildings 100 m apart along x, in 3 files."""
    from tests.test_ortho_tiles import write_tile
    root = tmp_path / "The Hague" / "mini_cleaner"
    for t in range(3):
        boxes = {f"b{4 * t + k}": (100 * (4 * t + k), 0, 100 * (4 * t + k) + 10, 10, 5)
                 for k in range(4)}
        for lod in ("LOD1", "LOD2"):
            write_tile(root / lod / "Source B" / f"10-{t}-0.city.json", boxes)
    return root


def test_datamodule_grid_split_keeps_blocks_whole_and_applies_the_buffer(tmp_path):
    import numpy as np

    from src.dataset.mesh_datamodule import MeshDataModule
    from src.dataset.splits import grid_labels
    root = _line_subset(tmp_path)
    kw = dict(lod_in="LOD1", lod_out="LOD2", batch_size=1, seed=3,
              train_val_test_split=(0.5, 0.25, 0.25), split="grid", split_block_m=200)
    plain = MeshDataModule(root, **kw)
    plain.setup()
    xy = plain.full_dataset.xy
    assert xy.shape == (12, 2) and xy[0] == pytest.approx((5.0, 5.0))
    labels = grid_labels(xy, 200)
    blocks = [{labels[i] for i in getattr(plain, f"{n}_dataset").indices}
              for n in ("train", "val", "test")]
    assert not (blocks[0] & blocks[1] or blocks[0] & blocks[2] or blocks[1] & blocks[2])

    buffered = MeshDataModule(root, split_buffer_m=150, **kw)
    buffered.setup()
    ev = buffered.val_dataset.indices + buffered.test_dataset.indices
    assert ev == plain.val_dataset.indices + plain.test_dataset.indices   # eval untouched
    tr = buffered.train_dataset.indices
    assert buffered.buffer_dropped == len(plain.train_dataset.indices) - len(tr) > 0
    d = np.linalg.norm(xy[tr][:, None] - xy[ev][None], axis=-1)
    assert d.min() >= 150


def test_grid_without_a_block_size_and_random_with_a_buffer_are_refused(tmp_path):
    from src.dataset.mesh_datamodule import MeshDataModule
    with pytest.raises(ValueError, match="split_block_m"):
        MeshDataModule(tmp_path, split="grid")
    with pytest.raises(ValueError, match="buffer"):
        MeshDataModule(tmp_path, split_buffer_m=50)
