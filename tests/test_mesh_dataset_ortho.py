"""MeshDataset's optional orthophoto field, built end to end on a tiny subset.

raw tiles -> `src.ortho_parse` -> MeshDataset(ortho_dir=...). Off by default, so
every existing config and checkpoint sees exactly the old items.
"""
import pickle
import shutil
from pathlib import Path

import pytest
import torch

from src.dataset.mesh_datamodule import MeshDataModule
from src.dataset.mesh_dataset import MeshDataset, mesh_collate_fn
from src.dataset.ortho import OrthoReader
from src.ortho_parse import main as parse_main
from tests.test_ortho_parse import GSD, write_raw_tile
from tests.test_ortho_tiles import write_tile

S = 32
BOXES = {"a": (80100, 450100, 80120, 450130, 8), "b": (80500, 450500, 80510, 450512, 5),
         "c": (80700, 450300, 80716, 450310, 6), "d": (80300, 450800, 80312, 450820, 7)}


@pytest.fixture(scope="module")
def subset(tmp_path_factory):
    hague = tmp_path_factory.mktemp("data") / "The Hague"
    raw = hague / "raw" / "ortho"
    write_raw_tile(raw, 80000, 451000)
    root = hague / "mini_cleaner"
    for lod in ("LOD1", "LOD2"):
        write_tile(root / lod / "Source B" / "10-1-2.city.json", BOXES)
    parse_main([str(root), "--raw", str(raw), "--out", str(root / "ortho"),
                "--gsd", str(GSD), "--workers", "1"])
    return root


def _ds(root, **kw):
    return MeshDataset(root, lod_in="LOD1", lod_out="LOD2", **kw)


def test_without_ortho_items_are_unchanged(subset):
    item = _ds(subset)[0]
    assert set(item) == {"cond", "tgt", "id", "center", "scale"}


def test_image_is_the_aligned_crop_of_the_items_own_frame(subset):
    ds = _ds(subset, ortho_dir="ortho", image_size=S, image_margin=0.1)
    item = ds[0]
    img = item["image"]
    assert img.dtype == torch.uint8 and tuple(img.shape) == (3, S, S)
    assert img.float().mean() > 10                       # real texture, not padding
    direct = OrthoReader(subset / "ortho", size=S, beta=0.1).crop(
        "Source B/10-1-2.tif", item["center"].numpy(), item["scale"].numpy())
    assert torch.equal(img, direct)


def test_absolute_ortho_dir_is_used_as_is(subset):
    ds = _ds(subset, ortho_dir=str(subset / "ortho"), image_size=S)
    assert tuple(ds[1]["image"].shape) == (3, S, S)


def test_amt_length_pass_reads_no_images(subset, monkeypatch):
    calls = []
    monkeypatch.setattr(OrthoReader, "crop", lambda self, *a: calls.append(a))
    _ds(subset, ortho_dir="ortho", image_size=S, tokenization="amt")
    assert calls == []


@pytest.mark.parametrize("kw, match", [
    ({"image_margin": 0.3}, "beta_max"),
    ({"margin_hi": (0.0, 0.0, 0.2)}, "margin"),
])
def test_settings_the_mosaics_were_not_built_for_are_refused(subset, kw, match):
    with pytest.raises(ValueError, match=match):
        _ds(subset, ortho_dir="ortho", image_size=S, **kw)


def test_a_different_frame_lod_is_refused(subset, tmp_path):
    root = tmp_path / "copy"
    shutil.copytree(subset, root)
    shutil.copytree(root / "LOD1", root / "LOD1_synth")
    with pytest.raises(ValueError, match="lod_in"):
        MeshDataset(root, lod_in="LOD1_synth", lod_out="LOD2", ortho_dir="ortho")


def test_missing_mosaic_is_reported_at_construction(subset, tmp_path):
    root = tmp_path / "copy"
    shutil.copytree(subset, root)
    (root / "ortho" / "Source B" / "10-1-2.tif").unlink()
    with pytest.raises(FileNotFoundError, match="10-1-2.tif"):
        _ds(root, ortho_dir="ortho")


def test_dataset_pickles_after_reading_an_image(subset):
    ds = _ds(subset, ortho_dir="ortho", image_size=S)
    first = ds[0]["image"]
    clone = pickle.loads(pickle.dumps(ds))
    assert torch.equal(clone[0]["image"], first)


def test_collate_stacks_images_only_when_present(subset):
    with_img = _ds(subset, ortho_dir="ortho", image_size=S)
    batch = mesh_collate_fn([with_img[0], with_img[1]])
    assert tuple(batch["image"].shape) == (2, 3, S, S)
    plain = mesh_collate_fn([_ds(subset)[0], _ds(subset)[1]])
    assert "image" not in plain


def test_datamodule_threads_the_image_settings(subset):
    dm = MeshDataModule(subset, lod_in="LOD1", lod_out="LOD2", batch_size=2,
                        train_val_test_split=(0.5, 0.25, 0.25), ortho_dir="ortho",
                        image_size=16, image_margin=0.05)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert tuple(batch["image"].shape) == (2, 3, 16, 16)


def test_config_defaults_keep_images_off():
    from src.utils.initialization import load_config

    cfg = load_config(Path("configs/mesh-v5-full-opt-coord.yaml"), [])
    assert cfg.mesh_data.ortho_dir is None
    assert (cfg.mesh_data.image_size, cfg.mesh_data.image_margin) == (256, 0.1)
