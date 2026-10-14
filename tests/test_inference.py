import json
import sys

import lightning as L
import numpy as np
import pytest
import torch

import src.inference as inf
from src.dataset.mesh_dataset import NUM_BINS, detokenize, specials
from src.models.mesh_transformer import MeshTransformerModule
from src.utils.config import Config

BOS, EOS, PAD = specials(NUM_BINS)

# A 10 x 8 x 6 m box at Dutch RD magnitudes, so a frame bug cannot hide near 0.
ORIGIN = np.array([80000.0, 450000.0, 2.0])
CORNERS = ORIGIN + np.array([[0, 0, 0], [10, 0, 0], [10, 8, 0], [0, 8, 0],
                             [0, 0, 6], [10, 0, 6], [10, 8, 6], [0, 8, 6]], dtype=float)
BOX_FACES = [[[0, 3, 2, 1]], [[4, 5, 6, 7]], [[0, 1, 5, 4]], [[1, 2, 6, 5]],
             [[2, 3, 7, 6]], [[3, 0, 4, 7]]]


def _box_cityjson(ids, shift=10.0):
    """One CityJSON tile with a box per id, each `shift` metres further along x."""
    verts, objects = [], {}
    for k, obj_id in enumerate(ids):
        base = len(verts)
        verts.extend((CORNERS + [k * shift, 0, 0]).tolist())
        boundaries = [[[i + base for i in ring] for ring in face] for face in BOX_FACES]
        objects[obj_id] = {"type": "Building", "geometry": [
            {"type": "Solid", "lod": "1", "boundaries": [boundaries]}]}
    return {"type": "CityJSON", "version": "1.1", "CityObjects": objects,
            "vertices": verts}


def _write_tile(path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_box_cityjson(ids)), encoding="utf-8")
    return path


def _model(max_seq_len=256, tokenization="coord"):
    torch.manual_seed(0)
    return MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2,
                                 num_layers=1, dropout=0.0, max_seq_len=max_seq_len,
                                 tokenization=tokenization).eval()


def _echo(model):
    """Make `model.generate` answer every LOD1 with itself, deterministically."""
    def generate(cond, cond_pad_mask=None, **kwargs):
        rows = [[BOS, *c[~m].tolist(), EOS] for c, m in zip(cond, cond_pad_mask)]
        width = max(len(r) for r in rows)
        return torch.tensor([r + [PAD] * (width - len(r)) for r in rows])
    model.generate = generate
    return model


def _cfg(tmp_path, **inference):
    cfg = Config()
    cfg.mesh_data.max_faces = 20          # keeps a random model's budget short
    cfg.mesh_eval.beam_size = 1
    cfg.inference.output_dir = str(tmp_path / "out")
    for key, value in inference.items():
        setattr(cfg.inference, key, value)
    return cfg


def _save_ckpt(model, path):
    torch.save({"state_dict": model.state_dict(),
                "hyper_parameters": dict(model.hparams),
                "pytorch-lightning_version": L.__version__}, path)
    return path


# ----------------------------------------------------------------------
# Model-agnostic helpers carried over from the graph-era inference
# ----------------------------------------------------------------------

def test_write_combined_offsets_vertex_indices(tmp_path):
    for i in range(2):
        cj = {"type": "CityJSON", "version": "1.1",
              "CityObjects": {f"b{i}": {"type": "Building", "geometry": [
                  {"type": "Solid", "lod": "2", "boundaries": [[[[0, 1, 2]]]]}]}},
              "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]}
        (tmp_path / f"b{i}.city.json").write_text(json.dumps(cj), encoding="utf-8")

    inf._write_combined(tmp_path)

    combined = json.loads((tmp_path / "all_buildings.city.json").read_text(encoding="utf-8"))
    assert len(combined["vertices"]) == 6
    assert combined["metadata"]["datasetLod"] == "2"
    rings = [obj["geometry"][0]["boundaries"][0][0][0]
             for obj in combined["CityObjects"].values()]
    assert sorted(rings) == [[0, 1, 2], [3, 4, 5]]


def test_write_combined_does_not_swallow_its_own_output(tmp_path):
    for i in range(2):
        cj = {"type": "CityJSON", "version": "1.1",
              "CityObjects": {f"b{i}": {"type": "Building", "geometry": []}},
              "vertices": [[0, 0, 0]]}
        (tmp_path / f"b{i}.city.json").write_text(json.dumps(cj), encoding="utf-8")

    inf._write_combined(tmp_path)
    inf._write_combined(tmp_path)

    combined = json.loads((tmp_path / "all_buildings.city.json").read_text(encoding="utf-8"))
    assert sorted(combined["CityObjects"]) == ["b0", "b1"]
    assert len(combined["vertices"]) == 2


def test_resolve_checkpoint_local(tmp_path):
    ckpt = tmp_path / "last.ckpt"
    ckpt.touch()
    cfg = Config()
    cfg.inference.checkpoint_path = str(ckpt)
    assert inf._resolve_checkpoint(cfg) == str(ckpt)


def test_resolve_checkpoint_missing_local_exits(tmp_path):
    cfg = Config()
    cfg.inference.checkpoint_path = str(tmp_path / "nope.ckpt")
    with pytest.raises(SystemExit):
        inf._resolve_checkpoint(cfg)


def test_resolve_checkpoint_passes_url_through():
    cfg = Config()
    cfg.inference.checkpoint_path = "https://example.com/model.ckpt"
    # fsspec URLs go straight to Lightning, no existence check
    assert inf._resolve_checkpoint(cfg) == "https://example.com/model.ckpt"


def test_resolve_checkpoint_wandb_downloads(tmp_path, monkeypatch):
    art_dir = tmp_path / "downloaded"
    art_dir.mkdir()
    (art_dir / "model.ckpt").touch()

    class FakeArtifact:
        def download(self, root):
            return str(art_dir)

    class FakeApi:
        def artifact(self, ref):
            seen["ref"] = ref
            return FakeArtifact()

    seen = {}
    monkeypatch.setitem(sys.modules, "wandb", type("wandb", (), {"Api": FakeApi})())

    cfg = Config()
    cfg.logging.wandb_entity = "me"
    cfg.logging.project_name = "lod-generation"
    cfg.inference.checkpoint_path = "wandb://model-abc123:best"

    assert inf._resolve_checkpoint(cfg) == str(art_dir / "model.ckpt")
    # bare name:alias gets qualified from cfg.logging
    assert seen["ref"] == "me/lod-generation/model-abc123:best"


# ----------------------------------------------------------------------
# LOD1 input and conditioning
# ----------------------------------------------------------------------

def test_load_lod1_reads_a_single_file(tmp_path):
    path = _write_tile(tmp_path / "tile.city.json", ["a", "b"])
    meshes = inf.load_lod1(path)
    assert sorted(meshes) == ["a", "b"]
    assert len(meshes["a"][1]) == 12          # six quads, triangulated


def test_load_lod1_walks_a_directory_and_skips_unreadable_tiles(tmp_path, caplog):
    _write_tile(tmp_path / "in" / "x" / "one.city.json", ["a"])
    _write_tile(tmp_path / "in" / "two.json", ["b"])
    (tmp_path / "in" / "broken.city.json").write_text("{not json", encoding="utf-8")

    meshes = inf.load_lod1(tmp_path / "in")

    assert sorted(meshes) == ["a", "b"]
    assert "broken.city.json" in caplog.text


def test_load_lod1_rejects_a_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        inf.load_lod1(tmp_path / "nowhere")


def test_condition_tokens_are_the_training_frame(tmp_path):
    cfg = _cfg(tmp_path)
    verts, faces = inf.load_lod1(_write_tile(tmp_path / "t.city.json", ["a"]))["a"]

    tokens, center, scale = inf.condition_tokens(verts, faces, cfg.mesh_data, "coord")

    back, _ = detokenize(tokens, NUM_BINS)
    half_bin = 0.5 * scale.max() / (NUM_BINS - 1)
    got = np.unique(np.round(back * scale + center, 6), axis=0)
    assert np.allclose(np.sort(got, axis=0), np.sort(CORNERS, axis=0), atol=half_bin + 1e-6)
    # +z headroom from the config, exactly as MeshDataset applies it
    assert scale[2] == pytest.approx(10.0 * 1.1)


# ----------------------------------------------------------------------
# Generation and write-back
# ----------------------------------------------------------------------

def test_echo_model_writes_the_input_building_back_in_world_coordinates(tmp_path):
    cfg = _cfg(tmp_path)
    meshes = inf.load_lod1(_write_tile(tmp_path / "t.city.json", ["bag/0363.1", "b"]))
    out = tmp_path / "out"

    status = inf.infer_buildings(_echo(_model()), cfg, meshes, out)

    assert status == {"bag/0363.1": "ok", "b": "ok"}
    cj = json.loads((out / "bag_0363.1.city.json").read_text(encoding="utf-8"))
    assert list(cj["CityObjects"]) == ["bag/0363.1"]
    got = np.unique(np.asarray(cj["vertices"]), axis=0)
    half_bin = 0.5 * 11.0 / (NUM_BINS - 1)
    assert np.allclose(np.sort(got, axis=0), np.sort(CORNERS, axis=0), atol=half_bin + 1e-6)
    assert (out / "bag_0363.1.obj").is_file()


def test_an_empty_generation_is_recorded_as_failed_and_the_rest_still_written(tmp_path):
    cfg = _cfg(tmp_path)
    meshes = inf.load_lod1(_write_tile(tmp_path / "t.city.json", ["a", "b"]))
    model = _model()

    def generate(cond, cond_pad_mask=None, **kwargs):
        out = torch.full((2, 2), EOS)
        out[:, 0] = BOS
        return out
    model.generate = generate

    status = inf.infer_buildings(model, cfg, meshes, tmp_path / "out")

    assert status == {"a": "failed", "b": "failed"}
    assert not list((tmp_path / "out").glob("*.city.json"))


def test_a_condition_beyond_the_position_table_is_skipped(tmp_path):
    cfg = _cfg(tmp_path)
    meshes = inf.load_lod1(_write_tile(tmp_path / "t.city.json", ["a"]))
    # 12 triangles = 108 condition tokens against a 64-row learned table
    model = _echo(_model(max_seq_len=64))

    status = inf.infer_buildings(model, cfg, meshes, tmp_path / "out")

    assert status == {"a": "skipped"}


def test_batches_are_split_by_batch_size(tmp_path):
    cfg = _cfg(tmp_path, batch_size=2)
    meshes = inf.load_lod1(_write_tile(tmp_path / "t.city.json", ["a", "b", "c"]))
    model = _echo(_model())
    inner, sizes = model.generate, []
    model.generate = lambda cond, cond_pad_mask=None, **kw: (
        sizes.append(len(cond)) or inner(cond, cond_pad_mask, **kw))

    inf.infer_buildings(model, cfg, meshes, tmp_path / "out")

    assert sizes == [2, 1]


def test_decoding_follows_the_run_config(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.mesh_eval.beam_size = 3
    cfg.mesh_eval.length_penalty = 0.5
    meshes = inf.load_lod1(_write_tile(tmp_path / "t.city.json", ["a"]))
    model = _echo(_model())
    inner, seen = model.generate, {}
    model.generate = lambda cond, cond_pad_mask=None, **kw: (
        seen.update(kw) or inner(cond, cond_pad_mask, **kw))

    inf.infer_buildings(model, cfg, meshes, tmp_path / "out")

    assert seen == {"max_new_tokens": 9 * 20 + 1, "temperature": 0.0,
                    "beam_size": 3, "length_penalty": 0.5}


# ----------------------------------------------------------------------
# run_inference: config checks, checkpoint loading, manifest
# ----------------------------------------------------------------------

def test_run_inference_end_to_end_from_a_checkpoint(tmp_path):
    ckpt = _save_ckpt(_model(), tmp_path / "m.ckpt")
    tile = _write_tile(tmp_path / "in" / "t.city.json", ["a", "b"])
    cfg = _cfg(tmp_path, checkpoint_path=str(ckpt), input_path=str(tile))

    out_dir = inf.run_inference(cfg)

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["status"]) == {"a", "b"}
    assert set(manifest["status"].values()) <= {"ok", "failed", "skipped"}
    assert manifest["checkpoint"] == str(ckpt)
    assert manifest["decoding"]["beam_size"] == 1
    assert "commit" in manifest["git"]
    assert manifest["config"]["inference"]["input_path"] == str(tile)
    assert out_dir.parent == tmp_path / "out"


def test_run_inference_refuses_a_tokenization_mismatch(tmp_path):
    ckpt = _save_ckpt(_model(tokenization="amt"), tmp_path / "m.ckpt")
    tile = _write_tile(tmp_path / "t.city.json", ["a"])
    cfg = _cfg(tmp_path, checkpoint_path=str(ckpt), input_path=str(tile))

    with pytest.raises(ValueError, match="tokenization"):
        inf.run_inference(cfg)


@pytest.mark.parametrize("field, value, match", [
    ("config_set", "mesh_vqvae", "config_set"),
    ("checkpoint_path", None, "checkpoint_path"),
    ("input_path", None, "input_path"),
    ("tokenizer", "vqvae", "vqvae"),
])
def test_run_inference_fails_fast_on_config(tmp_path, field, value, match):
    cfg = _cfg(tmp_path, checkpoint_path="unused.ckpt", input_path="unused")
    if field == "config_set":
        cfg.config_set = value
    elif field == "tokenizer":
        cfg.mesh_model.tokenizer = value
    else:
        setattr(cfg.inference, field, value)

    with pytest.raises(ValueError, match=match):
        inf.run_inference(cfg)


def test_main_dispatches_inference(tmp_path, monkeypatch):
    import main as entry

    config = tmp_path / "c.yaml"
    config.write_text("config_set: mesh\n", encoding="utf-8")
    seen = {}
    monkeypatch.setattr(entry, "run_inference", lambda cfg: seen.update(cfg=cfg))
    monkeypatch.setattr(sys, "argv", ["main.py", "--inference", "--config", str(config),
                                      "inference.input_path=tiles"])

    entry.main()

    assert seen["cfg"].mode == "inference"
    assert seen["cfg"].inference.input_path == "tiles"
