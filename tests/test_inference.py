import json

from src.utils.config import Config
import src.inference as inf


def test_write_combined_offsets_vertex_indices(tmp_path):
    for i in range(2):
        cj = {"type": "CityJSON", "version": "1.1",
              "CityObjects": {f"b{i}": {"type": "Building", "geometry": [
                  {"type": "Solid", "lod": "1", "boundaries": [[[[0, 1, 2]]]]}]}},
              "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]}
        (tmp_path / f"gen_{i}.city.json").write_text(json.dumps(cj), encoding="utf-8")

    inf._write_combined(tmp_path)

    combined = json.loads((tmp_path / "all_buildings.city.json").read_text(encoding="utf-8"))
    assert len(combined["vertices"]) == 6
    rings = [obj["geometry"][0]["boundaries"][0][0][0]
             for obj in combined["CityObjects"].values()]
    assert sorted(rings) == [[0, 1, 2], [3, 4, 5]]


def test_run_inference_routes_through_generative_eval(tmp_path, monkeypatch):
    ckpt = tmp_path / "fake.ckpt"
    ckpt.touch()

    cfg = Config()
    cfg.inference.checkpoint_path = str(ckpt)
    cfg.inference.batch_size = 3
    cfg.inference.output_dir = str(tmp_path / "generated")
    cfg.generative_eval.num_batches = 99  # must be overridden to a single batch

    class FakeModel:
        n_max = 6
        def to(self, device): return self
        def eval(self): return self

    monkeypatch.setattr(inf.CityJSONDiffusionModule, "load_from_checkpoint",
                        staticmethod(lambda p: FakeModel()))
    monkeypatch.setattr(inf, "_load_datamodule", lambda c: None)

    seen = {}
    def fake_eval(model, datamodule, eval_cfg, loggers, save_dir):
        seen.update(cfg=eval_cfg, loggers=loggers, save_dir=save_dir)
        return {"gen/rejection_rate": 0.0}

    monkeypatch.setattr(inf, "run_generative_eval", fake_eval)

    metrics = inf.run_inference(cfg)

    assert metrics == {"gen/rejection_rate": 0.0}
    assert seen["loggers"] == []                     # no wandb_run_id -> local only
    assert seen["cfg"].num_batches == 1
    assert seen["cfg"].batch_size == 3
    assert seen["cfg"].log_n_samples == 3            # every graph in the batch persisted
    assert seen["cfg"].save_dir == cfg.inference.output_dir
    assert cfg.generative_eval.num_batches == 99     # caller's config untouched
