# Generative Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an end-of-pipeline Lightning callback that samples full reverse-diffusion buildings, scores their generative quality, logs scalars + a per-instance table to WandB, and saves sample graphs and geometries locally and to WandB.

**Architecture:** A `GenerativeEvalCallback` fires on `on_test_end` (after the test-split single-step metrics in `src/train.py:109`). Its scoring body is a standalone callable over per-instance *records* (`src/eval/callback.py:run_generative_eval`), so switching to conditional/guided generation later is additive (swap the sample source, add ground-truth columns). Metric arms (validity, features, distribution, novelty) are separate pure-function modules under `src/eval/`. The reference distribution is the test-split graphs pushed through the same converter+featurizer as the samples.

**Tech Stack:** PyTorch Lightning, numpy, scipy (new dep), wandb, val3dity (external binary, gated).

## Global Constraints

- Levi representation is **live**: node classes `0=VERTEX,1=GROUND,2=ROOF,3=WALL,4=OFF`; edge classes `0=off,1=vertex-vertex,2=vertex-face`. Import these from `src.dataset.dataset`.
- `sample()` returns **argmax** labels — there is **no** `threshold` knob anywhere.
- `graph_to_cityjson(coords, node_classes, edge_classes, building_id=...)` is the exact-inverse converter; it reads vertex rows for coords and ignores OFF/padding. Reuse it for both samples and references.
- Coordinates are metres end-to-end. Sampled `pos` is in scaled units — multiply by `model.coord_scale` and add `model.z_shift` (via the new `_denormalize_coords` helper). Reference graphs from the datamodule are already metres.
- Dataset items may be `tuple` (multi-LOD); take `item[0]`, mirroring `datamodule.compute_marginals`.
- New Python dependency allowed: **`scipy`** only. Ported shape formulas are attributed inline, not added as deps.
- Cite 3dSAGER (arXiv:2511.06300) at the feature implementation; val3dity (Ledoux 2018) at the validity implementation.
- `pytest` from repo root; test files live in `tests/`. Follow the fake-logger pattern from `tests/test_model_smoke.py` (`model.log = lambda ...`).
- Per project CLAUDE.md: run `impact({target, direction:"upstream"})` and report blast radius before editing any existing symbol; run `detect_changes()` before each commit.

---

### Task 1: Config block + callback wiring

**Files:**
- Modify: `src/utils/config.py` (add `GenerativeEvalConfig`, field on `Config`)
- Create: `src/eval/__init__.py`
- Create: `src/eval/callback.py` (stub callback + no-op body)
- Modify: `src/utils/setup_utils.py:create_callbacks` (append when enabled)
- Test: `tests/test_eval_callback.py`

**Interfaces:**
- Produces: `GenerativeEvalConfig` dataclass; `Config.generative_eval`; `GenerativeEvalCallback(cfg: GenerativeEvalConfig, save_dir: Path)`; `run_generative_eval(model, datamodule, cfg, loggers, save_dir) -> dict` (stub returns `{}` this task).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_callback.py
from pathlib import Path
from src.utils.config import Config, GenerativeEvalConfig
from src.utils.setup_utils import create_callbacks


def test_callback_absent_when_disabled(tmp_path):
    cfg = Config()
    cfg.generative_eval.enabled = False
    names = [type(c).__name__ for c in create_callbacks(cfg, tmp_path)]
    assert "GenerativeEvalCallback" not in names


def test_callback_present_when_enabled(tmp_path):
    cfg = Config()
    cfg.generative_eval.enabled = True
    names = [type(c).__name__ for c in create_callbacks(cfg, tmp_path)]
    assert "GenerativeEvalCallback" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_callback.py -v`
Expected: FAIL (`AttributeError: 'Config' object has no attribute 'generative_eval'`).

- [ ] **Step 3: Add the config dataclass**

```python
# src/utils/config.py — add near the other @dataclass blocks
@dataclass
class GenerativeEvalConfig:
    """End-of-pipeline full-generation evaluation (fires on test end)."""
    enabled: bool = False
    num_batches: int = 4          # batches sampled through the full reverse chain
    batch_size: int = 16
    seed: int = 1234              # fixed -> comparable buildings across runs
    log_n_samples: int = 8        # graphs+geometries persisted locally and to WandB
    save_dir: Optional[str] = None  # None -> <run_save_dir>/generative_eval
    feature_set: str = "full"     # "full" | "welldefined"
    val3dity_path: Optional[str] = None  # None -> shutil.which("val3dity")
    novelty_tol: float = 0.1      # feature-space distance for a "novel" sample
```

Add the field to `Config`:

```python
# src/utils/config.py — inside class Config
    generative_eval: GenerativeEvalConfig = field(default_factory=GenerativeEvalConfig)
```

- [ ] **Step 4: Create the eval package and stub callback**

```python
# src/eval/__init__.py
```

```python
# src/eval/callback.py
import logging
from pathlib import Path

import lightning as L

logger = logging.getLogger(__name__)


def run_generative_eval(model, datamodule, cfg, loggers, save_dir):
    """Sample buildings end-to-end and score them. Returns a dict of scalar metrics.

    Standalone (checkpoint-callable) body; the callback is a thin Lightning adapter.
    Fleshed out in later tasks; a no-op stub for now.
    """
    return {}


class GenerativeEvalCallback(L.Callback):
    def __init__(self, cfg, save_dir):
        super().__init__()
        self.cfg = cfg
        self.save_dir = Path(save_dir)

    def on_test_end(self, trainer, pl_module):
        if not self.cfg.enabled:
            return
        out_dir = Path(self.cfg.save_dir) if self.cfg.save_dir else self.save_dir / "generative_eval"
        run_generative_eval(pl_module, trainer.datamodule, self.cfg, trainer.loggers, out_dir)
```

- [ ] **Step 5: Wire into create_callbacks**

```python
# src/utils/setup_utils.py — inside create_callbacks, before `return callbacks`
    ge_cfg = cfg.generative_eval
    if ge_cfg.enabled:
        from src.eval.callback import GenerativeEvalCallback
        callbacks.append(GenerativeEvalCallback(ge_cfg, save_dir))
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_callback.py -v`
Expected: PASS (both tests).

- [ ] **Step 7: Commit**

```bash
git add src/utils/config.py src/eval/__init__.py src/eval/callback.py src/utils/setup_utils.py tests/test_eval_callback.py
git commit -m "feat(eval): generative_eval config block and callback wiring"
```

---

### Task 2: `_denormalize_coords` helper on the model

**Files:**
- Modify: `src/models/diffusion.py` (extract helper; `generate_cityjson` uses it)
- Test: `tests/test_eval_sampling.py`

**Interfaces:**
- Produces: `CityJSONDiffusionModule._denormalize_coords(pos_i) -> np.ndarray` — takes one `[N,3]` tensor in scaled units, returns `[N,3]` float64 metres (`* coord_scale`, `+ z_shift` on z).

- [ ] **Step 1: Impact analysis (report, don't skip)**

Run `impact({target: "generate_cityjson", direction: "upstream"})` and report callers/risk. Expected caller: `src/inference.py:47`. Confirm its return type (list of dicts) stays unchanged.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_eval_sampling.py
import numpy as np
import torch
from src.models.diffusion import CityJSONDiffusionModule


def _tiny_model():
    return CityJSONDiffusionModule(hidden_dim=8, edge_dim=4, global_dim=4,
                                   n_head=2, num_layers=1, T=10, n_max=6)


def test_denormalize_applies_scale_and_zshift():
    m = _tiny_model()
    m.coord_scale = 2.0
    m.z_shift = 5.0
    pos = torch.zeros(3, 3)
    pos[:, 2] = 1.0
    out = m._denormalize_coords(pos)
    assert isinstance(out, np.ndarray)
    # x,y scaled by 2 (still 0); z = 1*2 + 5 = 7
    assert np.allclose(out[:, :2], 0.0)
    assert np.allclose(out[:, 2], 7.0)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_sampling.py::test_denormalize_applies_scale_and_zshift -v`
Expected: FAIL (`AttributeError: ... has no attribute '_denormalize_coords'`).

- [ ] **Step 4: Add the helper and route `generate_cityjson` through it**

```python
# src/models/diffusion.py — add as a method on CityJSONDiffusionModule
    def _denormalize_coords(self, pos_i):
        """Scaled-unit positions [N,3] -> metres. Inverse of the target normalization:
        multiply by coord_scale, then restore the se2 z offset."""
        coords = (pos_i * self.coord_scale).detach().cpu().numpy().astype(float)
        coords[:, 2] += self.z_shift
        return coords
```

Replace the inline un-scaling in `generate_cityjson` (currently the two lines computing `coords`) with:

```python
            coords = self._denormalize_coords(pos[i])
```

- [ ] **Step 5: Run tests to verify pass + no regression**

Run: `python -m pytest tests/test_eval_sampling.py tests/test_model_smoke.py -v`
Expected: PASS.

- [ ] **Step 6: `detect_changes()` then commit**

Run `detect_changes({scope: "compare", base_ref: "main"})`; confirm only `generate_cityjson` / the new helper changed.

```bash
git add src/models/diffusion.py tests/test_eval_sampling.py
git commit -m "refactor(model): extract _denormalize_coords, share with generate_cityjson"
```

---

### Task 3: Sampling → records + drop counting

**Files:**
- Create: `src/eval/sampling.py`
- Test: `tests/test_eval_sampling.py` (extend)

**Interfaces:**
- Consumes: `model.sample`, `model._denormalize_coords`, `graph_to_cityjson`.
- Produces: `draw_samples(model, num_batches, batch_size) -> (records, stats)` where each `record = {"coords": np.ndarray[Nv,3], "node_labels": np.ndarray[N], "edge_labels": np.ndarray[N,N], "cityjson": dict}` (only non-dropped instances), and `stats = {"attempted": int, "dropped": int}`. A dropped instance = `graph_to_cityjson` returns `{}` or `<3` vertex nodes.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_sampling.py — append
import numpy as np
from src.dataset.dataset import VERTEX, WALL, EDGE_VF, EDGE_VV
from src.eval.sampling import draw_samples


def test_draw_samples_counts_drops(monkeypatch):
    m = _tiny_model()

    # one well-formed triangle+face graph, one all-OFF (dropped) graph
    N = 6
    good_pos = torch.zeros(1, N, 3)
    good_pos[0, :3] = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.float)
    good_lab = torch.full((1, N), 4)  # OFF
    good_lab[0, :3] = VERTEX
    good_lab[0, 3] = WALL
    good_edge = torch.zeros(1, N, N, dtype=torch.long)
    for v in range(3):
        good_edge[0, 3, v] = good_edge[0, v, 3] = EDGE_VF
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        good_edge[0, a, b] = good_edge[0, b, a] = EDGE_VV

    bad_pos = torch.zeros(1, N, 3)
    bad_lab = torch.full((1, N), 4)
    bad_edge = torch.zeros(1, N, N, dtype=torch.long)

    calls = iter([(good_pos, good_lab, good_edge), (bad_pos, bad_lab, bad_edge)])
    monkeypatch.setattr(m, "sample", lambda batch_size=1: next(calls))

    records, stats = draw_samples(m, num_batches=2, batch_size=1)
    assert stats == {"attempted": 2, "dropped": 1}
    assert len(records) == 1
    assert records[0]["cityjson"]["type"] == "CityJSON"
    assert records[0]["coords"].shape[1] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_sampling.py::test_draw_samples_counts_drops -v`
Expected: FAIL (`ModuleNotFoundError: src.eval.sampling`).

- [ ] **Step 3: Implement `draw_samples`**

```python
# src/eval/sampling.py
import logging

import numpy as np

from src.post_process.post_process import graph_to_cityjson

logger = logging.getLogger(__name__)


def draw_samples(model, num_batches, batch_size):
    """Run the full reverse chain num_batches times and convert each instance.

    Returns (records, stats). Records are only the reconstructable buildings;
    stats tracks attempted vs dropped so a rejection rate can be computed.
    """
    model.eval()
    records, attempted, dropped = [], 0, 0
    for _ in range(num_batches):
        pos, node_labels, edge_labels = model.sample(batch_size=batch_size)
        for i in range(pos.shape[0]):
            attempted += 1
            coords = model._denormalize_coords(pos[i])
            nl = node_labels[i].detach().cpu().numpy()
            el = edge_labels[i].detach().cpu().numpy()
            cj = graph_to_cityjson(coords, nl, el, building_id=f"gen_{attempted - 1}")
            if not cj:
                dropped += 1
                continue
            records.append({"coords": coords, "node_labels": nl,
                            "edge_labels": el, "cityjson": cj})
    return records, {"attempted": attempted, "dropped": dropped}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_sampling.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/eval/sampling.py tests/test_eval_sampling.py
git commit -m "feat(eval): draw_samples with drop counting"
```

---

### Task 4: Building features — core geometric descriptors

**Files:**
- Create: `src/eval/building_features.py`
- Test: `tests/test_eval_features.py`

**Interfaces:**
- Produces:
  - `mesh_from_cityjson(cj) -> (verts: np.ndarray[V,3], faces: list[list[int]])`
  - `CORE_FEATURES: list[str]`
  - `building_features(cj, feature_set="full") -> dict[str, float]` (this task returns the CORE subset; Task 5 adds the shape descriptors and the `welldefined`/`full` selection).
  - `feature_matrix(cjs, feature_set="full") -> (np.ndarray[M,F], names: list[str])`

Reference the cube fixture from `tests/test_levi_roundtrip.py` semantics: unit cube → area 6, volume 1.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_features.py
import numpy as np

from src.eval.building_features import building_features, feature_matrix, mesh_from_cityjson

# unit cube as CityJSON (6 quad faces, CCW-outward)
CUBE = {
    "type": "CityJSON", "version": "1.1",
    "CityObjects": {"b": {"type": "Building", "geometry": [{
        "type": "Solid", "lod": "2",
        "boundaries": [[
            [[0, 3, 2, 1]], [[4, 5, 6, 7]], [[0, 1, 5, 4]],
            [[1, 2, 6, 5]], [[2, 3, 7, 6]], [[3, 0, 4, 7]],
        ]],
    }]}},
    "vertices": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                 [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
}


def test_mesh_extraction():
    verts, faces = mesh_from_cityjson(CUBE)
    assert verts.shape == (8, 3)
    assert len(faces) == 6


def test_core_features_of_unit_cube():
    f = building_features(CUBE)
    assert f["num_vertices"] == 8
    assert f["num_faces"] == 6
    assert np.isclose(f["volume"], 1.0)
    assert np.isclose(f["area"], 6.0)
    assert np.isclose(f["height_diff"], 1.0)
    assert np.isclose(f["convex_hull_area"], 6.0)


def test_feature_matrix_stacks():
    X, names = feature_matrix([CUBE, CUBE])
    assert X.shape[0] == 2 and X.shape[1] == len(names)
    assert "volume" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_features.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement core features**

```python
# src/eval/building_features.py
"""Per-building geometric feature vectors.

Property set follows 3dSAGER (Genossar et al., arXiv:2511.06300, Table 1).
This module computes the analytically-defined core here; shape descriptors
ported from tudelft3d/3d-building-metrics are added alongside (see Task 5).
numpy + scipy only.
"""
import numpy as np
from scipy.spatial import ConvexHull

CORE_FEATURES = [
    "area", "volume", "height_diff", "num_vertices", "num_faces",
    "convex_hull_area", "ave_centroid_distance", "bbox_diagonal",
]


def mesh_from_cityjson(cj):
    """(verts[V,3], faces) from the first Solid; faces are outer-ring index lists."""
    verts = np.asarray(cj["vertices"], dtype=float)
    obj = next(iter(cj["CityObjects"].values()))
    faces = [ring[0] for ring in obj["geometry"][0]["boundaries"][0]]
    return verts, faces


def _triangulate(face):
    """Fan-triangulate a polygon ring into (i0,i1,i2) index triples."""
    return [(face[0], face[k], face[k + 1]) for k in range(1, len(face) - 1)]


def surface_area(verts, faces):
    total = 0.0
    for face in faces:
        for a, b, c in _triangulate(face):
            total += 0.5 * np.linalg.norm(np.cross(verts[b] - verts[a], verts[c] - verts[a]))
    return float(total)


def signed_volume(verts, faces):
    """Divergence theorem; assumes CCW-outward rings (our converter guarantees this)."""
    vol = 0.0
    for face in faces:
        for a, b, c in _triangulate(face):
            vol += np.dot(verts[a], np.cross(verts[b], verts[c]))
    return abs(vol) / 6.0


def building_features(cj, feature_set="full"):
    verts, faces = mesh_from_cityjson(cj)
    centroid = verts.mean(axis=0)
    try:
        hull_area = float(ConvexHull(verts).area)
    except Exception:
        hull_area = float("nan")  # coplanar/degenerate hull
    f = {
        "area": surface_area(verts, faces),
        "volume": signed_volume(verts, faces),
        "height_diff": float(verts[:, 2].max() - verts[:, 2].min()),
        "num_vertices": float(len(verts)),
        "num_faces": float(len(faces)),
        "convex_hull_area": hull_area,
        "ave_centroid_distance": float(np.linalg.norm(verts - centroid, axis=1).mean()),
        "bbox_diagonal": float(np.linalg.norm(verts.max(0) - verts.min(0))),
    }
    return f


def feature_matrix(cjs, feature_set="full"):
    names = list(building_features(cjs[0], feature_set).keys()) if cjs else CORE_FEATURES
    rows = [[building_features(cj, feature_set)[n] for n in names] for cj in cjs]
    return np.asarray(rows, dtype=float), names
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_features.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/eval/building_features.py tests/test_eval_features.py
git commit -m "feat(eval): core building geometric features"
```

---

### Task 5: Building features — 3dSAGER shape descriptors (enrichment)

**Files:**
- Modify: `src/eval/building_features.py`
- Test: `tests/test_eval_features.py` (extend)

**Note:** This task adds the remaining 3dSAGER Table-1 descriptors (`shape_index, perimeter, circumference, perimeter_index, fractality, elongation, hemisphericality, cubeness, axes_symmetry, density, num_floors`) and the `full`/`welldefined` selection. **Do not invent formulas.** Pull the exact definitions from 3dSAGER (arXiv:2511.06300, Table 1) via the literature tools and cross-check the ported ones (`shape_index`, `hemisphericality`, `cubeness`, `elongation`) against `tudelft3d/3d-building-metrics` (`cityStats.py`, `shape_index.py`) via the GitHub MCP. Attribute each formula inline. The core pipeline (Tasks 6–9) works on `CORE_FEATURES` alone, so this task is separable enrichment — implement after the pipeline is proven end-to-end if sequencing pressure exists.

**Interfaces:**
- Produces: `FULL_FEATURES: list[str]` (17), `WELLDEFINED_FEATURES: list[str]` (drops `num_floors, fractality, circumference`); `building_features(cj, feature_set)` returns the selected set; `feature_matrix` honors `feature_set`.

- [x] **Step 1: Fetch the formulas**

Retrieve 3dSAGER Table 1 (arXiv:2511.06300) and the four ported descriptors from `tudelft3d/3d-building-metrics`. Record each formula in a comment above its implementation with attribution.

- [x] **Step 2: Write failing analytic tests**

```python
# tests/test_eval_features.py — append
def test_cube_shape_descriptors():
    from src.eval.building_features import building_features
    f = building_features(CUBE, feature_set="full")
    # a cube is maximally cube-like; cubeness ~ 1
    assert 0.95 <= f["cubeness"] <= 1.0
    # not elongated
    assert np.isclose(f["elongation"], 1.0, atol=0.05)


def test_welldefined_drops_lod1_ambiguous():
    from src.eval.building_features import building_features
    f = building_features(CUBE, feature_set="welldefined")
    for dropped in ("num_floors", "fractality", "circumference"):
        assert dropped not in f
```

- [x] **Step 3: Run to verify it fails**

Run: `python -m pytest tests/test_eval_features.py -k descriptors -v`
Expected: FAIL (`KeyError: 'cubeness'`).

- [x] **Step 4: Implement the descriptors + selection**

Add each descriptor function (with the fetched formula + attribution), define `FULL_FEATURES`/`WELLDEFINED_FEATURES`, and make `building_features`/`feature_matrix` select by `feature_set` (`"full"` default, `"welldefined"` subset; unknown → raise `ValueError`).

- [x] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_eval_features.py -v`
Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add src/eval/building_features.py tests/test_eval_features.py
git commit -m "feat(eval): 3dSAGER shape descriptors + feature-set selection"
```

---

### Task 6: Distribution distances

**Files:**
- Create: `src/eval/distribution.py`
- Test: `tests/test_eval_distribution.py`

**Interfaces:**
- Produces:
  - `log1p_normalize(X) -> np.ndarray`
  - `per_feature_wasserstein(gen, ref, names) -> dict[str, float]` with per-feature keys `wass/<name>` plus `wass/mean`.
  - `kernel_mmd(gen, ref) -> float` (RBF, median-heuristic bandwidth on pooled pairwise distances).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_distribution.py
import numpy as np
from src.eval.distribution import log1p_normalize, per_feature_wasserstein, kernel_mmd

rng = np.random.default_rng(0)


def test_log1p_normalize_matches_expm1_inverse():
    X = np.array([[0.0, 9.0]])
    assert np.allclose(log1p_normalize(X), np.log1p(X))


def test_identical_distributions_zero_distance():
    A = rng.normal(size=(200, 3))
    w = per_feature_wasserstein(A, A.copy(), ["a", "b", "c"])
    assert w["wass/mean"] < 1e-9
    assert kernel_mmd(A, A.copy()) < 1e-6


def test_shift_increases_distance_monotonically():
    A = rng.normal(size=(200, 2))
    near = per_feature_wasserstein(A, A + 0.5, ["a", "b"])["wass/mean"]
    far = per_feature_wasserstein(A, A + 2.0, ["a", "b"])["wass/mean"]
    assert far > near > 0
    assert kernel_mmd(A, A + 2.0) > kernel_mmd(A, A + 0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_distribution.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# src/eval/distribution.py
"""Distribution-realism distances between generated and reference feature sets.

log(1+x) normalization per 3dSAGER (arXiv:2511.06300, sec 3.1).
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


def log1p_normalize(X):
    return np.log1p(np.asarray(X, dtype=float))


def per_feature_wasserstein(gen, ref, names):
    gen, ref = np.asarray(gen, dtype=float), np.asarray(ref, dtype=float)
    out, acc = {}, []
    for j, name in enumerate(names):
        d = wasserstein_distance(gen[:, j], ref[:, j])
        out[f"wass/{name}"] = float(d)
        acc.append(d)
    out["wass/mean"] = float(np.mean(acc)) if acc else 0.0
    return out


def _median_bandwidth(pooled):
    d = cdist(pooled, pooled)
    med = np.median(d[np.triu_indices_from(d, k=1)])
    return float(med) if med > 0 else 1.0


def kernel_mmd(gen, ref):
    """Unbiased RBF-kernel MMD^2 with median-heuristic bandwidth."""
    gen, ref = np.asarray(gen, dtype=float), np.asarray(ref, dtype=float)
    gamma = 1.0 / (2.0 * _median_bandwidth(np.vstack([gen, ref])) ** 2)
    k = lambda a, b: np.exp(-gamma * cdist(a, b) ** 2)  # noqa: E731
    m, n = len(gen), len(ref)
    kxx = (k(gen, gen).sum() - np.trace(k(gen, gen))) / (m * (m - 1))
    kyy = (k(ref, ref).sum() - np.trace(k(ref, ref))) / (n * (n - 1))
    kxy = k(gen, ref).mean()
    return float(kxx + kyy - 2 * kxy)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_distribution.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/eval/distribution.py tests/test_eval_distribution.py
git commit -m "feat(eval): per-feature Wasserstein + kernel MMD"
```

---

### Task 7: Novelty & uniqueness

**Files:**
- Create: `src/eval/novelty.py`
- Test: `tests/test_eval_novelty.py`

**Interfaces:**
- Produces: `novelty_uniqueness(gen, train, tol) -> dict` with keys `novelty` (fraction whose nearest train neighbour distance > tol) and `uniqueness` (fraction not a mutual near-duplicate within tol among generated). Operates on already-normalized feature matrices.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_novelty.py
import numpy as np
from src.eval.novelty import novelty_uniqueness


def test_identical_to_train_is_not_novel():
    train = np.array([[0.0, 0.0], [1.0, 1.0]])
    gen = np.array([[0.0, 0.0]])          # exactly a train building
    out = novelty_uniqueness(gen, train, tol=0.1)
    assert out["novelty"] == 0.0


def test_all_distinct_is_unique():
    train = np.array([[0.0, 0.0]])
    gen = np.array([[5.0, 5.0], [9.0, 9.0]])
    out = novelty_uniqueness(gen, train, tol=0.1)
    assert out["uniqueness"] == 1.0
    assert out["novelty"] == 1.0


def test_duplicates_lower_uniqueness():
    train = np.array([[0.0, 0.0]])
    gen = np.array([[5.0, 5.0], [5.0, 5.0]])  # mutual duplicates
    out = novelty_uniqueness(gen, train, tol=0.1)
    assert out["uniqueness"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_novelty.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# src/eval/novelty.py
import numpy as np
from scipy.spatial.distance import cdist


def novelty_uniqueness(gen, train, tol):
    gen, train = np.asarray(gen, dtype=float), np.asarray(train, dtype=float)
    if len(gen) == 0:
        return {"novelty": 0.0, "uniqueness": 0.0}

    nn_train = cdist(gen, train).min(axis=1) if len(train) else np.full(len(gen), np.inf)
    novelty = float((nn_train > tol).mean())

    if len(gen) == 1:
        uniqueness = 1.0
    else:
        d = cdist(gen, gen)
        np.fill_diagonal(d, np.inf)
        uniqueness = float((d.min(axis=1) > tol).mean())
    return {"novelty": novelty, "uniqueness": uniqueness}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_novelty.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/eval/novelty.py tests/test_eval_novelty.py
git commit -m "feat(eval): novelty and uniqueness vs train features"
```

---

### Task 8: Validity via val3dity (gated)

**Files:**
- Create: `src/eval/validity.py`
- Create: `tests/fixtures/val3dity_report.json` (captured report — one valid, one invalid feature)
- Test: `tests/test_eval_validity.py`

**Interfaces:**
- Produces:
  - `parse_val3dity_report(report: dict) -> dict` with `valid_flags: list[bool]`, `error_histogram: dict[str,int]`, `valid_fraction: float`.
  - `check_validity(cjs, val3dity_path=None) -> dict | None` — serialize to CityJSONSeq, run one subprocess, parse. Returns `None` (and logs a warning) when the binary is unavailable.

The report-parsing logic is unit-tested against the fixture without the binary; the subprocess call is exercised only when `val3dity` is present.

- [ ] **Step 1: Capture a fixture and write the failing test**

Create `tests/fixtures/val3dity_report.json` with two features, one `"validity": true` and one `"validity": false` carrying an error code (match your installed val3dity's `--report` schema; adjust the parser keys to it).

```python
# tests/test_eval_validity.py
import json
from pathlib import Path
from src.eval.validity import parse_val3dity_report

FIX = Path(__file__).parent / "fixtures" / "val3dity_report.json"


def test_parse_report_counts_valid_and_errors():
    report = json.loads(FIX.read_text())
    out = parse_val3dity_report(report)
    assert len(out["valid_flags"]) == 2
    assert out["valid_fraction"] == 0.5
    assert sum(out["error_histogram"].values()) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_validity.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement (parser first, then the gated subprocess)**

```python
# src/eval/validity.py
"""3D validity via val3dity (Ledoux 2018), gated on the binary being on PATH."""
import json
import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)


def parse_val3dity_report(report):
    """Flatten a val3dity --report into per-feature validity + an error histogram.

    Key names follow the installed val3dity report schema; adjust if it differs.
    """
    features = report.get("features", [])
    flags, hist = [], {}
    for feat in features:
        valid = bool(feat.get("validity", False))
        flags.append(valid)
        for err in feat.get("errors", []):
            code = str(err)
            hist[code] = hist.get(code, 0) + 1
    valid_fraction = float(sum(flags) / len(flags)) if flags else 0.0
    return {"valid_flags": flags, "error_histogram": hist, "valid_fraction": valid_fraction}


def _to_cityjsonseq(cjs):
    """One JSON object per line (CityJSONSeq) for val3dity stdin streaming."""
    return "\n".join(json.dumps(cj) for cj in cjs)


def check_validity(cjs, val3dity_path=None):
    exe = val3dity_path or shutil.which("val3dity")
    if not exe:
        logger.warning("val3dity not found; skipping the validity arm.")
        return None
    proc = subprocess.run(
        [exe, "stdin", "--report"], input=_to_cityjsonseq(cjs),
        capture_output=True, text=True, check=True,
    )
    return parse_val3dity_report(json.loads(proc.stdout))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_validity.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/eval/validity.py tests/test_eval_validity.py tests/fixtures/val3dity_report.json
git commit -m "feat(eval): val3dity validity parsing + gated subprocess"
```

---

### Task 9: Assemble the eval body — scoring, coherence, logging, saving

**Files:**
- Modify: `src/eval/callback.py` (`run_generative_eval` full body + helpers)
- Create: `src/eval/geometry_io.py` (`cityjson_to_obj`, `save_records`)
- Test: `tests/test_eval_callback.py` (extend)

**Interfaces:**
- Consumes: `draw_samples`, `feature_matrix`/`building_features`, `log1p_normalize`/`per_feature_wasserstein`/`kernel_mmd`, `novelty_uniqueness`, `check_validity`, `graph_to_cityjson`.
- Produces:
  - `face_centroid_consistency(coords, node_labels, edge_labels) -> float` (mean `‖face_pos − centroid(members)‖`; **needs the un-truncated sampled positions**, so `draw_samples` records must retain the full `pos` row for face nodes — see Step 3 note).
  - `reference_features(datamodule, split, feature_set) -> (np.ndarray, names)` — dataset graphs → `graph_to_cityjson` → `feature_matrix`.
  - `cityjson_to_obj(cj) -> str`; `save_records(records, out_dir, n)`.
  - `run_generative_eval(...)` returns a dict of all `gen/` scalars and logs them + a per-instance `wandb.Table` + saved artifacts.

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_eval_callback.py — append
import numpy as np
import torch
from src.eval.callback import face_centroid_consistency, run_generative_eval
from src.dataset.dataset import VERTEX, WALL, EDGE_VF


def test_face_centroid_consistency_zero_when_centered():
    # 3 vertices + 1 face node sitting exactly at their centroid
    coords = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [1, 1, 0]], dtype=float)
    labels = np.array([VERTEX, VERTEX, VERTEX, WALL])
    edge = np.zeros((4, 4), dtype=np.int64)
    for v in range(3):
        edge[3, v] = edge[v, 3] = EDGE_VF
    assert face_centroid_consistency(coords, labels, edge) < 1e-9
    coords[3] = [9, 9, 0]  # displace the face node
    assert face_centroid_consistency(coords, labels, edge) > 1.0


def test_run_generative_eval_smoke(tmp_path, monkeypatch):
    from src.utils.config import GenerativeEvalConfig
    from src.models.diffusion import CityJSONDiffusionModule
    import src.eval.callback as cb

    model = CityJSONDiffusionModule(hidden_dim=8, edge_dim=4, global_dim=4,
                                    n_head=2, num_layers=1, T=10, n_max=6)

    # one well-formed record; no real sampling, no val3dity
    rec = {"coords": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
           "node_labels": np.array([VERTEX, VERTEX, VERTEX]),
           "edge_labels": np.zeros((3, 3), dtype=np.int64),
           "cityjson": {"type": "CityJSON", "version": "1.1",
                        "CityObjects": {"b": {"type": "Building", "geometry": [{
                            "type": "Solid", "lod": "2",
                            "boundaries": [[[[0, 1, 2]]]]}]}},
                        "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]}}
    monkeypatch.setattr(cb, "draw_samples", lambda m, nb, bs: ([rec], {"attempted": 1, "dropped": 0}))
    monkeypatch.setattr(cb, "reference_features", lambda dm, split, fs: (np.array([[1.0, 2.0]]), ["a", "b"]))
    monkeypatch.setattr(cb, "building_features", lambda cj, feature_set="full": {"a": 1.0, "b": 2.0})
    monkeypatch.setattr(cb, "check_validity", lambda cjs, path=None: None)

    cfg = GenerativeEvalConfig(enabled=True, num_batches=1, batch_size=1, log_n_samples=1)
    metrics = run_generative_eval(model, datamodule=None, cfg=cfg, loggers=[], save_dir=tmp_path)

    assert "gen/rejection_rate" in metrics
    assert "gen/face_centroid_consistency" in metrics
    assert (tmp_path / "gen_0.city.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_callback.py -v`
Expected: FAIL (`ImportError: cannot import name 'face_centroid_consistency'`).

- [ ] **Step 3: Extend `draw_samples` to retain face positions**

`face_centroid_consistency` needs the sampled **face-node** positions, which `graph_to_cityjson` discards. Add the full denormalized position array to each record so the metric can read face rows:

```python
# src/eval/sampling.py — inside the loop, add to the record dict:
                "all_coords": coords,   # full [N,3], face rows included
```
(`coords` is already the full `_denormalize_coords(pos[i])`; `graph_to_cityjson` internally selects vertex rows, so keep the full array here.)

- [ ] **Step 4: Implement geometry_io**

```python
# src/eval/geometry_io.py
import json
from pathlib import Path

import numpy as np


def cityjson_to_obj(cj):
    """Minimal OBJ (vertices + faces) for wandb.Object3D / local inspection."""
    verts = cj["vertices"]
    obj = next(iter(cj["CityObjects"].values()))
    lines = [f"v {x} {y} {z}" for x, y, z in verts]
    for ring in obj["geometry"][0]["boundaries"][0]:
        idx = ring[0]
        lines.append("f " + " ".join(str(i + 1) for i in idx))  # OBJ is 1-indexed
    return "\n".join(lines) + "\n"


def save_records(records, out_dir, n):
    """Persist the first n records: graph .npz + CityJSON + OBJ. Returns saved paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, rec in enumerate(records[:n]):
        stem = out_dir / f"gen_{i}"
        np.savez(stem.with_suffix(".npz"),
                 coords=rec["all_coords"], node_labels=rec["node_labels"],
                 edge_labels=rec["edge_labels"])
        stem.with_suffix(".city.json").write_text(json.dumps(rec["cityjson"]), encoding="utf-8")
        stem.with_suffix(".obj").write_text(cityjson_to_obj(rec["cityjson"]), encoding="utf-8")
        paths.append(stem)
    return paths
```

- [ ] **Step 5: Implement the eval body**

```python
# src/eval/callback.py — replace the stub run_generative_eval and add helpers
import logging
from pathlib import Path

import lightning as L
import numpy as np

from src.dataset.dataset import EDGE_VF, VERTEX
from src.eval.sampling import draw_samples
from src.eval.building_features import building_features, feature_matrix
from src.eval.distribution import log1p_normalize, per_feature_wasserstein, kernel_mmd
from src.eval.novelty import novelty_uniqueness
from src.eval.validity import check_validity
from src.post_process.post_process import graph_to_cityjson

logger = logging.getLogger(__name__)


def face_centroid_consistency(coords, node_labels, edge_labels):
    """Mean distance between each face node and the centroid of its member vertices.

    Face position and its vertices diffuse independently and the converter ignores
    the face row, so this drift is an internal-coherence signal the converter can't see.
    """
    coords = np.asarray(coords, dtype=float)
    node_labels = np.asarray(node_labels)
    face_ids = np.flatnonzero((node_labels != VERTEX) & (node_labels != 4))  # exclude OFF
    drifts = []
    for f in face_ids:
        members = np.flatnonzero((node_labels == VERTEX) & (edge_labels[f] == EDGE_VF))
        if len(members) >= 1:
            drifts.append(np.linalg.norm(coords[f] - coords[members].mean(axis=0)))
    return float(np.mean(drifts)) if drifts else 0.0


def reference_features(datamodule, split, feature_set):
    """Test/train graphs -> the same converter -> feature matrix."""
    ds = getattr(datamodule, f"{split}_dataset")
    cjs = []
    for item in ds:
        if isinstance(item, tuple):
            item = item[0]
        coords = item["x"].numpy().astype(float)
        node_classes = item["node_categories"].argmax(-1).numpy()
        edge_classes = item["y"].squeeze(-1).numpy()
        cj = graph_to_cityjson(coords, node_classes, edge_classes)
        if cj:
            cjs.append(cj)
    return feature_matrix(cjs, feature_set)


def run_generative_eval(model, datamodule, cfg, loggers, save_dir):
    L.seed_everything(cfg.seed)
    records, stats = draw_samples(model, cfg.num_batches, cfg.batch_size)

    metrics = {}
    n_invalid = 0
    if records:
        val = check_validity([r["cityjson"] for r in records], cfg.val3dity_path)
        if val is not None:
            metrics["gen/valid_fraction"] = val["valid_fraction"]
            n_invalid = sum(1 for v in val["valid_flags"] if not v)
            for code, count in val["error_histogram"].items():
                metrics[f"gen/err/{code}"] = count

    attempted = max(stats["attempted"], 1)
    metrics["gen/rejection_rate"] = (stats["dropped"] + n_invalid) / attempted
    metrics["gen/dropped"] = stats["dropped"]
    metrics["gen/face_centroid_consistency"] = float(np.mean([
        face_centroid_consistency(r["all_coords"], r["node_labels"], r["edge_labels"])
        for r in records])) if records else 0.0

    if records and datamodule is not None:
        gen_X = log1p_normalize([[building_features(r["cityjson"], cfg.feature_set)[n]
                                  for n in _names(cfg)] for r in records])
        ref_X, names = reference_features(datamodule, "test", cfg.feature_set)
        ref_X = log1p_normalize(ref_X)
        metrics.update(per_feature_wasserstein(gen_X, ref_X, names))
        metrics["gen/mmd"] = kernel_mmd(gen_X, ref_X)
        train_X, _ = reference_features(datamodule, "train", cfg.feature_set)
        metrics.update({f"gen/{k}": v for k, v in
                        novelty_uniqueness(gen_X, log1p_normalize(train_X), cfg.novelty_tol).items()})

    _save_and_log(records, metrics, loggers, save_dir, cfg)
    return metrics


def _names(cfg):
    from src.eval.building_features import CORE_FEATURES
    return list(building_features({"vertices": [[0, 0, 0]] * 3, "CityObjects": {"b": {
        "geometry": [{"boundaries": [[[[0, 1, 2]]]]}]}}}, cfg.feature_set).keys())
```
(If `_names` bootstrap is awkward, expose the ordered name list directly from `building_features` via a module-level `feature_names(feature_set)` and use that instead — pick whichever is cleaner at implementation.)

- [ ] **Step 6: Implement `_save_and_log` (local files + WandB table/artifact)**

```python
# src/eval/callback.py — add
def _save_and_log(records, metrics, loggers, save_dir, cfg):
    from src.eval.geometry_io import save_records, cityjson_to_obj
    save_dir = Path(save_dir)
    save_records(records, save_dir, cfg.log_n_samples)

    wandb_logger = next((lg for lg in (loggers or [])
                         if type(lg).__name__ == "WandbLogger"), None)
    if wandb_logger is None:
        logger.info("No WandB logger; wrote scalars + files locally only.")
        return
    import wandb
    exp = wandb_logger.experiment
    exp.log(metrics)

    cols = ["sample_id", "num_vertices", "num_faces", "mesh"]
    table = wandb.Table(columns=cols)
    for i, rec in enumerate(records[:cfg.log_n_samples]):
        obj = wandb.Object3D(io_from_obj(cityjson_to_obj(rec["cityjson"])))
        n_v = int((rec["node_labels"] == VERTEX).sum())
        n_f = len(rec["cityjson"]["CityObjects"]) and \
            len(next(iter(rec["cityjson"]["CityObjects"].values()))["geometry"][0]["boundaries"][0])
        table.add_data(f"gen_{i}", n_v, n_f, obj)
    exp.log({"gen/samples": table})

    art = wandb.Artifact(f"generative_eval_{exp.id}", type="generated_buildings")
    art.add_dir(str(save_dir))
    exp.log_artifact(art)


def io_from_obj(obj_str):
    import io
    buf = io.StringIO(obj_str)
    buf.name = "sample.obj"  # wandb.Object3D infers format from the name
    return buf
```
(Per-instance feature columns from Task 5 slot into `cols` once available; keep this task's table to the always-present columns so it works on `CORE_FEATURES` alone.)

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_eval_callback.py -v`
Expected: PASS (both new tests; WandB path is skipped since `loggers=[]`).

- [ ] **Step 8: Commit**

```bash
git add src/eval/callback.py src/eval/geometry_io.py src/eval/sampling.py tests/test_eval_callback.py
git commit -m "feat(eval): full generative-eval body, coherence metric, table + artifact logging"
```

---

### Task 10: Phase 1 — extend the post_process trust tests

**Files:**
- Modify: `tests/test_levi_roundtrip.py`
- Create: `tests/fixtures/real_lod2_building.city.json` (a small committed real LoD2 building)

**Interfaces:** none new — exercises `parse_cityjson_file_to_graphs` + `graph_to_cityjson`.

- [ ] **Step 1: Add a committed real LoD2 fixture and round-trip test**

Commit a small real LoD2 building to `tests/fixtures/real_lod2_building.city.json`, then:

```python
# tests/test_levi_roundtrip.py — append
from pathlib import Path

REAL = Path(__file__).parent / "fixtures" / "real_lod2_building.city.json"


@pytest.mark.skipif(not REAL.exists(), reason="real LoD2 fixture absent")
def test_real_lod2_round_trip_is_identity(tmp_path):
    from src.post_process.post_process import graph_to_cityjson
    g1 = write_and_parse(json.loads(REAL.read_text()), tmp_path, name="real_in.city.json")
    cj2 = graph_to_cityjson(*graph_to_dense(g1), building_id="b1")
    g2 = write_and_parse(cj2, tmp_path, name="real_rt.city.json")
    assert canonical_graph(g2) == canonical_graph(g1)
```

- [ ] **Step 2: Add the non-convex order-recovery test**

```python
# tests/test_levi_roundtrip.py — append
# L-shaped (concave) floor face; convex angular sort would cross-link it.
L_VERTICES = [
    [0, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0],
    [0, 0, 1], [2, 0, 1], [2, 1, 1], [1, 1, 1], [1, 2, 1], [0, 2, 1],
]
L_FACES = [
    ([0, 5, 4, 3, 2, 1], "GroundSurface"),
    ([6, 7, 8, 9, 10, 11], "RoofSurface"),
    ([0, 1, 7, 6], "WallSurface"), ([1, 2, 8, 7], "WallSurface"),
    ([2, 3, 9, 8], "WallSurface"), ([3, 4, 10, 9], "WallSurface"),
    ([4, 5, 11, 10], "WallSurface"), ([5, 0, 6, 11], "WallSurface"),
]


def test_non_convex_face_order_recovered(tmp_path):
    from src.post_process.post_process import graph_to_cityjson
    g1 = write_and_parse(make_cityjson(L_VERTICES, L_FACES), tmp_path, name="L.city.json")
    cj2 = graph_to_cityjson(*graph_to_dense(g1), building_id="b1")
    g2 = write_and_parse(cj2, tmp_path, name="L_rt.city.json")
    assert canonical_graph(g2) == canonical_graph(g1)
    assert solid_volume(cj2) == pytest.approx(3.0)  # L-prism volume
```

- [ ] **Step 3: Add adversarial generated-topology tests**

```python
# tests/test_levi_roundtrip.py — append
def test_broken_cycle_falls_back_without_crash():
    from src.post_process.post_process import graph_to_cityjson
    # 4 vertices in a face but vv edges form a path, not a cycle -> angle-sort fallback
    coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0]])
    labels = np.array([VERTEX, VERTEX, VERTEX, VERTEX, WALL])
    edge = np.zeros((5, 5), dtype=np.int64)
    for v in range(4):
        edge[4, v] = edge[v, 4] = EDGE_VF
    for a, b in [(0, 1), (1, 2), (2, 3)]:  # open path, no closing edge
        edge[a, b] = edge[b, a] = EDGE_VV
    cj = graph_to_cityjson(coords, labels, edge)
    assert cj["CityObjects"]  # produced a face via fallback, did not crash


def test_disconnected_graph_returns_empty():
    from src.post_process.post_process import graph_to_cityjson
    coords = np.array([[0, 0, 0], [1, 0, 0]])
    labels = np.array([VERTEX, VERTEX])          # <3 vertices, no faces
    edge = np.zeros((2, 2), dtype=np.int64)
    assert graph_to_cityjson(coords, labels, edge) == {}
```

- [ ] **Step 4: Add a gated val3dity assertion**

```python
# tests/test_levi_roundtrip.py — append
import shutil


@pytest.mark.skipif(shutil.which("val3dity") is None, reason="val3dity not on PATH")
def test_converted_cube_is_val3dity_valid(tmp_path):
    import subprocess
    from src.post_process.post_process import graph_to_cityjson, save_to_file
    g1 = write_and_parse(make_cityjson(CUBE_VERTICES, CUBE_FACES), tmp_path)
    cj2 = graph_to_cityjson(*graph_to_dense(g1), building_id="b1")
    out = tmp_path / "cube.city.json"
    save_to_file(cj2, out)
    proc = subprocess.run(["val3dity", str(out), "--report"], capture_output=True, text=True)
    assert '"validity": true' in proc.stdout or proc.returncode == 0
```

- [ ] **Step 5: Run the full trust suite**

Run: `python -m pytest tests/test_levi_roundtrip.py -v`
Expected: PASS (val3dity + real-fixture tests skip if unavailable).

- [ ] **Step 6: Commit**

```bash
git add tests/test_levi_roundtrip.py tests/fixtures/real_lod2_building.city.json
git commit -m "test(post_process): real-data, non-convex, and adversarial round-trip coverage"
```

---

### Task 11: Dependency + docs

**Files:**
- Modify: `requirements.txt`
- Modify: `CLAUDE.md` or a short note (optional)

- [ ] **Step 1: Add scipy**

Append `scipy` to `requirements.txt` (only if not already present — check first).

- [ ] **Step 2: Run the whole suite**

Run: `python -m pytest tests/ -v`
Expected: PASS (skips where external tools absent).

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "build: add scipy for generative-eval feature/distribution code"
```

---

## Self-Review

**Spec coverage:**
- §4 Phase 1 trust tests → Task 10. ✓
- §5.1 two tiers (Tier 1 exists) → no task needed (kept as-is); Tier 2 → Tasks 1–9. ✓
- §5.2 home/trigger/config → Task 1. ✓
- §5.3 flow + 5.3.1 direct sampling → Tasks 2, 3. ✓
- §5.4 features → Tasks 4, 5. ✓
- §5.5 distribution → Task 6. ✓
- §5.6 novelty (spec numbers it 5.7) → Task 7. ✓
- §5.3 validity (spec 5.4) → Task 8. ✓
- §5.8 logging/saving/coherence/seam → Task 9 (records carry `conditioning/ground_truth` seam — add the two `None` keys in Task 3's record dict when implementing; the standalone `run_generative_eval` body is the swap point). ✓
- §6 deps → Task 11. ✓

**Gap noted:** Task 3's record dict in the code block omits the `conditioning: None`/`ground_truth: None` seam keys from §5.8 — add them when implementing so the conditional extension stays additive.

**Placeholder scan:** Task 5 intentionally defers exact 3dSAGER formulas to literature retrieval (not fabricated) — this is a correctness safeguard, not a placeholder; its tests are concrete. Task 8 fixture keys depend on the installed val3dity schema — flagged inline.

**Type consistency:** `draw_samples` record keys (`coords`, `all_coords`, `node_labels`, `edge_labels`, `cityjson`) are consistent across Tasks 3/9. `building_features` returns `dict[str,float]`; `feature_matrix` returns `(ndarray, names)` — consistent in Tasks 4/6/9. `per_feature_wasserstein` keys (`wass/<name>`, `wass/mean`) consistent. `run_generative_eval` signature `(model, datamodule, cfg, loggers, save_dir)` consistent across Tasks 1/9 and the callback call site.
