# SE(2) Mode + Levi Face Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Face nodes get diffused ring-centroid positions; `se2` becomes an additional equivariance mode (xy-only CoM, absolute z standardized by a train-empirical `z_shift`).

**Architecture:** Per spec `docs/superpowers/specs/2026-07-19-se2-levi-model-design.md`. All changes are surgical: one `xy_only` flag on `remove_mean_with_mask` threaded through regnn/noise, a real-node mask in `_centre_positions`, centroids at parse time, and `z_shift` mirroring the `coord_scale` contract.

**Tech Stack:** torch, Lightning, pytest.

## Global Constraints

- Default config (`equivariance="so2"`) must be numerically identical to current behavior except: face slots carry centroids, and `_centre_positions` centers over real nodes instead of vertex nodes.
- Real-node mask = `1 − node_categories[..., -1]` (last class is Off/Virtual in both 5-class and legacy 2-class conventions).
- Centered targets must keep an exactly-zero mean over all N slots (OFF slots pinned to 0) so the all-ones projections in `noise.py`/regnn stay consistent — no loss floor.
- `z_shift` is subtracted in metres before the `/ coord_scale` division; inverted in `generate_cityjson`. It is 0.0 unless `equivariance == "se2"`.
- Blast radius (established by grep; gitnexus index stale — reindex crashes on known FTS bug): `remove_mean_with_mask` ← noise.py ×3, regnn ×2, PositionsMLP; `_centre_positions` ← `_prepare` only; `equivariance` currently regnn-internal, must be threaded config→model.

---

### Task 1: Centroid positions + real-mask centering

**Files:**
- Modify: `src/dataset/dataset.py` (parse: face coords = ring centroid)
- Modify: `src/models/diffusion.py` (`_centre_positions`, `_prepare`)
- Test: `tests/test_levi_roundtrip.py` (update cube assertion), `tests/test_centering.py` (new)

**Interfaces:**
- Produces: raw graph dict `x[face]` = mean of that face's ring vertex coords (after any `normalize_coords` shift).
- Produces: `_centre_positions(R0, node_categories, xy_only=False, z_shift=0.0)` — subtracts `[mean_x, mean_y, mean_z]` (or `[mean_x, mean_y, z_shift]` when `xy_only`) computed over real nodes, zeroes OFF slots.

- [ ] **Step 1: Write failing tests**

In `tests/test_levi_roundtrip.py`, replace the two face-coordinate assertions in `test_parse_cube_levi_structure`:

```python
    # face nodes sit at their ring centroid
    expected_centroids = torch.tensor(
        [np.mean([CUBE_VERTICES[v] for v in ring], axis=0) for ring, _ in CUBE_FACES],
        dtype=torch.float32,
    )
    assert torch.allclose(g["x"][8:], expected_centroids)
```

New `tests/test_centering.py`:

```python
"""_centre_positions: real-node centering that keeps the all-ones subspace exact.

Targets must have zero mean over ALL N slots (the noise model and network
project positions with an all-ones mask), while OFF slots stay exactly zero.
"""
import torch

from src.models.diffusion import CityJSONDiffusionModule

B, N = 2, 6


def _cats(labels, num_classes):
    return torch.nn.functional.one_hot(torch.tensor(labels), num_classes).float().expand(B, -1, -1).clone()


def test_centre_positions_translates_faces_and_zeroes_off():
    R0 = torch.randn(B, N, 3)
    # 3 vertices, 2 faces, 1 off (class 4 == last)
    cats = _cats([0, 0, 0, 1, 3, 4], 5)

    out = CityJSONDiffusionModule._centre_positions(R0, cats)

    real = out[:, :5]
    assert torch.allclose(real.mean(dim=1), torch.zeros(B, 3), atol=1e-6)
    assert torch.all(out[:, 5] == 0)                       # OFF pinned to zero
    assert torch.allclose(out.mean(dim=1), torch.zeros(B, 3), atol=1e-6)  # all-N mean 0
    # faces are translated, not zeroed: relative geometry preserved
    rel_before = R0[:, 3] - R0[:, 0]
    rel_after = out[:, 3] - out[:, 0]
    assert torch.allclose(rel_before, rel_after, atol=1e-6)


def test_centre_positions_xy_only_keeps_absolute_z_minus_shift():
    R0 = torch.randn(B, N, 3)
    cats = _cats([0, 0, 0, 1, 3, 4], 5)

    out = CityJSONDiffusionModule._centre_positions(R0, cats, xy_only=True, z_shift=5.0)

    real = out[:, :5]
    assert torch.allclose(real[..., :2].mean(dim=1), torch.zeros(B, 2), atol=1e-6)
    assert torch.allclose(real[..., 2], R0[:, :5, 2] - 5.0, atol=1e-6)  # z untouched by centering
    assert torch.all(out[:, 5] == 0)


def test_centre_positions_matches_legacy_for_two_class_batches():
    """With 2-class categories (Active/Virtual) and zero virtual coords, the
    real mask equals the old node_mask and results are unchanged."""
    R0 = torch.zeros(B, N, 3)
    R0[:, :4] = torch.randn(B, 4, 3)
    cats = _cats([0, 0, 0, 0, 1, 1], 2)

    out = CityJSONDiffusionModule._centre_positions(R0, cats)

    mean = R0[:, :4].mean(dim=1, keepdim=True)
    assert torch.allclose(out[:, :4], R0[:, :4] - mean, atol=1e-6)
    assert torch.all(out[:, 4:] == 0)
```

- [ ] **Step 2: Run, expect failures** — `python -m pytest tests/test_levi_roundtrip.py tests/test_centering.py -v`: centroid assertion fails (zeros), centering tests fail (old signature takes `node_mask` and zeroes faces).

- [ ] **Step 3: Implement**

`src/dataset/dataset.py`, in `parse_cityjson_file_to_graphs` after `x[:n_vertices] = coords`:

```python
        for f_i, (ring, _) in enumerate(faces):
            x[n_vertices + f_i] = coords[[idx_map[v] for v in ring]].mean(axis=0)
```

and update the function docstring line "(classes GROUND/ROOF/WALL, zero coords)" to "(classes GROUND/ROOF/WALL, positioned at their ring centroid)".

`src/models/diffusion.py`, replace `_centre_positions` and its call:

```python
    @staticmethod
    def _centre_positions(R0, node_categories, xy_only=False, z_shift=0.0):
        """Centre real (vertex + face) nodes, leaving Off nodes at 0.

        The network is constrained to emit zero-CoM coordinates (PositionsMLP and
        every layer re-centre over all N slots), so the target must live on the
        same subspace or the coordinate loss has an irreducible floor. Centering
        over the real nodes and pinning Off slots to zero makes the all-N mean
        exactly zero. The real mask is 1 - P(last class): the last class is
        Off/Virtual in both the 5-class and legacy 2-class conventions.

        Under se2 (`xy_only=True`) only the xy mean is removed — z keeps its
        absolute value minus `z_shift` (the train-split mean vertex height), the
        moment-matching analogue of the discrete marginal priors.
        """
        real = (1.0 - node_categories[..., -1]).unsqueeze(-1)     # [B, N, 1]
        num_real = real.sum(dim=1, keepdim=True).clamp(min=1)
        mean = (R0 * real).sum(dim=1, keepdim=True) / num_real
        if xy_only:
            mean = torch.cat(
                (mean[..., :2], torch.full_like(mean[..., 2:], z_shift)), dim=-1
            )
        return (R0 - mean) * real
```

In `_prepare` (equivariance/z_shift attrs arrive in Task 3; keep defaults for now):

```python
        R0 = self._centre_positions(batch["x"], batch["node_categories"]) / self.coord_scale
```

- [ ] **Step 4: Run** — `python -m pytest tests/ -v`: all pass (legacy 2-class tests unchanged via the equivalence test's argument).
- [ ] **Step 5: Commit** — `git add -A src tests; git commit -m "feat: face nodes carry diffused ring centroids; centre over real nodes"`

---

### Task 2: `xy_only` CoM + `se2` in layers/regnn/noise

**Files:**
- Modify: `src/models/layers.py` (`remove_mean_with_mask`, `PositionsMLP`)
- Modify: `src/models/regnn.py` (accept "se2", thread `xy_only`, docstring)
- Modify: `src/models/noise.py` (`GraphNoiseModel(xy_only_com=...)`, 3 call sites)
- Test: `tests/test_equivariance.py` (new), `tests/test_model_smoke.py` (add "se2" param)

**Interfaces:**
- Produces: `remove_mean_with_mask(x, node_mask, xy_only=False)`; `PositionsMLP(hidden_dim, eps=1e-5, xy_only=False)`; `rEGNNTransformer(..., equivariance="se2")` valid; `GraphNoiseModel(..., xy_only_com=False)`.

- [ ] **Step 1: Write failing tests**

`tests/test_equivariance.py`:

```python
"""Symmetry regressions for the so2 / se2 modes.

Yaw rotation about z must rotate the position output and leave X/E logits
unchanged; xy translation of the input must be quotiented out by the CoM
projection; under se2 the projection must not touch z.
"""
import math

import pytest
import torch

from src.models.layers import remove_mean_with_mask
from src.models.regnn import rEGNNTransformer

B, N = 2, 6


def _rot_z(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _inputs():
    torch.manual_seed(0)
    X = torch.nn.functional.one_hot(torch.randint(0, 5, (B, N)), 5).float()
    E = torch.nn.functional.one_hot(torch.randint(0, 3, (B, N, N)), 3).float()
    E = 0.5 * (E + E.transpose(1, 2))
    R = torch.randn(B, N, 3)
    t = torch.rand(B, 1)
    return X, E, R, t


def _net(equivariance):
    torch.manual_seed(1)
    return rEGNNTransformer(num_node_classes=5, num_edge_classes=3, hidden_dim=8,
                            edge_dim=4, global_dim=4, n_head=2, num_layers=2,
                            equivariance=equivariance).eval()


@pytest.mark.parametrize("equivariance", ["so2", "se2"])
def test_yaw_rotation_equivariance(equivariance):
    net = _net(equivariance)
    X, E, R, t = _inputs()
    Q = _rot_z(0.7)

    with torch.no_grad():
        pos1, E1, X1 = net(X, R, E, t)
        pos2, E2, X2 = net(X, R @ Q.T, E, t)

    assert torch.allclose(pos2, pos1 @ Q.T, atol=1e-4)
    assert torch.allclose(X2, X1, atol=1e-4)
    assert torch.allclose(E2, E1, atol=1e-4)


@pytest.mark.parametrize("equivariance", ["so2", "se2"])
def test_xy_translation_is_quotiented(equivariance):
    net = _net(equivariance)
    X, E, R, t = _inputs()
    shift = torch.tensor([3.0, -2.0, 0.0])

    with torch.no_grad():
        pos1, E1, X1 = net(X, R, E, t)
        pos2, E2, X2 = net(X, R + shift, E, t)

    assert torch.allclose(pos2, pos1, atol=1e-4)
    assert torch.allclose(X2, X1, atol=1e-4)


def test_se2_network_uses_absolute_z():
    """z translation is NOT a symmetry of se2: output must change."""
    net = _net("se2")
    X, E, R, t = _inputs()

    with torch.no_grad():
        pos1, _, X1 = net(X, R, E, t)
        pos2, _, X2 = net(X, R + torch.tensor([0.0, 0.0, 4.0]), E, t)

    assert not torch.allclose(X2, X1, atol=1e-4)


def test_remove_mean_xy_only_leaves_z():
    torch.manual_seed(0)
    x = torch.randn(B, N, 3)
    mask = torch.ones(B, N)

    out = remove_mean_with_mask(x, mask, xy_only=True)

    assert torch.allclose(out[..., :2].mean(dim=1), torch.zeros(B, 2), atol=1e-6)
    assert torch.allclose(out[..., 2], x[..., 2], atol=1e-6)
```

In `tests/test_model_smoke.py`, extend the parametrize to `["so2", "se2", "o3"]`.

- [ ] **Step 2: Run, expect failures** — se2 raises `Unknown equivariance`; `xy_only` kwarg unknown.

- [ ] **Step 3: Implement**

`layers.py`:

```python
def remove_mean_with_mask(x, node_mask, xy_only=False):
    """Project onto the zero centre-of-mass subspace, ignoring padded nodes.

    Args:
        x (Tensor): [B, N, D]
        node_mask (Tensor): [B, N], 1 for active nodes.
        xy_only (bool): se2 mode — remove the mean of the first two components
            only; the last (z) passes through untouched.
    """
    mask = node_mask.unsqueeze(-1).to(x.dtype)
    num_nodes = mask.sum(dim=1, keepdim=True).clamp(min=1)
    mean = (x * mask).sum(dim=1, keepdim=True) / num_nodes
    if xy_only:
        mean = torch.cat((mean[..., :2], torch.zeros_like(mean[..., 2:])), dim=-1)
    return (x - mean) * mask
```

`PositionsMLP.__init__` gains `xy_only=False` (stored), and its last line becomes
`return remove_mean_with_mask(new_pos, node_mask, xy_only=self.xy_only)`.

`regnn.py`:
- `mask_graph(X, E, pos, node_mask, xy_only=False)`; its `remove_mean_with_mask` call passes `xy_only`.
- `NodeEdgeBlock`: accept `"se2"`; `node_geom_dim`/`pair_geom_dim` and the feature branch test `equivariance in ("so2", "se2")`; velocity line becomes `remove_mean_with_mask(vel, node_mask, xy_only=self.equivariance == "se2")`.
- `XEyTransformerLayer`: pass equivariance through (already does); its `mask_graph` call passes `xy_only=self.self_attn.equivariance == "se2"`.
- `rEGNNTransformer`: `self.xy_only = equivariance == "se2"`; both `PositionsMLP(pos_mlp_dim, xy_only=self.xy_only)`; both `mask_graph(..., xy_only=self.xy_only)`; docstring gains the `"se2"` entry (so2 features + xy-only translation quotient, absolute z).

`noise.py`: `GraphNoiseModel.__init__(..., xy_only_com=False)` stored; the three `remove_mean_with_mask(..., node_mask)` calls become `remove_mean_with_mask(..., node_mask, xy_only=self.xy_only_com)`.

- [ ] **Step 4: Run** — `python -m pytest tests/ -v`: all pass.
- [ ] **Step 5: Commit** — `git commit -m "feat: se2 equivariance mode — xy-only zero-CoM in network and noise"`

---

### Task 3: Thread config + `z_shift` end to end

**Files:**
- Modify: `src/utils/config.py` (`ModelConfig.equivariance`, `DataConfig.z_shift`)
- Modify: `src/dataset/datamodule.py` (`compute_z_shift`)
- Modify: `src/models/diffusion.py` (ctor args `equivariance`, `z_shift`; `_prepare`; `generate_cityjson` z inversion; pass to network + noise)
- Modify: `src/utils/setup_utils.py`, `src/train.py`, `configs/*.yaml`
- Test: `tests/test_z_shift.py` (new)

**Interfaces:**
- Produces: `CityJSONDiffusionModule(..., equivariance="so2", z_shift=0.0)` (hparams-saved); `CityJSONDataModule.compute_z_shift() -> float`.

- [ ] **Step 1: Write failing tests**

`tests/test_z_shift.py`:

```python
"""se2 z standardization: train-empirical mean out in _prepare, back in decode.

The Gaussian forward process converges to N(0, I) regardless of the data, so
matching the empirical z distribution means standardizing with train-split
moments (the analogue of the discrete marginal priors), not sampling a
histogram. z_shift rides in checkpoint hparams exactly like coord_scale.
"""
import torch

from src.dataset.datamodule import CityJSONDataModule
from src.models.diffusion import CityJSONDiffusionModule

B, N = 2, 6


def _batch():
    torch.manual_seed(0)
    x = torch.zeros(B, N, 3)
    x[:, :4] = torch.randn(B, 4, 3) + torch.tensor([0.0, 0.0, 10.0])
    cats = torch.zeros(B, N, 5)
    cats[:, :4, 0] = 1.0
    cats[:, 4:, 4] = 1.0
    return {
        "x": x,
        "node_categories": cats,
        "y": torch.randint(0, 3, (B, N, N, 1)),
        "node_mask": (cats[..., 0] == 1).float(),
    }


def _model(**kw):
    return CityJSONDiffusionModule(hidden_dim=8, edge_dim=4, global_dim=4, n_head=2,
                                   num_layers=1, T=10, n_max=N, **kw)


def test_compute_z_shift_is_pooled_vertex_z_mean():
    items = []
    for zs in ([1.0, 3.0], [5.0]):
        n = len(zs)
        x = torch.zeros(N, 3)
        x[:n, 2] = torch.tensor(zs)
        mask = torch.zeros(N)
        mask[:n] = 1.0
        items.append({"x": x, "node_mask": mask})

    dm = CityJSONDataModule(dataset_dir="unused", lods=1)
    dm.train_dataset = items

    assert abs(dm.compute_z_shift() - 3.0) < 1e-6


def test_prepare_se2_standardizes_z_and_centres_xy_only():
    model = _model(equivariance="se2", z_shift=10.0, coord_scale=2.0)
    batch = _batch()

    R0 = model._prepare(batch)[0]

    real = R0[:, :4]
    assert torch.allclose(real[..., :2].mean(dim=1), torch.zeros(B, 2), atol=1e-5)
    expected_z = (batch["x"][:, :4, 2] - 10.0) / 2.0
    assert torch.allclose(real[..., 2], expected_z, atol=1e-5)
    assert torch.all(R0[:, 4:] == 0)


def test_generate_cityjson_restores_z_shift(monkeypatch):
    import src.post_process.post_process as pp

    model = _model(equivariance="se2", z_shift=7.0, coord_scale=2.0)

    pos = torch.ones(1, 4, 3)
    node_labels = torch.zeros(1, 4, dtype=torch.long)
    edge_labels = torch.zeros(1, 4, 4, dtype=torch.long)
    monkeypatch.setattr(model, "sample",
                        lambda batch_size=1: (pos, node_labels, edge_labels))

    seen = {}

    def fake(coords, node_classes, edge_classes, building_id):
        seen["coords"] = coords.copy()
        return {"type": "CityJSON"}

    monkeypatch.setattr(pp, "graph_to_cityjson", fake)
    model.generate_cityjson(batch_size=1)

    assert torch.allclose(torch.as_tensor(seen["coords"][:, :2]),
                          torch.full((4, 2), 2.0), atol=1e-6)
    assert torch.allclose(torch.as_tensor(seen["coords"][:, 2]),
                          torch.full((4,), 9.0), atol=1e-6)   # 1*2 + 7


def test_z_shift_survives_checkpoint(tmp_path):
    import lightning as L

    model = _model(equivariance="se2", z_shift=4.5, coord_scale=1.0)
    ckpt = tmp_path / "m.ckpt"
    trainer = L.Trainer(logger=False, enable_checkpointing=False, accelerator="cpu")
    trainer.strategy.connect(model)
    trainer.save_checkpoint(ckpt)

    restored = CityJSONDiffusionModule.load_from_checkpoint(ckpt, map_location="cpu")
    assert restored.z_shift == 4.5
    assert restored.equivariance == "se2"
```

- [ ] **Step 2: Run, expect failures** — unknown ctor kwargs, missing `compute_z_shift`.

- [ ] **Step 3: Implement**

- `config.py`: `ModelConfig.equivariance: str = "so2"  # "so2" | "se2" | "o3"`; `DataConfig.z_shift: Optional[float] = None  # se2 only; None = train-split mean vertex z`.
- `datamodule.py`, next to `compute_coord_scale` (mirror its docstring conventions):

```python
    def compute_z_shift(self):
        """Pooled mean of vertex-node z over the train split (se2 only).

        The se2 mode keeps absolute heights; subtracting the train-split mean
        makes the z channel zero-mean so the N(0, 1) position prior matches the
        data through the whole chain. Stored in checkpoint hparams like
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
```

- `diffusion.py`: ctor gains `equivariance="so2", z_shift=0.0` (validated: se2 requires nothing extra; non-se2 forces `z_shift = 0.0`); store both; pass `equivariance=equivariance` to `rEGNNTransformer` and `xy_only_com=(equivariance == "se2")` to `GraphNoiseModel`; docstring entries. `_prepare`:

```python
        R0 = self._centre_positions(
            batch["x"], batch["node_categories"],
            xy_only=self.equivariance == "se2", z_shift=self.z_shift,
        ) / self.coord_scale
```

  `generate_cityjson`: after `coords = (pos[i] * self.coord_scale).cpu().numpy()` add `coords[:, 2] += self.z_shift`.
- `setup_utils.create_model`: pass `equivariance=cfg.model.equivariance` and a new `z_shift: float = 0.0` parameter through.
- `train.py`: read the current `coord_scale` block (around line 53) and mirror it:

```python
    z_shift = cfg.data.z_shift
    if z_shift is None:
        z_shift = datamodule.compute_z_shift() if cfg.model.equivariance == "se2" else 0.0
```

  and pass `z_shift=z_shift` into `create_model`. Log it beside coord_scale.
- `configs/*.yaml`: add `equivariance: so2` under `model:` (train/inference/default), comment `# so2 | se2 | o3`.

- [ ] **Step 4: Run full suite** — `python -m pytest tests/ -v`: all pass.
- [ ] **Step 5: Commit** — `git commit -m "feat: thread se2 equivariance and z_shift through config, datamodule and module"`

---

### Task 4: Closeout

- [ ] Full suite green; `git diff main --stat` sanity check (only expected files).
- [ ] Update memory `levi-graph-representation.md` (face coords are now centroids; se2/z_shift contract).
- [ ] Note gitnexus index remains stale (known FTS-extension crash) — flag to user.
