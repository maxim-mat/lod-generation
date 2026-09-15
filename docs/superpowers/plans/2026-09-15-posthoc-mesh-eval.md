# Post-hoc Mesh Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score one `MeshTransformerModule` checkpoint on a data split under several decoding strategies, writing a per-object CSV per strategy plus everything needed to reproduce it.

**Architecture:** One additive change to `generate` (top-p, min-p, per-row seeds). New metric helpers (`roof_submesh`, `hausdorff_max`, torch Sinkhorn). A new module `src/eval/posthoc_eval.py` that expands the strategy grid, scores objects, writes artifacts and exposes a CLI. The eval YAML holds only its own settings and points at a training config.

**Tech Stack:** Python, PyTorch, Lightning, OmegaConf, numpy, trimesh, manifold3d, pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-posthoc-mesh-eval-design.md`

## Global Constraints

- Work on the current branch `mesh-transformer`; never create a branch.
- No new dependencies. Sinkhorn is hand-written in torch (POT and geomloss are not installed).
- Before editing an existing function, run `impact` upstream on it (gitnexus MCP, or `node .gitnexus/run.cjs impact "<symbol>" --direction upstream --repo .`) and report the result. Stop on HIGH/CRITICAL.
- Before every commit, run `detect_changes({scope: "all"})` (or `node .gitnexus/run.cjs detect-changes --scope all --repo .`). A `partial` or `truncated` result is not clean; re-run.
- Stage only the files the task names. The working tree has unrelated untracked files; never `git add -A`.
- Tests never load the full dataset. The only real-data test reads one tile per LOD with `max_files=1`.
- Run only the test files named in a step, not the whole suite. Do not run the eval script itself.
- Keep reproducibility artifacts: seeds, config dumps, run metadata, error handling around I/O, docstrings on shared functions.
- Cite papers at the point of implementation: top-p arXiv:1904.09751, min-p arXiv:2407.01082, Sinkhorn arXiv:1306.0895, Sinkhorn divergence arXiv:1810.08278.
- If a test from this plan fails for a reason the plan did not predict, stop and report. Never loosen a threshold to make it pass.

## File Structure

| File | Responsibility |
|---|---|
| `src/models/mesh_transformer.py` | `filter_logits`, `_gumbel_argmax`; `generate` gains `top_p`, `min_p`, `seeds` |
| `src/eval/mesh_metrics.py` | `hausdorff_max`, `roof_submesh` (numpy only) |
| `src/eval/sinkhorn.py` | debiased Sinkhorn divergence on point clouds and meshes (torch) |
| `src/utils/config.py` | `StrategyConfig`, `SinkhornConfig`, `PosthocEvalConfig` |
| `src/eval/posthoc_eval.py` | strategy expansion, config loading/validation, `score_object`, `run_posthoc_eval`, artifacts, CLI |
| `configs/posthoc-eval.yaml` | shipped default eval config |
| `tests/test_sampling_filters.py` | Task 1 |
| `tests/test_roof_and_hausdorff.py` | Task 2 |
| `tests/test_sinkhorn.py` | Task 3 |
| `tests/test_posthoc_eval.py` | Tasks 4, 5, 7, 8 |
| `tests/test_metric_sanity_real.py` | Task 6 |

---

### Task 1: Top-p, min-p and per-row seeded sampling in `generate`

**Files:**
- Modify: `src/models/mesh_transformer.py` (add helpers after `invalid_logits_mask`, ends ~line 206; replace `generate`, ~lines 1419-1495)
- Test: `tests/test_sampling_filters.py`

**Interfaces:**
- Produces: `filter_logits(logits: Tensor[B, V], temperature: float, top_p: float | None = None, min_p: float | None = None) -> Tensor[B, V]` (scaled logits, `-inf` where filtered out).
- Produces: `MeshTransformerModule.generate(cond, cond_pad_mask=None, max_new_tokens=None, temperature=1.0, beam_size=1, length_penalty=1.0, use_cache=True, top_p=None, min_p=None, seeds=None) -> Tensor[B, L]`. `seeds` is a sequence of one int per row; it is ignored by greedy and beam decoding.

- [ ] **Step 1: Re-check blast radius**

Run `impact` upstream on `generate` in `src/models/mesh_transformer.py`. Expected: LOW, flagged as a lower bound. Known text-search callers are `src/eval/mesh_eval.py::run_mesh_eval`, `tests/test_beam_search.py`, `tests/test_kv_cache.py`, `tests/test_mesh_amt.py`, `tests/test_mesh_transformer.py` and `tests/test_mesh_code_transformer.py`. All new arguments default to off.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_sampling_filters.py`:

```python
"""Top-p / min-p filtering and per-row seeded sampling for `generate`.

Nucleus (top-p) sampling: Holtzman et al., "The Curious Case of Neural Text
Degeneration", arXiv:1904.09751. Min-p: Nguyen et al., "Turning Up the Heat:
Min-p Sampling for Creative and Coherent LLM Outputs", arXiv:2407.01082.

The network is replaced by a stub with a known next-token distribution, so the
assertions test the sampler, not an untrained model's noise.
"""
import pytest
import torch

from src.dataset.mesh_dataset import specials, vocab_size
from src.models.mesh_transformer import MeshTransformerModule, filter_logits

NUM_BINS = 8
V = vocab_size(NUM_BINS)
BOS, EOS, PAD = specials(NUM_BINS)
CAP = 12

PROBS = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
LOGITS = PROBS.log()


def _kept(filtered):
    return torch.isfinite(filtered[0]).nonzero().flatten().tolist()


def test_top_p_keeps_the_smallest_set_reaching_the_mass():
    assert _kept(filter_logits(LOGITS, 1.0, top_p=0.75)) == [0, 1]
    assert _kept(filter_logits(LOGITS, 1.0, top_p=0.85)) == [0, 1, 2]


def test_min_p_keeps_tokens_above_a_fraction_of_the_top_probability():
    assert _kept(filter_logits(LOGITS, 1.0, min_p=0.2)) == [0, 1, 2]   # p >= 0.10
    assert _kept(filter_logits(LOGITS, 1.0, min_p=0.5)) == [0, 1]      # p >= 0.25


def test_filters_compose_as_an_intersection():
    assert _kept(filter_logits(LOGITS, 1.0, top_p=0.85, min_p=0.5)) == [0, 1]


def test_temperature_is_applied_before_the_filter():
    # At T=2 the probabilities flatten to ~[0.379, 0.294, 0.208, 0.120], so the
    # min-p threshold 0.5 * 0.379 = 0.19 now admits token 2 as well.
    out = filter_logits(LOGITS, 2.0, min_p=0.5)
    assert _kept(out) == [0, 1, 2]
    assert torch.allclose(out[0, 0], LOGITS[0, 0] / 2.0)


def test_masked_tokens_stay_masked_and_a_row_is_never_emptied():
    logits = torch.tensor([[float("-inf"), 0.0, float("-inf"), -50.0]])
    assert _kept(filter_logits(logits, 1.0, top_p=1e-9, min_p=1.0)) == [1]


@pytest.mark.parametrize("temperature, kw", [
    (0.0, {}), (1.0, {"top_p": 0.0}), (1.0, {"top_p": 1.5}),
    (1.0, {"min_p": -0.1}), (1.0, {"min_p": 1.1}),
])
def test_invalid_arguments_raise(temperature, kw):
    with pytest.raises(ValueError):
        filter_logits(LOGITS, temperature, **kw)


class _Die(torch.nn.Module):
    """Network stub: every step is the same distribution over the 8 coordinate
    tokens. `skew` raises token 0's logit; everything else is unreachable."""

    max_seq_len = CAP

    def __init__(self, skew=0.0):
        super().__init__()
        self.skew = skew

    def generation_budget(self, n_cond):
        return CAP - 1

    def forward(self, cond, tgt, *a, **k):
        out = torch.full((tgt.shape[0], tgt.shape[1] - 1, V), -1e30)
        out[:, :, :NUM_BINS] = 0.0
        out[:, :, 0] = self.skew
        return out


def _module(skew=0.0):
    m = MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2,
                              num_layers=1, dropout=0.0, max_seq_len=CAP).eval()
    m.network = _Die(skew)
    return m


def _gen(m, seeds, **kw):
    """`use_cache=False`: the stub implements `forward` only."""
    kw.setdefault("temperature", 1.0)
    cond = torch.zeros(len(seeds), 4, dtype=torch.long)
    with torch.no_grad():
        return m.generate(cond, use_cache=False, seeds=seeds, **kw)


def test_same_seeds_give_the_same_tokens():
    m = _module()
    assert torch.equal(_gen(m, [3, 4]), _gen(m, [3, 4]))


def test_different_seeds_give_different_tokens():
    out = _gen(_module(), [3, 4])
    assert not torch.equal(out[0], out[1])


def test_a_rows_sample_does_not_depend_on_its_batch_neighbours():
    m = _module()
    together = _gen(m, [3, 4, 5])
    assert torch.equal(together[1], _gen(m, [4])[0])
    assert torch.equal(together[[2, 0, 1]], _gen(m, [5, 3, 4]))


@pytest.mark.parametrize("kw", [{"top_p": 0.3}, {"min_p": 0.5}])
def test_a_tight_filter_leaves_only_the_top_token(kw):
    # skew 2.0: p(token 0) = e^2 / (e^2 + 7) = 0.513, every other token 0.070.
    m = _module(skew=2.0)
    assert (_gen(m, [3, 4], **kw)[:, 1:] == 0).all()
    with torch.no_grad():
        unseeded = m.generate(torch.zeros(2, 4, dtype=torch.long),
                              temperature=1.0, use_cache=False, **kw)
    assert (unseeded[:, 1:] == 0).all()


def test_without_a_filter_the_skewed_die_still_rolls_other_faces():
    assert (_gen(_module(skew=2.0), [3, 4])[:, 1:] != 0).any()


def test_unseeded_sampling_still_draws_from_the_global_rng():
    """`seeds=None` without a filter must stay token-identical to before this
    change: one `torch.multinomial` per step on the global RNG."""
    m = _module()
    cond = torch.zeros(2, 4, dtype=torch.long)
    torch.manual_seed(0)
    with torch.no_grad():
        got = m.generate(cond, temperature=1.0, use_cache=False)
    torch.manual_seed(0)
    probs = torch.zeros(2, V)
    probs[:, :NUM_BINS] = 1.0 / NUM_BINS
    expect = torch.stack([torch.multinomial(probs, 1).squeeze(-1)
                          for _ in range(CAP - 1)], dim=1)
    assert torch.equal(got[:, 1:], expect)


@pytest.mark.parametrize("kw", [{"top_p": 0.9}, {"min_p": 0.1}])
def test_filters_with_beam_search_are_rejected(kw):
    with pytest.raises(ValueError, match="beam"):
        _module().generate(torch.zeros(1, 4, dtype=torch.long), temperature=0.0,
                           beam_size=2, use_cache=False, **kw)


def test_filters_without_temperature_are_rejected():
    with pytest.raises(ValueError, match="temperature"):
        _module().generate(torch.zeros(1, 4, dtype=torch.long), temperature=0.0,
                           top_p=0.9, use_cache=False)


def test_seed_count_must_match_the_batch():
    with pytest.raises(ValueError, match="seeds"):
        _module().generate(torch.zeros(3, 4, dtype=torch.long), temperature=1.0,
                           seeds=[1, 2], use_cache=False)
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_sampling_filters.py -v`
Expected: collection error, `ImportError: cannot import name 'filter_logits'`.

- [ ] **Step 4: Add the helpers**

In `src/models/mesh_transformer.py`, directly after the `return mask` that ends `invalid_logits_mask`, insert:

```python
def filter_logits(logits, temperature, top_p=None, min_p=None):
    """Temperature, then nucleus and/or min-p truncation, for one decoding step.

    Temperature is applied first and the filter reads the tempered
    probabilities -- the HuggingFace logits-warper order -- so a higher
    temperature widens what the filter admits.

    * top-p (Holtzman et al., arXiv:1904.09751): keep the smallest
      highest-probability set whose cumulative mass reaches ``top_p``.
    * min-p (Nguyen et al., arXiv:2407.01082): keep tokens with
      ``p >= min_p * max(p)``.

    Given both, a token must pass both. The most likely token always survives,
    so combined with `invalid_logits_mask` a row can never be left with nothing
    to sample.

    Args:
        logits: [B, V], already masked with ``-inf`` where illegal.
        temperature (float): > 0. Greedy decoding never calls this.
        top_p (float, optional): in (0, 1].
        min_p (float, optional): in [0, 1].

    Returns:
        Tensor: [B, V] tempered logits, ``-inf`` where filtered out.
    """
    if temperature <= 0:
        raise ValueError(f"filter_logits needs temperature > 0, got {temperature}.")
    if top_p is not None and not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}.")
    if min_p is not None and not 0.0 <= min_p <= 1.0:
        raise ValueError(f"min_p must be in [0, 1], got {min_p}.")

    scaled = logits.float() / temperature
    probs = torch.softmax(scaled, dim=-1)
    keep = torch.ones_like(probs, dtype=torch.bool)
    if top_p is not None:
        sorted_p, order = probs.sort(dim=-1, descending=True)
        # Mass strictly before each token: a token stays while the set so far
        # has not yet reached top_p, which keeps the token that crosses it.
        before = sorted_p.cumsum(dim=-1) - sorted_p
        keep &= torch.zeros_like(keep).scatter(-1, order, before < top_p)
    if min_p is not None:
        keep &= probs >= min_p * probs.max(dim=-1, keepdim=True).values
    keep.scatter_(-1, probs.argmax(dim=-1, keepdim=True), True)
    return scaled.masked_fill(~keep, float("-inf"))


def _gumbel_argmax(logits, generators):
    """One categorical draw per row, each from that row's own generator.

    Gumbel-max: ``argmax(logits + G)`` with ``G = -log(-log(U))`` is an exact
    sample from ``softmax(logits)``. Drawing ``U`` per row, rather than one
    ``torch.multinomial`` over the batch, is what makes a row's sample depend on
    its own seed only -- not on which buildings share its batch or their order.
    """
    u = torch.stack([torch.rand(logits.shape[-1], generator=g, device=logits.device)
                     for g in generators])
    # rand is in [0, 1): clamp the 0 end so the double log stays finite.
    gumbel = -torch.log(-torch.log(u.clamp_min(torch.finfo(u.dtype).tiny)))
    return (logits.float() + gumbel).argmax(dim=-1)
```

- [ ] **Step 5: Replace `generate`**

Replace the whole `generate` method (from `@torch.no_grad()` above `def generate` through its `return tgt`) with:

```python
    @torch.no_grad()
    def generate(self, cond, cond_pad_mask=None, max_new_tokens=None,
                 temperature=1.0, beam_size=1, length_penalty=1.0,
                 use_cache=True, top_p=None, min_p=None, seeds=None):
        """Decode an LOD2 token sequence from LOD1 tokens.

        Args:
            max_new_tokens: defaults to `network.generation_budget(len(cond))`,
                and is clamped to it. Callers size this from the task
                (`mesh_eval` asks for ``9 * max_faces + 1``) and cannot know
                what the position scheme allows, so the clamp lives here --
                the one place that has both numbers.
            temperature: <= 0 is greedy argmax, > 0 samples.
            beam_size: 1 keeps the greedy/sampling loop. > 1 runs beam
                search, which is deterministic and therefore rejects a
                sampling temperature. Note the reference does *not* do this:
                MeshAnything V2 passes `num_beams=1` and offers top-k/top-p
                as its only alternative to greedy.
            use_cache: incremental decoding. A pure speedup -- pinned
                token-identical to the uncached path by
                `tests/test_kv_cache.py` -- so False is an escape hatch,
                not a mode.
            length_penalty: beam scores are divided by ``length ** penalty``,
                as in HF `generate`. 0.0 is raw sum-of-log-probs, which is
                biased toward short sequences -- and short here means a mesh
                with fewer triangles, so the default is 1.0 (mean log-prob
                per token) rather than the usual 0.
            top_p, min_p: truncate the sampling distribution; see
                `filter_logits`. Need ``temperature > 0`` and ``beam_size == 1``.
            seeds: one int per row. Each row then samples from its own
                generator, so its tokens do not depend on its batch neighbours.
                Ignored by greedy and beam decoding. None samples from the
                global RNG exactly as before these arguments existed.

        Returns:
            Tensor: [B, L] generated tokens, BOS-led, EOS-terminated where the
            model chose to stop. Decode with
            `src.dataset.mesh_dataset.detokenize`.
        """
        bos, eos = self.bos, self.eos
        if beam_size < 1:
            raise ValueError(f"beam_size must be >= 1, got {beam_size}.")
        if beam_size > 1 and temperature > 0:
            raise ValueError(
                f"beam_size={beam_size} with temperature={temperature} asks for a "
                "deterministic argmax over sequences and a random draw at the "
                "same time. Use temperature<=0 with beams, or beam_size=1 to "
                "sample.")
        filtered = top_p is not None or min_p is not None
        if filtered and beam_size > 1:
            raise ValueError(
                f"top_p/min_p truncate a sampling distribution, but beam_size="
                f"{beam_size} runs a deterministic beam search that never samples.")
        if filtered and temperature <= 0:
            raise ValueError(
                f"top_p/min_p need temperature > 0 to sample, got {temperature}; "
                "temperature <= 0 is greedy argmax, which no filter can change.")
        if seeds is not None and len(seeds) != cond.shape[0]:
            raise ValueError(f"seeds has {len(seeds)} entries for a batch of "
                             f"{cond.shape[0]} rows; pass one seed per row.")
        budget = self.network.generation_budget(cond.shape[1])
        if budget < 1:
            raise ValueError(
                f"A condition of {cond.shape[1]} tokens leaves no room to "
                f"generate against a budget of {self.network.max_seq_len}. "
                "Lower mesh_data.max_faces, or set "
                "mesh_model.pos_embed=sinusoidal, which has no ceiling.")
        if max_new_tokens is None:
            max_new_tokens = budget
        elif max_new_tokens > budget:
            logger.info("Generation budget %d -> %d (position limit, "
                        "condition is %d tokens)",
                        max_new_tokens, budget, cond.shape[1])
            max_new_tokens = budget
        if beam_size > 1:
            return self._beam_search(cond, cond_pad_mask, max_new_tokens,
                                     beam_size, length_penalty, use_cache)
        tgt = torch.full((cond.shape[0], 1), bos, dtype=torch.long, device=cond.device)
        done = torch.zeros(cond.shape[0], dtype=torch.bool, device=cond.device)
        cache = self.network.new_cache(cond, cond_pad_mask) if use_cache else None
        generators = None
        if seeds is not None and temperature > 0:
            generators = [torch.Generator(device=cond.device).manual_seed(int(s))
                          for s in seeds]

        for _ in range(max_new_tokens):
            logits = self._step_logits(cond, tgt, cond_pad_mask, cache)

            if temperature <= 0:
                nxt = logits.argmax(dim=-1)
            elif not filtered and generators is None:
                nxt = torch.multinomial(
                    torch.softmax(logits / temperature, dim=-1), num_samples=1).squeeze(-1)
            else:
                warped = filter_logits(logits, temperature, top_p=top_p, min_p=min_p)
                if generators is None:
                    nxt = torch.multinomial(torch.softmax(warped, dim=-1),
                                            num_samples=1).squeeze(-1)
                else:
                    nxt = _gumbel_argmax(warped, generators)
            nxt = torch.where(done, torch.full_like(nxt, self.pad), nxt)
            tgt = torch.cat([tgt, nxt[:, None]], dim=1)
            done |= nxt == eos
            if bool(done.all()):
                break
        return tgt
```

- [ ] **Step 6: Run the new tests and confirm they pass**

Run: `python -m pytest tests/test_sampling_filters.py -v`
Expected: all PASS.

- [ ] **Step 7: Run the existing decoding tests**

Run: `python -m pytest tests/test_beam_search.py tests/test_kv_cache.py tests/test_mesh_amt.py tests/test_mesh_transformer.py tests/test_mesh_eval.py -v`
Expected: all PASS, same as before the change.

- [ ] **Step 8: Commit**

Run `detect_changes({scope: "all"})` and confirm only `generate`, `filter_logits` and `_gumbel_argmax` changed. Then:

```bash
git add src/models/mesh_transformer.py tests/test_sampling_filters.py
git commit -m "feat(mesh): top-p, min-p and per-row seeded sampling in generate"
```

### Task 2: `roof_submesh` and `hausdorff_max` in `mesh_metrics`

**Files:**
- Modify: `src/eval/mesh_metrics.py` (add `hausdorff_max` after `hausdorff_p95`; add `roof_submesh` after `_mean_roof_z`)
- Test: `tests/test_roof_and_hausdorff.py`

**Interfaces:**
- Produces: `hausdorff_max(d_ab: np.ndarray, d_ba: np.ndarray) -> float` (NaN if either is empty).
- Produces: `roof_submesh(verts, faces, up=_UP) -> tuple[np.ndarray [V', 3], np.ndarray [F', 3] int64]`, compacted; both empty when nothing faces up.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_roof_and_hausdorff.py`:

```python
"""Roof extraction and the textbook Hausdorff distance for the post-hoc eval.

Roof means a winding-derived unit normal with z > 0.1: the rule
`mesh_to_cityjson` uses to label RoofSurface. Winding decides it, not position.
"""
import numpy as np
import trimesh

from src.eval.mesh_metrics import hausdorff_max, roof_submesh


def _box():
    """2 m cube centred on the origin, outward-wound, 12 triangles."""
    b = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    return np.asarray(b.vertices, dtype=float), np.asarray(b.faces, dtype=np.int64)


def test_roof_of_an_outward_box_is_its_top_face():
    v, f = roof_submesh(*_box())
    assert f.shape == (2, 3)
    assert len(v) == 4                      # unreferenced vertices dropped
    assert np.allclose(v[:, 2], 1.0)
    assert f.max() == len(v) - 1            # indices point into the compacted verts


def test_flipped_winding_reads_the_floor_as_roof():
    v, f = _box()
    rv, rf = roof_submesh(v, f[:, ::-1])
    assert rf.shape == (2, 3)
    assert np.allclose(rv[:, 2], -1.0)


def test_a_walls_only_mesh_has_no_roof():
    v, f = _box()
    normals = trimesh.Trimesh(vertices=v, faces=f, process=False).face_normals
    rv, rf = roof_submesh(v, f[np.abs(normals[:, 2]) < 0.5])
    assert rv.shape == (0, 3) and rf.shape == (0, 3)


def test_an_empty_mesh_has_no_roof():
    rv, rf = roof_submesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    assert rv.shape == (0, 3) and rf.shape == (0, 3)


def test_hausdorff_max_is_the_worse_direction():
    assert hausdorff_max(np.array([0.1, 0.4]), np.array([0.2, 0.3])) == 0.4
    assert np.isnan(hausdorff_max(np.zeros(0), np.array([1.0])))
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_roof_and_hausdorff.py -v`
Expected: collection error, `ImportError: cannot import name 'hausdorff_max'`.

- [ ] **Step 3: Implement**

In `src/eval/mesh_metrics.py`, directly after `hausdorff_p95`:

```python
def hausdorff_max(d_ab, d_ba):
    """Textbook Hausdorff distance: the worst deviation in either direction, metres.

    Reported next to `hausdorff_p95`, which exists because one stray vertex
    decides this number. With both, the gap between them shows how much of the
    worst case is a single outlier.
    """
    if len(d_ab) == 0 or len(d_ba) == 0:
        return float("nan")
    return float(max(d_ab.max(), d_ba.max()))
```

Directly after `_mean_roof_z`:

```python
def roof_submesh(verts, faces, up=_UP):
    """The upward-facing triangles of a mesh, compacted to their own vertices.

    Roof means a winding-derived unit normal with ``z > up``: the same cut
    `mesh_to_cityjson` uses to label RoofSurface and `_mean_roof_z` uses here.
    Winding decides it, so a triangle wound backwards reads as ground and a
    backwards floor reads as roof -- exactly as in the CityJSON write-back.

    Returns:
        tuple: (verts [V', 3], faces [F', 3] int64) with unreferenced vertices
        dropped; both empty when nothing faces up.
    """
    empty = (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    v = np.asarray(verts, dtype=float).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(f) == 0:
        return empty

    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cross = np.cross(b - a, c - a)
    norm = np.linalg.norm(cross, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        nz = np.where(norm > 0, cross[:, 2] / norm, 0.0)

    roof = f[nz > up]
    if len(roof) == 0:
        return empty
    used, inverse = np.unique(roof, return_inverse=True)
    # reshape: numpy 2.0.0 returned `inverse` in the input's shape, others flat.
    return v[used], inverse.reshape(-1, 3).astype(np.int64)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_roof_and_hausdorff.py tests/test_mesh_metrics.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

Run `detect_changes({scope: "all"})`. Then:

```bash
git add src/eval/mesh_metrics.py tests/test_roof_and_hausdorff.py
git commit -m "feat(eval): roof_submesh and hausdorff_max metric helpers"
```

---

### Task 3: Debiased Sinkhorn divergence

**Files:**
- Create: `src/eval/sinkhorn.py`
- Test: `tests/test_sinkhorn.py`

**Interfaces:**
- Consumes: `src.eval.mesh_metrics.sample_surface(verts, faces, n, seed) -> np.ndarray [n, 3]`.
- Produces: `sinkhorn_divergence(x, y, blur=0.1, max_iters=500, tol=1e-6, device="cpu") -> float`, the square root of 2·S in input units (NaN if either cloud is empty).
- Produces: `mesh_sinkhorn(gen, ref, n_points=2048, blur=0.1, max_iters=500, tol=1e-6, seed=0, device="cpu") -> float`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sinkhorn.py`:

```python
"""Debiased Sinkhorn divergence, reported as sqrt(2 S) so it reads in metres.

With cost |x - y|^2 / 2, translating one cloud by d adds exactly d^2 / 2 to
OT_eps(a, b) and nothing to the two self terms, so the reported value of a
translated copy is d at any blur -- a closed-form target, not a tuned one.
"""
import numpy as np
import trimesh

from src.eval.sinkhorn import mesh_sinkhorn, sinkhorn_divergence


def _cloud(n=200, seed=0):
    return np.random.default_rng(seed).random((n, 3))


def _box():
    b = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    return np.asarray(b.vertices, dtype=float), np.asarray(b.faces, dtype=np.int64)


def test_a_cloud_against_itself_is_zero():
    x = _cloud()
    assert sinkhorn_divergence(x, x, blur=0.05) == 0.0


def test_a_translated_copy_scores_the_translation():
    x = _cloud()
    got = sinkhorn_divergence(x, x + np.array([1.0, 0.0, 0.0]), blur=0.05,
                              max_iters=2000, tol=1e-9)
    assert abs(got - 1.0) < 0.02


def test_world_coordinates_do_not_cost_precision():
    """EPSG:28992 puts y near 450,000; float32 squared distances there lose the
    metre scale. The solver shifts to a shared origin and runs in float64."""
    v, f = _box()
    world = v + np.array([80_000.0, 450_000.0, 0.0])
    lifted = world + np.array([0.0, 0.0, 0.5])
    got = mesh_sinkhorn((world, f), (lifted, f), n_points=256, blur=0.05,
                        max_iters=2000, tol=1e-9)
    assert abs(got - 0.5) < 0.02


def test_identical_meshes_score_zero_because_both_sides_share_a_seed():
    v, f = _box()
    assert mesh_sinkhorn((v, f), (v.copy(), f.copy()), n_points=256) == 0.0


def test_empty_input_is_nan():
    assert np.isnan(sinkhorn_divergence(np.zeros((0, 3)), _cloud()))
    v, f = _box()
    assert np.isnan(mesh_sinkhorn((np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)),
                                  (v, f), n_points=64))
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_sinkhorn.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'src.eval.sinkhorn'`.

- [ ] **Step 3: Implement**

Create `src/eval/sinkhorn.py`:

```python
"""Debiased Sinkhorn divergence between point clouds, in metres.

Entropic optimal transport solved in the log domain (Cuturi, "Sinkhorn
Distances: Lightspeed Computation of Optimal Transport", arXiv:1306.0895). The
entropic bias is removed with the Sinkhorn divergence
``S = OT(a, b) - OT(a, a) / 2 - OT(b, b) / 2``, using epsilon-scaling and
averaged dual updates (Feydy et al., "Interpolating between Optimal Transport
and MMDs using Sinkhorn Divergences", arXiv:1810.08278).

Cost is ``|x - y|^2 / 2``, so ``sqrt(2 S)`` tends to the 2-Wasserstein distance
as the blur shrinks and is exactly ``d`` for a copy translated by ``d``.

Torch rather than numpy: three n x n log-domain solves per pair are too slow in
CPU numpy at n = 2048, and `mesh_metrics` is numpy-only by its own rule.
"""
import math

import numpy as np
import torch

from src.eval.mesh_metrics import sample_surface


def _eps_schedule(diameter, blur, scaling=0.5):
    """Epsilons ``(diameter * scaling^k)^2`` stepping down to ``blur^2``.

    Starting at the blur is correct but slow: a cold solve at small epsilon
    needs on the order of diameter^2 / epsilon iterations. Annealing from the
    diameter reaches the same fixed point in a handful of steps per octave.
    """
    diameter = max(float(diameter), blur)
    steps = int(math.ceil(math.log(diameter / blur) / math.log(1.0 / scaling)))
    return [(diameter * scaling ** k) ** 2 for k in range(steps)] + [blur ** 2]


def _ot_eps(x, y, schedule, max_iters, tol):
    """Entropic OT cost ``<a, f> + <b, g>`` at the dual optimum, uniform weights."""
    log_a = torch.full((len(x),), -math.log(len(x)), dtype=x.dtype, device=x.device)
    log_b = torch.full((len(y),), -math.log(len(y)), dtype=x.dtype, device=x.device)
    cost = 0.5 * torch.cdist(x, y).pow(2)
    f = torch.zeros(len(x), dtype=x.dtype, device=x.device)
    g = torch.zeros(len(y), dtype=x.dtype, device=x.device)

    def step(eps, f, g):
        f_new = -eps * torch.logsumexp(log_b[None, :] + (g[None, :] - cost) / eps, dim=1)
        g_new = -eps * torch.logsumexp(log_a[:, None] + (f[:, None] - cost) / eps, dim=0)
        # Averaged (symmetric) updates: the plain alternating scheme oscillates.
        return 0.5 * (f + f_new), 0.5 * (g + g_new)

    for eps in schedule[:-1]:
        f, g = step(eps, f, g)
    for _ in range(max_iters):
        f_new, g_new = step(schedule[-1], f, g)
        delta = max(float((f_new - f).abs().max()), float((g_new - g).abs().max()))
        f, g = f_new, g_new
        if delta < tol:
            break
    return float((log_a.exp() * f).sum() + (log_b.exp() * g).sum())


def sinkhorn_divergence(x, y, blur=0.1, max_iters=500, tol=1e-6, device="cpu"):
    """``sqrt(2 S_eps(x, y))`` in the units of the input.

    Args:
        x, y: [n, 3] and [m, 3] numpy arrays in the same units (metres here).
        blur (float): entropic length scale, ``eps = blur ** 2``.
        max_iters (int): iteration cap at the final epsilon.
        tol (float): stop when no dual potential moves more than this, in
            squared input units.
        device: torch device for the solve. Always float64.

    Returns:
        float: ``nan`` if either cloud is empty. Identical inputs return exactly
        0, because the three solves then run on bit-identical tensors.
    """
    x = np.asarray(x, dtype=float).reshape(-1, 3)
    y = np.asarray(y, dtype=float).reshape(-1, 3)
    if len(x) == 0 or len(y) == 0:
        return float("nan")

    # One shared origin, then float64: at EPSG:28992 magnitudes (y ~ 450,000)
    # float32 squared distances cannot resolve a metre.
    origin = np.minimum(x.min(axis=0), y.min(axis=0))
    xt = torch.as_tensor(x - origin, dtype=torch.float64, device=device)
    yt = torch.as_tensor(y - origin, dtype=torch.float64, device=device)
    extent = np.maximum(x.max(axis=0), y.max(axis=0)) - origin
    schedule = _eps_schedule(np.linalg.norm(extent), blur)

    s = (_ot_eps(xt, yt, schedule, max_iters, tol)
         - 0.5 * _ot_eps(xt, xt, schedule, max_iters, tol)
         - 0.5 * _ot_eps(yt, yt, schedule, max_iters, tol))
    return math.sqrt(2.0 * max(s, 0.0))


def mesh_sinkhorn(gen, ref, n_points=2048, blur=0.1, max_iters=500, tol=1e-6,
                  seed=0, device="cpu"):
    """`sinkhorn_divergence` between area-weighted surface samples of two meshes.

    Both meshes are sampled with the SAME seed. Identical meshes then give
    identical clouds and score exactly 0; independent seeds would leave a
    self-score of sampling noise on the order of the sample spacing.

    Returns:
        float: ``nan`` when either mesh has no positive surface area.
    """
    p = sample_surface(*gen, n=n_points, seed=seed)
    q = sample_surface(*ref, n=n_points, seed=seed)
    if len(p) == 0 or len(q) == 0:
        return float("nan")
    return sinkhorn_divergence(p, q, blur=blur, max_iters=max_iters, tol=tol,
                               device=device)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_sinkhorn.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

Run `detect_changes({scope: "all"})`. Then:

```bash
git add src/eval/sinkhorn.py tests/test_sinkhorn.py
git commit -m "feat(eval): debiased Sinkhorn divergence for mesh surfaces"
```

---

### Task 4: Eval config, loading, validation and strategy expansion

**Files:**
- Modify: `src/utils/config.py` (add three dataclasses after `GenerativeEvalConfig`, before `Config`)
- Create: `src/eval/posthoc_eval.py`
- Create: `configs/posthoc-eval.yaml`
- Test: `tests/test_posthoc_eval.py`

**Interfaces:**
- Produces (`src.utils.config`): `StrategyConfig`, `SinkhornConfig(n_points=2048, blur_m=0.1, max_iters=500, tol=1e-6)`, `PosthocEvalConfig`.
- Produces (`src.eval.posthoc_eval`):
  - `METRICS = ("chamfer", "hausdorff", "vol_symdiff", "sinkhorn", "roof_chamfer")`, `REFERENCES = ("lod1", "lod2")`, `SPLITS = ("train", "val", "test")`
  - `DecodeRun(name: str, kind: str, params: dict)` with `.generate_kwargs() -> dict`
  - `expand_strategies(strategies: list[StrategyConfig]) -> list[DecodeRun]`
  - `validate_posthoc_config(cfg: PosthocEvalConfig) -> list[DecodeRun]`
  - `load_posthoc_config(path: Path, overrides: list[str]) -> PosthocEvalConfig`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_posthoc_eval.py`:

```python
"""Post-hoc eval: config, strategy grid, per-object scoring, artifacts, CLI.

Synthetic meshes and an untrained tiny model only. The real-data check lives in
tests/test_metric_sanity_real.py.
"""
from pathlib import Path

import numpy as np
import pytest
from omegaconf.errors import OmegaConfBaseException

from src.eval.posthoc_eval import (expand_strategies, load_posthoc_config,
                                   validate_posthoc_config)
from src.utils.config import PosthocEvalConfig, StrategyConfig

ROOT = Path(__file__).resolve().parents[1]


def _cfg(**kw):
    kw.setdefault("reference_config", "unused.yaml")
    kw.setdefault("checkpoint", "unused.ckpt")
    return PosthocEvalConfig(**kw)


# ---------------------------------------------------------------- Task 4

def test_default_strategies_expand_to_the_seven_named_runs():
    names = [r.name for r in expand_strategies(_cfg().strategies)]
    assert names == ["greedy", "beam_k3_lp1.0", "beam_k5_lp1.0",
                     "top_p_p0.95_t0.6", "top_p_p0.95_t1.2",
                     "min_p_p0.05_t0.6", "min_p_p0.05_t1.2"]


def test_each_run_maps_to_the_generate_arguments_it_names():
    runs = {r.name: r.generate_kwargs() for r in expand_strategies(_cfg().strategies)}
    assert runs["greedy"] == {"temperature": 0.0}
    assert runs["beam_k5_lp1.0"] == {"temperature": 0.0, "beam_size": 5,
                                     "length_penalty": 1.0}
    assert runs["top_p_p0.95_t1.2"] == {"temperature": 1.2, "top_p": 0.95}
    assert runs["min_p_p0.05_t0.6"] == {"temperature": 0.6, "min_p": 0.05}


def test_beam_length_penalty_defaults_to_one():
    runs = expand_strategies([StrategyConfig(kind="beam", beam_size=[3])])
    assert [r.name for r in runs] == ["beam_k3_lp1.0"]


@pytest.mark.parametrize("strategy, match", [
    (StrategyConfig(kind="nucleus"), "Unknown strategy kind"),
    (StrategyConfig(kind="greedy", temperature=[0.6]), "takes no"),
    (StrategyConfig(kind="beam", beam_size=[3], top_p=[0.9]), "takes no"),
    (StrategyConfig(kind="top_p", top_p=[0.9]), "temperature"),
    (StrategyConfig(kind="min_p", min_p=[0.05], temperature=[0.0]), "temperature"),
    (StrategyConfig(kind="beam", beam_size=[]), "beam_size"),
])
def test_bad_strategies_are_rejected(strategy, match):
    with pytest.raises(ValueError, match=match):
        expand_strategies([strategy])


def test_duplicate_run_names_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        expand_strategies([StrategyConfig(kind="greedy"), StrategyConfig(kind="greedy")])


@pytest.mark.parametrize("kw, match", [
    ({"metrics": ["chamfer", "emd"]}, "metric"),
    ({"references": ["lod3"]}, "reference"),
    ({"split": "holdout"}, "split"),
    ({"strategies": []}, "strategy"),
    ({"batch_size": 0}, "batch_size"),
])
def test_validation_fails_fast(kw, match):
    with pytest.raises(ValueError, match=match):
        validate_posthoc_config(_cfg(**kw))


def test_loading_merges_yaml_and_overrides(tmp_path):
    path = tmp_path / "eval.yaml"
    path.write_text("reference_config: configs/mesh-v5-full-coord.yaml\n"
                    "checkpoint: some.ckpt\n"
                    "strategies:\n  - {kind: beam, beam_size: [2]}\n")
    cfg = load_posthoc_config(path, ["max_objects=16", "sinkhorn.blur_m=0.2"])
    assert cfg.max_objects == 16 and cfg.sinkhorn.blur_m == 0.2
    assert [r.name for r in validate_posthoc_config(cfg)] == ["beam_k2_lp1.0"]


def test_loading_rejects_unknown_keys(tmp_path):
    path = tmp_path / "eval.yaml"
    path.write_text("reference_config: a.yaml\ncheckpoint: b.ckpt\n"
                    "strategies:\n  - {kind: beam, beams: [3]}\n")
    with pytest.raises(OmegaConfBaseException):
        load_posthoc_config(path, [])


def test_loading_requires_a_checkpoint(tmp_path):
    path = tmp_path / "eval.yaml"
    path.write_text("reference_config: a.yaml\n")
    with pytest.raises(OmegaConfBaseException):
        load_posthoc_config(path, [])


def test_the_shipped_config_is_valid():
    cfg = load_posthoc_config(ROOT / "configs" / "posthoc-eval.yaml", [])
    assert len(validate_posthoc_config(cfg)) == 7
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_posthoc_eval.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'src.eval.posthoc_eval'`.

- [ ] **Step 3: Add the config dataclasses**

In `src/utils/config.py`, directly before `@dataclass\nclass Config:`:

```python
@dataclass
class StrategyConfig:
    """One decoding strategy for `src.eval.posthoc_eval`.

    Every param is a list, and lists expand as a cartesian product within the
    strategy. Params that do not belong to `kind` must stay null:
      greedy -- none
      beam   -- beam_size, length_penalty (defaults to [1.0])
      top_p  -- top_p, temperature
      min_p  -- min_p, temperature
    """
    kind: str = MISSING
    beam_size: Optional[List[int]] = None
    length_penalty: Optional[List[float]] = None
    top_p: Optional[List[float]] = None
    min_p: Optional[List[float]] = None
    temperature: Optional[List[float]] = None


@dataclass
class SinkhornConfig:
    """Debiased Sinkhorn divergence settings (`src.eval.sinkhorn`)."""
    n_points: int = 2048      # surface samples per mesh, same seed both sides
    blur_m: float = 0.1       # entropic length scale; eps = blur_m ** 2
    max_iters: int = 500      # cap at the final epsilon
    tol: float = 1e-6         # dual-potential change, m^2


def _default_strategies():
    return [StrategyConfig(kind="greedy"),
            StrategyConfig(kind="beam", beam_size=[3, 5], length_penalty=[1.0]),
            StrategyConfig(kind="top_p", top_p=[0.95], temperature=[0.6, 1.2]),
            StrategyConfig(kind="min_p", min_p=[0.05], temperature=[0.6, 1.2])]


@dataclass
class PosthocEvalConfig:
    """Post-hoc scoring of one mesh-transformer checkpoint under many decoders.

    Holds only eval settings. Data, split seed, tokenizer and model shape all
    come from `reference_config`, the run's own training config, so the split
    and the vocabulary cannot drift from what the checkpoint was trained on.
    """
    reference_config: str = MISSING
    checkpoint: str = MISSING
    # Dot-list applied to the reference, e.g. a relocated dataset_dir.
    reference_overrides: List[str] = field(default_factory=list)
    split: str = "val"
    # None scores the whole split; else a seeded subsample via eval_indices.
    max_objects: Optional[int] = None
    batch_size: int = 8
    seed: int = 1234
    device: str = "auto"
    # None -> <save_dir>/<experiment_name>/<run_name>/posthoc_eval
    output_dir: Optional[str] = None
    strategies: List[StrategyConfig] = field(default_factory=_default_strategies)
    metrics: List[str] = field(default_factory=lambda: [
        "chamfer", "hausdorff", "vol_symdiff", "sinkhorn", "roof_chamfer"])
    references: List[str] = field(default_factory=lambda: ["lod1", "lod2"])
    n_points: int = 4096      # surface samples for the chamfer family
    sinkhorn: SinkhornConfig = field(default_factory=SinkhornConfig)
```

- [ ] **Step 4: Create the module with expansion, validation and loading**

Create `src/eval/posthoc_eval.py`:

```python
"""Post-hoc evaluation of a mesh-transformer checkpoint across decoding strategies.

Usage:
    python -m src.eval.posthoc_eval --config configs/posthoc-eval.yaml [dot.overrides]

`src/eval/mesh_eval.py` scores greedy or beam decoding on a small subsample
during training. This scores a finished checkpoint on a whole split under a grid
of strategies, one CSV row per object, and writes everything needed to
reproduce any row: resolved configs, checkpoint hash, git state, versions,
per-object seeds.
"""
import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from src.utils.config import PosthocEvalConfig

logger = logging.getLogger(__name__)

METRICS = ("chamfer", "hausdorff", "vol_symdiff", "sinkhorn", "roof_chamfer")
REFERENCES = ("lod1", "lod2")
SPLITS = ("train", "val", "test")

# Which StrategyConfig fields each kind reads, in run-name order.
STRATEGY_PARAMS = {
    "greedy": (),
    "beam": ("beam_size", "length_penalty"),
    "top_p": ("top_p", "temperature"),
    "min_p": ("min_p", "temperature"),
}
_ALL_PARAMS = ("beam_size", "length_penalty", "top_p", "min_p", "temperature")
_PARAM_DEFAULTS = {"length_penalty": [1.0]}
_ABBREV = {"beam_size": "k", "length_penalty": "lp", "top_p": "p", "min_p": "p",
           "temperature": "t"}


@dataclass
class DecodeRun:
    """One point of the strategy grid: a name for the output folder plus params."""
    name: str
    kind: str
    params: dict

    def generate_kwargs(self):
        """Keyword arguments for `MeshTransformerModule.generate`, seeds excluded."""
        p = self.params
        if self.kind == "greedy":
            return {"temperature": 0.0}
        if self.kind == "beam":
            return {"temperature": 0.0, "beam_size": int(p["beam_size"]),
                    "length_penalty": float(p["length_penalty"])}
        return {"temperature": float(p["temperature"]), self.kind: float(p[self.kind])}


def expand_strategies(strategies):
    """Cartesian product of each strategy's list-valued params.

    Raises:
        ValueError: unknown kind, a param that does not belong to the kind, a
            missing or empty required list, a non-positive temperature, or two
            runs with the same name.
    """
    runs = []
    for s in strategies:
        if s.kind not in STRATEGY_PARAMS:
            raise ValueError(f"Unknown strategy kind {s.kind!r}; expected one of "
                             f"{sorted(STRATEGY_PARAMS)}.")
        allowed = STRATEGY_PARAMS[s.kind]
        for key in _ALL_PARAMS:
            if key not in allowed and getattr(s, key) is not None:
                raise ValueError(f"Strategy {s.kind!r} takes no {key!r}; "
                                 f"allowed: {list(allowed)}.")
        grid = []
        for key in allowed:
            values = getattr(s, key)
            if values is None:
                values = _PARAM_DEFAULTS.get(key)
            if not values:
                raise ValueError(f"Strategy {s.kind!r} needs a non-empty {key!r} list.")
            if key == "temperature" and min(values) <= 0:
                raise ValueError(f"Strategy {s.kind!r} samples, so every temperature "
                                 f"must be > 0; got {list(values)}.")
            grid.append([(key, v) for v in values])
        for combo in itertools.product(*grid):
            name = "_".join([s.kind] + [f"{_ABBREV[k]}{v}" for k, v in combo])
            runs.append(DecodeRun(name=name, kind=s.kind, params=dict(combo)))

    names = [r.name for r in runs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"Duplicate strategy runs: {dupes}.")
    return runs


def validate_posthoc_config(cfg):
    """Everything checkable without touching a model, data or checkpoint.

    Returns:
        list[DecodeRun]: the expanded strategy grid.
    """
    bad = sorted(set(cfg.metrics) - set(METRICS))
    if bad or not cfg.metrics:
        raise ValueError(f"Unknown or empty metric list {list(cfg.metrics)}; "
                         f"expected a subset of {list(METRICS)}.")
    bad = sorted(set(cfg.references) - set(REFERENCES))
    if bad or not cfg.references:
        raise ValueError(f"Unknown or empty reference list {list(cfg.references)}; "
                         f"expected a subset of {list(REFERENCES)}.")
    if cfg.split not in SPLITS:
        raise ValueError(f"Unknown split {cfg.split!r}; expected one of {list(SPLITS)}.")
    if cfg.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {cfg.batch_size}.")
    if not cfg.strategies:
        raise ValueError("At least one strategy is required.")
    return expand_strategies(cfg.strategies)


def load_posthoc_config(path, overrides):
    """YAML + dot-list overrides merged onto the `PosthocEvalConfig` schema.

    Unknown keys and missing `reference_config` / `checkpoint` raise an OmegaConf
    error here, before anything expensive is loaded.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Eval config not found: {path}")
    merged = OmegaConf.merge(OmegaConf.structured(PosthocEvalConfig),
                             OmegaConf.load(path),
                             OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_object(merged)
```

- [ ] **Step 5: Create the shipped default config**

Create `configs/posthoc-eval.yaml`:

```yaml
# Post-hoc evaluation across decoding strategies.
# Usage: python -m src.eval.posthoc_eval --config configs/posthoc-eval.yaml [dot.overrides]
#
# Holds only eval settings. Data, split seed, tokenizer and model shape come from
# `reference_config`, the run's own training config.

reference_config: configs/mesh-v5-full-coord.yaml
checkpoint: outputs/initial-runs/mesh-v5-full-coord/checkpoints/last.ckpt
# Dot-list applied to the reference, e.g.
#   ["mesh_data.dataset_dir=data/The Hague/mini_remote"]
reference_overrides: []

split: val
# null scores the whole split (~26k objects on cleaner). Set e.g. 256 for a
# first pass; the subsample is seeded, so it is the same buildings every time.
max_objects: null
batch_size: 8
seed: 1234
device: auto
output_dir: null      # null -> <save_dir>/<experiment_name>/<run_name>/posthoc_eval

# List-valued params expand as a cartesian product within each strategy.
strategies:
  - {kind: greedy}
  - {kind: beam,  beam_size: [3, 5], length_penalty: [1.0]}
  - {kind: top_p, top_p: [0.95], temperature: [0.6, 1.2]}
  - {kind: min_p, min_p: [0.05], temperature: [0.6, 1.2]}

metrics: [chamfer, hausdorff, vol_symdiff, sinkhorn, roof_chamfer]
references: [lod1, lod2]
n_points: 4096
sinkhorn: {n_points: 2048, blur_m: 0.1, max_iters: 500, tol: 1.0e-6}
```

- [ ] **Step 6: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_posthoc_eval.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

Run `detect_changes({scope: "all"})`. Then:

```bash
git add src/utils/config.py src/eval/posthoc_eval.py configs/posthoc-eval.yaml tests/test_posthoc_eval.py
git commit -m "feat(eval): post-hoc eval config and strategy grid expansion"
```

---

### Task 5: `score_object` — every metric against every reference

**Files:**
- Modify: `src/eval/posthoc_eval.py`
- Test: `tests/test_posthoc_eval.py` (append)

**Interfaces:**
- Consumes: `surface_distances`, `chamfer_distance`, `hausdorff_p95`, `hausdorff_max`, `roof_submesh`, `is_watertight_mesh` from `src.eval.mesh_metrics`; `exact_pair_volumes` from `src.analysis.exact_volume`; `mesh_sinkhorn` from `src.eval.sinkhorn`; `SinkhornConfig`.
- Produces: `score_object(gen, refs: dict[str, mesh], metrics, n_points=4096, sinkhorn_cfg=None, seed=0, device="cpu") -> dict`. Columns: `n_faces_gen`, `watertight_gen`, then per reference suffix `_vs_<name>`: `chamfer_m`, `hausdorff_p95_m`, `hausdorff_max_m`, `symdiff_m3`, `rel_symdiff`, `vol_ok`, `vol_note`, `sinkhorn_m`, `roof_chamfer_m`. A metric's columns appear only when it is selected.
- Produces: `METRIC_COLUMNS = ("chamfer_m", "hausdorff_p95_m", "hausdorff_max_m", "symdiff_m3", "rel_symdiff", "sinkhorn_m", "roof_chamfer_m")`.

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_posthoc_eval.py`:

```python
# ---------------------------------------------------------------- Task 5

import trimesh

from src.eval.posthoc_eval import METRICS, score_object
from src.utils.config import SinkhornConfig

SINK = SinkhornConfig(n_points=256, blur_m=0.05, max_iters=2000, tol=1e-9)


def _box(dz=0.0):
    b = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    return (np.asarray(b.vertices, dtype=float) + np.array([0.0, 0.0, dz]),
            np.asarray(b.faces, dtype=np.int64))


def _score(gen, ref, metrics=METRICS):
    return score_object(gen, {"gt": ref}, metrics, n_points=1024, sinkhorn_cfg=SINK)


def test_a_box_against_itself_scores_zero_everywhere():
    row = _score(_box(), _box())
    for key in ("chamfer_m", "hausdorff_p95_m", "hausdorff_max_m", "sinkhorn_m",
                "roof_chamfer_m"):
        assert row[f"{key}_vs_gt"] < 1e-9, key
    assert row["vol_ok_vs_gt"] is True
    assert row["rel_symdiff_vs_gt"] < 1e-9
    assert row["watertight_gen"] == 1.0 and row["n_faces_gen"] == 12


def test_a_lifted_box_scores_the_known_geometry():
    """2 m cube lifted 0.5 m: overlap 2x2x1.5 = 6 m^3, so the symmetric
    difference is 8 + 8 - 2*6 = 4 m^3 and the union 10 m^3. The roofs are
    parallel 2x2 squares 0.5 m apart, so their chamfer is exactly 0.5 m."""
    row = _score(_box(0.5), _box())
    assert row["symdiff_m3_vs_gt"] == pytest.approx(4.0, abs=1e-9)
    assert row["rel_symdiff_vs_gt"] == pytest.approx(0.4, abs=1e-9)
    assert row["roof_chamfer_m_vs_gt"] == pytest.approx(0.5, abs=1e-6)
    assert row["sinkhorn_m_vs_gt"] == pytest.approx(0.5, abs=0.02)
    assert row["chamfer_m_vs_gt"] > 0
    assert row["hausdorff_max_m_vs_gt"] >= row["hausdorff_p95_m_vs_gt"]


def test_an_open_mesh_has_no_volume_but_still_has_distances():
    v, f = _box()
    row = _score((v, f[:-2]), _box())
    assert row["vol_ok_vs_gt"] is False and np.isnan(row["symdiff_m3_vs_gt"])
    assert row["vol_note_vs_gt"]
    assert row["watertight_gen"] == 0.0
    assert np.isfinite(row["chamfer_m_vs_gt"])


def test_an_empty_generation_is_nan_not_an_exception():
    empty = (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    row = _score(empty, _box())
    for key in ("chamfer_m", "hausdorff_max_m", "sinkhorn_m", "roof_chamfer_m",
                "symdiff_m3"):
        assert np.isnan(row[f"{key}_vs_gt"]), key
    assert row["n_faces_gen"] == 0 and row["watertight_gen"] == 0.0


def test_only_requested_metrics_and_references_produce_columns():
    row = score_object(_box(), {"lod1": _box(), "lod2": _box(0.5)}, ["chamfer"],
                       n_points=256)
    metric_cols = sorted(k for k in row if "_vs_" in k)
    assert metric_cols == ["chamfer_m_vs_lod1", "chamfer_m_vs_lod2"]
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_posthoc_eval.py -v`
Expected: collection error, `ImportError: cannot import name 'score_object'`.

- [ ] **Step 3: Implement**

In `src/eval/posthoc_eval.py`, add to the imports:

```python
import numpy as np

from src.analysis.exact_volume import exact_pair_volumes
from src.eval.mesh_metrics import (chamfer_distance, hausdorff_max, hausdorff_p95,
                                   is_watertight_mesh, roof_submesh,
                                   surface_distances)
from src.eval.sinkhorn import mesh_sinkhorn
```

Change the config import line to `from src.utils.config import PosthocEvalConfig, SinkhornConfig`. After `SPLITS`, add:

```python
# Per-reference numeric columns that summaries average. Order is CSV order.
METRIC_COLUMNS = ("chamfer_m", "hausdorff_p95_m", "hausdorff_max_m", "symdiff_m3",
                  "rel_symdiff", "sinkhorn_m", "roof_chamfer_m")
```

At the end of the file, add:

```python
def score_object(gen, refs, metrics, n_points=4096, sinkhorn_cfg=None, seed=0,
                 device="cpu"):
    """Every selected metric of one generated mesh against every reference.

    Args:
        gen: ``(verts [V, 3] metres, faces [F, 3])``.
        refs: reference name -> mesh; columns get the suffix ``_vs_<name>``.
        metrics: subset of `METRICS`.
        n_points: surface samples for chamfer, Hausdorff and roof chamfer.
        sinkhorn_cfg: `SinkhornConfig`; None uses its defaults.
        seed: surface-sampling seed shared by every sampled metric.
        device: torch device for the Sinkhorn solve.

    Returns:
        dict: column -> value. Undefined metrics are ``nan``, never an
        exception: a free-running generation is often empty or open.
    """
    sink = sinkhorn_cfg or SinkhornConfig()
    row = {"n_faces_gen": int(len(np.asarray(gen[1]).reshape(-1, 3))),
           "watertight_gen": float(is_watertight_mesh(gen[1]))}
    nan = float("nan")

    for name, ref in refs.items():
        sfx = f"_vs_{name}"
        if "chamfer" in metrics or "hausdorff" in metrics:
            d_ab, d_ba = surface_distances(gen, ref, n=n_points, seed=seed)
            if "chamfer" in metrics:
                row["chamfer_m" + sfx] = chamfer_distance(d_ab, d_ba)
            if "hausdorff" in metrics:
                row["hausdorff_p95_m" + sfx] = hausdorff_p95(d_ab, d_ba)
                row["hausdorff_max_m" + sfx] = hausdorff_max(d_ab, d_ba)
        if "vol_symdiff" in metrics:
            # Exact CSG; `ok` is False unless both are closed, outward solids.
            vol = exact_pair_volumes(gen, ref)
            ok = bool(vol["ok"])
            row["symdiff_m3" + sfx] = float(vol["symdiff_m3"]) if ok else nan
            row["rel_symdiff" + sfx] = float(vol["rel_symdiff"]) if ok else nan
            row["vol_ok" + sfx] = ok
            row["vol_note" + sfx] = vol["note"]
        if "sinkhorn" in metrics:
            row["sinkhorn_m" + sfx] = mesh_sinkhorn(
                gen, ref, n_points=sink.n_points, blur=sink.blur_m,
                max_iters=sink.max_iters, tol=sink.tol, seed=seed, device=device)
        if "roof_chamfer" in metrics:
            roof_gen, roof_ref = roof_submesh(*gen), roof_submesh(*ref)
            if len(roof_gen[1]) and len(roof_ref[1]):
                row["roof_chamfer_m" + sfx] = chamfer_distance(
                    *surface_distances(roof_gen, roof_ref, n=n_points, seed=seed))
            else:
                row["roof_chamfer_m" + sfx] = nan
    return row
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_posthoc_eval.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

Run `detect_changes({scope: "all"})`. Then:

```bash
git add src/eval/posthoc_eval.py tests/test_posthoc_eval.py
git commit -m "feat(eval): per-object scoring against LOD1 and LOD2 references"
```

---

### Task 6: Real-data sanity and baseline check

**Files:**
- Create: `tests/test_metric_sanity_real.py`

**Interfaces:**
- Consumes: `MeshDataset(dataset_dir, lod_in, lod_out, num_bins, margin_lo, margin_hi, max_faces, max_files, tokenization)`, `detokenize(tokens, num_bins)`, `score_object`, `METRICS`, `SinkhornConfig`.

This test reads **one tile per LOD** (`max_files=1`). In `data/The Hague/cleaner` the first tile of both `LOD1` and `LOD2` is `Source B/10-232-568.city.json` (3.6 MB and 4.6 MB), so the pairs intersect. It never calls `MeshDataModule.setup()`.

- [ ] **Step 1: Write the test**

Create `tests/test_metric_sanity_real.py`:

```python
"""Sanity and baseline check of every post-hoc metric on real buildings.

Two readings from the same scores:
  * sanity -- each mesh against itself must score ~0 on every metric; anything
    else is a metric bug, not a model finding;
  * baseline -- LOD1 against its own LOD2 is the do-nothing prediction. Its
    numbers are printed (pytest -s) as the floor a generation must beat.

Reads ONE tile per LOD (max_files=1), never the datamodule. Meshes pass through
the same tokenizer round trip the eval scores against. Skipped when the corpus
is not on disk.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dataset.mesh_dataset import MeshDataset, detokenize
from src.eval.posthoc_eval import METRICS, score_object
from src.utils.config import SinkhornConfig

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "The Hague" / "cleaner"
NUM_BINS = 128
N_OBJECTS = 8
N_POINTS = 4096
# Smaller than the eval default so the test stays under a minute on CPU. The
# self-score assertions do not depend on it: identical inputs give exactly 0.
SINK = SinkhornConfig(n_points=512, blur_m=0.1, max_iters=200, tol=1e-6)

pytestmark = pytest.mark.skipif(
    not ((DATA / "LOD1").is_dir() and (DATA / "LOD2").is_dir()),
    reason=f"real corpus not present under {DATA}")

DISTANCES = ("chamfer_m", "hausdorff_p95_m", "hausdorff_max_m")


@pytest.fixture(scope="module")
def pairs():
    # The mesh-v5 runs' data settings: LOD1 in, LOD2 out, 128 bins, +z margin 0.1.
    ds = MeshDataset(DATA, lod_in="LOD1", lod_out="LOD2", num_bins=NUM_BINS,
                     margin_lo=[0.0, 0.0, 0.0], margin_hi=[0.0, 0.0, 0.1],
                     max_faces=200, max_files=1, tokenization="coord")
    assert len(ds) >= N_OBJECTS, f"only {len(ds)} pairs in the first tile"
    out = []
    for i in range(N_OBJECTS):
        item = ds[i]
        centre, scale = item["center"].numpy(), item["scale"].numpy()
        v1, f1 = detokenize(item["cond"].numpy(), NUM_BINS)
        v2, f2 = detokenize(item["tgt"].numpy(), NUM_BINS)
        out.append((ds.ids[i], (v1 * scale + centre, f1), (v2 * scale + centre, f2)))
    return out


@pytest.fixture(scope="module")
def scores(pairs):
    def s(a, b):
        return score_object(a, {"x": b}, METRICS, n_points=N_POINTS, sinkhorn_cfg=SINK)
    return [{"id": oid, "lod1_self": s(l1, l1), "lod2_self": s(l2, l2),
             "lod1_lod2": s(l1, l2), "lod2_lod1": s(l2, l1)}
            for oid, l1, l2 in pairs]


@pytest.mark.parametrize("which", ["lod1_self", "lod2_self"])
def test_every_metric_scores_a_mesh_against_itself_as_zero(scores, which):
    for r in scores:
        row = r[which]
        for key in DISTANCES:
            value = row[f"{key}_vs_x"]
            # Not exactly 0: the proximity query runs on world coordinates.
            assert np.isfinite(value) and value < 1e-6, (r["id"], key, value)
        roof = row["roof_chamfer_m_vs_x"]
        assert np.isnan(roof) or roof < 1e-6, (r["id"], roof)
        assert row["sinkhorn_m_vs_x"] < 0.1 * SINK.blur_m, r["id"]
        if row["vol_ok_vs_x"]:
            assert row["rel_symdiff_vs_x"] < 1e-9, r["id"]


def test_volume_checks_are_not_vacuous(scores):
    assert any(r["lod2_self"]["vol_ok_vs_x"] for r in scores)


def test_lod1_against_lod2_is_a_well_formed_baseline(scores):
    keys = DISTANCES + ("roof_chamfer_m", "sinkhorn_m", "symdiff_m3", "rel_symdiff")
    for r in scores:
        pair, own = r["lod1_lod2"], r["lod1_self"]
        assert np.isfinite(pair["chamfer_m_vs_x"]) and np.isfinite(pair["sinkhorn_m_vs_x"])
        for key in keys:
            value = pair[f"{key}_vs_x"]
            if not np.isfinite(value):
                continue
            # Zero is allowed: 31.8% of cleaner pairs are effectively identical.
            assert value >= 0.0, (r["id"], key)
            base = own.get(f"{key}_vs_x", np.nan)
            if np.isfinite(base):
                assert value >= base - 1e-6, (r["id"], key)
        assert pair["hausdorff_max_m_vs_x"] >= pair["hausdorff_p95_m_vs_x"]
        assert pair["chamfer_m_vs_x"] <= pair["hausdorff_max_m_vs_x"]


def test_swapping_the_arguments(scores):
    """Exact CSG and same-seed Sinkhorn agree tightly after a swap. The chamfer
    family only agrees statistically: `surface_distances` samples the two sides
    with seeds `seed` and `seed + 1`, so a swap resamples both clouds.
    `hausdorff_max_m` is not compared -- a max over samples is not stable."""
    for r in scores:
        a, b = r["lod1_lod2"], r["lod2_lod1"]
        if a["vol_ok_vs_x"] and b["vol_ok_vs_x"]:
            assert a["symdiff_m3_vs_x"] == pytest.approx(b["symdiff_m3_vs_x"],
                                                          rel=1e-6, abs=1e-6)
        assert a["sinkhorn_m_vs_x"] == pytest.approx(b["sinkhorn_m_vs_x"],
                                                      rel=0.02, abs=0.01)
        assert a["chamfer_m_vs_x"] == pytest.approx(b["chamfer_m_vs_x"],
                                                     rel=0.05, abs=0.01)
        assert a["hausdorff_p95_m_vs_x"] == pytest.approx(b["hausdorff_p95_m_vs_x"],
                                                           rel=0.10, abs=0.02)


def test_print_the_do_nothing_baseline(scores):
    table = pd.DataFrame([{"id": r["id"], **{k.removesuffix("_vs_x"): v
                                            for k, v in r["lod1_lod2"].items()}}
                          for r in scores]).set_index("id")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\nLOD1 vs LOD2 (do-nothing baseline), per object:")
        print(table.to_string())
        print("\nmeans:")
        print(table.select_dtypes("number").mean().to_string())
    assert len(table) == N_OBJECTS
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_metric_sanity_real.py -v -s`
Expected: all PASS and a printed baseline table. Tasks 2, 3 and 5 are already implemented, so this passes straight away; it guards the metrics against regressions. If a self-score assertion fails, stop and report: it means a metric is wrong on real coordinates, and the threshold must not be loosened.

- [ ] **Step 3: Commit**

Run `detect_changes({scope: "all"})`. Then:

```bash
git add tests/test_metric_sanity_real.py
git commit -m "test(eval): real-data metric sanity and LOD1-vs-LOD2 baseline"
```

---

### Task 7: `run_posthoc_eval` — generation loop, CSVs and artifacts

**Files:**
- Modify: `src/eval/posthoc_eval.py`
- Test: `tests/test_posthoc_eval.py` (append)

**Interfaces:**
- Consumes: `eval_indices`, `_resolve`, `_to_metres` from `src.eval.mesh_eval`; `mesh_collate_fn`, `specials`, `detokenize`, `amt_detokenize` from `src.dataset.mesh_dataset`; `MeshTransformerModule._prepare`, `._item_cond`, `.generate`, `.decode_tokens`, `.tokenization`, `.num_bins`, `.eos`.
- Produces:
  - `object_seed(seed: int, object_id: str) -> int`
  - `generation_budget(ref_cfg: Config) -> int`
  - `checkpoint_info(path) -> dict` with keys `path`, `sha256`, `epoch`, `global_step`, `hyper_parameters`
  - `run_posthoc_eval(cfg, ref_cfg, model, dataset, out_dir, checkpoint=None, split_sizes=None, device="cpu") -> list[dict]` (summary rows, one per run)

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_posthoc_eval.py`:

```python
# ---------------------------------------------------------------- Task 7

import json

import pandas as pd
import torch

from src.dataset.mesh_dataset import specials
from src.eval.posthoc_eval import (checkpoint_info, object_seed, run_posthoc_eval)
from src.models.mesh_transformer import MeshTransformerModule
from src.utils.config import Config

NUM_BINS = 32
BOS, EOS, PAD = specials(NUM_BINS)


class _Dataset:
    """Two random-token (cond, tgt) pairs. Same stub as tests/test_mesh_eval.py:
    the meshes are noise, which is the point -- scoring must survive noise."""

    def __init__(self):
        self.ids = ["a", "b"]
        rng = np.random.default_rng(0)
        self.items = []
        for k in range(2):
            n_faces = 4 + k
            self.items.append({
                "cond": torch.from_numpy(rng.integers(0, NUM_BINS, 9 * n_faces)),
                "tgt": torch.cat([torch.tensor([BOS]),
                                  torch.from_numpy(rng.integers(0, NUM_BINS, 9 * n_faces)),
                                  torch.tensor([EOS])]),
                "id": self.ids[k],
                "center": torch.zeros(3),
                "scale": torch.ones(3) * 10.0,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _model():
    torch.manual_seed(1)
    return MeshTransformerModule(num_bins=NUM_BINS, d_model=16, n_head=2,
                                 num_layers=1, dropout=0.0, max_seq_len=128)


def _small_cfg(**kw):
    kw.setdefault("batch_size", 2)
    kw.setdefault("n_points", 256)
    kw.setdefault("sinkhorn", SinkhornConfig(n_points=128, max_iters=50))
    return _cfg(**kw)


OBJECT_COLUMNS = [
    "object_id", "seed", "n_faces_lod1", "n_faces_lod2", "eos_emitted", "decode_ok",
    "gen_seconds", "n_faces_gen", "watertight_gen",
] + [f"{m}_vs_{ref}" for ref in ("lod1", "lod2")
     for m in ("chamfer_m", "hausdorff_p95_m", "hausdorff_max_m", "symdiff_m3",
               "rel_symdiff", "vol_ok", "vol_note", "sinkhorn_m", "roof_chamfer_m")]


def test_object_seed_is_stable_and_distinct():
    assert object_seed(1234, "a") == object_seed(1234, "a")
    assert object_seed(1234, "a") != object_seed(1234, "b")
    assert object_seed(1234, "a") != object_seed(7, "a")
    assert object_seed(1234, "a") == 2078608377   # zlib.crc32(b"1234:a"), unsalted


def test_checkpoint_info_hashes_and_reads_metadata(tmp_path):
    path = tmp_path / "x.ckpt"
    torch.save({"epoch": 3, "global_step": 99,
                "hyper_parameters": {"tokenization": "coord"}}, path)
    info = checkpoint_info(path)
    assert info["epoch"] == 3 and info["global_step"] == 99
    assert info["hyper_parameters"] == {"tokenization": "coord"}
    assert len(info["sha256"]) == 64


def test_end_to_end_writes_one_row_per_object_per_run(tmp_path):
    ckpt = tmp_path / "x.ckpt"
    torch.save({"epoch": 0, "global_step": 0, "hyper_parameters": {}}, ckpt)
    cfg = _small_cfg()
    out = tmp_path / "eval"
    summary = run_posthoc_eval(cfg, Config(), _model(), _Dataset(), out,
                               checkpoint=checkpoint_info(ckpt),
                               split_sizes={"train": 0, "val": 2, "test": 0})

    names = [r.name for r in expand_strategies(cfg.strategies)]
    assert [row["run"] for row in summary] == names
    for name in names:
        df = pd.read_csv(out / name / "objects.csv")
        assert df["object_id"].tolist() == ["a", "b"]
        assert list(df.columns) == OBJECT_COLUMNS
        run = json.loads((out / name / "run.json").read_text())
        assert run["name"] == name and run["batch_size"] == 2

    for artifact in ("eval_config.yaml", "reference_config.yaml", "environment.json",
                     "checkpoint.json", "split.json", "summary.csv"):
        assert (out / artifact).is_file(), artifact
    table = pd.read_csv(out / "summary.csv")
    assert len(table) == 7
    assert {"watertight_pct", "decode_rate", "eos_rate", "chamfer_m_vs_lod2_mean",
            "chamfer_m_vs_lod2_median", "vol_defined_pct_vs_lod1",
            "roof_defined_pct_vs_lod2"} <= set(table.columns)
    split = json.loads((out / "split.json").read_text())
    assert split["n_evaluated"] == 2 and split["split_sizes"]["val"] == 2


def test_a_rerun_reproduces_every_row(tmp_path):
    cfg = _small_cfg(strategies=[
        StrategyConfig(kind="top_p", top_p=[0.95], temperature=[1.2]),
        StrategyConfig(kind="beam", beam_size=[3])])
    for sub in ("a", "b"):
        run_posthoc_eval(cfg, Config(), _model(), _Dataset(), tmp_path / sub)
    for name in ("top_p_p0.95_t1.2", "beam_k3_lp1.0"):
        a = pd.read_csv(tmp_path / "a" / name / "objects.csv").drop(columns="gen_seconds")
        b = pd.read_csv(tmp_path / "b" / name / "objects.csv").drop(columns="gen_seconds")
        pd.testing.assert_frame_equal(a, b)


def test_max_objects_takes_a_seeded_subsample(tmp_path):
    cfg = _small_cfg(max_objects=1, strategies=[StrategyConfig(kind="greedy")],
                     metrics=["chamfer"])
    run_posthoc_eval(cfg, Config(), _model(), _Dataset(), tmp_path / "e")
    assert len(pd.read_csv(tmp_path / "e" / "greedy" / "objects.csv")) == 1


def test_an_empty_selection_is_rejected(tmp_path):
    class _Empty(_Dataset):
        def __len__(self):
            return 0
    with pytest.raises(ValueError, match="no objects"):
        run_posthoc_eval(_small_cfg(), Config(), _model(), _Empty(), tmp_path / "e")
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_posthoc_eval.py -v`
Expected: collection error, `ImportError: cannot import name 'checkpoint_info'`.

- [ ] **Step 3: Confirm the pinned seed value**

Run: `python -c "import zlib; print(zlib.crc32(b'1234:a'))"`
Expected: `2078608377`, the literal in `test_object_seed_is_stable_and_distinct`. The test pins the seed rule so recorded seeds can never silently change meaning.

- [ ] **Step 4: Implement**

In `src/eval/posthoc_eval.py`, extend the imports so the top of the file reads:

```python
import csv
import hashlib
import itertools
import json
import logging
import platform
import subprocess
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.analysis.exact_volume import exact_pair_volumes
from src.dataset.mesh_dataset import (amt_detokenize, detokenize, mesh_collate_fn,
                                      specials)
from src.eval.mesh_eval import _resolve, _to_metres, eval_indices
from src.eval.mesh_metrics import (chamfer_distance, hausdorff_max, hausdorff_p95,
                                   is_watertight_mesh, roof_submesh,
                                   surface_distances)
from src.eval.sinkhorn import mesh_sinkhorn
from src.utils.config import PosthocEvalConfig, SinkhornConfig

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
# Surface-sampling seed for every metric, as in `run_mesh_eval`. Recorded in run.json.
SURFACE_SEED = 0
```

At the end of the file, add:

```python
def object_seed(seed, object_id):
    """Sampling seed for one object: ``crc32("<seed>:<object_id>")``.

    The same for every strategy, so strategies are compared on common random
    numbers, and independent of batch position. crc32 rather than `hash()`,
    which Python salts per process.
    """
    return zlib.crc32(f"{seed}:{object_id}".encode())


def generation_budget(ref_cfg):
    """Tokens to allow per object: the `MeshEvalCallback` rule, 9 per face + EOS.

    `generate` clamps this to what the position scheme allows.
    """
    faces = ref_cfg.mesh_model.n_max_triangles or ref_cfg.mesh_data.max_faces or 200
    return 9 * int(faces) + 1


def checkpoint_info(path):
    """Path, sha256 and Lightning metadata of a checkpoint, for checkpoint.json."""
    path = Path(path)
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(),
            "epoch": ckpt.get("epoch"), "global_step": ckpt.get("global_step"),
            "hyper_parameters": dict(ckpt.get("hyper_parameters") or {})}


def _git_state():
    """``(info, diff)``: commit, dirty flag and status lines, plus ``git diff HEAD``."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                               text=True, check=True).stdout
    try:
        status = git("status", "--porcelain")
        info = {"commit": git("rev-parse", "HEAD").strip(),
                "dirty": bool(status.strip()), "status": status.splitlines()}
        return info, git("diff", "HEAD")
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning("git state unavailable: %s", exc)
        return {"commit": None, "error": str(exc)}, ""


def _versions():
    out = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
           "cuda_available": torch.cuda.is_available()}
    for pkg in ("numpy", "pandas", "trimesh", "manifold3d", "lightning", "omegaconf"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _write_artifacts(out_dir, cfg, ref_cfg, ids, n_split, checkpoint, split_sizes, device):
    """Everything besides the per-run CSVs that a reproduction needs."""
    (out_dir / "eval_config.yaml").write_text(
        OmegaConf.to_yaml(OmegaConf.structured(cfg)), encoding="utf-8")
    (out_dir / "reference_config.yaml").write_text(
        OmegaConf.to_yaml(OmegaConf.structured(ref_cfg)), encoding="utf-8")

    git, diff = _git_state()
    _write_json(out_dir / "environment.json", {
        "git": git, "versions": _versions(), "device": str(device),
        "hostname": platform.node(), "platform": platform.platform(), "argv": sys.argv,
        "note": "Untracked files are listed under git.status but are not "
                "captured in git_diff.patch."})
    if diff:
        (out_dir / "git_diff.patch").write_text(diff, encoding="utf-8")
    if checkpoint is not None:
        _write_json(out_dir / "checkpoint.json", checkpoint)
    _write_json(out_dir / "split.json", {
        "split": cfg.split, "dataset_dir": ref_cfg.mesh_data.dataset_dir,
        "split_sizes": split_sizes, "n_split": n_split,
        "max_objects": cfg.max_objects, "n_evaluated": len(ids),
        "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()})


def _append_csv(path, rows):
    """Append rows, writing the header only when the file is new."""
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            writer.writeheader()
        writer.writerows(rows)


def _summarize(run, rows, cfg):
    """One summary.csv row: rates, nanmean/nanmedian of each metric, defined %."""
    df = pd.DataFrame(rows)
    out = {"run": run.name, "kind": run.kind,
           **{f"param_{k}": v for k, v in run.params.items()},
           "n_objects": len(df),
           "watertight_pct": 100.0 * df["watertight_gen"].mean(),
           "decode_rate": df["decode_ok"].mean(), "eos_rate": df["eos_emitted"].mean()}
    for ref in cfg.references:
        for col in METRIC_COLUMNS:
            name = f"{col}_vs_{ref}"
            if name in df:
                out[f"{name}_mean"] = df[name].mean()
                out[f"{name}_median"] = df[name].median()
        if "vol_symdiff" in cfg.metrics:
            out[f"vol_defined_pct_vs_{ref}"] = 100.0 * df[f"vol_ok_vs_{ref}"].mean()
        if "roof_chamfer" in cfg.metrics:
            out[f"roof_defined_pct_vs_{ref}"] = (
                100.0 * np.isfinite(df[f"roof_chamfer_m_vs_{ref}"].astype(float)).mean())
    return out


def _score_batch(model, batch, out, cond, tgt, cond_pad, cfg, seeds, seconds, device):
    """Decode each generated sequence and score it against its references."""
    inverse = amt_detokenize if model.tokenization == "amt" else detokenize
    rows = []
    for k, oid in enumerate(batch["ids"]):
        centre = batch["center"][k].cpu().numpy()
        scale = batch["scale"][k].cpu().numpy()
        item_cond = model._item_cond((cond, cond_pad), k)
        gen = _to_metres(model, out[k].cpu(), centre, scale, item_cond)
        # Both references go through the tokenizer round trip, as in run_mesh_eval.
        lod2 = _to_metres(model, tgt[k].cpu(), centre, scale, item_cond)
        v1, f1 = inverse(batch["cond"][k].cpu().numpy(), model.num_bins)
        lod1 = (v1 * scale + centre, f1)
        refs = {"lod1": lod1, "lod2": lod2}

        row = {"object_id": str(oid), "seed": seeds[k],
               "n_faces_lod1": len(lod1[1]), "n_faces_lod2": len(lod2[1]),
               "eos_emitted": bool((out[k] == model.eos).any()),
               "decode_ok": bool(len(gen[1]) > 0), "gen_seconds": seconds}
        row.update(score_object(gen, {name: refs[name] for name in cfg.references},
                                cfg.metrics, n_points=cfg.n_points,
                                sinkhorn_cfg=cfg.sinkhorn, seed=SURFACE_SEED,
                                device=device))
        rows.append(row)
    return rows


def run_posthoc_eval(cfg, ref_cfg, model, dataset, out_dir, checkpoint=None,
                     split_sizes=None, device="cpu"):
    """Generate and score every selected object under every expanded strategy.

    Args:
        cfg: `PosthocEvalConfig`.
        ref_cfg: the resolved training `Config` the checkpoint belongs to.
        model: a `MeshTransformerModule`, already on ``device``.
        dataset: `MeshDataset`, a `Subset` of one, or anything with the same
            item contract.
        out_dir: this evaluation's directory; created if missing.
        checkpoint: `checkpoint_info` dict, written to checkpoint.json.
        split_sizes: split name -> size, recorded in split.json.
        device: torch device for generation and the Sinkhorn solve.

    Returns:
        list[dict]: one summary row per run, as written to summary.csv.
    """
    runs = validate_posthoc_config(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = (np.arange(len(dataset)) if cfg.max_objects is None
               else eval_indices(len(dataset), cfg.max_objects, cfg.seed))
    if len(indices) == 0:
        raise ValueError(f"The {cfg.split} split selected no objects to evaluate.")
    ids = []
    for i in indices:
        full, orig = _resolve(dataset, int(i))
        ids.append(str(full.ids[orig]))
    _write_artifacts(out_dir, cfg, ref_cfg, ids, len(dataset), checkpoint,
                     split_sizes, device)

    max_new_tokens = generation_budget(ref_cfg)
    pad = specials(model.num_bins)[2]
    model.eval()
    summary = []
    for run in runs:
        run_dir = out_dir / run.name
        run_dir.mkdir(exist_ok=True)
        kwargs = run.generate_kwargs()
        _write_json(run_dir / "run.json", {
            **asdict(run), "generate_kwargs": kwargs, "max_new_tokens": max_new_tokens,
            "batch_size": cfg.batch_size, "seed": cfg.seed,
            "seed_rule": "zlib.crc32(f'{seed}:{object_id}'), same for every strategy",
            "surface_seed": SURFACE_SEED})
        logger.info("[posthoc] %s: %d objects", run.name, len(indices))

        rows = []
        for start in range(0, len(indices), cfg.batch_size):
            chunk = indices[start:start + cfg.batch_size]
            batch = mesh_collate_fn([dataset[int(i)] for i in chunk], pad=pad)
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            seeds = [object_seed(cfg.seed, oid) for oid in batch["ids"]]

            started = time.perf_counter()
            with torch.no_grad():
                cond, tgt, cond_pad, _ = model._prepare(batch)
                out = model.generate(cond, cond_pad, max_new_tokens=max_new_tokens,
                                     seeds=seeds, **kwargs)
            seconds = (time.perf_counter() - started) / len(chunk)

            batch_rows = _score_batch(model, batch, out, cond, tgt, cond_pad, cfg,
                                      seeds, seconds, device)
            # Flushed per batch: a crash keeps every finished object.
            _append_csv(run_dir / "objects.csv", batch_rows)
            rows.extend(batch_rows)

        summary.append(_summarize(run, rows, cfg))
        pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)
    return summary
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_posthoc_eval.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

Run `detect_changes({scope: "all"})`. Then:

```bash
git add src/eval/posthoc_eval.py tests/test_posthoc_eval.py
git commit -m "feat(eval): post-hoc run loop with per-object CSVs and reproducibility artifacts"
```

---

### Task 8: CLI — load the reference run, checkpoint and split

**Files:**
- Modify: `src/eval/posthoc_eval.py`
- Test: `tests/test_posthoc_eval.py` (append)

**Interfaces:**
- Consumes: `load_config(config_path: Path, overrides: list[str]) -> Config` from `src.utils.initialization`; `MeshDataModule`; `MeshTransformerModule.load_from_checkpoint`; `_load_vqvae(cfg) -> MeshVQVAE | None` from `src.train_mesh`; `checkpoint_info`, `run_posthoc_eval`.
- Produces:
  - `check_tokenization(hparams: dict, ref_cfg: Config) -> None` (raises `ValueError`)
  - `resolve_device(name: str) -> str`
  - `build_split(ref_cfg, split) -> tuple[Dataset, dict[str, int]]`
  - `load_model(ref_cfg, checkpoint: Path, device: str) -> MeshTransformerModule`
  - `main(argv: list[str] | None = None) -> Path` (the evaluation directory)

Order inside `main` is the fail-fast contract. Validate the eval config, load the reference config, check the checkpoint exists, hash it and check tokenization, and only then load data and weights.

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_posthoc_eval.py`:

```python
# ---------------------------------------------------------------- Task 8

import src.eval.posthoc_eval as posthoc
from src.eval.posthoc_eval import check_tokenization, main, resolve_device


def test_a_tokenization_mismatch_is_refused():
    ref = Config()
    ref.mesh_data.tokenization = "coord"
    check_tokenization({"tokenization": "coord"}, ref)
    # Checkpoints from before AMT existed carry no key and were coordinate runs.
    check_tokenization({}, ref)
    with pytest.raises(ValueError, match="tokenization"):
        check_tokenization({"tokenization": "amt"}, ref)


def test_resolve_device():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cpu", "cuda")


def _write(tmp_path, text):
    path = tmp_path / "eval.yaml"
    path.write_text(text)
    return path


def test_main_rejects_a_bad_metric_before_reading_anything(tmp_path):
    path = _write(tmp_path, "reference_config: does/not/exist.yaml\n"
                            "checkpoint: nope.ckpt\nmetrics: [emd]\n")
    with pytest.raises(ValueError, match="metric"):
        main(["--config", str(path)])


def test_main_rejects_a_non_mesh_reference(tmp_path):
    ref = tmp_path / "ref.yaml"
    ref.write_text("config_set: diffusion\ndata:\n  dataset_dir: x\n")
    path = _write(tmp_path, f"reference_config: {ref.as_posix()}\ncheckpoint: nope.ckpt\n")
    with pytest.raises(ValueError, match="config_set"):
        main(["--config", str(path)])


def test_main_checks_the_checkpoint_before_loading_data(tmp_path, monkeypatch):
    ref = (ROOT / "configs" / "mesh-v5-full-coord.yaml").as_posix()
    missing = (tmp_path / "nope.ckpt").as_posix()
    path = _write(tmp_path, f"reference_config: {ref}\ncheckpoint: {missing}\n")
    monkeypatch.setattr(posthoc, "build_split", lambda *a, **k: pytest.fail(
        "the dataset was loaded before the checkpoint was checked"))
    with pytest.raises(FileNotFoundError, match="nope.ckpt"):
        main(["--config", str(path)])


def test_main_checks_tokenization_before_loading_data(tmp_path, monkeypatch):
    ckpt = tmp_path / "amt.ckpt"
    torch.save({"epoch": 0, "global_step": 0,
                "hyper_parameters": {"tokenization": "amt"}}, ckpt)
    ref = (ROOT / "configs" / "mesh-v5-full-coord.yaml").as_posix()   # coord run
    path = _write(tmp_path, f"reference_config: {ref}\ncheckpoint: {ckpt.as_posix()}\n")
    monkeypatch.setattr(posthoc, "build_split", lambda *a, **k: pytest.fail(
        "the dataset was loaded before tokenization was checked"))
    with pytest.raises(ValueError, match="tokenization"):
        main(["--config", str(path)])
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_posthoc_eval.py -v`
Expected: collection error, `ImportError: cannot import name 'check_tokenization'`.

- [ ] **Step 3: Implement**

In `src/eval/posthoc_eval.py`, add to the imports:

```python
import argparse
from datetime import datetime

import lightning as L

from src.utils.initialization import load_config
```

At the end of the file, add:

```python
def check_tokenization(hparams, ref_cfg):
    """Refuse a checkpoint whose token spelling differs from the reference config.

    Nothing downstream reconciles the two, and a mismatch does not raise by
    itself: a coordinate sequence decoded as AMT keeps its vertices and silently
    re-triangulates them into ~1.8x the faces (see notebooks/test-mesh-normal.ipynb).
    A checkpoint without the key predates AMT, so it is a coordinate run.
    """
    saved = hparams.get("tokenization", "coord")
    wanted = ref_cfg.mesh_data.tokenization
    if saved != wanted:
        raise ValueError(
            f"Checkpoint was trained with tokenization={saved!r} but the reference "
            f"config sets mesh_data.tokenization={wanted!r}. Point reference_config "
            "at the config this checkpoint was trained with.")


def resolve_device(name):
    """``"auto"`` -> cuda when available, else cpu; anything else unchanged."""
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def build_split(ref_cfg, split):
    """The requested split, rebuilt exactly as `train_mesh` built it.

    Same arguments and seed as training, so `random_split` reproduces the
    partition the checkpoint was trained against.

    Returns:
        tuple: (dataset, {split name: size}).
    """
    from src.dataset.mesh_datamodule import MeshDataModule

    m = ref_cfg.mesh_data
    if not m.dataset_dir:
        raise ValueError("reference config has no mesh_data.dataset_dir.")
    datamodule = MeshDataModule(
        dataset_dir=m.dataset_dir, lod_in=m.lod_in, lod_out=m.lod_out,
        num_bins=m.num_bins, margin_lo=list(m.margin_lo), margin_hi=list(m.margin_hi),
        max_faces=m.max_faces, max_files=m.max_files, tokenization=m.tokenization,
        batch_size=ref_cfg.training.batch_size,
        train_val_test_split=tuple(ref_cfg.training.train_val_test_split),
        num_workers=0, seed=ref_cfg.seed)
    datamodule.setup()
    sizes = {name: len(getattr(datamodule, f"{name}_dataset")) for name in SPLITS}
    return getattr(datamodule, f"{split}_dataset"), sizes


def load_model(ref_cfg, checkpoint, device):
    """The checkpoint's module, with its frozen VQ-VAE when the run used one."""
    from src.models.mesh_transformer import MeshTransformerModule
    from src.train_mesh import _load_vqvae

    vqvae = _load_vqvae(ref_cfg)
    extra = {} if vqvae is None else {"vqvae": vqvae}
    model = MeshTransformerModule.load_from_checkpoint(str(checkpoint),
                                                       map_location="cpu", **extra)
    return model.to(device).eval()


def main(argv=None):
    """CLI entry point. Returns the evaluation directory."""
    parser = argparse.ArgumentParser(
        description="Score a mesh-transformer checkpoint under several decoding strategies.")
    parser.add_argument("--config", "-c", type=Path, required=True,
                        help="Post-hoc eval YAML, e.g. configs/posthoc-eval.yaml")
    parser.add_argument("overrides", nargs="*",
                        help="Dot-list overrides, e.g. max_objects=256")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Cheapest checks first: nothing below touches data until all of these pass.
    cfg = load_posthoc_config(args.config, args.overrides)
    validate_posthoc_config(cfg)
    ref_cfg = load_config(Path(cfg.reference_config), list(cfg.reference_overrides))
    if ref_cfg.config_set != "mesh":
        raise ValueError(f"reference config has config_set={ref_cfg.config_set!r}; "
                         "post-hoc eval scores mesh-transformer runs (config_set: mesh).")
    ckpt = Path(cfg.checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    info = checkpoint_info(ckpt)
    check_tokenization(info["hyper_parameters"], ref_cfg)

    L.seed_everything(cfg.seed, workers=True)
    device = resolve_device(cfg.device)
    dataset, sizes = build_split(ref_cfg, cfg.split)
    model = load_model(ref_cfg, ckpt, device)

    base = (Path(cfg.output_dir) if cfg.output_dir else
            Path(ref_cfg.logging.save_dir, ref_cfg.logging.experiment_name,
                 ref_cfg.logging.run_name, "posthoc_eval"))
    out_dir = base / ckpt.stem / datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.info("[posthoc] %s, %s split (%d objects) -> %s",
                ckpt.name, cfg.split, len(dataset), out_dir)
    run_posthoc_eval(cfg, ref_cfg, model, dataset, out_dir, checkpoint=info,
                     split_sizes=sizes, device=device)
    logger.info("[posthoc] done: %s", out_dir)
    return out_dir


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_posthoc_eval.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

Run `detect_changes({scope: "all"})`. Then:

```bash
git add src/eval/posthoc_eval.py tests/test_posthoc_eval.py
git commit -m "feat(eval): post-hoc eval CLI with fail-fast checkpoint and tokenizer checks"
```

---

### Task 9: Final verification

**Files:** none changed.

- [ ] **Step 1: Run every test this plan touches or depends on**

Run: `python -m pytest tests/test_sampling_filters.py tests/test_roof_and_hausdorff.py tests/test_sinkhorn.py tests/test_posthoc_eval.py tests/test_metric_sanity_real.py tests/test_beam_search.py tests/test_kv_cache.py tests/test_mesh_amt.py tests/test_mesh_transformer.py tests/test_mesh_eval.py tests/test_mesh_metrics.py -v`
Expected: all PASS. Report the real-data baseline means printed by `test_print_the_do_nothing_baseline`.

- [ ] **Step 2: Check the graph against the branch**

Run `detect_changes({scope: "compare", base_ref: "main"})`. Expected: changes confined to `generate`, the new helpers in `mesh_transformer.py` and `mesh_metrics.py`, the three new config dataclasses and the two new eval modules.

- [ ] **Step 3: Record the decisions**

Leave a short session note covering the non-obvious decisions:
- common random numbers via the crc32 seed;
- Sinkhorn sampling both sides with the same seed;
- the float64 shared-origin requirement;
- winding-based roof extraction;
- why chamfer only agrees statistically under an argument swap.

Do not run the eval script itself unless asked.

## Self-Review

- **Spec coverage:**
  - §1 inference: Task 1.
  - §2 config, expansion, fail-fast, tokenization check, VQ-VAE loading, generation budget, per-object seed: Tasks 4, 7, 8.
  - §3 metrics (chamfer, both Hausdorffs, exact symmetric difference, Sinkhorn, roof chamfer, context columns, summary): Tasks 2, 3, 5, 7.
  - §4 artifacts: Task 7.
  - §5 files: all mapped.
  - §6 synthetic tests: Tasks 1–5, 7, 8.
  - §6 real-data test: Task 6.
- **Deliberate details:**
  - `split_sizes` is only known in the CLI, which builds the datamodule. `run_posthoc_eval` accepts it as an argument.
  - Roof/Hausdorff tests live in their own file, as the spec's file table says.
- **Type consistency:** `DecodeRun.generate_kwargs()` keys match `generate`'s parameters. `score_object`'s column order matches `OBJECT_COLUMNS`. `SinkhornConfig` field names (`n_points`, `blur_m`, `max_iters`, `tol`) are used identically in Tasks 3, 5, 6 and 7.
