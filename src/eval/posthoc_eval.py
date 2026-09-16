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
