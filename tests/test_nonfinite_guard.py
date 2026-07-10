"""A non-finite training loss must skip the step, not poison the weights.

Once a single forward pass produces NaN or inf, backward fills every gradient
with NaN and AdamW writes NaN into every parameter and both moment buffers. The
model never recovers: run gptwstyp logged NaN for the remaining ~75,000 steps
and only stopped because EarlyStopping checks for finite metrics.

Skipping the optimizer step turns a run-killer into a logged anomaly. This is a
safety net, not a fix -- the overflow it catches is addressed by the pooled-std
and coordinate-scale commits.
"""
import logging

import pytest
import torch

from src.models.diffusion import NUM_EDGE_CLASSES, CityJSONDiffusionModule

B, N = 2, 6
DIFFUSION_LOGGER = "src.models.diffusion"


@pytest.fixture
def model():
    return CityJSONDiffusionModule(hidden_dim=8, edge_dim=4, global_dim=4,
                                   n_head=2, num_layers=1, T=10, n_max=N)


@pytest.fixture
def batch():
    return {
        "x": torch.randn(B, N, 3),
        "node_categories": torch.nn.functional.one_hot(torch.randint(0, 2, (B, N)), 2).float(),
        "y": torch.randint(0, NUM_EDGE_CLASSES, (B, N, N, 1)),
        "node_mask": torch.ones(B, N),
    }


def _logged(model, monkeypatch):
    """Capture self.log() calls; there is no Trainer attached in a unit test."""
    calls = {}
    monkeypatch.setattr(model, "log", lambda name, value, **kw: calls.__setitem__(name, value))
    return calls


def _poison(model, monkeypatch, value=float("nan")):
    bad = torch.tensor(value)
    monkeypatch.setattr(model, "_shared_step", lambda *a, **k: (bad, bad, bad, bad, None, None, None))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_loss_skips_the_optimizer_step(model, batch, monkeypatch, value):
    """Returning None makes Lightning skip backward and the optimizer step."""
    _logged(model, monkeypatch)
    _poison(model, monkeypatch, value)

    assert model.training_step(batch, batch_idx=0) is None


def test_nonfinite_loss_increments_a_counter_logged_for_wandb(model, batch, monkeypatch):
    calls = _logged(model, monkeypatch)
    _poison(model, monkeypatch)

    model.training_step(batch, batch_idx=0)
    assert calls["train_nonfinite_skips"] == 1.0

    model.training_step(batch, batch_idx=1)
    assert calls["train_nonfinite_skips"] == 2.0


def test_nonfinite_loss_is_warned_about_with_context(model, batch, monkeypatch, caplog):
    _logged(model, monkeypatch)
    _poison(model, monkeypatch)

    with caplog.at_level(logging.WARNING, logger=DIFFUSION_LOGGER):
        model.training_step(batch, batch_idx=7)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "a skipped batch must be visible in the logs"
    message = warnings[0].getMessage()
    assert "batch 7" in message
    assert "skip" in message.lower()


def test_counter_is_logged_on_healthy_steps_so_the_wandb_series_is_continuous(model, batch, monkeypatch):
    """A metric logged only on failure shows up in wandb as isolated points."""
    calls = _logged(model, monkeypatch)

    loss = model.training_step(batch, batch_idx=0)

    assert loss is not None and torch.isfinite(loss)
    assert calls["train_nonfinite_skips"] == 0.0


def test_healthy_step_does_not_increment_the_counter(model, batch, monkeypatch):
    calls = _logged(model, monkeypatch)

    model.training_step(batch, batch_idx=0)
    model.training_step(batch, batch_idx=1)

    assert calls["train_nonfinite_skips"] == 0.0
    assert torch.isfinite(calls["train_loss"])
