"""The compatibility gate. Every rejection here is a run that would either
crash hours in or train to a loss floor it cannot get under.

Cheap and pure: no model, no data, no trainer.
"""
import pytest
from omegaconf import OmegaConf

from src.utils.config import Config, validate_combination


def _cfg(**kw):
    cfg = OmegaConf.structured(Config)
    cfg.config_set = "mesh_diffusion"
    for k, v in kw.items():
        OmegaConf.update(cfg, f"mesh_diffusion.{k}", v, force_add=True)
    return cfg


def test_default_arm_is_legal():
    validate_combination(_cfg())  # morton / sinusoidal / mse / unet / ddpm / noise


def test_unordered_with_mse_raises():
    with pytest.raises(ValueError, match="hungarian"):
        validate_combination(_cfg(order="none", pos_embed="none", loss="mse"))


def test_unordered_with_pos_embed_raises():
    with pytest.raises(ValueError, match="pos_embed"):
        validate_combination(
            _cfg(order="none", pos_embed="sinusoidal", loss="hungarian"))


def test_no_pos_embed_with_mse_raises():
    with pytest.raises(ValueError, match="hungarian"):
        validate_combination(_cfg(order="morton", pos_embed="none", loss="mse"))


def test_unet_requires_sorted_order():
    with pytest.raises(ValueError, match="unet"):
        validate_combination(_cfg(denoiser="unet", order="none",
                                  pos_embed="none", loss="hungarian"))


def test_flow_requires_velocity_target():
    with pytest.raises(ValueError, match="velocity"):
        validate_combination(_cfg(process="flow", target="noise"))
    validate_combination(_cfg(process="flow", target="velocity"))


def test_d3pm_requires_bins_and_original():
    with pytest.raises(ValueError, match="state: bins"):
        validate_combination(_cfg(process="d3pm", target="original",
                                  loss="ce", state="quantized"))
    with pytest.raises(ValueError, match="original"):
        validate_combination(_cfg(process="d3pm", target="noise",
                                  loss="ce", state="bins"))
    validate_combination(_cfg(process="d3pm", target="original",
                              loss="ce", state="bins"))


def test_gaussian_process_rejects_the_integer_state():
    with pytest.raises(ValueError, match="no Gaussian path"):
        validate_combination(_cfg(process="ddpm", target="original",
                                  loss="ce", state="bins"))


def test_ce_needs_an_alphabet_and_the_x0_target():
    with pytest.raises(ValueError, match="discrete alphabet"):
        validate_combination(_cfg(loss="ce", state="continuous",
                                  target="original"))
    with pytest.raises(ValueError, match="signal-free"):
        validate_combination(_cfg(loss="ce", state="quantized", target="noise"))
    validate_combination(_cfg(loss="ce", state="quantized", target="original"))


def test_flow_with_ce_is_rejected():
    with pytest.raises(ValueError, match="incompatible"):
        validate_combination(_cfg(process="flow", target="velocity",
                                  loss="ce", state="quantized"))


def test_onehot_with_a_regression_loss_is_rejected():
    with pytest.raises(ValueError, match="indicator channels"):
        validate_combination(_cfg(loss="mse", state="onehot", target="original"))


def test_ddpm_rejects_velocity():
    with pytest.raises(ValueError, match="target"):
        validate_combination(_cfg(process="ddpm", target="velocity"))


def test_sorted_plus_hungarian_warns_but_passes(caplog):
    validate_combination(_cfg(order="morton", pos_embed="sinusoidal",
                              loss="hungarian", denoiser="transformer"))
    assert any("wasteful" in r.message.lower() or "wasteful" in r.getMessage().lower()
               for r in caplog.records)


def test_every_shipped_diffusion_config_is_a_legal_arm():
    """Every configs/mesh-diff-*.yaml must survive the gate.

    This is the cheapest possible guard against the failure the gate exists
    for: a config committed months ago, launched overnight, and rejected at
    startup after the GPU was already reserved.
    """
    from pathlib import Path
    from omegaconf import OmegaConf

    paths = sorted(Path("configs").glob("mesh-diff-*.yaml"))
    assert paths, "no mesh-diff configs found"
    for path in paths:
        cfg = OmegaConf.merge(OmegaConf.structured(Config),
                              OmegaConf.load(path))
        validate_combination(cfg)


@pytest.mark.parametrize("order,pos_embed", [("morton", "none"), ("none", "none")])
def test_ce_needs_an_order_and_a_positional_encoding(order, pos_embed):
    """`loss: ce` follows `loss: mse` in the compatibility matrix.

    A categorical readout is still scored slot against slot, so swapping the
    regression head for a softmax does nothing about the permutation the model
    cannot see. Missed once already: the D4 gate was keyed on `mse` alone,
    which let every unordered categorical arm through.
    """
    with pytest.raises(ValueError, match="hungarian"):
        validate_combination(_cfg(loss="ce", state="quantized", target="original",
                                  order=order, pos_embed=pos_embed,
                                  denoiser="transformer"))
