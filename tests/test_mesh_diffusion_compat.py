"""The compatibility gate. Every rejection here is a run that would either
crash hours in or train to a loss floor it cannot get under.

Cheap and pure: no model, no data, no trainer.
"""
import math

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


def test_no_pos_embed_with_mse_raises_for_the_transformer():
    with pytest.raises(ValueError, match="hungarian"):
        validate_combination(_cfg(order="morton", pos_embed="none", loss="mse",
                                  denoiser="transformer"))


def test_unet_with_file_order_warns_but_passes(caplog):
    """D5, relaxed. The convolution needs adjacent slots to be spatially
    adjacent, and file order on this corpus already is: 72.4% of consecutive
    faces share a corner against 40.9% for a random permutation, because
    CityJSON groups faces by surface. What it lacks is canonicality -- it is a
    property of the writer, not the geometry -- so this warns rather than
    passing silently. It is the a4-unet-hungarian arm.
    """
    validate_combination(_cfg(denoiser="unet", order="none",
                              pos_embed="none", loss="hungarian"))
    assert any("not" in r.getMessage() and "canonical" in r.getMessage()
               for r in caplog.records)


def test_pos_embed_is_flagged_as_inert_for_the_unet(caplog):
    """`ConditionalMeshUNet` has no positional input; a convolution is
    translation-equivariant along the face axis. The knob stays legal so a1 and
    a2 differ on `denoiser` alone, but a logged config must not imply it did
    something."""
    validate_combination(_cfg(denoiser="unet", order="morton",
                              pos_embed="sinusoidal", loss="mse"))
    assert any("ignored under denoiser: unet" in r.getMessage()
               for r in caplog.records)


def test_unet_may_drop_pos_embed_under_mse():
    """The pos_embed half of D4 binds the transformer only -- requiring it of
    the U-Net would be requiring a no-op."""
    validate_combination(_cfg(denoiser="unet", order="morton",
                              pos_embed="none", loss="mse"))
    with pytest.raises(ValueError, match="pos_embed: sinusoidal"):
        validate_combination(_cfg(denoiser="transformer", order="morton",
                                  pos_embed="none", loss="mse"))


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


# Init-time scale of each loss term, with the zero-init head predicting 0.
# E[x0^2] = 0.0958 measured over real faces on The Hague/mini_cleaner; the rest
# are analytic. Kept as constants so this test needs no dataset -- re-derive the
# 0.0958 if the corpus changes. See MeshDiffusionConfig.presence_weight.
_COORD_MS = 0.0958
_INIT_SCALE = {
    "noise":    (1.0, 1.0),                        # eps ~ N(0,1) on all 10 channels
    "original": (_COORD_MS, 0.25),                 # x0; presence target +-0.5
    "velocity": (_COORD_MS + 1.0, 1.25),           # x0 - z
    "ce":       (math.log(128), math.log(2)),      # ln K vs ln 2
}


def _parity_weight(cfg):
    """presence_weight that puts the two terms on equal footing at init."""
    d = cfg.mesh_diffusion
    coord, presence = _INIT_SCALE["ce" if d.loss == "ce" else d.target]
    return coord / presence


def test_shipped_configs_equalise_presence_weight():
    """Every arm must weight presence to init parity for its own regime.

    A flat presence_weight across arms is not a neutral default: the two loss
    terms start at scales set by the target and the readout, so b3 and b4 --
    which share a target and differ only in readout, and are the grid's
    cleanest control -- would differ by 18x in how hard the model is pushed to
    get face count right. That lands on exactly the quantity the geometric
    metrics are most sensitive to.
    """
    from pathlib import Path
    from omegaconf import OmegaConf

    paths = sorted(Path("configs").glob("mesh-diff-*.yaml"))
    checked = 0
    for path in paths:
        if path.name in ("mesh-diff-base.yaml", "mesh-diff-smoke.yaml"):
            continue
        cfg = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.load(path))
        want = _parity_weight(cfg)
        got = cfg.mesh_diffusion.presence_weight
        assert abs(got - want) / want < 0.01, (
            f"{path.name}: presence_weight {got} but its regime "
            f"(loss={cfg.mesh_diffusion.loss}, target={cfg.mesh_diffusion.target}) "
            f"needs {want:.3f} for init parity")
        checked += 1
    assert checked >= 13
