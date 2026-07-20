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
