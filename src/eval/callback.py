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
