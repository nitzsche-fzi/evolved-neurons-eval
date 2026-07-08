from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import yaml


class BaseMetricsSummary(pl.Callback, ABC):
    """Shared helper for YAML metric-summary callbacks."""

    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename
        self.path: Optional[Path] = None

    @staticmethod
    def _collect_scalars(metrics: dict) -> dict:
        out: dict = {}
        for k, v in metrics.items():
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                continue
        return out

    def setup(self, trainer, pl_module, stage) -> None:
        if trainer.is_global_zero:
            self.path = Path(trainer.log_dir).resolve() / self.filename if trainer.log_dir else None
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def build_payload(self) -> Optional[dict]:
        """Return YAML payload or None when nothing should be written."""
        raise NotImplementedError

    def _write_to_file(self, trainer: pl.Trainer):
        if not trainer.is_global_zero or self.path is None:
            return
        payload = self.build_payload()
        if payload is None:
            return
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=True)


class ValMetricsSummary(BaseMetricsSummary):
    """Track best validation snapshot for a monitored metric."""

    def __init__(self, filename: str, monitor: str = "val_loss", mode: str = "min"):
        super().__init__(filename=filename)
        assert mode in ("min", "max"), "mode must be 'min' or 'max'"
        self.monitor = monitor
        self.mode = mode
        self._best: dict = {}
        self._better = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        metrics = self._collect_scalars(dict(trainer.callback_metrics))
        if self.monitor not in metrics:
            return
        score = metrics[self.monitor]
        if not self._best or self._better(score, self._best[self.monitor]):
            self._best = {**metrics, "epoch": int(trainer.current_epoch)}
            self._write_to_file(trainer)

    def build_payload(self) -> Optional[dict]:
        if not self._best:
            return None
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "best": self._best,
        }


class TestMetricsSummary(BaseMetricsSummary):
    """Persist `test_*` metrics from test stage, optionally filtered by ckpt name."""

    def __init__(self, filename: str, checkpoint_name: Optional[str] = None):
        super().__init__(filename=filename)
        self.checkpoint_name = checkpoint_name
        self._test: dict = {}

    def _matches_checkpoint(self, trainer: pl.Trainer) -> bool:
        if not self.checkpoint_name:
            return True
        ckpt_path = getattr(trainer, "ckpt_path", None)
        if not ckpt_path:
            return False
        name = Path(str(ckpt_path)).name
        return self.checkpoint_name in name

    def on_test_end(self, trainer, pl_module) -> None:
        if not self._matches_checkpoint(trainer):
            return
        metrics = self._collect_scalars(dict(trainer.callback_metrics))
        test_metrics = {k: v for k, v in metrics.items() if k.startswith("test")}
        if not test_metrics:
            return
        self._test = test_metrics
        self._write_to_file(trainer)

    def build_payload(self) -> Optional[dict]:
        if not self._test:
            return None
        # Add to existing yaml if it was created from BestMetricsSummary earlier
        payload: dict = {}
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
                if isinstance(loaded, dict):
                    payload = loaded
        payload["test"] = self._test
        return payload


class BestValTracker(pl.Callback):
    """In-memory tracker for the best validation metric across epochs.
       Optionally snapshots other metrics at that epoch."""

    def __init__(
        self,
        monitor: str,
        mode: str,
        extras: Optional[tuple[str, ...]] = None,
        composite_lambda: Optional[float] = None,
        spike_energy_weight: Optional[float] = None,
    ):
        super().__init__()
        assert mode in ("min", "max")
        self.monitor = monitor
        self.mode = mode
        self.extras = extras
        self.composite_lambda = composite_lambda
        self.spike_energy_weight = spike_energy_weight
        self._better = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)
        self.best: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.best_extras: dict[str, float] = {}

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        if self.composite_lambda is not None:
            acc = metrics.get("val_acc")
            sr = metrics.get("val_spike_rate")
            if acc is None or sr is None:
                return
            acc_f = float(acc.detach().cpu()) if hasattr(acc, "detach") else float(acc)
            sr_f = float(sr.detach().cpu()) if hasattr(sr, "detach") else float(sr)
            w = 1.0 if self.spike_energy_weight is None else float(self.spike_energy_weight)
            v = acc_f - self.composite_lambda * w * sr_f
            self.log(self.monitor, v, prog_bar=False, on_epoch=True)
        else:
            v = metrics.get(self.monitor)
            if v is None:
                return
            v = float(v.detach().cpu()) if hasattr(v, "detach") else float(v)
        if self.best is None or self._better(v, self.best):
            self.best = v
            self.best_epoch = int(trainer.current_epoch)
            if self.extras is None:
                return
            snap: dict[str, float] = {}
            for k in self.extras:
                x = metrics.get(k)
                if x is None:
                    continue
                x = float(x.detach().cpu()) if hasattr(x, "detach") else float(x)
                snap[k] = x
            self.best_extras = snap
