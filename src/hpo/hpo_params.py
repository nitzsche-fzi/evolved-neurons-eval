"""Map Optuna ``trial.params`` names from ``best_params.yaml`` to ``TaskConfig`` fields."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.data.task_config import TaskConfig

# Same routing as hpo_tune_framing._EXTRA_KEYS
_EXTRA_KEYS: frozenset[str] = frozenset({
    "dt",
    "n_steps",
    "threshold",
    "time_jitter",
    "event_dropping",
    "random_time_scale",
    "noise",
    "random_start_offset",
    "random_image_scale",
    "random_image_offset",
})

# Optuna-only / gate flags — not written to cfg
_SKIP_KEYS: frozenset[str] = frozenset({
    "time_skew_on",
    "time_jitter_on",
    "event_dropping_on",
    "image_scale_on",
    "image_offset_on",
})


def load_hpo_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_hpo_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert ``best.params`` keys to names used by ``TaskConfig`` / ``cfg.extra``."""
    out: dict[str, Any] = {}

    for k, v in params.items():
        if k in _SKIP_KEYS or v is None:
            continue
        if k in (
            "lr",
            "weight_decay",
            "spike_rate_reg",
            "surrogate_alpha",
            "lr_min_factor",
            "lr_scheduler",
            "inp_mean",
            "inp_var",
        ):
            out[k] = v
            continue
        if k in _EXTRA_KEYS:
            out[k] = v
            continue

    if "noise_p" in params and params["noise_p"] is not None:
        out["noise"] = params["noise_p"]
    if "time_jitter_std_us" in params and params["time_jitter_std_us"] is not None:
        out["time_jitter"] = params["time_jitter_std_us"]

    if "time_skew_halfwidth" in params and params["time_skew_halfwidth"] is not None:
        s = float(params["time_skew_halfwidth"])
        out["random_time_scale"] = (1.0 - s, 1.0 + s)
    if "image_scale_halfwidth" in params and params["image_scale_halfwidth"] is not None:
        s = float(params["image_scale_halfwidth"])
        out["random_image_scale"] = (1.0 - s, 1.0 + s)
    if "image_offset_halfwidth" in params and params["image_offset_halfwidth"] is not None:
        o = float(params["image_offset_halfwidth"])
        out["random_image_offset"] = (-o, o)

    for k in ("dt", "n_steps", "threshold", "event_dropping", "random_start_offset"):
        if k in params and params[k] is not None:
            out[k] = params[k]

    return out


def apply_hpo_params_to_cfg(cfg: TaskConfig, params: dict[str, Any]) -> TaskConfig:
    for k, v in normalize_hpo_params(params).items():
        if k in _EXTRA_KEYS:
            cfg.extra[k] = v
        else:
            setattr(cfg, k, v)
    return cfg
