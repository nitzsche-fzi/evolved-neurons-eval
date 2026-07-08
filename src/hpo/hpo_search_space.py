"""Optuna search spaces for ``hpo_tune.py``."""
from __future__ import annotations

import argparse

import optuna

from src.data.task_config import TASK_CONFIGS

# Log-uniform bounds: upper limit keeps penalty from dominating typical val CE.
SPIKE_RATE_REG_BOUNDS: dict[str, tuple[float, float]] = {
    "dvsgesture": (0.01, 5.0),  # upper bound maybe a bit high
    "braille": (0.01, 10.0),  # upper bound maybe a bit low
    "shd": (0.1, 20.0),
}


def suggest_spike_rate_reg(trial: optuna.Trial, task: str) -> float:
    lo, hi = SPIKE_RATE_REG_BOUNDS[task]
    return trial.suggest_float("spike_rate_reg", lo, hi, log=True)


def suggest_hyperparameters(trial: optuna.Trial, args: argparse.Namespace) -> dict:
    """Dispatch to the right search space for the current ``--search-mode``."""
    if args.search_mode == "dataset":
        return _suggest_dataset_hyperparameters(trial, args)
    if args.search_mode == "size_adapt":
        return _suggest_size_adapt_hyperparameters(trial, args)
    return _suggest_neuron_hyperparameters(trial, args)


def _suggest_size_adapt_hyperparameters(trial: optuna.Trial, args: argparse.Namespace) -> dict:
    """Narrow search: lr and spike_rate_reg only, relative to 1× baseline."""
    baseline = args.baseline_params
    if "lr" not in baseline:
        raise ValueError("Baseline params must contain 'lr' for size_adapt.")
    if "spike_rate_reg" not in baseline:
        raise ValueError("Baseline params must contain 'spike_rate_reg' for size_adapt.")

    lo = float(args.adapt_range_min)
    hi = float(args.adapt_range_max)
    lr_base = float(baseline["lr"])
    sr_base = float(baseline["spike_rate_reg"])
    return {
        "lr": trial.suggest_float("lr", lr_base * lo, lr_base * hi, log=True),
        "spike_rate_reg": trial.suggest_float(
            "spike_rate_reg", sr_base * lo, sr_base * hi, log=True,
        ),
    }


def _suggest_neuron_hyperparameters(trial: optuna.Trial, args: argparse.Namespace) -> dict:
    """Training-side hparams for a fixed (task, neuron) pair."""
    params: dict = {
        "lr": trial.suggest_float("lr", 5e-5, 5e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
        "spike_rate_reg": suggest_spike_rate_reg(trial, args.task),
        "surrogate_alpha": trial.suggest_float("surrogate_alpha", 0.5, 5.0),
    }

    sched = args.lr_scheduler
    if sched == "cosine":
        params["lr_min_factor"] = trial.suggest_float(
            "lr_min_factor", 1e-3, 0.3, log=True
        )

    params.update(_suggest_neuron_framing(trial, args.task))

    aug_magnitude = {
        "random_time_scale": _suggest_time_scale_magnitude,
        "event_dropping": _suggest_event_dropping_magnitude,
        "time_jitter": _suggest_time_jitter_magnitude,
        "noise": _suggest_noise_magnitude,
        "random_image_scale": _suggest_image_scale_magnitude,
        "random_image_offset": _suggest_image_offset_magnitude,
    }
    for aug in TASK_CONFIGS[args.task]().extra.get("use_augmentations", ()):
        params[aug] = aug_magnitude[aug](trial)
    return params


def _suggest_neuron_framing(trial: optuna.Trial, task: str) -> dict:
    """Per-neuron temporal-framing search space."""
    if task == "braille":
        return {"n_steps": trial.suggest_categorical("n_steps", [64, 96, 128, 192, 256])}
    if task == "dvsgesture":
        return {
            "dt": trial.suggest_categorical("dt", [4000, 6000, 8000, 10000, 16000]),
            "n_steps": trial.suggest_categorical("n_steps", [50, 75, 100, 150, 200]),
        }
    if task == "shd":
        return {"dt": trial.suggest_categorical("dt", [4, 6, 8, 10, 12, 16])}
    return {}


def _suggest_dataset_hyperparameters(trial: optuna.Trial, args: argparse.Namespace) -> dict:
    """Data-pipeline hparams in ``TaskConfig.extra``."""
    if args.task == "braille":
        return _suggest_braille_dataset(trial)
    if args.task == "shd":
        return _suggest_shd_dataset(trial)
    if args.task == "dvsgesture":
        return _suggest_dvs_dataset(trial)
    raise ValueError(
        f"--search-mode dataset is not implemented for --task {args.task!r}. "
        "Add a search spec in hpo_search_space._suggest_dataset_hyperparameters."
    )


def _suggest_braille_dataset(trial: optuna.Trial) -> dict:
    return {
        "threshold": trial.suggest_categorical("threshold", [1, 2, 5, 10]),
        "random_time_scale": _suggest_time_scale(trial),
        "event_dropping": _suggest_event_dropping(trial),
        "time_jitter": _suggest_time_jitter(trial),
        "noise": _suggest_noise(trial),
        "random_start_offset": trial.suggest_categorical("random_start_offset", [False, True]),
    }


def _suggest_shd_dataset(trial: optuna.Trial) -> dict:
    return {
        "random_time_scale": _suggest_time_scale(trial),
        "event_dropping": _suggest_event_dropping(trial),
        "time_jitter": _suggest_time_jitter(trial),
        "noise": _suggest_noise(trial),
    }


def _suggest_dvs_dataset(trial: optuna.Trial) -> dict:
    return {
        "random_time_scale": _suggest_time_scale(trial),
        "event_dropping": _suggest_event_dropping(trial),
        "time_jitter": _suggest_time_jitter(trial),
        "noise": _suggest_noise(trial),
        "random_start_offset": trial.suggest_categorical("random_start_offset", [False, True]),
        "random_image_scale": _suggest_image_scale(trial),
        "random_image_offset": _suggest_image_offset(trial),
    }


def _suggest_time_scale_magnitude(trial: optuna.Trial) -> tuple[float, float]:
    s = trial.suggest_float("time_skew_halfwidth", 0.0, 0.3)
    return (1.0 - s, 1.0 + s)


def _suggest_time_scale(trial: optuna.Trial) -> tuple[float, float]:
    if not trial.suggest_categorical("time_skew_on", [False, True]):
        return (1.0, 1.0)
    return _suggest_time_scale_magnitude(trial)


def _suggest_event_dropping_magnitude(trial: optuna.Trial) -> float:
    return trial.suggest_float("event_dropping", 0.0, 0.5)


def _suggest_event_dropping(trial: optuna.Trial) -> float:
    if not trial.suggest_categorical("event_dropping_on", [False, True]):
        return 0.0
    return _suggest_event_dropping_magnitude(trial)


def _suggest_time_jitter_magnitude(trial: optuna.Trial) -> float:
    return trial.suggest_float("time_jitter_std_us", 1e3, 5e4, log=True)


def _suggest_time_jitter(trial: optuna.Trial) -> float:
    if not trial.suggest_categorical("time_jitter_on", [False, True]):
        return 0.0
    return _suggest_time_jitter_magnitude(trial)


def _suggest_noise_magnitude(trial: optuna.Trial) -> float:
    return trial.suggest_float("noise_p", 1e-5, 1e-3, log=True)


def _suggest_noise(trial: optuna.Trial) -> float:
    if not trial.suggest_categorical("noise_on", [False, True]):
        return 0.0
    return _suggest_noise_magnitude(trial)


def _suggest_image_scale_magnitude(trial: optuna.Trial) -> tuple[float, float]:
    s = trial.suggest_float("image_scale_halfwidth", 0.0, 0.1)
    return (1.0 - s, 1.0 + s)


def _suggest_image_scale(trial: optuna.Trial) -> tuple[float, float]:
    if not trial.suggest_categorical("image_scale_on", [False, True]):
        return (1.0, 1.0)
    return _suggest_image_scale_magnitude(trial)


def _suggest_image_offset_magnitude(trial: optuna.Trial) -> tuple[float, float]:
    o = trial.suggest_float("image_offset_halfwidth", 0.0, 0.2)
    return (-o, o)


def _suggest_image_offset(trial: optuna.Trial) -> tuple[float, float]:
    if not trial.suggest_categorical("image_offset_on", [False, True]):
        return (0.0, 0.0)
    return _suggest_image_offset_magnitude(trial)
