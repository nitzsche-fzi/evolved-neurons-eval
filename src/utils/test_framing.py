"""Offline test-set framing stats for energy estimation."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any, Optional

import tonic
import tonic.transforms as transforms
import yaml

from braille_dataset import Braille  # pyright: ignore[reportMissingImports]
from esn.augmentations import AudioTransform, FrameTransform  # pyright: ignore[reportMissingImports]

from src.data.task_config import TASK_CONFIGS, TaskConfig
from src.hpo.hpo_params import apply_hpo_params_to_cfg, load_hpo_yaml

_FRAMING_CACHE: dict[tuple[Any, ...], int] = {}


def _load_run_info(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def task_config_from_run_info(run_info_path: Path) -> Optional[TaskConfig]:
    """Rebuild the eval ``TaskConfig`` (HPO framing, no train augmentations)."""
    info = _load_run_info(run_info_path)
    args_dict = info.get("args")
    if not isinstance(args_dict, dict):
        return None

    task = args_dict.get("task")
    if task not in TASK_CONFIGS:
        return None

    cfg = TASK_CONFIGS[task]()
    hpo_path = args_dict.get("hpo_params")
    if hpo_path:
        params = load_hpo_yaml(hpo_path).get("params") or {}
        if isinstance(params, dict):
            apply_hpo_params_to_cfg(cfg, params)

    args = Namespace(**args_dict)
    if getattr(args, "dataset_path", None):
        cfg.dataset_path = args.dataset_path
    if getattr(args, "noise", None) is not None:
        cfg.extra["noise"] = args.noise

    return cfg


def _framing_cache_key(
    cfg: TaskConfig,
    *,
    n_hidden_neurons: int,
    cv_folds: int,
    cv_fold_idx: int,
) -> tuple[Any, ...]:
    extra = cfg.extra
    return (
        cfg.name,
        cfg.dataset_path,
        int(n_hidden_neurons),
        int(cv_folds),
        int(cv_fold_idx),
        extra.get("dt"),
        extra.get("n_steps"),
        extra.get("threshold"),
        tuple(extra.get("desired_sensor_size") or ()),
        extra.get("squeeze_thresh", 1),
    )


def _sample_timesteps(sample) -> int:
    if hasattr(sample, "is_sparse") and sample.is_sparse:
        return int(sample.shape[0])
    return int(sample.shape[0])


def _build_test_dataset(cfg: TaskConfig, *, cv_folds: int, cv_fold_idx: int):
    """Build the deterministic test dataset used during eval (no augmentations)."""
    extra = cfg.extra
    if cfg.name == "shd":
        transform = transforms.Compose([
            AudioTransform(
                original_sensor_size=[700, 1, 1],
                desired_sensor_size=extra["desired_sensor_size"],
                dt=extra["dt"],
                random_time_scale=[1.0, 1.0],
                squeeze_thresh=extra.get("squeeze_thresh", 1),
            ),
        ])
        return tonic.datasets.SHD(save_to=cfg.dataset_path, train=False, transform=transform)

    if cfg.name == "dvsgesture":
        transform = transforms.Compose([
            FrameTransform(
                original_sensor_size=tonic.datasets.DVSGesture.sensor_size,
                desired_sensor_size=extra["desired_sensor_size"],
                dt=extra["dt"],
                n_steps=extra["n_steps"],
                random_start_offset=False,
                noise=0.0,
                random_time_scale=[1.0, 1.0],
                random_image_scale=[1.0, 1.0],
                random_image_offset=[0.0, 0.0],
            ),
        ])
        return tonic.datasets.DVSGesture(
            save_to=cfg.dataset_path,
            train=False,
            transform=transform,
        )

    if cfg.name == "braille":
        assert cv_folds in (0, 1, 5)
        transform = transforms.Compose([
            transforms.ToFrame(sensor_size=Braille.sensor_size, n_time_bins=extra["n_steps"]),
        ])
        return Braille(
            save_to=cfg.dataset_path,
            threshold=extra["threshold"],
            split="test",
            transform=transform,
        )
    raise ValueError(f"Unsupported task: {cfg.name}")


def compute_test_neuron_timesteps(
    cfg: TaskConfig,
    *,
    n_hidden_neurons: int,
    cv_folds: int = 0,
    cv_fold_idx: int = 0,
) -> Optional[int]:
    """Total hidden neuron updates over the deterministic test set (``n_hidden * sum T_i``)."""
    key = _framing_cache_key(
        cfg,
        n_hidden_neurons=n_hidden_neurons,
        cv_folds=cv_folds,
        cv_fold_idx=cv_fold_idx,
    )
    if key in _FRAMING_CACHE:
        return _FRAMING_CACHE[key]

    dataset = _build_test_dataset(cfg, cv_folds=cv_folds, cv_fold_idx=cv_fold_idx)
    n_samples = len(dataset)
    n_steps = cfg.extra.get("n_steps")
    if n_steps is not None:
        total = int(n_hidden_neurons) * int(n_steps) * n_samples
    else:
        total_timesteps = sum(_sample_timesteps(dataset[i][0]) for i in range(n_samples))
        total = int(n_hidden_neurons) * total_timesteps

    _FRAMING_CACHE[key] = total
    return total


def test_neuron_timesteps_from_run_info(
    run_info_path: Path,
    *,
    n_hidden_neurons: int,
) -> Optional[int]:
    """Resolve offline test-set neuron-timesteps for one eval run."""
    info = _load_run_info(run_info_path)
    args_dict = info.get("args") if isinstance(info.get("args"), dict) else {}
    cfg = task_config_from_run_info(run_info_path)
    if cfg is None:
        return None
    return compute_test_neuron_timesteps(
        cfg,
        n_hidden_neurons=n_hidden_neurons,
        cv_folds=int(args_dict.get("cv_folds") or 0),
        cv_fold_idx=int(args_dict.get("cv_fold_idx") or 0),
    )
