"""Compute activation-filtered inp_mean / inp_var for neuromorphic weight init.

Example:
    python3 compute_inp_stats.py --task braille
    python3 compute_inp_stats.py --task shd --dataset-path /shared/datasets
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import tonic
from torch.utils.data import DataLoader

from braille_dataset import Braille  # pyright: ignore[reportMissingImports]

from src.data.task_config import TASK_CONFIGS, TaskConfig
from src.data.datasets import (
    BrailleDataModule,
    DVSGestureDataModule,
    SHDDataModule,
)


@dataclass(frozen=True)
class InputStats:
    inp_mean: float
    inp_var: float
    dataset_size: int
    num_steps: int
    feature_dim: int
    max_activity: float
    activity_threshold: float
    n_selected_steps: int
    initializer_ok: bool
    mean_events_per_sample: float
    braille_mean_events_by_threshold: dict[int, float] | None = None


def _mean_raw_events(cfg: TaskConfig, cv_fold_idx: int) -> float:
    """Mean raw event count per sample on the train split (no transforms)."""
    path = cfg.dataset_path
    if cfg.name == "shd":
        ds = tonic.datasets.SHD(save_to=path, train=True)
    elif cfg.name == "dvsgesture":
        ds = tonic.datasets.DVSGesture(save_to=path, train=True)
    elif cfg.name == "braille":
        split = f"train{cv_fold_idx + 1}"
        ds = Braille(
            save_to=path,
            threshold=cfg.extra["threshold"],
            split=split,
        )
    else:
        raise ValueError(f"Unsupported task: {cfg.name}")

    counts = np.array([len(ds[i][0]) for i in range(len(ds))], dtype=np.float64)
    return float(counts.mean())


def _braille_mean_events_by_threshold(dataset_path: str) -> dict[int, float]:
    """Mean raw event count per sample for each threshold (full dataset)."""
    means: dict[int, float] = {}
    for th in Braille.THRESHOLDS:
        ds = Braille(save_to=dataset_path, threshold=th, split=None)
        counts = np.array([len(ds.data[i]) for i in range(len(ds))], dtype=np.float64)
        means[int(th)] = float(counts.mean())
    return means


def _build_stats_loader(cfg: TaskConfig, num_workers: int, cv_fold_idx: int) -> DataLoader:
    """Training split, deterministic transforms, val/test collation."""
    if cfg.name == "shd":
        dm = SHDDataModule(cfg, num_workers)
        dataset = dm._build_dataset(train=True, train_augment=False)
        collate_fn = dm._val_collate()
    elif cfg.name == "dvsgesture":
        dm = DVSGestureDataModule(cfg, num_workers)
        dataset = dm._build_dataset(train=True, train_augment=False)
        collate_fn = dm._val_collate()
    elif cfg.name == "braille":
        cv_folds = 0 if cv_fold_idx == 0 else 5
        dm = BrailleDataModule(cfg, num_workers, cv_folds, cv_fold_idx)
        split = f"train{cv_fold_idx + 1}"
        dataset = dm._build_dataset(split=split, train_augment=False)
        collate_fn = dm._val_collate()
    else:
        raise ValueError(f"Unsupported task: {cfg.name}")

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )


def _activation_filtered_mean_var(
    loader: DataLoader, activation_fraction: float
) -> tuple[float, float, int, int, float, float, int]:
    """Core stats pass shared by the report and the in-loop helper.

    Flattens all time steps, keeps the steps whose total activity is at least
    ``activation_fraction`` of the global max, and reduces over the selected
    steps. Returns ``(inp_mean, inp_var, num_steps, feature_dim, max_activity,
    activity_threshold, n_selected_steps)``.
    """
    all_activities: list[np.ndarray] = []
    all_steps: list[np.ndarray] = []

    for batch in loader:
        inputs, _ = batch
        x = inputs.reshape(inputs.shape[0], inputs.shape[1], -1).cpu().numpy()
        steps = x.reshape(-1, x.shape[-1])
        all_activities.append(steps.sum(axis=1))
        all_steps.append(steps)

    if not all_steps:
        raise RuntimeError("No batches produced; is the dataset empty or missing?")

    activities = np.concatenate(all_activities, axis=0)
    steps = np.concatenate(all_steps, axis=0)
    num_steps, feature_dim = steps.shape

    max_activity = float(np.max(activities))
    activity_threshold = activation_fraction * max_activity
    selected = steps[activities >= activity_threshold]

    inp_mean = float(selected.mean())
    inp_var = float(selected.var(ddof=0))
    return (
        inp_mean,
        inp_var,
        num_steps,
        feature_dim,
        max_activity,
        activity_threshold,
        int(selected.shape[0]),
    )


def compute_inp_mean_var(
    cfg: TaskConfig,
    *,
    num_workers: int = 4,
    activation_fraction: float = 0.2,
    cv_fold_idx: int = 0,
) -> tuple[float, float]:
    """Activation-filtered ``(inp_mean, inp_var)`` for neuromorphic weight init.

    Lightweight counterpart to :func:`compute_inp_stats` meant for use inside the
    HPO / training loop: it skips the expensive ``mean_events`` and per-threshold
    passes and returns only the two values consumed by
    ``neuromorphic_dense_layer``. The stats depend solely on the framing knobs in
    ``cfg.extra`` (n_steps / dt / threshold), so callers should cache by framing.
    """
    loader = _build_stats_loader(cfg, num_workers, cv_fold_idx)
    inp_mean, inp_var, *_ = _activation_filtered_mean_var(loader, activation_fraction)
    return inp_mean, inp_var


def compute_inp_stats(
    cfg: TaskConfig,
    *,
    num_workers: int = 4,
    activation_fraction: float = 0.2,
    cv_fold_idx: int = 0,
) -> InputStats:
    mean_events = _mean_raw_events(cfg, cv_fold_idx)
    braille_by_threshold = (
        _braille_mean_events_by_threshold(cfg.dataset_path)
        if cfg.name == "braille"
        else None
    )

    loader = _build_stats_loader(cfg, num_workers, cv_fold_idx)
    (
        inp_mean,
        inp_var,
        num_steps,
        feature_dim,
        max_activity,
        activity_threshold,
        n_selected,
    ) = _activation_filtered_mean_var(loader, activation_fraction)

    var_bound = cfg.input_size * inp_mean**2
    initializer_ok = inp_var <= var_bound + 1e-9

    return InputStats(
        inp_mean=inp_mean,
        inp_var=inp_var,
        dataset_size=len(loader.dataset),
        num_steps=num_steps,
        feature_dim=feature_dim,
        max_activity=max_activity,
        activity_threshold=activity_threshold,
        n_selected_steps=n_selected,
        initializer_ok=initializer_ok,
        mean_events_per_sample=mean_events,
        braille_mean_events_by_threshold=braille_by_threshold,
    )


def _format_report(
    task: str,
    stats: InputStats,
    cfg: TaskConfig,
    activation_fraction: float,
) -> str:
    lines = [
        f"[{task}]",
        f"  dataset samples        : {stats.dataset_size}",
        f"  mean events / sample   : {stats.mean_events_per_sample:.2f}",
        f"  flattened steps        : {stats.num_steps} (dim {stats.feature_dim}, cfg.input_size={cfg.input_size})",
        f"  activation fraction    : {activation_fraction}",
        f"  max step activity      : {stats.max_activity:.6f}",
        f"  activity threshold     : {stats.activity_threshold:.6f}",
        f"  selected steps         : {stats.n_selected_steps}",
        f"  inp_mean               : {stats.inp_mean:.6f}",
        f"  inp_var                : {stats.inp_var:.6f}",
        f"  config inp_mean        : {cfg.inp_mean}",
        f"  config inp_var         : {cfg.inp_var}",
        f"  initializer bound      : inp_var <= input_size * inp_mean^2 = {cfg.input_size * stats.inp_mean**2:.6f}",
        f"  initializer feasible   : {stats.initializer_ok}",
    ]
    if not stats.initializer_ok:
        lines.append(
            "  WARNING: inp_var exceeds the neuromorphic_dense_layer bound; "
            "weight init will raise ValueError."
        )
    lines.append("")
    lines.append("  Paste into task_config.py:")
    lines.append(f"        inp_mean={stats.inp_mean:.6g},")
    lines.append(f"        inp_var={stats.inp_var:.6g},")
    if stats.braille_mean_events_by_threshold is not None:
        items = ", ".join(
            f"{k}: {v:.2f}"
            for k, v in sorted(stats.braille_mean_events_by_threshold.items())
        )
        lines.append(f'        "mean_events_per_sample": {{{items}}},')
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=sorted(TASK_CONFIGS), default="braille")
    p.add_argument("--dataset-path", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--activation-fraction",
        type=float,
        default=0.2,
        help="Keep steps with activity >= fraction * max_activity (default: 0.2).",
    )
    p.add_argument(
        "--cv-fold-idx",
        type=int,
        default=0,
        help="Braille train split index (train{cv_fold_idx+1}); ignored for other tasks.",
    )
    p.add_argument(
        "--all-tasks",
        action="store_true",
        help="Compute stats for every registered task.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tasks = sorted(TASK_CONFIGS) if args.all_tasks else [args.task]

    for task in tasks:
        cfg = TASK_CONFIGS[task]()
        if args.dataset_path is not None:
            cfg.dataset_path = args.dataset_path
        if args.batch_size is not None:
            cfg.batch_size = args.batch_size

        stats = compute_inp_stats(
            cfg,
            num_workers=args.num_workers,
            activation_fraction=args.activation_fraction,
            cv_fold_idx=args.cv_fold_idx,
        )
        print(_format_report(task, stats, cfg, args.activation_fraction))


if __name__ == "__main__":
    main()
