"""Aggregate size-sweep eval runs into accuracy/energy summary tables."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from src.data.task_config import TASK_CONFIGS
from src.utils.aggregate_runs import TEST_ENERGY_KEY, aggregate_runs

REPORT_CKPT = "best_acc"


def format_size_mult_tag(mult: float) -> str:
    return f"{mult:g}".replace(".", "p")


def mult_dir_name(mult: float) -> str:
    return f"mult_{format_size_mult_tag(mult)}"


def hidden_size_for_mult(baseline_hidden_size: int, mult: float) -> int:
    return max(1, round(baseline_hidden_size * mult))


def resolve_report_seeds(
    log_dir: Path,
    *,
    n_runs: Optional[int] = None,
    report_seeds: Optional[list[int]] = None,
) -> Optional[list[int]]:
    """Pick seeds for summary stats (None = use all available runs)."""
    if report_seeds is not None:
        return list(report_seeds)
    if n_runs is None or n_runs <= 0:
        return None
    full = aggregate_runs(log_dir)
    seeds = sorted({int(s) for s in full.get("seeds", []) if s is not None})
    return seeds[: int(n_runs)]


def _energy_fields(test_energy: dict[str, Any]) -> dict[str, Any]:
    """Flatten ``best_acc.test_energy`` from ``aggregate_runs`` for sweep tables."""
    stats = test_energy.get("energy_test_set_uj")
    mean = std = None
    if isinstance(stats, dict):
        mean = stats.get("mean")
        std = stats.get("std")
    return {
        "energy_idle_pj": test_energy.get("energy_idle_pj"),
        "energy_spike_pj": test_energy.get("energy_spike_pj"),
        "energy_test_set_uj_mean": mean,
        "energy_test_set_uj_std": std,
        "test_set_size": test_energy.get("test_set_size"),
        "test_neuron_timesteps": test_energy.get("test_neuron_timesteps"),
    }


def eval_metrics_for_dir(
    log_dir: Path,
    *,
    n_runs: Optional[int] = None,
    report_seeds: Optional[list[int]] = None,
) -> dict[str, Any]:
    """Load mean/std test metrics and test-set energy from a run_eval log directory."""
    seeds = resolve_report_seeds(
        log_dir,
        n_runs=n_runs,
        report_seeds=report_seeds,
    )
    summary = aggregate_runs(log_dir, seeds=seeds)
    per_ckpt = summary.get(REPORT_CKPT, {})
    test_energy = per_ckpt.get(TEST_ENERGY_KEY, {}) if isinstance(per_ckpt, dict) else {}
    empty_energy = _energy_fields({})

    if not per_ckpt:
        return {
            "n_runs": 0,
            "seeds_used": seeds or [],
            "test_acc_mean": None,
            "test_acc_std": None,
            "test_spike_rate_mean": None,
            "test_spike_rate_std": None,
            **empty_energy,
        }
    acc = per_ckpt.get("test_acc", {})
    sr = per_ckpt.get("test_spike_rate", {})
    return {
        "n_runs": summary.get("n_runs", 0),
        "seeds_used": seeds if seeds is not None else summary.get("seeds", []),
        "test_acc_mean": acc.get("mean"),
        "test_acc_std": acc.get("std"),
        "test_spike_rate_mean": sr.get("mean"),
        "test_spike_rate_std": sr.get("std"),
        **_energy_fields(test_energy if isinstance(test_energy, dict) else {}),
    }


def build_size_point_record(
    *,
    task: str,
    neuron: str,
    size_mult: float,
    baseline_hidden_size: int,
    n_hidden_layers: int,
    eval_log_dir: Path,
    hpo_params_path: Optional[Path] = None,
    eval_source: str = "sweep",
    n_runs: Optional[int] = None,
    report_seeds: Optional[list[int]] = None,
) -> dict[str, Any]:
    cfg = TASK_CONFIGS[task]()
    hidden_size = hidden_size_for_mult(baseline_hidden_size, size_mult)
    n_hidden = hidden_size * n_hidden_layers
    n_steps = int(cfg.extra.get("n_steps", 0))

    metrics = eval_metrics_for_dir(
        eval_log_dir,
        n_runs=n_runs,
        report_seeds=report_seeds,
    )

    return {
        "task": task,
        "neuron": neuron,
        "size_mult": float(size_mult),
        "hidden_size": hidden_size,
        "n_hidden_layers": n_hidden_layers,
        "n_hidden_neurons": n_hidden,
        "n_timesteps": n_steps,
        "eval_log_dir": str(eval_log_dir.resolve()),
        "eval_source": eval_source,
        "hpo_params_path": str(hpo_params_path.resolve()) if hpo_params_path else None,
        **metrics,
    }


def write_sweep_summary(
    records: list[dict[str, Any]],
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": REPORT_CKPT,
        "points": records,
    }
    yaml_path = out_dir / "sweep_summary.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    csv_path = out_dir / "sweep_summary.csv"
    fieldnames = [
        "task", "neuron", "size_mult", "hidden_size", "n_hidden_neurons",
        "n_runs", "seeds_used", "test_acc_mean", "test_acc_std",
        "test_spike_rate_mean", "test_spike_rate_std",
        "test_set_size", "test_neuron_timesteps",
        "energy_idle_pj", "energy_spike_pj",
        "energy_test_set_uj_mean", "energy_test_set_uj_std",
        "eval_source", "eval_log_dir",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            row_out = dict(row)
            seeds = row_out.get("seeds_used")
            if isinstance(seeds, list):
                row_out["seeds_used"] = ",".join(str(s) for s in seeds)
            writer.writerow(row_out)
    return yaml_path, csv_path
