"""Aggregate metrics across repeated training runs.

Each `train.py` invocation writes its outputs into a fresh
`<log_dir>/lightning_logs/version_<N>/` folder containing
`best_acc_metrics.yaml` and `best_loss_metrics.yaml` (and, when produced
by recent versions of `train.py`, a `run_info.yaml` recording the seed).

This module scans those per-version YAML files, computes mean / std /
min / max / median across runs for every numeric metric and writes
``summary.yaml`` (aggregate stats only, no per-run ``values`` lists),
``summary_detailed.yaml`` (full dump including ``values``), and optionally
``summary.csv`` into the ``lightning_logs/`` folder.

Usage:
    python3 -m src.utils.aggregate_runs <log_dir>
    python3 -m src.utils.aggregate_runs <log_dir> --csv

``<log_dir>`` accepts either the parent directory that contains
``lightning_logs/`` (e.g. ``results/n1d1/esn_integrator/shd``) or the
``lightning_logs`` directory itself.
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import yaml

from src.utils.energy import (
    energy_test_set_for_run,
    get_neuron_energy,
    pj_to_uj,
    resolve_run_energy_config,
)


# Metric files written by `train.py`. Each maps to the kind of checkpoint
# whose `test_*` metrics are stored in that file.
METRIC_FILES: dict[str, str] = {
    "best_acc": "best_acc_metrics.yaml",
    "best_loss": "best_loss_metrics.yaml",
}

# Display order for metrics inside each checkpoint block in summary.yaml.
METRIC_KEY_ORDER: tuple[str, ...] = (
    "test_acc",
    "test_spike_rate",
    "test_f1",
    "best_val_acc",
    "best_val_spike_rate",
)

ENERGY_CKPT = "best_acc"
TEST_ENERGY_KEY = "test_energy"


def _resolve_lightning_logs(log_dir: Path) -> Path:
    """Accept either ``.../lightning_logs`` or its parent."""
    log_dir = log_dir.expanduser().resolve()
    if log_dir.name == "lightning_logs":
        return log_dir
    candidate = log_dir / "lightning_logs"
    if candidate.is_dir():
        return candidate
    return log_dir  # fall through; caller will get an empty result


def _load_yaml(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def _flatten_run_metrics(payload: dict) -> dict[str, float]:
    """Extract scalar metrics from a single best_*_metrics.yaml payload.

    Produces flat keys like ``test_acc`` and ``best_val_acc`` so that runs
    with mismatching checkpoint files can still be merged sensibly.
    """
    flat: dict[str, float] = {}
    test = payload.get("test") if isinstance(payload, dict) else None
    if isinstance(test, dict):
        for k, v in test.items():
            # Keep the summary focused: accuracy and spike-rate only.
            # Drop cross-entropy and any loss values.
            k = str(k)
            if "ce" in k or "loss" in k:
                continue
            if k in ("test_spike_count", "test_n_samples", "test_neuron_timesteps"):
                continue
            try:
                flat[k] = float(v)
            except (TypeError, ValueError):
                continue
    best = payload.get("best") if isinstance(payload, dict) else None
    if isinstance(best, dict):
        for k, v in best.items():
            # Do not include best_epoch in the aggregate summary.
            k = str(k)
            if k == "epoch":
                continue
            # Drop CE/loss metrics.
            if "ce" in k or "loss" in k:
                continue
            # Drop validation F1 
            if k.startswith("val_") and "f1" in k:
                continue
            # Drop all train metrics.
            if k.startswith("train_"):
                continue
            try:
                flat[f"best_{k}"] = float(v)
            except (TypeError, ValueError):
                continue
    return flat


def _ordered_metric_keys(keys: Iterable[str]) -> list[str]:
    """Return keys in METRIC_KEY_ORDER, then any unknown keys alphabetically."""
    key_set = set(keys)
    ordered = [k for k in METRIC_KEY_ORDER if k in key_set]
    ordered.extend(sorted(key_set - set(METRIC_KEY_ORDER)))
    return ordered


def _stats(values: list[float]) -> dict[str, float]:
    """Sample statistics for a list of floats. Empty -> empty dict."""
    n = len(values)
    if n == 0:
        return {}
    mean = sum(values) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    sorted_vals = sorted(values)
    if n % 2 == 1:
        median = sorted_vals[n // 2]
    else:
        median = 0.5 * (sorted_vals[n // 2 - 1] + sorted_vals[n // 2])
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "median": median,
        "values": values,
    }


def _collect_run(version_dir: Path) -> Optional[dict]:
    """Read one ``version_*`` directory; return None if it has no metrics."""
    found_any = False
    per_ckpt: dict[str, dict[str, float]] = {}
    for ckpt_key, filename in METRIC_FILES.items():
        payload = _load_yaml(version_dir / filename)
        if payload is None:
            continue
        flat = _flatten_run_metrics(payload)
        if flat:
            per_ckpt[ckpt_key] = flat
            found_any = True
    if not found_any:
        return None

    info = _load_yaml(version_dir / "run_info.yaml") or {}
    seed = info.get("seed")
    timestamp = info.get("timestamp")

    return {
        "version": version_dir.name,
        "seed": seed,
        "timestamp": timestamp,
        "metrics": per_ckpt,
    }


def _collect_runs(lightning_logs: Path) -> list[dict]:
    if not lightning_logs.is_dir():
        return []
    versions = sorted(
        (p for p in lightning_logs.iterdir() if p.is_dir() and p.name.startswith("version_")),
        key=lambda p: int(p.name.split("_", 1)[1]) if p.name.split("_", 1)[1].isdigit() else -1,
    )
    runs = []
    for v in versions:
        run = _collect_run(v)
        if run is not None:
            runs.append(run)
    return runs


def _aggregate_per_ckpt(runs: list[dict], ckpt_key: str) -> dict[str, dict[str, float]]:
    """For one checkpoint kind, compute stats over every metric seen."""
    per_metric_values: dict[str, list[float]] = {}
    for r in runs:
        flat = r["metrics"].get(ckpt_key, {})
        for k, v in flat.items():
            per_metric_values.setdefault(k, []).append(v)
    stats = {k: _stats(v) for k, v in per_metric_values.items()}
    return {k: stats[k] for k in _ordered_metric_keys(stats)}


def _attach_run_energy(runs: list[dict], lightning_logs: Path) -> dict:
    """Compute per-run and aggregate hidden-network test-set energy from ``best_acc`` metrics."""
    if not runs:
        return {}

    config: Optional[dict] = None
    energy_values: list[float] = []
    for run in runs:
        version_dir = lightning_logs / run["version"]
        run_config = resolve_run_energy_config(version_dir / "run_info.yaml")
        if config is None:
            config = run_config
        metrics = run["metrics"].get(ENERGY_CKPT, {})
        energy_pj = energy_test_set_for_run(config=run_config, metrics=metrics)
        if energy_pj is not None:
            energy_uj = pj_to_uj(energy_pj)
            energy_values.append(energy_uj)
            run["metrics"].setdefault(ENERGY_CKPT, {})[TEST_ENERGY_KEY] = {
                "energy_test_set_uj": energy_uj,
            }

    if config is None:
        return {}

    energy_block: dict = {
        "hidden_size": config.get("hidden_size"),
        "n_hidden_layers": config.get("n_hidden_layers"),
        "n_hidden_neurons": config.get("n_hidden_neurons"),
        "n_timesteps": config.get("n_timesteps"),
        "test_set_size": config.get("test_set_size"),
        "test_neuron_timesteps": config.get("test_neuron_timesteps"),
    }
    neuron = config.get("neuron")
    if neuron is not None:
        try:
            constants = get_neuron_energy(str(neuron))
            energy_block["energy_idle_pj"] = constants.energy_idle
            energy_block["energy_spike_pj"] = constants.energy_spike
        except ValueError:
            pass
    energy_block["energy_test_set_uj"] = _stats(energy_values) if energy_values else None
    return energy_block


def next_seed_start(log_dir: Path) -> int:
    """Return max recorded seed + 1 (legacy helper).

    ``run_eval.py`` uses ``seed_start + completed_run_count`` instead; keep this
    for callers that need gap-aware continuation by seed value.
    """
    summary = aggregate_runs(log_dir)
    seeds = [int(s) for s in summary.get("seeds", []) if s is not None]
    return max(seeds) + 1 if seeds else 1


def aggregate_runs(log_dir: Path, *, seeds: Optional[list[int]] = None) -> dict:
    """Compute the multi-run summary for `<log_dir>/lightning_logs/`.

    If ``seeds`` is given, only runs whose ``run_info.yaml`` seed is in that list
    are included (useful for subsampling a fixed seed subset for reporting).
    """
    lightning_logs = _resolve_lightning_logs(log_dir)
    runs = _collect_runs(lightning_logs)
    if seeds is not None:
        seed_set = {int(s) for s in seeds}
        runs = [r for r in runs if r.get("seed") is not None and int(r["seed"]) in seed_set]

    summary: dict = {
        "log_dir": str(lightning_logs),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_runs": len(runs),
    }
    for ckpt_key in METRIC_FILES:
        summary[ckpt_key] = _aggregate_per_ckpt(runs, ckpt_key)
    energy_block = _attach_run_energy(runs, lightning_logs)
    if energy_block:
        summary.setdefault("best_acc", {})[TEST_ENERGY_KEY] = energy_block
    # Keep these near the end for readability in the YAML
    summary["versions"] = [r["version"] for r in runs]
    summary["seeds"] = [r["seed"] for r in runs]
    summary["runs"] = runs
    return summary


def _summary_without_values(summary: dict) -> dict:
    """Copy of ``summary`` with per-metric ``values`` lists removed."""
    out = copy.deepcopy(summary)
    for ckpt_key in METRIC_FILES:
        per_metric = out.get(ckpt_key)
        if not isinstance(per_metric, dict):
            continue
        for stats in per_metric.values():
            if isinstance(stats, dict):
                stats.pop("values", None)
    best_acc = out.get("best_acc")
    if isinstance(best_acc, dict):
        test_energy = best_acc.get(TEST_ENERGY_KEY)
        if isinstance(test_energy, dict):
            per_sample = test_energy.get("energy_test_set_uj")
            if isinstance(per_sample, dict):
                per_sample.pop("values", None)
    return out


def write_summary(summary: dict, out_dir: Path) -> tuple[Path, Path]:
    """Write ``summary.yaml`` (compact) and ``summary_detailed.yaml`` (full)."""
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = out_dir / "summary_detailed.yaml"
    compact_path = out_dir / "summary.yaml"
    with open(detailed_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    with open(compact_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_summary_without_values(summary), f, sort_keys=False)
    return compact_path, detailed_path


def write_summary_csv(summary: dict, out_dir: Path) -> Path:
    """Wide CSV with one row per run for quick inspection in pandas/Excel."""
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.csv"
    runs: list[dict] = summary.get("runs", [])

    metric_keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for r in runs:
        for ckpt_key, flat in r["metrics"].items():
            for k in flat:
                key = (ckpt_key, k)
                if key not in seen:
                    seen.add(key)
                    metric_keys.append(key)

    fieldnames = ["version", "seed", "timestamp", "best_acc/test_energy/energy_test_set_uj"] + [
        f"{c}/{m}" for c, m in metric_keys
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in runs:
            best_acc_test_energy = (r.get("metrics") or {}).get("best_acc", {}).get(TEST_ENERGY_KEY) or {}
            row = {
                "version": r.get("version"),
                "seed": r.get("seed"),
                "timestamp": r.get("timestamp"),
                "best_acc/test_energy/energy_test_set_uj": best_acc_test_energy.get("energy_test_set_uj", ""),
            }
            for ckpt_key, k in metric_keys:
                row[f"{ckpt_key}/{k}"] = r["metrics"].get(ckpt_key, {}).get(k, "")
            writer.writerow(row)
    return path


def _format_stat_line(name: str, stats: dict) -> str:
    return (
        f"  {name:<22} n={stats['n']:>3}  "
        f"mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
        f"min={stats['min']:.4f}  max={stats['max']:.4f}  "
        f"median={stats['median']:.4f}"
    )


def print_summary(summary: dict, metrics_to_print: Iterable[str] = ("test_acc", "best_val_acc")) -> None:
    """Pretty-print the headline stats. Intended for stdout."""
    print(f"\n[aggregate] log_dir   : {summary['log_dir']}")
    print(f"[aggregate] n_runs    : {summary['n_runs']}")
    if summary["n_runs"] == 0:
        print("[aggregate] No runs with metrics found.")
        return
    seeds = [s for s in summary["seeds"] if s is not None]
    if seeds:
        print(f"[aggregate] seeds     : {seeds}")
    for ckpt_key in METRIC_FILES:
        per_metric = summary.get(ckpt_key, {})
        if not per_metric:
            continue
        print(f"[aggregate] {ckpt_key} checkpoint:")
        for metric in metrics_to_print:
            stats = per_metric.get(metric)
            if stats:
                print(_format_stat_line(metric, stats))
    best_acc = summary.get("best_acc") or {}
    test_energy = best_acc.get(TEST_ENERGY_KEY) or {}
    energy_stats = test_energy.get("energy_test_set_uj")
    if isinstance(energy_stats, dict) and energy_stats.get("n", 0) > 0:
        print("[aggregate] best_acc.test_energy:")
        print(_format_stat_line("energy_test_set_uj", energy_stats))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "log_dir",
        type=str,
        help="Either a folder containing lightning_logs/ or the lightning_logs/ "
             "folder itself.",
    )
    p.add_argument("--csv", action="store_true", help="Also write summary.csv next to summary.yaml.")
    p.add_argument("--quiet", action="store_true", help="Suppress the printed summary.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    summary = aggregate_runs(log_dir)
    out_dir = _resolve_lightning_logs(log_dir)
    compact_path, detailed_path = write_summary(summary, out_dir)
    if args.csv:
        csv_path = write_summary_csv(summary, out_dir)
        print(f"[aggregate] wrote {csv_path}")
    if not args.quiet:
        print_summary(summary)
    print(f"[aggregate] wrote {compact_path}")
    print(f"[aggregate] wrote {detailed_path}")


if __name__ == "__main__":
    main()
