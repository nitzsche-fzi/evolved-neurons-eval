#!/usr/bin/env python3
"""Run network-size sweeps from a baseline HPO configuration.

For each requested size multiplier, this script can fine-tune the learning
rate and spike-rate regularization, run repeated evaluations, and write a
compact sweep summary with accuracy, spike-rate, and energy metrics.

Use ``--mode`` to run the full workflow (``all``), only fine-tune (``tune``),
only evaluate existing configurations (``eval``), or only rebuild summaries
from existing evaluation runs (``summarize``).

Examples:
    python3 size_sweep.py --task dvsgesture --neuron n1d1 \\
        --hpo-params results/hpo/n1d1/dvsgesture/n1d1-dvsgesture/best_params.yaml \\
        --baseline-eval-dir results/eval/n1d1/dvsgesture/final \\
        --mode all --n-runs 1 --devices [0]

    python3 size_sweep.py ... --mode all --n-runs 3 --devices [0]

    python3 size_sweep.py ... --mode summarize --n-runs 3
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
import yaml

from src.data.task_config import TASK_CONFIGS
from src.utils.aggregate_runs import aggregate_runs
from src.utils.energy import get_supported_neurons
from src.hpo.hpo_params import load_hpo_yaml
from src.utils.sweep_aggregate import (
    build_size_point_record,
    hidden_size_for_mult,
    mult_dir_name,
    write_sweep_summary,
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

HERE = Path(__file__).resolve().parent
HPO_SCRIPT = HERE / "hpo_tune.py"
RUN_EVAL_SCRIPT = HERE / "run_eval.py"
DEFAULT_SIZE_MULTS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
MODES = ("all", "tune", "eval", "summarize")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--task", choices=sorted(TASK_CONFIGS), required=True)
    p.add_argument("--neuron", choices=get_supported_neurons(), required=True)
    p.add_argument("--hpo-params", type=str, required=True, help="best_params.yaml from full neuron HPO (parent for size fine-tune).")
    p.add_argument("--mode", choices=MODES, default="all", help="Pipeline stages to run.")
    p.add_argument("--baseline-eval-dir", type=str, default=None, help="Existing run_eval log dir for 1×.")
    p.add_argument("--output-dir", type=str, default=None, help="Sweep output root (default: results/size_sweep/<task>/<neuron>).")
    p.add_argument("--dataset-path", type=str, default=None)

    p.add_argument("--size-multipliers", type=float, nargs="+", default=list(DEFAULT_SIZE_MULTS), metavar="MULT")
    p.add_argument("--baseline-hidden-size", type=int, default=256)
    p.add_argument("--n-hidden-layers", type=int, default=2)

    p.add_argument("--n-runs", type=int, default=3, help="Target count of eval runs per size.")
    p.add_argument("--seed-start", type=int, default=None, help="Base seed for run_eval (default: 1).")

    p.add_argument("--storage", type=str, default=None)
    p.add_argument("--fine-tune-trials", type=int, default=20)
    p.add_argument("--fine-tune-epochs", type=int, default=150)

    p.add_argument("--devices", type=str, default="1")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--stop-on-error", action="store_true")
    args = p.parse_args()

    if args.mode == "eval" and args.n_runs <= 0:
        p.error("--mode eval requires --n-runs > 0")
    return args


def _do_tune(mode: str) -> bool:
    return mode in ("all", "tune")


def _do_eval(mode: str) -> bool:
    return mode in ("all", "eval")


def _do_summarize(mode: str) -> bool:
    return mode in ("all", "eval", "summarize")


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    return (HERE / "results" / "size_sweep" / args.task / args.neuron).resolve()


def _run_cmd(cmd: list[str], *, cwd: Path) -> int:
    print("[size_sweep] $", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(cwd), check=False).returncode


def _hpo_params_for_mult(
    args: argparse.Namespace,
    out_root: Path,
    mult: float,
) -> Path:
    if mult == 1.0:
        return Path(args.hpo_params).expanduser().resolve()

    mult_root = out_root / mult_dir_name(mult)
    hpo_dir = mult_root / "hpo"
    best_path = hpo_dir / "best_params.yaml"
    if best_path.is_file():
        print(f"[size_sweep] Reusing fine-tune params: {best_path}")
        return best_path

    if not _do_tune(args.mode):
        if not best_path.is_file():
            raise FileNotFoundError(
                f"Missing fine-tuned params for mult={mult:g}: {best_path}. "
                f"Run with --mode tune or --mode all first."
            )
        return best_path

    hpo_dir.mkdir(parents=True, exist_ok=True)
    hidden_size = hidden_size_for_mult(args.baseline_hidden_size, mult)
    storage = args.storage
    if storage is None:
        db_path = out_root / "optuna" / f"size_sweep_{args.neuron}_{args.task}.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{db_path.as_posix()}"

    cmd = [
        sys.executable,
        str(HPO_SCRIPT),
        "--search-mode", "size_adapt",
        "--task", args.task,
        "--neuron", args.neuron,
        "--baseline-params", str(Path(args.hpo_params).expanduser().resolve()),
        "--size-mult", str(mult),
        "--hidden-size", str(hidden_size),
        "--n-hidden-layers", str(args.n_hidden_layers),
        "--n-trials", str(args.fine_tune_trials),
        "--n-epochs", str(args.fine_tune_epochs),
        "--output-dir", str(hpo_dir),
        "--storage", storage,
        "--devices", args.devices,
        "--num-workers", str(args.num_workers),
        "--lr-scheduler", "cosine",
    ]
    if args.dataset_path is not None:
        cmd.extend(["--dataset-path", args.dataset_path])
    rc = _run_cmd(cmd, cwd=HERE)
    if rc != 0:
        raise RuntimeError(f"size_adapt HPO failed for mult={mult:g} (exit {rc})")
    if not best_path.is_file():
        raise FileNotFoundError(f"HPO finished but {best_path} was not written.")
    return best_path


def _completed_eval_runs(eval_dir: Path) -> int:
    return int(aggregate_runs(eval_dir).get("n_runs", 0))


def _eval_log_dir(args: argparse.Namespace, out_root: Path, mult: float) -> tuple[Path, str]:
    if mult == 1.0 and args.baseline_eval_dir:
        path = Path(args.baseline_eval_dir).expanduser().resolve()
        return path, "baseline_eval"
    return out_root / mult_dir_name(mult) / "eval", "sweep_eval"


def _run_eval_for_mult(
    args: argparse.Namespace,
    out_root: Path,
    mult: float,
    hpo_params_path: Path,
) -> None:
    if not _do_eval(args.mode) or args.n_runs <= 0:
        return

    eval_dir, _ = _eval_log_dir(args, out_root, mult)
    eval_dir.mkdir(parents=True, exist_ok=True)
    hidden_size = hidden_size_for_mult(args.baseline_hidden_size, mult)

    cmd = [
        sys.executable,
        str(RUN_EVAL_SCRIPT),
        "--task", args.task,
        "--neuron", args.neuron,
        "--log-dir", str(eval_dir),
        "--hpo-params", str(hpo_params_path),
        "--hidden-size", str(hidden_size),
        "--n-hidden-layers", str(args.n_hidden_layers),
        "--n-runs", str(args.n_runs),
        "--devices", args.devices,
        "--num-workers", str(args.num_workers),
    ]
    if args.dataset_path is not None:
        cmd.extend(["--dataset-path", args.dataset_path])
    if args.seed_start is not None:
        cmd.extend(["--seed-start", str(args.seed_start)])
    if args.stop_on_error:
        cmd.append("--stop-on-error")

    rc = _run_cmd(cmd, cwd=HERE)
    if rc != 0:
        raise RuntimeError(f"run_eval failed for mult={mult:g} (exit {rc})")


def _write_results(out_root: Path, payload: dict) -> None:
    path = out_root / "size_sweep_results.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def main() -> None:
    args = parse_args()
    out_root = _resolve_output_dir(args)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[size_sweep] mode={args.mode}  output={out_root}")

    baseline_hpo = Path(args.hpo_params).expanduser().resolve()
    if not baseline_hpo.is_file():
        raise SystemExit(f"--hpo-params not found: {baseline_hpo}")

    baseline_data = load_hpo_yaml(baseline_hpo)

    result_points: list[dict] = []

    for mult in args.size_multipliers:
        print(f"\n{'=' * 60}\n[size_sweep] mult={mult:g}\n{'=' * 60}")
        hpo_path = _hpo_params_for_mult(args, out_root, mult)
        eval_dir, eval_source = _eval_log_dir(args, out_root, mult)

        _run_eval_for_mult(args, out_root, mult, hpo_path)

        result_points.append({
            "size_mult": mult,
            "mult_dir": mult_dir_name(mult),
            "hidden_size": hidden_size_for_mult(args.baseline_hidden_size, mult),
            "hpo_params": str(hpo_path),
            "eval_log_dir": str(eval_dir),
            "eval_source": eval_source,
            "eval_runs_completed": _completed_eval_runs(eval_dir),
            "eval_target_runs": args.n_runs if _do_eval(args.mode) else None,
        })

    results = {
        "task": args.task,
        "neuron": args.neuron,
        "mode": args.mode,
        "baseline_hpo_params": str(baseline_hpo),
        "baseline_eval_dir": str(Path(args.baseline_eval_dir).resolve())
        if args.baseline_eval_dir
        else None,
        "baseline_hidden_size": args.baseline_hidden_size,
        "size_multipliers": list(args.size_multipliers),
        "baseline_hpo_n_epochs": baseline_data.get("n_epochs"),
        "points": result_points,
    }
    _write_results(out_root, results)
    print(f"[size_sweep] Results -> {out_root / 'size_sweep_results.yaml'}")

    if not _do_summarize(args.mode):
        return

    records = [
        build_size_point_record(
            task=args.task,
            neuron=args.neuron,
            size_mult=pt["size_mult"],
            baseline_hidden_size=args.baseline_hidden_size,
            n_hidden_layers=args.n_hidden_layers,
            eval_log_dir=Path(pt["eval_log_dir"]),
            hpo_params_path=Path(pt["hpo_params"]),
            eval_source=pt["eval_source"],
            n_runs=args.n_runs,
        )
        for pt in result_points
    ]
    for pt, rec in zip(result_points, records):
        pt["n_runs_reported"] = rec["n_runs"]
        pt["seeds_used"] = rec.get("seeds_used")

    yaml_path, csv_path = write_sweep_summary(records, out_root)
    _write_results(out_root, results)
    print(f"[size_sweep] Wrote {yaml_path}")
    print(f"[size_sweep] Wrote {csv_path}")


if __name__ == "__main__":
    main()
