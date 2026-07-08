"""Run `train.py` multiple times with varying seeds and aggregate results.

Use this to estimate the spread (mean / std) of final accuracy for a
given configuration. Each run is launched as a separate subprocess so
that CUDA state, RNG state and Python module state cannot leak between
runs.

``--n-runs`` is the target number of completed runs in ``<log_dir>``
(``lightning_logs/version_*`` folders with metrics). Before each run the
log directory is scanned; the next seed is ``--seed-start`` plus the
number of completed runs (plus in-flight claims when several processes
share the same ``--log-dir``). The loop stops once the target is met.

Examples:
    # Up to 20 completed runs (seeds 1..20 when starting empty)
    python3 run_eval.py --task shd --neuron n1d1 --n-runs 20 --devices [0]

    # Resume / parallelise: same command on multiple GPUs adds runs until 20 exist
    python3 run_eval.py --task dvsgesture --neuron n1d1 --log-dir results/eval/n1d1/dvsgesture/final \\
        --n-runs 20 --devices [0]

After every completed run the cumulative summaries are refreshed at
``<log_dir>/lightning_logs/summary.yaml`` (compact),
``summary_detailed.yaml`` (includes per-metric ``values`` lists), and
``summary.csv`` if ``--csv`` is set, so you can inspect partial results
at any time.
"""
from __future__ import annotations

import argparse
import fcntl
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from src.utils.aggregate_runs import (
    aggregate_runs,
    print_summary,
    write_summary,
    write_summary_csv,
)
from src.data.task_config import TASK_CONFIGS
from src.models.network import NEURON_SPECS
from src.hpo.hpo_params import load_hpo_yaml

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

HERE = Path(__file__).resolve().parent
TRAIN_SCRIPT = HERE / "train.py"
EVAL_READOUT = "esn_integrator"
LOCK_FILE = ".run_eval.lock"
CLAIMS_DIR = ".run_eval_claims"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse the multi-run options; everything else is forwarded to train.py."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = p.add_argument_group("multi-run options")
    g.add_argument("--n-runs", type=int, required=True, help="Target number of runs (parallel-safe).")
    g.add_argument("--seed-start", type=int, default=1, help="Base seed; number of already completed runs is added to every new run.")
    g.add_argument("--stop-on-error", action="store_true", help="Abort if any single run fails. Default: log failure and continue.")
    g.add_argument("--csv", action="store_true", help="Also write summary.csv after each run (default: yaml only).",)

    g2 = p.add_argument_group("training args needed up front (forwarded to train.py)")
    g2.add_argument("--task", choices=sorted(TASK_CONFIGS), default=None)
    g2.add_argument("--neuron", choices=sorted(NEURON_SPECS), default=None)
    g2.add_argument("--log-dir", type=str, default=None, help="Override default results path. Same semantics as train.py.",)
    g2.add_argument("--hpo-params", type=str, default=None,
        help="Path to an HPO `best_params.yaml` produced by hpo_tune.py. "
             "Any explicit CLI flags passed to run_eval.py override the YAML.",
    )

    args, passthrough = p.parse_known_args()
    return args, passthrough


def _load_hpo_params_flags(path: str, task: str, neuron: Optional[str]) -> tuple[list[str], Optional[str]]:
    p = Path(path).expanduser().resolve()
    data = load_hpo_yaml(p)
    yaml_task = data.get("task")
    yaml_neuron = data.get("neuron")
    if task and yaml_task and yaml_task != task:
        raise SystemExit(f"--hpo-params task mismatch: yaml={yaml_task} cli={task}")
    if neuron and yaml_neuron and yaml_neuron != neuron:
        raise SystemExit(f"--hpo-params neuron mismatch: yaml={yaml_neuron} cli={neuron}")

    return ["--hpo-params", str(p)], yaml_task, yaml_neuron


def resolve_log_dir(args: argparse.Namespace) -> Path:
    """Mirror train.py's default log dir resolution."""
    if args.log_dir is not None:
        return Path(args.log_dir).expanduser().resolve()
    return (HERE / "results" / "eval" / args.neuron / args.task).resolve()


def completed_run_count(log_dir: Path) -> int:
    """Number of version_* folders with aggregate-able metrics."""
    return int(aggregate_runs(log_dir).get("n_runs", 0))


@contextmanager
def _eval_dir_lock(log_dir: Path) -> Iterator[None]:
    log_dir.mkdir(parents=True, exist_ok=True)
    lock_path = log_dir / LOCK_FILE
    with open(lock_path, "w", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def _pending_claim_seeds(log_dir: Path) -> set[int]:
    claims_root = log_dir / CLAIMS_DIR
    if not claims_root.is_dir():
        return set()
    pending: set[int] = set()
    for path in claims_root.iterdir():
        if not path.name.startswith("seed_"):
            continue
        try:
            pending.add(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return pending


def _claim_path(log_dir: Path, seed: int) -> Path:
    return log_dir / CLAIMS_DIR / f"seed_{seed}"


def allocate_next_seed(log_dir: Path, *, seed_start: int, target_runs: int) -> Optional[int]:
    """Pick the next seed under lock, or None if the target is already met."""
    with _eval_dir_lock(log_dir):
        while True:
            n_done = completed_run_count(log_dir)
            if n_done >= target_runs:
                return None
            pending = _pending_claim_seeds(log_dir)
            if n_done + len(pending) >= target_runs:
                return None
            seed = seed_start + n_done + len(pending)
            claims_root = log_dir / CLAIMS_DIR
            claims_root.mkdir(parents=True, exist_ok=True)
            claim = _claim_path(log_dir, seed)
            try:
                claim.touch(exist_ok=False)
            except FileExistsError:
                continue
            return seed


def release_seed_claim(log_dir: Path, seed: int) -> None:
    _claim_path(log_dir, seed).unlink(missing_ok=True)


def build_train_command(
    seed: int,
    args: argparse.Namespace,
    passthrough: list[str],
    log_dir: Path,
) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--task", args.task,
        "--neuron", args.neuron,
        "--readout", EVAL_READOUT,
        "--log-dir", str(log_dir),
        "--seed", str(seed),
        *passthrough,
    ]
    return cmd


def refresh_summary(log_dir: Path, also_csv: bool) -> Optional[dict]:
    """Re-aggregate and persist the summary; safe to call after each run."""
    summary = aggregate_runs(log_dir)
    lightning_logs = log_dir / "lightning_logs"
    if summary["n_runs"] == 0 or not lightning_logs.is_dir():
        return summary
    write_summary(summary, lightning_logs)
    if also_csv:
        write_summary_csv(summary, lightning_logs)
    return summary


def main() -> None:
    args, passthrough = parse_args()
    assert args.neuron or args.hpo_params, "--neuron is required unless --hpo-params provides it."

    hpo_flags: list[str] = []
    if args.hpo_params:
        hpo_flags, yaml_task, yaml_neuron = _load_hpo_params_flags(args.hpo_params, args.task, args.neuron)
        if args.task is None:
            args.task = yaml_task
        if args.neuron is None:
            args.neuron = yaml_neuron

    log_dir = resolve_log_dir(args)
    log_dir.mkdir(parents=True, exist_ok=True)

    n_existing = completed_run_count(log_dir)
    print(f"[run_eval] log_dir:       {log_dir}")
    print(f"[run_eval] target_runs:   {args.n_runs}  (existing: {n_existing})")
    print(f"[run_eval] seed_start:    {args.seed_start}")
    if args.hpo_params:
        print(f"[run_eval] HPO params:    {Path(args.hpo_params).expanduser().resolve()}")
    forwarded = " ".join([*hpo_flags, *passthrough]) if (hpo_flags or passthrough) else "(none)"
    print(f"[run_eval] forwarded train args: {forwarded}\n")

    if n_existing >= args.n_runs:
        print(f"[run_eval] Target already met ({n_existing}/{args.n_runs}); nothing to do.")
        refresh_summary(log_dir, args.csv)
        return

    failures: list[tuple[int, int]] = []  # (seed, returncode)
    run_idx = 0
    while True:
        seed = allocate_next_seed(log_dir, seed_start=args.seed_start, target_runs=args.n_runs)
        if seed is None: break

        n_done = completed_run_count(log_dir)
        run_idx += 1
        print(
            f"\n=========== [run_eval] run {run_idx}  "
            f"seed={seed}  progress={n_done}/{args.n_runs} ==========="
        )
        cmd = build_train_command(seed, args, [*hpo_flags, *passthrough], log_dir)
        print("[run_eval] $", " ".join(cmd))
        try:
            result = subprocess.run(cmd, cwd=str(HERE), check=False)
        except KeyboardInterrupt:
            print("\n[run_eval] interrupted by user; aggregating completed runs and exiting.")
            break
        finally:
            release_seed_claim(log_dir, seed)

        if result.returncode != 0:
            failures.append((seed, result.returncode))
            print(f"[run_eval] seed={seed} FAILED with exit code {result.returncode}.")
            if args.stop_on_error:
                print("[run_eval] --stop-on-error set; aborting remaining runs.")
                break
            continue

        summary = refresh_summary(log_dir, args.csv)
        if summary is not None and summary["n_runs"] > 0:
            print_summary(summary)

        if completed_run_count(log_dir) >= args.n_runs:
            break

    print("\n=========== [run_eval] all done ===========")
    summary = refresh_summary(log_dir, args.csv)
    n_final = completed_run_count(log_dir)
    print(f"[run_eval] completed runs: {n_final}/{args.n_runs}")
    if summary is not None and n_final > 0:
        print_summary(summary)
        logs = (log_dir / "lightning_logs").resolve()
        print(f"\n[run_eval] summary          -> {logs / 'summary.yaml'}")
        print(f"[run_eval] summary_detailed -> {logs / 'summary_detailed.yaml'}")
    if failures:
        print(f"[run_eval] {len(failures)} run(s) failed: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()