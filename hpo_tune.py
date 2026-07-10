"""Optuna-based hyperparameter optimisation.

Three search modes:
  * ``--search-mode neuron`` (default): tune training-side hparams
    (lr, weight_decay, spike_rate_reg, surrogate_alpha, scheduler params) plus
    the *temporal framing* (n_steps / dt) for a given (task, neuron) pair. The
    augmentation *magnitudes* listed in ``TaskConfig.extra["use_augmentations"]``
    are always co-tuned here (the dataset phase decides *which* augmentations
    help; their strength is a regulariser that trades off against the neuron
    hparams, so it is re-tuned per neuron). Framing lives here because the useful
    temporal resolution / sequence length depends on the neuron's dynamics; when
    it is sampled, the neuromorphic-init ``inp_mean`` / ``inp_var`` are
    recomputed for that framing (cached by framing).
  * ``--search-mode dataset``: tune the neuron-independent data-pipeline hparams
    in ``TaskConfig.extra`` (signal encoding such as Braille ``threshold``, plus
    augmentations) for a fixed neuron. Use this once per dataset, with a
    fast/stable neuron (e.g. ``esn_lifbox``), before doing per-neuron sweeps.
  * ``--search-mode size_adapt``: narrow fine-tune around a 1× ``best_params.yaml``
    (``--baseline-params``); only ``lr`` and ``spike_rate_reg`` are searched
    within ±2× of the baseline values. Used by ``size_sweep.py``.

Example (per-neuron):
    python3 hpo_tune.py --task shd --neuron n1d1 --seed 42 --devices [0]

Example (dataset, Braille):
    python3 hpo_tune.py --task braille --neuron esn_lifbox --search-mode dataset \
        --objective val_acc --cv-folds 5 --n-trials 80 --n-epochs 30 --devices [0]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.study import MaxTrialsCallback
from optuna.trial import FrozenTrial, TrialState
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import yaml

from compute_inp_stats import compute_inp_mean_var
from src.data.datasets import build_datamodule
from src.data.task_config import TASK_CONFIGS
from src.models.network import NEURON_SPECS
from src.utils.helpers import parse_devices_arg
from src.hpo.hpo_objective import selection_lambda_for, spike_energy_weight
from src.hpo.hpo_params import load_hpo_yaml
from src.hpo.hpo_search_space import suggest_hyperparameters
from src.utils.metrics import BestValTracker
from train import apply_cli_overrides, build_model

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

DEFAULT_STORAGE = "sqlite:///results/hpo/optuna.db"


# Keys that live in ``TaskConfig.extra`` (the data-pipeline dict) rather than as
# top-level ``TaskConfig`` attributes. When a trial samples one of these, the
# value is routed into ``cfg.extra[k]`` instead of ``setattr(cfg, k, v)``.
_EXTRA_KEYS: frozenset[str] = frozenset({
    # framing
    "dt",
    "n_steps",
    "threshold",
    # event-level augmentations
    "time_jitter",
    "event_dropping",
    "random_time_scale",
    # task-specific augmentations
    "noise",
    "random_start_offset",
    "random_image_scale",
    "random_image_offset",
})


# Framing knobs that change the first-layer input distribution. When a trial
# samples any of these (neuron mode now tunes the temporal framing, dataset mode
# still tunes Braille's `threshold`), the neuromorphic-init stats must be
# recomputed for that framing instead of reusing the committed task_config
# constants (which only match the default framing).
_FRAMING_KEYS: frozenset[str] = frozenset({"dt", "n_steps", "threshold", "desired_sensor_size"})

# (task, cv_folds, cv_fold_idx, framing) -> (inp_mean, inp_var). The stats depend
# only on framing, so caching keeps the cost to one pass per unique framing
# regardless of how many trials sample it.
_INP_STATS_CACHE: dict[tuple, tuple[float, float]] = {}


def _framing_cache_key(cfg, cv_folds: int, cv_fold_idx: int) -> tuple:
    framing = tuple(
        (k, tuple(v) if isinstance(v, (list, tuple)) else v)
        for k, v in sorted(cfg.extra.items())
        if k in _FRAMING_KEYS
    )
    return (cfg.name, int(cv_folds), int(cv_fold_idx), framing)


def _apply_recomputed_inp_stats(
    args: argparse.Namespace, cfg, cv_folds: int, cv_fold_idx: int
) -> None:
    """Set ``cfg.inp_mean`` / ``cfg.inp_var`` for the trial's framing (cached)."""
    key = _framing_cache_key(cfg, cv_folds, cv_fold_idx)
    if key not in _INP_STATS_CACHE:
        _INP_STATS_CACHE[key] = compute_inp_mean_var(
            cfg, num_workers=args.num_workers, cv_fold_idx=cv_fold_idx,
        )
    cfg.inp_mean, cfg.inp_var = _INP_STATS_CACHE[key]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=sorted(TASK_CONFIGS), default="dvsgesture")
    p.add_argument("--neuron", choices=sorted(NEURON_SPECS), required=True)
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--n-hidden-layers", type=int, default=2)
    p.add_argument("--study-name", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--dataset-path", type=str, default=None)

    p.add_argument("--search-mode", choices=("neuron", "dataset", "size_adapt"), default="neuron", help="What to optimise.")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--objective", choices=("val_acc", "val_comp"), default="val_comp", help="Optimisation target.")
    p.add_argument("--selection-lambda", type=float, default=None, help="λ for --objective val_comp (default 0.5).")
    # Optuna / lightning settings
    p.add_argument("--storage", type=str, default=os.environ.get("HPO_STORAGE", DEFAULT_STORAGE), 
        help=f"Optuna storage URL (default: $HPO_STORAGE or {DEFAULT_STORAGE}).",
    )
    p.add_argument("--devices", type=str, default="1", help="Either 1 (single GPU) or [N] (GPU index).")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--disable-pruning", action="store_true", help="Disable Optuna pruning callback and pruner.")
    p.add_argument("--pruning-warmup-steps", type=int, default=10, help="Epochs before pruning can stop trials.")
    p.add_argument("--pruning-startup-trials", type=int, default=10, help="Completed trials before pruning becomes active.")
    # Data and network settings
    p.add_argument("--cv-folds", type=int, default=0, help="If > 1, use k-fold cross validation. ")
    p.add_argument("--lr-scheduler", choices=("none", "cosine"), default="cosine")
    p.add_argument("--lr-min-factor", type=float, default=None)
    # For --search-mode size_adapt
    p.add_argument("--baseline-params", type=str, default=None, help="Path to base best_params.yaml (required for --search-mode size_adapt).")
    p.add_argument("--size-mult", type=float, default=None, help="Network size multiplier label for size_adapt study naming (e.g. 2.0).")
    p.add_argument("--adapt-range-min", type=float, default=0.5, help="Lower bound as a factor of the baseline lr / spike_rate_reg (size_adapt).")
    p.add_argument("--adapt-range-max", type=float, default=2.0, help="Upper bound as a factor of the baseline lr / spike_rate_reg (size_adapt).")

    return p.parse_args()


def _load_baseline_params(path: str) -> dict:
    data = load_hpo_yaml(path)
    params = data.get("params")
    if not isinstance(params, dict) or not params:
        raise ValueError(f"No 'params' dict in baseline YAML: {path}")
    return dict(params)


def _is_pruning_enabled(args: argparse.Namespace) -> bool:
    return not args.disable_pruning


def _build_pruner(args: argparse.Namespace) -> optuna.pruners.BasePruner:
    if not _is_pruning_enabled(args):
        return optuna.pruners.NopPruner()
    return optuna.pruners.MedianPruner(
        n_startup_trials=max(0, args.pruning_startup_trials),
        n_warmup_steps=max(0, args.pruning_warmup_steps),
    )


def _early_stopping_enabled(trial: optuna.Trial, args: argparse.Namespace) -> bool:
    """Match Optuna pruner: no early stopping until startup trials have completed."""
    n_complete = len(
        trial.study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    )
    return n_complete >= args.pruning_startup_trials


def _train_for_config(
    args: argparse.Namespace,
    cfg,
    trial_args: argparse.Namespace,
    accelerator: str,
    devices: object,
    monitor: str,
    mode: str,
    cv_folds: int,
    cv_fold_idx: int,
    trial: optuna.Trial,
    enable_pruning: bool = True,
    recompute_inp_stats: bool = False,
) -> float:
    """Train a model for the given (cfg, fold) and return its best validation score."""
    if recompute_inp_stats:
        _apply_recomputed_inp_stats(args, cfg, cv_folds, cv_fold_idx)
    model = build_model(cfg, trial_args)
    dm = build_datamodule(
        cfg, args.num_workers,
        cv_folds=cv_folds, cv_fold_idx=cv_fold_idx,
    )

    comp_lambda = None
    energy_weight = None
    if args.objective == "val_comp":
        comp_lambda = selection_lambda_for(args.selection_lambda)
        energy_weight = spike_energy_weight(args.neuron)
        monitor = "val_comp"
        mode = "max"
        extras = ("val_acc", "val_spike_rate", "val_loss")
    else:
        extras = ("val_spike_rate",)

    best_val = BestValTracker(monitor, mode, extras, comp_lambda, energy_weight)
    callbacks: list = [best_val]
    if _early_stopping_enabled(trial, args):
        callbacks.append(EarlyStopping(monitor=monitor, mode=mode, patience=50))
    if enable_pruning and _is_pruning_enabled(args):
        callbacks.append(PyTorchLightningPruningCallback(trial, monitor=monitor))

    trainer = pl.Trainer(
        max_epochs=cfg.n_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=5,
        gradient_clip_val=None,
    )
    trainer.fit(model, datamodule=dm)

    if best_val.best is None:
        raise optuna.TrialPruned(f"No '{monitor}' was logged during validation.")
    if best_val.best_epoch is not None:
        trial.set_user_attr("best_epoch", best_val.best_epoch)
    for k, v in best_val.best_extras.items():
        trial.set_user_attr(k, v)
    if comp_lambda is not None:
        trial.set_user_attr("selection_lambda", comp_lambda)
    if energy_weight is not None:
        trial.set_user_attr("spike_energy_weight", energy_weight)
    return float(best_val.best)


def _build_trial_cfg(args: argparse.Namespace, params: dict) -> object:
    """Apply HPO-sampled params on top of the CLI-overridden TaskConfig.

    Keys listed in ``_EXTRA_KEYS`` are written into ``cfg.extra`` (the data
    pipeline reads from there); everything else is treated as a top-level
    ``TaskConfig`` attribute.
    """
    cfg = TASK_CONFIGS[args.task]()
    cfg = apply_cli_overrides(cfg, args)
    cfg.n_epochs = args.n_epochs
    for k, v in params.items():
        if k in _EXTRA_KEYS:
            cfg.extra[k] = v
        else:
            setattr(cfg, k, v)
    return cfg


def make_objective(args: argparse.Namespace, accelerator: str, devices: object):
    monitor = "val_comp" if args.objective == "val_comp" else "val_acc"
    mode = "max"

    def objective(trial: optuna.Trial):
        pl.seed_everything(args.seed + trial.number, workers=True)

        sampled = suggest_hyperparameters(trial, args)
        if args.search_mode == "size_adapt":
            merged = {**args.baseline_params, **sampled}
        else:
            merged = sampled
        cfg = _build_trial_cfg(args, merged)
        # Recompute network init stats only when the trial actually changed
        # the framing; otherwise the committed task_config constants are correct.
        recompute_stats = bool(_FRAMING_KEYS & set(merged))

        if args.cv_folds <= 1:
            return _train_for_config(
                args, cfg, args, accelerator, devices, monitor, mode,
                cv_folds=0, cv_fold_idx=0, trial=trial,
                recompute_inp_stats=recompute_stats,
            )
        # K-fold averaged objective. Pruning only on fold 0.
        scores: list[float] = []
        for fold in range(args.cv_folds):
            scores.append(
                _train_for_config(
                    args, cfg, args, accelerator, devices, monitor, mode,
                    cv_folds=args.cv_folds, cv_fold_idx=fold, trial=trial,
                    enable_pruning=(fold == 0),
                    recompute_inp_stats=recompute_stats,
                )
            )
        return sum(scores) / len(scores)
    return objective, mode


def _prepare_storage_url(storage: Optional[str]) -> Optional[str]:
    """Prepare Optuna storage URL, creating SQLite parent dirs when needed."""
    if not storage or not storage.startswith("sqlite:///"):
        return storage

    raw = storage[len("sqlite:///"):]
    db_path = Path(raw).expanduser()
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.as_posix()}"


def main() -> None:
    args = parse_args()
    args.readout = "esn_integrator" # expected by train.py
    args.devices = parse_devices_arg(args.devices)

    if args.search_mode == "size_adapt":
        if not args.baseline_params:
            raise SystemExit("--baseline-params is required for --search-mode size_adapt.")
        if args.size_mult is None:
            raise SystemExit("--size-mult is required for --search-mode size_adapt.")
        args.baseline_params_path = args.baseline_params
        args.baseline_params = _load_baseline_params(args.baseline_params)

    objective, _ = make_objective(args, "auto", args.devices)

    study_name = _resolve_study_name(args)
    storage_url = _prepare_storage_url(args.storage)

    direction = "maximize"
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage_url,
        load_if_exists=True,
        pruner=_build_pruner(args),
    )

    callbacks = [MaxTrialsCallback(args.n_trials, states=(TrialState.COMPLETE, TrialState.PRUNED)),]
    study.optimize(objective, n_trials=None, callbacks=callbacks, catch=(ValueError,),)

    best = study.best_trial
    print("\n[HPO] Best trial:")
    print(f"  value  : {best.value}")
    print(f"  number : {best.number}")
    print(f"  params : {best.params}")

    export_best_params(args, study)
    
def _resolve_study_name(args: argparse.Namespace) -> str:
    if args.study_name:
        return args.study_name
    if args.search_mode == "dataset":
        return f"dataset-{args.task}-{args.neuron}"
    if args.search_mode == "size_adapt":
        mult_tag = _format_size_mult_tag(args.size_mult)
        return f"{args.neuron}-{args.task}-size{mult_tag}x"
    return f"{args.neuron}-{args.task}"

def _format_size_mult_tag(mult: float) -> str:
    return f"{mult:g}".replace(".", "p")

def export_best_params(args: argparse.Namespace, study: optuna.Study) -> None:
    best = study.best_trial
    payload = _get_best_optuna_params(args, best)
    if args.search_mode == "size_adapt":
        payload["params"] = {**args.baseline_params, **best.params}
        payload["baseline_params_path"] = str(Path(args.baseline_params_path).resolve())
        payload["size_mult"] = float(args.size_mult)
        payload["adapt_range"] = [float(args.adapt_range_min), float(args.adapt_range_max)]
    payload["direction"]  = study.direction.name
    payload["study_name"] = study.study_name

    _attach_inp_stats_to_payload(args, payload)
    _save_params(args, payload)


def _get_best_optuna_params(args: argparse.Namespace, best: FrozenTrial) -> dict:
    payload = {
        "task": args.task,
        "neuron": args.neuron,
        "objective": args.objective,
        "direction": None,
        "params": dict(best.params),
        "trial_number": best.number,
        "n_trials": args.n_trials,
        "n_epochs": args.n_epochs,
        "study_name": None,
        "cv_folds": args.cv_folds,
        "pruning": {
            "enabled": _is_pruning_enabled(args),
            "pruner": "median",
            "warmup_steps": args.pruning_warmup_steps,
            "startup_trials": args.pruning_startup_trials,
        },
        "value": best.value,
        "selection_source": "optuna_best",
    }
    if args.objective == "val_comp":
        payload["selection_lambda"] = selection_lambda_for(args.selection_lambda)
        payload["spike_energy_weight"] = spike_energy_weight(args.neuron)
        payload["selection_formula"] = (
            "val_acc - lambda * spike_energy_weight * val_spike_rate"
        )
    payload["params"]["lr_scheduler"] = args.lr_scheduler
    return payload


def _attach_inp_stats_to_payload(args: argparse.Namespace, payload: dict) -> None:
    """Compute and store network weight init stats for the best trial's framing.

    Written into ``params`` (and ``optuna_best.params`` when present) so
    ``train.py --hpo-params`` can apply them without recomputing.
    """
    cfg = _build_trial_cfg(args, payload["params"])
    mean, var = compute_inp_mean_var(
        cfg, num_workers=args.num_workers, cv_fold_idx=0,
    )
    payload["params"]["inp_mean"] = mean
    payload["params"]["inp_var"] = var
    print(
        f"[HPO] neuromorphic-init stats for saved framing "
        f"(n_steps={cfg.extra.get('n_steps')}, dt={cfg.extra.get('dt')}, "
        f"threshold={cfg.extra.get('threshold')}): "
        f"inp_mean={mean:.6g}, inp_var={var:.6g}"
    )

def _save_params(args: argparse.Namespace, payload: dict, filename: str = "best_params.yaml"):
    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=True)
    print(f"[HPO] Saved best params to {out_path}")

def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    here = Path(__file__).resolve().parent
    return here / "results" / "hpo" / args.neuron / args.task / _resolve_study_name(args)


if __name__ == "__main__":
    main()
