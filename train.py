"""Train a Norse spiking neural network with neurons from `evolved-spiking-neurons` or Norse.

The network mirrors the dense architecture used in the genetic algorithm of
`evolving-spiking-neurons`: an input projection, a configurable number of equally
sized hidden spiking layers, and a non-spiking integrator readout. Tasks match
the GA configs (SHD and DVSGesture) and use the same tonic datasets and esn
augmentations.

Supported neurons (`--neuron`):
    * esn   : n1d1, n1d2, n1d3, n2d1, n2d2, n2d3, n3d1, n3d2, esn_lifbox
    * norse : norse_lifbox, norse_lif

Supported readouts (`--readout`):
    * esn_integrator (default) - the non-spiking Integrator from esn
    * norse_li                 - Norse LICell (leaky integrator with exp. decay)
    * norse_libox              - Norse LIBoxCell (box-shaped leaky integrator)

Example:
    python3 train.py --task shd --neuron n1d1
    python3 train.py --task shd --neuron norse_lifbox --readout norse_li
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import lightning.pytorch as pl
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint

from src.data.datasets import build_datamodule
from src.data.task_config import TaskConfig, TASK_CONFIGS
from src.utils.metrics import ValMetricsSummary, TestMetricsSummary
from src.utils.helpers import parse_devices_arg
from src.hpo.hpo_params import apply_hpo_params_to_cfg, load_hpo_yaml
from src.models.network import NEURON_SPECS, READOUT_FACTORIES, DenseSNN

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=sorted(TASK_CONFIGS), default="shd")
    p.add_argument("--neuron", choices=sorted(NEURON_SPECS), default="n1d1")
    p.add_argument("--readout", choices=sorted(READOUT_FACTORIES), default="esn_integrator")
    p.add_argument("--hidden-size", type=int, default=None, help="neurons per hidden layer (default: task-specific)")
    p.add_argument("--n-hidden-layers", type=int, default=2, help="number of spiking hidden layers (default 2)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--n-epochs", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--grad-clip-norm", type=float, default=None)
    p.add_argument("--spike-rate-reg", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lr-scheduler", choices=("none", "cosine", "warmup_cosine"), default="cosine")
    p.add_argument("--lr-warmup-epochs", type=int, default=None)
    p.add_argument("--lr-warmup-start-factor", type=float, default=None)
    p.add_argument("--lr-min-factor", type=float, default=None)
    p.add_argument("--surrogate-method", type=str, default=None)
    p.add_argument("--surrogate-alpha", type=float, default=None)
    p.add_argument("--dataset-path", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=str, default="1", help="Either 1 (single GPU) or [N] (GPU index).")
    p.add_argument("--log-dir", type=str, default=None)
    p.add_argument("--ckpt", type=str, default=None, help="Path to a Lightning checkpoint (.ckpt) to resume training from.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise", type=float, default=None, help="Augmentation noise. Use 0 to disable and None for defaults.")
    p.add_argument("--hpo-params", type=str, default=None,
        help="Path to hpo_tune.py best_params.yaml (applies params before other CLI overrides).",
    )
    return p.parse_args()


def apply_cli_overrides(cfg: TaskConfig, args: argparse.Namespace) -> TaskConfig:
    for attr in ("batch_size", "n_epochs", "lr", "weight_decay", "grad_clip_norm",
                 "spike_rate_reg", "lr_scheduler", "lr_warmup_epochs",
                 "lr_warmup_start_factor", "lr_min_factor",
                 "surrogate_method", "surrogate_alpha", "dataset_path"):
        cli_val = getattr(args, attr, None)
        if cli_val is not None:
            setattr(cfg, attr, cli_val)
    if getattr(args, "noise", None) is not None:
        cfg.extra["noise"] = args.noise
    return cfg


def build_model(cfg: TaskConfig, args: argparse.Namespace) -> DenseSNN:
    """Construct the DenseSNN with hidden size resolved from args/cfg defaults."""
    hidden_size = args.hidden_size if args.hidden_size is not None else cfg.default_hidden_size
    return DenseSNN(
        cfg=cfg,
        neuron_name=args.neuron,
        hidden_size=hidden_size,
        n_hidden_layers=args.n_hidden_layers,
        readout=args.readout,
    )


def build_trainer_callbacks(neuron_name: str) -> tuple[list, ModelCheckpoint, ModelCheckpoint]:
    """Standard training callbacks plus references to the two best-ckpt callbacks."""
    ckpt_val_loss = ModelCheckpoint(save_top_k=1, monitor="val_loss", filename=neuron_name + "-best-loss")
    ckpt_val_acc = ModelCheckpoint(save_top_k=1, monitor="val_acc", mode="max", filename=neuron_name + "-best-acc")
    callbacks = [
        ckpt_val_loss,
        ckpt_val_acc,
        ModelCheckpoint(save_top_k=1, monitor=None, every_n_epochs=1, filename=neuron_name + "-last"),
        ValMetricsSummary(filename="best_loss_metrics.yaml", monitor="val_loss", mode="min"),
        ValMetricsSummary(filename="best_acc_metrics.yaml", monitor="val_acc", mode="max"),
        TestMetricsSummary(filename="best_loss_metrics.yaml", checkpoint_name="best-loss"),
        TestMetricsSummary(filename="best_acc_metrics.yaml", checkpoint_name="best-acc"),
    ]
    return callbacks, ckpt_val_loss, ckpt_val_acc


def build_trainer(
    cfg: TaskConfig,
    output_dir,
    callbacks: list,
    accelerator: str,
    devices: object,
) -> pl.Trainer:
    return pl.Trainer(
        max_epochs=cfg.n_epochs,
        accelerator=accelerator,
        devices=devices,
        default_root_dir=output_dir,
        callbacks=callbacks,
        log_every_n_steps=10,
        gradient_clip_val=None,  # handled in configure_gradient_clipping
    )


def write_run_info(trainer: pl.Trainer, args: argparse.Namespace) -> None:
    if not trainer.is_global_zero or trainer.log_dir is None:
        return
    log_dir = Path(trainer.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
    }
    with open(log_dir / "run_info.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=True)


def main() -> None:
    args = parse_args()
    try:
        args.devices = parse_devices_arg(args.devices)
    except ValueError as exc:
        raise SystemExit(f"Invalid --devices value '{args.devices}': {exc}") from exc
    pl.seed_everything(args.seed, workers=True)
    
    output_dir = args.log_dir
    if output_dir is None:
        script_dir = Path(__file__).resolve().parent
        output_dir = script_dir / "results" / args.neuron / args.readout / args.task
        output_dir.mkdir(parents=True, exist_ok=True)

    cfg = TASK_CONFIGS[args.task]()
    if args.hpo_params is not None:
        hpo_data = load_hpo_yaml(args.hpo_params)
        cfg = apply_hpo_params_to_cfg(cfg, hpo_data.get("params", {}))
        if args.n_epochs is None and hpo_data.get("n_epochs") is not None:
            cfg.n_epochs = int(hpo_data["n_epochs"])
    cfg = apply_cli_overrides(cfg, args)

    model = build_model(cfg, args)
    dm = build_datamodule(
        cfg,
        num_workers=args.num_workers,
    )

    accelerator, devices = args.accelerator, args.devices
    callbacks, ckpt_val_loss, ckpt_val_acc = build_trainer_callbacks(args.neuron)
    trainer = build_trainer(cfg, output_dir, callbacks, accelerator, devices)
    write_run_info(trainer, args)

    ckpt_path = args.ckpt
    if ckpt_path is not None:
        ckpt_path = str(Path(ckpt_path).resolve())
        print(f"Resuming training from checkpoint: {ckpt_path}")

    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)

    best_loss_path = ckpt_val_loss.best_model_path
    best_acc_path  = ckpt_val_acc.best_model_path
    trainer.test(datamodule=dm, ckpt_path=best_loss_path, weights_only=False)
    trainer.test(datamodule=dm, ckpt_path=best_acc_path, weights_only=False)


if __name__ == "__main__":
    main()
