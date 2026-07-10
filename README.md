# ESN Evaluation
This repository contains the evaluation scripts used to train and compare evolved spiking neurons on event-based tasks, as well as found hyperparameter configurations and trained network checkpoints to reproduce the results reported in our publication.

The three main entry points are:
- `hpo_tune.py`: run Optuna hyperparameter optimization and write `best_params.yaml`.
- `run_eval.py`: run one configuration for multiple seeds and aggregate the results.
- `size_sweep.py`: evaluate a tuned neuron at different network sizes.

Additionally, `train.py` is the lower-level single-run script used by `run_eval.py`.

## Repository Overview
- `src/models/network.py`: dense SNN model, neuron registry, readouts.
- `src/data/`: task definitions and Lightning data modules.
- `src/hpo/`: Optuna search spaces and HPO parameter loading.
- `src/utils/`: metrics, energy accounting, aggregation helpers.
- `scripts/`: queue helpers for running batches of jobs.
- `experiments/configs/hpo/`: selected HPO parameter files used for evaluation in our publication.
- `experiments/checkpoints/`: lightning checkpoints with trained network per task and neuron with best accuracy, as used for evaluation in our publication.

Supported tasks:
- `shd`
- `dvsgesture`
- `braille`

Supported evolved-neuron names:
- `n1d1`, `n1d2`, `n1d3`
- `n2d1`, `n2d2`, `n2d3`
- `n3d1`, `n3d2`
- `esn_lifbox`

Norse baselines:
- `norse_lifbox`
- `norse_lif`

## Setup
Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This repository also expects two local packages to be importable:
- `esn`, from the companion evolved-spiking-neurons codebase: <https://anonymous.4open.science/r/spiking-neurons/>
- `braille_dataset`, from the companion Braille dataset repository: <https://anonymous.4open.science/r/spiking-braille/>

Please see those repositories for installation instructions. Make sure they are installed in the same environment.

## Datasets
All main scripts support passing a dataset cache/root directory with `--dataset-path`. SHD and DVSGesture are loaded through `tonic` and are stored under this path. The Braille task uses the external `braille_dataset` package and the same `--dataset-path` argument.

If you omit `--dataset-path`, the scripts use the task default from `src/data/task_config.py`.

## Quick Start
The usual workflow is:

1. Tune hyperparameters with `hpo_tune.py`.
2. Evaluate the tuned configuration with `run_eval.py`.
3. Optionally run a network-size sweep with `size_sweep.py`.

First, run a small Optuna search:

```bash
python hpo_tune.py \
  --task dvsgesture \
  --neuron n2d1 \
  --dataset-path data \
  --n-trials 30 \
  --n-epochs 50 \
  --devices "[0]"
```

Then evaluate the saved HPO configuration:

```bash
python run_eval.py \
  --task dvsgesture \
  --neuron n2d1 \
  --hpo-params results/hpo/n2d1/dvsgesture/n2d1-dvsgesture/best_params.yaml \
  --n-runs 3 \
  --dataset-path data \
  --devices "[0]"
```

`run_eval.py` forwards unknown training options to `train.py`, so arguments such as `--dataset-path`, `--hidden-size`, and `--n-epochs` can be passed through.

Evaluation writes Lightning logs, checkpoints, per-run metrics, and aggregate summaries under:

```text
results/eval/<neuron>/<task>/
```

unless `--log-dir` is set.
We suggest to use at least 10 eval runs (`--n-runs 10`) to get reliable means.

Finally, run a size sweep from the same HPO configuration:

```bash
python size_sweep.py \
  --task dvsgesture \
  --neuron n2d1 \
  --hpo-params results/hpo/n2d1/dvsgesture/n2d1-dvsgesture/best_params.yaml \
  --dataset-path data \
  --devices "[0]"
```

## Hyperparameter Optimization
Run an Optuna search for one task/neuron pair:

```bash
python hpo_tune.py \
  --task dvsgesture \
  --neuron n2d1 \
  --dataset-path data \
  --n-trials 30 \
  --n-epochs 50 \
  --devices "[0]"
```

By default, Optuna uses a local SQLite database:

```text
sqlite:///results/hpo/optuna.db
```

This path is relative to the working directory. Use `--storage` or the `HPO_STORAGE` environment variable to choose a different database.

The best trial parameters are written to:

```text
results/hpo/<neuron>/<task>/<study-name>/best_params.yaml
```

If no `--study-name` is supplied, the default study name is:

```text
<neuron>-<task>
```

You can evaluate a saved HPO configuration using:

```bash
python run_eval.py \
  --task dvsgesture \
  --neuron n2d1 \
  --hpo-params results/hpo/n2d1/dvsgesture/n2d1-dvsgesture/best_params.yaml \
  --n-runs 3 \
  --dataset-path data \
  --devices "[0]"
```

## Size Sweeps

After tuning a baseline configuration, run a size sweep:

```bash
python size_sweep.py \
  --task dvsgesture \
  --neuron n2d1 \
  --hpo-params results/hpo/n2d1/dvsgesture/n2d1-dvsgesture/best_params.yaml \
  --dataset-path data \
  --devices "[0]"
```

The sweep can fine-tune size-adapted hyperparameters, run repeated evaluations,
and write aggregate summaries under:

```text
results/size_sweep/<task>/<neuron>/
```

Useful modes:
- `--mode all`: fine-tune, evaluate, and summarize.
- `--mode tune`: only run size-adapted HPO.
- `--mode eval`: evaluate existing configurations.
- `--mode summarize`: rebuild summaries from existing evaluation logs.

## Reusing Provided HPO Parameters

The `experiments/configs/hpo/` directory contains selected `best_params.yaml` files. You can
pass one directly to `run_eval.py`:

```bash
python run_eval.py \
  --task dvsgesture \
  --neuron n2d2 \
  --hpo-params experiments/configs/hpo/dvsgesture/n2d2/best_params.yaml \
  --n-runs 3 \
  --dataset-path data \
  --devices "[0]"
```

## Notes
- `--devices 1` lets Lightning choose one available device. `--devices "[0]"`
  selects GPU index 0.
- `--num-workers` controls dataloader workers. Set it to `0` if multiprocessing
  causes issues on your platform.
- The code writes results into `results/` by default. This directory can become
  large during HPO and size sweeps, mainly due to lightning checkpoints.
