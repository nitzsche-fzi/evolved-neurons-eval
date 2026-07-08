#!/usr/bin/env bash
# Run neuron HPO for neurons sequentially on one task/GPU.
#
# Usage:
#   ./scripts/run_hpo_queue.sh <task> <device>
#
# Examples:
#   ./scripts/run_hpo_queue.sh shd 3
#   ./scripts/run_hpo_queue.sh dvs 0
#
# Optional env overrides:
#   HPO_STORAGE=sqlite:///...  # default: results/hpo/optuna.db
#   HPO_N_TRIALS=50  HPO_N_EPOCHS=200  HPO_SEED=42
#   HPO_PRUNING_WARMUP=50  HPO_PRUNING_STARTUP=20
#   CONTINUE_ON_ERROR=1   # do not stop the queue after a failed neuron
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

STORAGE="${HPO_STORAGE:-sqlite:///${ROOT_DIR}/results/hpo/optuna.db}"
SEED="${HPO_SEED:-42}"
N_TRIALS="${HPO_N_TRIALS:-250}"

# Neuron order (fixed)
#NEURONS=(n1d3 esn_lifbox n2d2 n1d2 n2d1 n3d1 n3d2)

usage() {
  echo "Usage: $0 <task> <device_number>" >&2
  echo "  task:   shd | dvs | braille" >&2
  echo "  device: GPU index, e.g. 3  ->  hpo_tune --devices [3]" >&2
  exit 1
}

if [[ $# -lt 2 ]]; then
  usage
fi

TASK="$1"
DEVICE="$2"
DEVICES="[${DEVICE}]"

case "${TASK}" in
  shd|dvs|braille) ;;
  *)
    echo "Unknown task: ${TASK}" >&2
    usage
    ;;
esac

# Task-specific defaults (override with HPO_N_EPOCHS / HPO_PRUNING_WARMUP if needed)
case "${TASK}" in
  shd)
    N_EPOCHS="${HPO_N_EPOCHS:-100}"
    PRUNING_WARMUP="${HPO_PRUNING_WARMUP:-50}"
    TASK_NAME=${TASK}
    OBJECTIVE="val_comp"
    NEURONS=(n2d2 n1d1 n3d1 esn_lifbox)
    ;;
  dvs)
    N_EPOCHS="${HPO_N_EPOCHS:-175}"
    PRUNING_WARMUP="${HPO_PRUNING_WARMUP:-50}"
    TASK_NAME="dvsgesture"
    OBJECTIVE="val_comp"
    NEURONS=(esn_lifbox)
    ;;
  braille)
    N_EPOCHS="${HPO_N_EPOCHS:-225}"
    PRUNING_WARMUP="${HPO_PRUNING_WARMUP:-50}"
    TASK_NAME=${TASK}
    OBJECTIVE="val_comp"
    NEURONS=(n2d2 n1d1 n3d1 esn_lifbox)
    ;;
esac
PRUNING_STARTUP="${HPO_PRUNING_STARTUP:-20}"

LOG_DIR="${ROOT_DIR}/results/hpo/logs/queue_${TASK}_gpu${DEVICE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "[queue] task=${TASK}  device=${DEVICES}  storage=${STORAGE}"
echo "[queue] n_trials=${N_TRIALS}  n_epochs=${N_EPOCHS}  pruning_warmup=${PRUNING_WARMUP}"
echo "[queue] neurons: ${NEURONS[*]}"
echo "[queue] logs: ${LOG_DIR}"
echo

FAILED=()
for neuron in "${NEURONS[@]}"; do
  study="${TASK}-${neuron}-final"
  log="${LOG_DIR}/${neuron}.log"
  echo "========== $(date -Is)  START  ${neuron}  study=${study} ==========" | tee "${log}"

  set +e
  python3 hpo_tune.py \
    --storage "${STORAGE}" \
    --study-name "${study}" \
    --task "${TASK_NAME}" \
    --neuron "${neuron}" \
    --search-mode neuron \
    --objective ${OBJECTIVE} \
    --n-trials "${N_TRIALS}" \
    --n-epochs "${N_EPOCHS}" \
    --pruning-warmup-steps "${PRUNING_WARMUP}" \
    --pruning-startup-trials "${PRUNING_STARTUP}" \
    --seed "${SEED}" \
    --devices "${DEVICES}" \
    2>&1 | tee -a "${log}"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ ${rc} -ne 0 ]]; then
    echo "========== $(date -Is)  FAILED ${neuron}  exit=${rc} ==========" | tee -a "${log}"
    FAILED+=("${neuron}")
    if [[ "${CONTINUE_ON_ERROR:-0}" != "1" ]]; then
      echo "[queue] Stopping (set CONTINUE_ON_ERROR=1 to run remaining neurons)." >&2
      exit "${rc}"
    fi
  else
    echo "========== $(date -Is)  DONE   ${neuron} ==========" | tee -a "${log}"
  fi
  echo
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "[queue] Finished with failures: ${FAILED[*]}" >&2
  exit 1
fi

echo "[queue] All ${#NEURONS[@]} runs completed successfully."
