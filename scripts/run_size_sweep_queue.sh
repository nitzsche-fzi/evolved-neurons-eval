#!/usr/bin/env bash
# Run size_sweep.py for one multiplier across all neurons on one task/GPU.
# Intended for manual parallelism: launch one queue per (task, multiplier, GPU).
#
# Usage:
#   ./scripts/run_size_sweep_queue.sh <task> <device> <multiplier>
#
# Examples:
#   ./scripts/run_size_sweep_queue.sh dvs 0 2
#   ./scripts/run_size_sweep_queue.sh shd 3 0.5
#   ./scripts/run_size_sweep_queue.sh braille 1 1
#
# Optional env overrides:
#   SWEEP_MODE=all              # size_sweep --mode (all | tune | eval | summarize)
#   SWEEP_N_RUNS=3              # target completed eval runs per neuron
#   SWEEP_SEED_START=1
#   HPO_STUDY_SUFFIX=final      # HPO dir: results/hpo/<neuron>/<task>/<task>-<neuron>-<suffix>/
#   SWEEP_FINE_TUNE_TRIALS=20
#   SWEEP_FINE_TUNE_EPOCHS=150
#   SWEEP_STORAGE=sqlite:///... # passed to size_sweep --storage (≠1× tune only)
#   CONTINUE_ON_ERROR=1         # do not stop the queue after a failed neuron
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

SWEEP_MODE="${SWEEP_MODE:-all}"
SWEEP_N_RUNS="${SWEEP_N_RUNS:-3}"
SWEEP_SEED_START="${SWEEP_SEED_START:-1}"
STUDY_SUFFIX="${HPO_STUDY_SUFFIX:-final}"
FINE_TUNE_TRIALS="${SWEEP_FINE_TUNE_TRIALS:-20}"

usage() {
  echo "Usage: $0 <task> <device_number> <multiplier>" >&2
  echo "  task:       shd | dvs | braille" >&2
  echo "  device:     GPU index, e.g. 3  ->  size_sweep --devices [3]" >&2
  echo "  multiplier: size grid point, e.g. 0.5 | 1 | 2 | 8" >&2
  echo "  env: SWEEP_MODE SWEEP_N_RUNS SWEEP_SEED_START HPO_STUDY_SUFFIX CONTINUE_ON_ERROR=1" >&2
  exit 1
}

if [[ $# -lt 3 ]]; then
  usage
fi

TASK="$1"
DEVICE="$2"
MULT="$3"
DEVICES="[${DEVICE}]"

case "${TASK}" in
  shd|dvs|braille) ;;
  *)
    echo "Unknown task: ${TASK}" >&2
    usage
    ;;
esac

if ! awk -v m="${MULT}" 'BEGIN { if (m + 0 != m) exit 1 }'; then
  echo "Invalid multiplier: ${MULT}" >&2
  usage
fi

NEURONS=(esn_lifbox n2d2 n3d1)

case "${TASK}" in
  shd)
    TASK_NAME="${TASK}"
    FINE_TUNE_EPOCHS="${SWEEP_FINE_TUNE_EPOCHS:-100}"
    ;;
  dvs)
    TASK_NAME="dvsgesture"
    FINE_TUNE_EPOCHS="${SWEEP_FINE_TUNE_EPOCHS:-175}"
    ;;
  braille)
    TASK_NAME="${TASK}"
    FINE_TUNE_EPOCHS="${SWEEP_FINE_TUNE_EPOCHS:-225}"
    ;;
esac

MULT_TAG="$(echo "${MULT}" | tr '.' 'p')"
LOG_DIR="${ROOT_DIR}/results/size_sweep/logs/queue_${TASK}_mult${MULT_TAG}_gpu${DEVICE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
QUEUE_LOG="${LOG_DIR}/queue.log"

echo "[queue] task=${TASK}  task_name=${TASK_NAME}  mult=${MULT}  device=${DEVICES}"
echo "[queue] mode=${SWEEP_MODE}  n_runs=${SWEEP_N_RUNS}  seed_start=${SWEEP_SEED_START}"
echo "[queue] hpo_study=*-${STUDY_SUFFIX}  fine_tune_trials=${FINE_TUNE_TRIALS}"
echo "[queue] neurons: ${NEURONS[*]}"
echo "[queue] logs: ${LOG_DIR}"
echo

FAILED=()
SKIPPED=()
SUCCEEDED=()
for neuron in "${NEURONS[@]}"; do
  study="${TASK}-${neuron}-${STUDY_SUFFIX}"
  hpo_yaml="${ROOT_DIR}/results/hpo/${neuron}/${TASK_NAME}/${study}/best_params.yaml"
  if [[ ! -f "${hpo_yaml}" ]]; then
    msg="[queue] skip ${neuron}: missing ${hpo_yaml}"
    echo "${msg}"
    echo "$(date -Is)  ${msg}" >> "${QUEUE_LOG}"
    SKIPPED+=("${neuron}")
    continue
  fi

  log="${LOG_DIR}/${neuron}.log"
  echo "========== $(date -Is)  START  ${neuron}  mult=${MULT}  hpo=${hpo_yaml} ==========" | tee "${log}"

  cmd=(
    python3 size_sweep.py
    --task "${TASK_NAME}"
    --neuron "${neuron}"
    --hpo-params "${hpo_yaml}"
    --size-multipliers "${MULT}"
    --mode "${SWEEP_MODE}"
    --n-runs "${SWEEP_N_RUNS}"
    --seed-start "${SWEEP_SEED_START}"
    --devices "${DEVICES}"
    --fine-tune-trials "${FINE_TUNE_TRIALS}"
    --fine-tune-epochs "${FINE_TUNE_EPOCHS}"
  )

  case "${MULT}" in
    1|1.0|1.00)
      baseline_eval="${ROOT_DIR}/results/eval/${neuron}/${TASK_NAME}/final"
      cmd+=(--baseline-eval-dir "${baseline_eval}")
      ;;
  esac

  if [[ -n "${SWEEP_STORAGE:-}" ]]; then
    cmd+=(--storage "${SWEEP_STORAGE}")
  fi

  set +e
  "${cmd[@]}" 2>&1 | tee -a "${log}"
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
    SUCCEEDED+=("${neuron}")
  fi
  echo
done

echo "[queue] Summary:"
if [[ ${#SUCCEEDED[@]} -gt 0 ]]; then
  echo "[queue]   completed (${#SUCCEEDED[@]}): ${SUCCEEDED[*]}"
fi
if [[ ${#SKIPPED[@]} -gt 0 ]]; then
  echo "[queue]   skipped (${#SKIPPED[@]}, no HPO): ${SKIPPED[*]}"
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "[queue]   failed (${#FAILED[@]}): ${FAILED[*]}"
fi
{
  echo "$(date -Is)  summary mult=${MULT} completed=${#SUCCEEDED[@]} skipped=${#SKIPPED[@]} failed=${#FAILED[@]}"
  [[ ${#SUCCEEDED[@]} -gt 0 ]] && echo "  completed: ${SUCCEEDED[*]}"
  [[ ${#SKIPPED[@]} -gt 0 ]] && echo "  skipped: ${SKIPPED[*]}"
  [[ ${#FAILED[@]} -gt 0 ]] && echo "  failed: ${FAILED[*]}"
} >> "${QUEUE_LOG}"

if [[ ${#FAILED[@]} -gt 0 ]]; then
  exit 1
fi

if [[ ${#SUCCEEDED[@]} -eq 0 ]]; then
  echo "[queue] No size_sweep runs started."
  exit 0
fi

echo "[queue] All ${#SUCCEEDED[@]} size_sweep run(s) completed successfully."
