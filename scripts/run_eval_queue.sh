#!/usr/bin/env bash
# Run multi-seed eval (run_eval.py) per neuron sequentially on one task/GPU.
# Loads best_params.yaml from HPO output dirs (same study names as run_hpo_queue.sh).
#
# Usage:
#   ./scripts/run_eval_queue.sh <task> <device> <n-runs>
#
# Examples:
#   ./scripts/run_eval_queue.sh shd 3 20
#   ./scripts/run_eval_queue.sh dvs 0 20
#
# Optional env overrides:
#   EVAL_SEED_START=1        # base seed (default: 1)
#   HPO_STUDY_SUFFIX=final   # study dir: <task>-<neuron>-<suffix> (default: final)
#   CONTINUE_ON_ERROR=1      # do not stop the queue after a failed neuron
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

SEED_START="${EVAL_SEED_START:-1}"
STUDY_SUFFIX="${HPO_STUDY_SUFFIX:-final}"

usage() {
  echo "Usage: $0 <task> <device_number> <n-runs>" >&2
  echo "  task:   shd | dvs | braille" >&2
  echo "  device: GPU index, e.g. 3  ->  run_eval --devices [3]" >&2
  echo "  n-runs: target completed runs per neuron (parallel-safe)" >&2
  exit 1
}

if [[ $# -lt 3 ]]; then
  usage
fi

TASK="$1"
DEVICE="$2"
N_RUNS="$3"
DEVICES="[${DEVICE}]"

case "${TASK}" in
  shd|dvs|braille) ;;
  *)
    echo "Unknown task: ${TASK}" >&2
    usage
    ;;
esac

# Task-specific defaults
case "${TASK}" in
  shd)
    TASK_NAME=${TASK}
    NEURONS=(n2d2 n3d1 n1d1 esn_lifbox)
    ;;
  dvs)
    TASK_NAME="dvsgesture"
    NEURONS=(n2d2 n3d1 n1d1 esn_lifbox)
    ;;
  braille)
    TASK_NAME=${TASK}
    NEURONS=(n2d2 n3d1 n1d1 esn_lifbox)
    ;;
esac

LOG_DIR="${ROOT_DIR}/results/eval/logs/queue_${TASK}_gpu${DEVICE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
QUEUE_LOG="${LOG_DIR}/queue.log"

echo "[queue] task=${TASK}  task_name=${TASK_NAME}  device=${DEVICES}"
echo "[queue] n_runs=${N_RUNS}  seed_start=${SEED_START}  hpo_study=*-${STUDY_SUFFIX}"
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

  echo "========== $(date -Is)  START  ${neuron}  hpo=${hpo_yaml} ==========" | tee "${log}"

  eval_log_dir="${ROOT_DIR}/results/eval/${neuron}/${TASK_NAME}/final"

  set +e
  python3 run_eval.py \
    --task "${TASK_NAME}" \
    --neuron "${neuron}" \
    --log-dir "${eval_log_dir}" \
    --hpo-params "${hpo_yaml}" \
    --n-runs "${N_RUNS}" \
    --seed-start "${SEED_START}" \
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
  echo "$(date -Is)  summary completed=${#SUCCEEDED[@]} skipped=${#SKIPPED[@]} failed=${#FAILED[@]}"
  [[ ${#SUCCEEDED[@]} -gt 0 ]] && echo "  completed: ${SUCCEEDED[*]}"
  [[ ${#SKIPPED[@]} -gt 0 ]] && echo "  skipped: ${SKIPPED[*]}"
  [[ ${#FAILED[@]} -gt 0 ]] && echo "  failed: ${FAILED[*]}"
} >> "${QUEUE_LOG}"

if [[ ${#FAILED[@]} -gt 0 ]]; then
  exit 1
fi

if [[ ${#SUCCEEDED[@]} -eq 0 ]]; then
  echo "[queue] No eval runs started."
  exit 0
fi

echo "[queue] All ${#SUCCEEDED[@]} eval run(s) completed successfully."
