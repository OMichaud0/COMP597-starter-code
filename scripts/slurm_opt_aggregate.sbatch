#!/bin/bash
#
# Background job: aggregate only.
# Submit with:
#   sbatch scripts/slurm_opt_aggregate.sbatch
# or with explicit run path:
#   sbatch --export=ALL,EXPERIMENT_DIR=/abs/path/to/opt-run-YYYYmmdd-HHMMSS scripts/slurm_opt_aggregate.sbatch
#
# IMPORTANT: update --account, partition, qos, and any environment activation.
#
#SBATCH -p gpu_debug_nvidia
#SBATCH --qos=gpu_debug_nvidia
#SBATCH --account=PI-YEAR
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --gpus=0
#SBATCH -t 0-02:00
#SBATCH -o slurm.aggregate.%N.%j.out
#SBATCH -e slurm.aggregate.%N.%j.err

set -euo pipefail

REPO_DIR="/home/2021/amoham92/courses/comp597/COMP597-starter-code"
RUN_NUM="${RUN_NUM:-1}"
RUN_ROOT="${REPO_DIR}/results/e2e_experiments/run-${RUN_NUM}"
LATEST_PTR_FILE="${RUN_ROOT}/latest_opt_run_path.txt"

cd "${REPO_DIR}"

# Optional: activate your Python environment
# source ~/.bashrc
# conda activate <your-env>

TARGET_DIR="${EXPERIMENT_DIR:-}"
if [[ -z "${TARGET_DIR}" ]]; then
  if [[ -f "${LATEST_PTR_FILE}" ]]; then
    TARGET_DIR="$(cat "${LATEST_PTR_FILE}")"
  else
    TARGET_DIR="$(ls -dt "${RUN_ROOT}"/opt-run-* 2>/dev/null | head -n1 || true)"
  fi
fi

if [[ -z "${TARGET_DIR}" ]]; then
  echo "[ERROR] Could not resolve target run folder for aggregation."
  exit 1
fi

echo "[INFO] Aggregating: ${TARGET_DIR}"
echo "[INFO] RUN_NUM=${RUN_NUM}"
python3 scripts/aggregate_opt_experiments.py "${TARGET_DIR}"
echo "[INFO] Completed aggregate job at $(date -Iseconds)"
