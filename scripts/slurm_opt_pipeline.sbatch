#!/bin/bash
#
# Single-job background pipeline:
# 1) run_opt_experiments.py
# 2) aggregate_opt_experiments.py
#
# Submit with:
#   sbatch scripts/slurm_opt_pipeline.sbatch
#
# IMPORTANT: update --account, partition, qos, and any environment activation.
#
#SBATCH -p gpu_debug_nvidia
#SBATCH --qos=gpu_debug_nvidia
#SBATCH --account=PI-YEAR
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH -t 0-12:00
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

set -euo pipefail

REPO_DIR="/home/2021/amoham92/courses/comp597/COMP597-starter-code"
RUN_NUM="${RUN_NUM:-1}"
RUN_ROOT="${REPO_DIR}/results/e2e_experiments/run-${RUN_NUM}"

cd "${REPO_DIR}"

# Optional: activate your Python environment
# source ~/.bashrc
# conda activate <your-env>

echo "[INFO] Starting OPT experiment matrix at $(date -Iseconds)"
echo "[INFO] RUN_NUM=${RUN_NUM}"
python3 scripts/run_opt_experiments.py \
  --repo-root . \
  --output-root "${RUN_ROOT}" \
  --launcher "python3 launch.py"

LATEST_RUN="$(ls -dt "${RUN_ROOT}"/opt-run-* 2>/dev/null | head -n1 || true)"
if [[ -z "${LATEST_RUN}" ]]; then
  echo "[ERROR] No opt-run-* folder found under ${RUN_ROOT}"
  exit 1
fi

echo "[INFO] Aggregating run folder: ${LATEST_RUN}"
python3 scripts/aggregate_opt_experiments.py "${LATEST_RUN}"
echo "[INFO] Completed at $(date -Iseconds)"
