#!/bin/bash
#
# Background job: run experiments only.
# Submit with:
#   sbatch scripts/slurm_opt_run.sbatch
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
#SBATCH -o slurm.run.%N.%j.out
#SBATCH -e slurm.run.%N.%j.err

set -euo pipefail

REPO_DIR="/home/2021/amoham92/courses/comp597/COMP597-starter-code"
RUN_NUM="${RUN_NUM:-1}"
RUN_ROOT="${REPO_DIR}/results/e2e_experiments/run-${RUN_NUM}"
LATEST_PTR_FILE="${RUN_ROOT}/latest_opt_run_path.txt"

cd "${REPO_DIR}"

# Optional: activate your Python environment
# source ~/.bashrc
# conda activate <your-env>

echo "[INFO] Starting run_opt_experiments.py at $(date -Iseconds)"
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

echo "${LATEST_RUN}" > "${LATEST_PTR_FILE}"
echo "[INFO] Saved latest run path to ${LATEST_PTR_FILE}"
echo "[INFO] Completed run job at $(date -Iseconds)"
