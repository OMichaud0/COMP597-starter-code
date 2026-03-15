#!/bin/bash
#
# Helper to submit two background jobs with dependency:
# 1) run job
# 2) aggregate job after run succeeds
#
# Usage:
#   bash scripts/submit_opt_background.sh
#
set -euo pipefail

RUN_NUM="${RUN_NUM:-1}"

RUN_JOB_ID="$(
  sbatch --parsable --export=ALL,RUN_NUM="${RUN_NUM}" scripts/slurm_opt_run.sbatch
)"
echo "[INFO] Submitted run job: ${RUN_JOB_ID} (RUN_NUM=${RUN_NUM})"

AGG_JOB_ID="$(
  sbatch --parsable --dependency=afterok:${RUN_JOB_ID} --export=ALL,RUN_NUM="${RUN_NUM}" scripts/slurm_opt_aggregate.sbatch
)"
echo "[INFO] Submitted aggregate job: ${AGG_JOB_ID} (afterok:${RUN_JOB_ID}, RUN_NUM=${RUN_NUM})"

echo "[INFO] Track with: squeue --user=${USER}"
