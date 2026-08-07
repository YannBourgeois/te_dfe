#!/bin/bash
#SBATCH --job-name=te_snpe_round
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/snpe_round_%A_%a.out
#SBATCH --error=logs/snpe_round_%A_%a.err

# ============================================================================
# 06_snpe_simulate_round.sh — SLURM array template for SNPE refinement rounds.
#
# This is submitted by 06_snpe_refine.py via:
#   sbatch --array=0-N \
#          --export=ALL,THETA_FILE=...,BASE_SEED=...,OUT_DIR=...,REPO_DIR=...
#          06_snpe_simulate_round.sh
#
# Each array task with index $SLURM_ARRAY_TASK_ID:
#   - reads THETA_FILE[task_id]
#   - runs 01_simulate.py with seed = BASE_SEED + task_id
#   - writes to OUT_DIR/sim_NNNNNN.npz
#
# Adjust the environment-activation block for your cluster.
# ============================================================================

set -euo pipefail

# Required env vars (set by 06_snpe_refine.py via --export)
: "${THETA_FILE:?THETA_FILE must be set}"
: "${BASE_SEED:?BASE_SEED must be set}"
: "${OUT_DIR:?OUT_DIR must be set}"
: "${REPO_DIR:?REPO_DIR must be set}"

# Activate environment — adjust for your cluster
module load slim/4.3 2>/dev/null || true
source /shared/projects/datepalm/test_DFE_deeplearning/.venv/bin/activate
source /shared/projects/datepalm/test_DFE_deeplearning/activate.sh 2>/dev/null || true
export PYTHONPATH="/shared/projects/datepalm/test_DFE_deeplearning/popgen-npe:${PYTHONPATH:-}"

mkdir -p "${REPO_DIR}/logs" "${OUT_DIR}"
cd "${REPO_DIR}"

SEED=$(( BASE_SEED + SLURM_ARRAY_TASK_ID ))
echo "[task ${SLURM_ARRAY_TASK_ID}] running seed ${SEED}, theta_row ${SLURM_ARRAY_TASK_ID}"

python 01_simulate.py \
    --seed       "${SEED}" \
    --out-dir    "${OUT_DIR}" \
    --theta-file "${THETA_FILE}" \
    --theta-row  "${SLURM_ARRAY_TASK_ID}"

exit 0
