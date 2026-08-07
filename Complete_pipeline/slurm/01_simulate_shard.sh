#!/bin/bash
#SBATCH --job-name=te_sim_shard
#SBATCH --output=logs/sim_shard_%A_%a.out
#SBATCH --error=logs/sim_shard_%A_%a.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
# ============================================================================
# 01_simulate_shard.sh — sharded training-sim production.
#
# Each task simulates a CHUNK of windows for one replicate, so the cluster is
# filled with many short jobs instead of fewer long ones (no waiting on the
# slowest window of a 20-window replicate). After the array finishes, collate:
#     python 01_collate_shard.py --shard-dir sim_shards/ --out-dir simulations/
# then proceed to 02_collate.py as usual.
#
# Array sizing: total tasks = N_REPLICATES * ceil(N_WINDOWS / CHUNK_SIZE).
#   10000 replicates x 20 windows, chunk 5 -> 40000 tasks.
# Most clusters cap array indices at 1000, so submit in BLOCKS using TASK_OFFSET:
#   the script computes the real task id as TASK_OFFSET + SLURM_ARRAY_TASK_ID.
# Submit block b (1000 tasks each) with:
#   sbatch --array=0-999%300 --export=ALL,TASK_OFFSET=$((b*1000)) 01_simulate_shard.sh
# for b = 0,1,2,...,39  (40 blocks -> 40000 tasks). The Python skips shards that
# already exist, so re-submitting a block only fills gaps.
#
# Time: a 5 Mb window is the unit of work; CHUNK_SIZE windows per task. Scale
# --time with CHUNK_SIZE (chunk 5 of 5 Mb windows ~ 1-1.5 h is typical; tune).
# ============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-te_npe}"
SHARD_DIR="${SHARD_DIR:-${REPO_DIR}/sim_shards}"

N_REPLICATES="${N_REPLICATES:-10000}"
N_WINDOWS="${N_WINDOWS:-20}"
CHUNK_SIZE="${CHUNK_SIZE:-5}"
MASTER_SEED="${MASTER_SEED:-12345}"
TASK_OFFSET="${TASK_OFFSET:-0}"
# Stratified window manifest (recommended): pre-sample windows covering the
# covariate space, then point the sharded sims at it so each replicate's
# windows span the (gene_density, recomb) range — restores the old
# per-replicate stratification. Build it once before launching:
#   python 00_sample_windows.py --n-replicates ${N_REPLICATES} \
#       --n-windows ${N_WINDOWS} --out window_manifest --seed 777
# then set WINDOW_MANIFEST=window_manifest.npz. Leave empty for independent
# (unstratified) per-window sampling.
WINDOW_MANIFEST="${WINDOW_MANIFEST:-}"

# Environment (adjust to your cluster).
if [ -z "${CONDA_PREFIX:-}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null \
        && conda activate "${CONDA_ENV}" || true
fi
module load slim/4.3 2>/dev/null || true

mkdir -p "${REPO_DIR}/logs" "${SHARD_DIR}"
cd "${REPO_DIR}"

TASK_ID=$(( TASK_OFFSET + SLURM_ARRAY_TASK_ID ))

MANIFEST_ARG=""
if [ -n "${WINDOW_MANIFEST}" ]; then
    MANIFEST_ARG="--window-manifest ${WINDOW_MANIFEST}"
fi

python 01_simulate_shard.py \
    --task-id      "${TASK_ID}" \
    --n-replicates "${N_REPLICATES}" \
    --n-windows    "${N_WINDOWS}" \
    --chunk-size   "${CHUNK_SIZE}" \
    --master-seed  "${MASTER_SEED}" \
    --out-dir      "${SHARD_DIR}" \
    ${MANIFEST_ARG}

exit 0
