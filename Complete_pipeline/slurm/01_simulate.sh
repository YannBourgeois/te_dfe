#!/bin/bash
#SBATCH --job-name=te_simulate
#SBATCH --array=0-499%200
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/sim_%A_%a.out
#SBATCH --error=logs/sim_%A_%a.err

# ============================================================================
# 01_simulate.sh — SLURM array script for parallel simulation.
#
# CAN BE CONFIGURED to run a second batch if 5000 sims:
#   #SBATCH --array=500-999%200  + SEEDS_PER_TASK=10 -> seeds 5000..9999
#
# To restart from scratch (seeds 0..4999), change to:
#   #SBATCH --array=0-499%200
#
# Multi-seed-per-task design lets you produce more than 1000 simulations
# while staying within the SLURM 1000-array-jobs cap on most clusters.
#
# Each array task with index $SLURM_ARRAY_TASK_ID runs SEEDS_PER_TASK
# simulations with seeds in [task*SEEDS_PER_TASK, (task+1)*SEEDS_PER_TASK).
#
# Examples:
#   1000 sims, 1 seed each:    #SBATCH --array=0-999    SEEDS_PER_TASK=1
#   5000 sims, 10 seeds each:  #SBATCH --array=0-499    SEEDS_PER_TASK=10
#   10000 sims, 10 seeds each: #SBATCH --array=0-999    SEEDS_PER_TASK=10
#   20000 sims, 20 seeds each: #SBATCH --array=0-999    SEEDS_PER_TASK=20
#
# Resume: the Python script skips replicates whose .npz already exists,
# so re-running this script will fill in only the missing seeds.
#
# Adjust:
#   --array=N-M%K     N=lower task, M=upper, K=max concurrent jobs
#   SEEDS_PER_TASK    how many seeds each task runs sequentially
#   --time            scale roughly linearly with SEEDS_PER_TASK
#   --mem             8G usually plenty for one SLiM at a time
#
# Adjust REPO_DIR and CONDA_ENV for your environment.
# ============================================================================

REPO_DIR="${REPO_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-te_npe}"
SIM_DIR="${SIM_DIR:-${REPO_DIR}/simulations}"
SEEDS_PER_TASK="${SEEDS_PER_TASK:-10}"

# Set KEEP_TREES=1 to also persist per-window tree sequences alongside
# the .npz tensors. They are saved under ${SIM_DIR}/trees/sim_NNNNNN/.
# Useful if you may want to re-extract channels / features later without
# rerunning SLiM. Storage cost is ~5-20 MB per window per sim — so a
# 5000-sim × 20-window run adds ~500 GB. Check your quota before
# enabling.
# Set KEEP_TREES_DIR=/path to override the default (${SIM_DIR}/trees).
KEEP_TREES="${KEEP_TREES:-0}"
KEEP_TREES_DIR="${KEEP_TREES_DIR:-}"

# Activate environment
if [ -n "${CONDA_PREFIX:-}" ]; then
    : # already in a conda env
else
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null \
        && conda activate "${CONDA_ENV}"
fi
module load slim/4.3
source /shared/projects/datepalm/test_DFE_deeplearning/.venv/bin/activate
source /shared/projects/datepalm/test_DFE_deeplearning/activate.sh
export PYTHONPATH="/shared/projects/datepalm/test_DFE_deeplearning/popgen-npe:$PYTHONPATH"

mkdir -p "${REPO_DIR}/logs" "${SIM_DIR}"
cd "${REPO_DIR}"

BASE_SEED=$(( SLURM_ARRAY_TASK_ID * SEEDS_PER_TASK ))
for ((i=0; i<SEEDS_PER_TASK; i++)); do
    SEED=$(( BASE_SEED + i ))
    echo "[task ${SLURM_ARRAY_TASK_ID}] running seed ${SEED}"

    # Skip if already done (resume support)
    OUT_FILE="${SIM_DIR}/sim_$(printf '%06d' ${SEED}).npz"
    if [ -f "${OUT_FILE}" ]; then
        echo "[task ${SLURM_ARRAY_TASK_ID}] seed ${SEED} already done, skip"
        continue
    fi

    EXTRA_ARGS=""
    if [ "${KEEP_TREES}" = "1" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --keep-trees"
        if [ -n "${KEEP_TREES_DIR}" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --keep-trees-dir ${KEEP_TREES_DIR}"
        fi
    fi

    python 01_simulate.py \
        --seed    "${SEED}" \
        --out-dir "${SIM_DIR}" \
        ${EXTRA_ARGS}
done

exit 0
