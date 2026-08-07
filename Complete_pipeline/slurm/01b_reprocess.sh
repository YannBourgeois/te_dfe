#!/bin/bash
#SBATCH --job-name=te_reprocess
#SBATCH --array=0-999%200
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --output=logs/reprocess_%A_%a.out
#SBATCH --error=logs/reprocess_%A_%a.err

# ============================================================================
# 01b_reprocess.sh — SLURM array script for re-extracting tensors from
# previously saved tree sequences.
#
# Prerequisite: you ran 01_simulate.sh with KEEP_TREES=1, so per-window
# tree sequences are saved under ${SIM_DIR}/trees/sim_NNNNNN/.
#
# Use case: you changed processor logic (new channels, new filtering,
# etc.) and want to update your .npz tensors without rerunning SLiM.
#
# Each array task with index $SLURM_ARRAY_TASK_ID re-extracts
# SEEDS_PER_TASK seeds: [task*SEEDS_PER_TASK, (task+1)*SEEDS_PER_TASK).
# Resume: skips seeds whose output .npz already exists.
#
# Adjust SEEDS_PER_TASK to match how 01_simulate.sh was run (so seed
# ranges line up). For 5000 sims with SEEDS_PER_TASK=10:
#   #SBATCH --array=0-499%200
#
# Re-extraction is much faster than SLiM (~1-5 seconds per window vs.
# minutes per simulation), so --time and --mem are modest.
#
# Required env vars:
#   TREES_DIR     directory where 01_simulate.sh stored trees (default
#                 ${REPO_DIR}/simulations/trees)
#   OUT_DIR       where to write fresh .npz tensors (default
#                 ${REPO_DIR}/simulations_v2)
# ============================================================================

REPO_DIR="${REPO_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-te_npe}"
TREES_DIR="${TREES_DIR:-${REPO_DIR}/simulations/trees}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/simulations_v2}"
SEEDS_PER_TASK="${SEEDS_PER_TASK:-10}"

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

mkdir -p "${REPO_DIR}/logs" "${OUT_DIR}"
cd "${REPO_DIR}"

BASE_SEED=$(( SLURM_ARRAY_TASK_ID * SEEDS_PER_TASK ))
for ((i=0; i<SEEDS_PER_TASK; i++)); do
    SEED=$(( BASE_SEED + i ))
    SIM_TREES_DIR="${TREES_DIR}/sim_$(printf '%06d' ${SEED})"

    # Skip seeds without saved trees (probably weren't simulated, or
    # KEEP_TREES wasn't set)
    if [ ! -d "${SIM_TREES_DIR}" ]; then
        echo "[task ${SLURM_ARRAY_TASK_ID}] seed ${SEED}: no trees dir, skip"
        continue
    fi

    OUT_FILE="${OUT_DIR}/sim_$(printf '%06d' ${SEED}).npz"
    if [ -f "${OUT_FILE}" ]; then
        echo "[task ${SLURM_ARRAY_TASK_ID}] seed ${SEED}: already done, skip"
        continue
    fi

    echo "[task ${SLURM_ARRAY_TASK_ID}] re-extracting seed ${SEED}"
    python 01b_reprocess.py \
        --seed       "${SEED}" \
        --trees-dir  "${TREES_DIR}" \
        --out-dir    "${OUT_DIR}"
done

exit 0
