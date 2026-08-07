#!/bin/bash
#SBATCH --job-name=te_collate
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G                    # needs to hold the full dataset in memory
#SBATCH --time=06:00:00
#SBATCH --output=logs/collate_%j.out
#SBATCH --error=logs/collate_%j.err

# ============================================================================
# 02_collate.sh — Aggregate per-job .npz files into one dataset.pt
# Submit AFTER 01_simulate.sh completes:
#   sbatch --dependency=afterok:<sim_jobid> slurm/02_collate.sh
# ============================================================================

REPO_DIR="${REPO_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-te_npe}"

if [ -z "${CONDA_PREFIX:-}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null \
        && conda activate "${CONDA_ENV}"
fi

module load slim/4.3
source /shared/projects/datepalm/test_DFE_deeplearning/.venv/bin/activate
source /shared/projects/datepalm/test_DFE_deeplearning/activate.sh
export PYTHONPATH="/shared/projects/datepalm/test_DFE_deeplearning/popgen-npe:$PYTHONPATH"

cd "${REPO_DIR}"
mkdir -p results
COLLATED_TRAINING_FOLDER="${COLLATED_TRAINING_FOLDER:-${REPO_DIR}/results}"
SIM_DIR="${SIM_DIR:-${REPO_DIR}/simulations}"


python 02_collate.py \
    --in-dir ${SIM_DIR}  \
    --out    ${COLLATED_TRAINING_FOLDER}/dataset.pt

exit $?
