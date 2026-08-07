#!/bin/bash
#SBATCH --job-name=te_train
#SBATCH --cpus-per-task=4
#SBATCH --account=datepalm
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --gres=gpu:1                 # request 1 L40S GPU
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ============================================================================
# 03_train.sh — Train SNPE on the collated dataset.
# Submit AFTER 02_collate.sh:
#   sbatch --dependency=afterok:<collate_jobid> slurm/03_train.sh
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

# Aggregator selection — override via env when submitting:
#   AGGREGATOR=deepsets     sbatch slurm/03_train.sh
#   AGGREGATOR=sigmoid_attn sbatch slurm/03_train.sh
# Optional variable-K / variable-N training (set VARIABLE_K=1):
#   matrix  representation uses K_MIN/K_MAX (window counts).
#   sumstat representation uses N_MIN/N_MAX (per-sim TE counts).
#   Passing both pairs is safe — 03_train uses the one matching REPRESENTATION.
# Examples:
#   # sumstat, variable-N, sigmoid attention (the date-palm setup):
#   VARIABLE_K=1 AGGREGATOR=sigmoid_attn N_MIN=20 N_MAX=500 sbatch slurm/03_train.sh
#   # matrix, variable-K:
#   VARIABLE_K=1 K_MIN=5 K_MAX=20 AGGREGATOR=sigmoid_attn sbatch slurm/03_train.sh
AGGREGATOR="${AGGREGATOR:-attention}"
VARIABLE_K="${VARIABLE_K:-0}"
K_MIN="${K_MIN:-5}"
K_MAX="${K_MAX:-20}"
N_MIN="${N_MIN:-20}"
N_MAX="${N_MAX:-500}"
COLLATED_TRAINING_FOLDER="${COLLATED_TRAINING_FOLDER:-results}"
RESULTS_DIR="${RESULTS_DIR:-results}"
N_HEADS="${N_HEADS:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EXTRA_ARGS=""
if [ "${VARIABLE_K}" = "1" ]; then
    # --variable-k is the master switch; the matrix path reads --k-*, the
    # sumstat path reads --n-*-train. Both are supplied so either works.
    EXTRA_ARGS="--variable-k --k-min ${K_MIN} --k-max ${K_MAX} \
--n-min-train ${N_MIN} --n-max-train ${N_MAX}"
fi

python 03_train.py \
    --dataset           "${COLLATED_TRAINING_FOLDER}"/dataset \
    --out               "${RESULTS_DIR}"/ \
    --n-attention-heads "${N_HEADS}" \
    --density-estimator nsf \
    --batch-size        "${BATCH_SIZE}" \
    --lr                5e-4 \
    --epochs            300 \
    --patience          30 \
    --aggregator-kind   "${AGGREGATOR}" \
    ${EXTRA_ARGS}

exit $?
