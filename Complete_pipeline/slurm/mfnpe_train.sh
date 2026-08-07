#!/bin/bash
#SBATCH --job-name=mfnpe
#SBATCH --cpus-per-task=4
#SBATCH --account=datepalm
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/mfnpe_%j.out
#SBATCH --error=logs/mfnpe_%j.err

# ============================================================================
# mfnpe_train.sh — ONE training job for the MF-NPE pipeline. Which of the three
# roles it plays is set by env vars (run_mfnpe.sh drives all three):
#
#   PRETRAIN   DATASET=<msprime set>  OUT=<dir>                    (no INIT_FROM)
#   FINETUNE   DATASET=<slim set>     OUT=<dir>  INIT_FROM=<pretrain dir>
#              SUBSET_N=<budget>      LR=<lower>
#   SCRATCH    DATASET=<slim set>     OUT=<dir>  SUBSET_N=<budget> (no INIT_FROM)
#
# The FINETUNE vs SCRATCH pair at the same SUBSET_N is the whole experiment:
# the gap between their curves is the value of the cheap simulations.
#
# ARCHITECTURE MUST MATCH between pretrain and fine-tune or --init-from fails
# loudly. All three roles therefore share the same arch env vars below.
# ============================================================================

# Training entrypoint: the MF-NPE fork by default (keeps the proven
# 03_train.py untouched). Override with TRAIN_SCRIPT=03_train.py to run
# the stock trainer through this same wrapper.
TRAIN_SCRIPT="${TRAIN_SCRIPT:-03_train_mfnpe.py}"
REPO_DIR="${REPO_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-te_npe}"

if [ -z "${CONDA_PREFIX:-}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null \
        && conda activate "${CONDA_ENV}"
fi

module load slim/4.3
source /shared/projects/datepalm/test_DFE_deeplearning/.venv/bin/activate
source /shared/projects/datepalm/test_DFE_deeplearning/activate.sh
export PYTHONPATH="/shared/projects/datepalm/test_DFE_deeplearning/popgen-npe:${PYTHONPATH:-}"

# Model selection MUST be set: te_dfe/config.py defaults to "simple" when
# TE_DFE_MODEL is unset, which silently trains DFE params against context
# data (7-col theta) without erroring. MF-NPE is the context pipeline.
export TE_DFE_MODEL="${TE_DFE_MODEL:-context}"
export MODEL="${MODEL:-$TE_DFE_MODEL}"
echo "[mfnpe] TE_DFE_MODEL=${TE_DFE_MODEL}"

cd "${REPO_DIR}"

DATASET="${DATASET:?set DATASET}"
OUT="${OUT:?set OUT}"
HEADS="${HEADS:-12}"
AGGREGATOR="${AGGREGATOR:-sigmoid_attn}"
LR="${LR:-5e-4}"
EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-30}"
BATCH="${BATCH:-8}"
VARIABLE_K="${VARIABLE_K:-1}"
N_MIN="${N_MIN:-20}"
N_MAX="${N_MAX:-1000}"

EXTRA=""
if [ "${VARIABLE_K}" = "1" ]; then
    EXTRA="--variable-k --n-min-train ${N_MIN} --n-max-train ${N_MAX}"
fi
if [ -n "${INIT_FROM:-}" ]; then
    EXTRA="${EXTRA} --init-from ${INIT_FROM}"
    [ "${FREEZE_EMBEDDING:-0}" = "1" ] && EXTRA="${EXTRA} --freeze-embedding"
fi
if [ -n "${SUBSET_N:-}" ]; then
    EXTRA="${EXTRA} --subset-n ${SUBSET_N}"
fi

mkdir -p "${OUT}" logs
echo "[mfnpe] dataset=${DATASET} out=${OUT} lr=${LR} ${EXTRA}"

python "${TRAIN_SCRIPT}" \
    --dataset           "${DATASET}" \
    --out               "${OUT}" \
    --n-attention-heads "${HEADS}" \
    --aggregator-kind   "${AGGREGATOR}" \
    --density-estimator nsf \
    --batch-size        "${BATCH}" \
    --lr                "${LR}" \
    --epochs            "${EPOCHS}" \
    --patience          "${PATIENCE}" \
    ${EXTRA}
