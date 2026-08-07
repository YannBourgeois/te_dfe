#!/bin/bash
#SBATCH --job-name=msp_collate
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/msp_collate_%j.out
#SBATCH --error=logs/msp_collate_%j.err

# --- defensive prologue -------------------------------------------------
# `set -u` can be INHERITED: bash enables options listed in an exported
# SHELLOPTS at startup, and `sbatch --export=ALL` carries it from the login
# shell. Commenting out `set -euo pipefail` here therefore does NOT disable
# it. Turn it off explicitly, and define PYTHONPATH before anything can
# expand it -- the slim modulefile prepends to it unguarded.
set +u
export PYTHONPATH="${PYTHONPATH:-}"


# ============================================================================
# STEP 2 — collate the s-proxy msprime shards into one dataset.
#
# Submit AFTER step 1 finishes:
#   sbatch --dependency=afterok:<step1_jobid> slurm/msprime_sproxy_collate.sh
# (For an array job, depend on the whole array: afterok:<arrayjobid>.)
#
# CRITICAL — CONFIG PARITY: this msprime set MUST be collated with the SAME
# config as the SLiM set it will be paired with (REPRESENTATION=sumstat,
# the same CONTEXT_PARAMS, and the same N_SUMSTAT_GLOBAL_COV / ENRICH_OE_DECILES
# setting). If they differ, --init-from will refuse the load at train time
# because trunk_in widths won't match. Simplest guarantee: don't change config
# between collating the two sets.
# ============================================================================
set -eo pipefail
REPO_DIR="${REPO_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-te_npe}"
SHARD_DIR="${SHARD_DIR:-${REPO_DIR}/sim_shards_msprime_sproxy}"
OUT="${OUT:-${REPO_DIR}/simulations_msprime/dataset}"
PATTERN="${PATTERN:-sim_*.npz}"         # shard glob; msprime shards are
                                        # sim_*_msprime.npz so the default works,
                                        # but '*_msprime.npz' also catches older
                                        # rep_* names

if [ -z "${CONDA_PREFIX:-}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null \
        && conda activate "${CONDA_ENV}"
fi
module load slim/4.3
source /shared/projects/datepalm/test_DFE_deeplearning/.venv/bin/activate
source /shared/projects/datepalm/test_DFE_deeplearning/activate.sh
export PYTHONPATH="/shared/projects/datepalm/test_DFE_deeplearning/popgen-npe:${PYTHONPATH:-}"

export TE_DFE_MODEL=context
export MODEL=context

cd "${REPO_DIR}"
mkdir -p logs "$(dirname "${OUT}")"

python 02_collate_msprime.py \
    --in-dir "${SHARD_DIR}/" \
    --out    "${OUT}" \
    --pattern "${PATTERN}"
