#!/bin/bash
#SBATCH --job-name=te_bench_collate
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/bench_collate_%j.out
#SBATCH --error=logs/bench_collate_%j.err

# ============================================================================
# 05a_collate.sh — assemble the per-chunk scenario_*.npz shards written by
# 05a_benchmark.sh into per-scenario test_*.npz files that 05b reads.
#
# MANDATORY between 05a and 05b: 05b globs test_*.npz, not scenario_*.npz.
# Detects matrix vs sumstat shards automatically; for sumstat it pools each
# scenario's per-TE features across all its window-chunks, then pads once.
#
# Chained:
#   sim_id=$(sbatch --parsable slurm/05a_benchmark.sh)
#   col_id=$(sbatch --parsable --dependency=afterok:${sim_id} slurm/05a_collate.sh)
#   sbatch --dependency=afterok:${col_id} slurm/05b_benchmark.sh
# ============================================================================

REPO_DIR="${REPO_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-te_npe}"
MODEL="${MODEL:-simple}"
SHARD_DIR="${SHARD_DIR:-${REPO_DIR}/test_shards}"
TEST_DIR="${TEST_DIR:-${REPO_DIR}/test_simulations}"

if [ -n "${CONDA_PREFIX:-}" ]; then
    :
else
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null \
        && conda activate "${CONDA_ENV}"
fi

source /shared/projects/datepalm/test_DFE_deeplearning/.venv/bin/activate 2>/dev/null || true
source /shared/projects/datepalm/test_DFE_deeplearning/activate.sh 2>/dev/null || true
export PYTHONPATH="/shared/projects/datepalm/test_DFE_deeplearning/popgen-npe:$PYTHONPATH"

mkdir -p "${REPO_DIR}/logs" "${TEST_DIR}"
cd "${REPO_DIR}"

python 05a_collate.py \
    --model     "${MODEL}" \
    --shard-dir "${SHARD_DIR}" \
    --out-dir   "${TEST_DIR}"

exit 0
