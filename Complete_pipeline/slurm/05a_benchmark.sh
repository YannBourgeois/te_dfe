#!/bin/bash
#SBATCH --job-name=te_bench_sim
#SBATCH --array=0-800%500
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/bench_sim_%A_%a.out
#SBATCH --error=logs/bench_sim_%A_%a.err

# ============================================================================
# 05a_benchmark.sh — SLURM array script for parallel BENCHMARK simulations.
#
# Drives 05a_benchmark_simulate.py, which uses the SAME scenario/chunk sharding
# model as 01_simulate_shard.py:
#   * a "scenario" is one draw of theta (the parameters to recover);
#   * each scenario's genome is split into chunks of CHUNK_SIZE windows;
#   * total array tasks = N_SCENARIOS * ceil(N_WINDOWS / CHUNK_SIZE);
#   * SLURM_ARRAY_TASK_ID maps to (scenario_id, chunk_id) inside the script;
#   * seeds are derived deterministically from MASTER_SEED, so the split is
#     reproducible and independent of chunking.
# Each task writes one shard:  scenario_NNNN_wSSSS-EEEE.npz  in SHARD_DIR.
# (05b then assembles the per-scenario shards and runs recovery / SBC.)
#
# IMPORTANT — size the array to match the parameters. Query it first:
#   N=$(python 05a_benchmark_simulate.py --model "${MODEL:-simple}" \
#         --n-scenarios ${N_SCENARIOS} --n-windows ${N_WINDOWS} \
#         --chunk-size ${CHUNK_SIZE} --print-array-size)
#   sbatch --array=0-$((N-1))%50 slurm/05a_benchmark.sh
# The #SBATCH --array above is only a default for N_SCENARIOS=50, N_WINDOWS=50,
# CHUNK_SIZE=10  (50 * 5 = 250 tasks -> 0-249).
#
# Submit AFTER 03_train.sh. The shards must be COLLATED into test_*.npz
# before 05b — do not skip that step (05b globs test_*.npz, not scenario_*):
#   sim_id=$(sbatch --parsable slurm/05a_benchmark.sh)
#   # after it finishes, assemble per-scenario test files:
#   python 05a_collate.py --model "${MODEL}" \
#       --shard-dir test_shards/ --out-dir test_simulations/
#   sbatch slurm/05b_benchmark.sh
# (or chain the collate in a small afterok job that runs 05a_collate, then 05b).
#
# Resume: the script skips any shard whose .npz already exists (use --overwrite
# to force). No per-seed loop here — one task = one (scenario, chunk) shard.
# ============================================================================

REPO_DIR="${REPO_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-te_npe}"
# SHARD_DIR holds the per-chunk scenario_*.npz this step writes; 05a_collate
# then assembles those into test_*.npz under TEST_DIR (which 05b reads). These
# MUST differ — writing shards into TEST_DIR makes 05b find scenario_*.npz that
# don't match its test_*.npz glob ("No test_*.npz").
SHARD_DIR="${SHARD_DIR:-${REPO_DIR}/test_shards}"
TEST_DIR="${TEST_DIR:-${REPO_DIR}/test_simulations}"
MODEL="${MODEL:-simple}"
N_SCENARIOS="${N_SCENARIOS:-50}"
N_WINDOWS="${N_WINDOWS:-50}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
MASTER_SEED="${MASTER_SEED:-20240601}"
TASK_OFFSET="${TASK_OFFSET:-0}"

if [ -n "${CONDA_PREFIX:-}" ]; then
    :
else
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null \
        && conda activate "${CONDA_ENV}"
fi

module load slim/4.3 2>/dev/null || true
source /shared/projects/datepalm/test_DFE_deeplearning/.venv/bin/activate 2>/dev/null || true
source /shared/projects/datepalm/test_DFE_deeplearning/activate.sh 2>/dev/null || true
export PYTHONPATH="/shared/projects/datepalm/test_DFE_deeplearning/popgen-npe:$PYTHONPATH"

mkdir -p "${REPO_DIR}/logs" "${SHARD_DIR}"
cd "${REPO_DIR}"

TASK_ID=$(( TASK_OFFSET + SLURM_ARRAY_TASK_ID ))

python 05a_benchmark_simulate.py \
    --model        "${MODEL}" \
    --task-id      "${TASK_ID}" \
    --n-scenarios  "${N_SCENARIOS}" \
    --n-windows    "${N_WINDOWS}" \
    --chunk-size   "${CHUNK_SIZE}" \
    --master-seed  "${MASTER_SEED}" \
    --out-dir      "${SHARD_DIR}"

exit 0
