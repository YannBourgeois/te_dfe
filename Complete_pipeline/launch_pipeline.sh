#!/bin/bash
# ============================================================================
# launch_pipeline.sh — Submit the full 5-stage pipeline to SLURM with
# dependencies. Each stage starts only after its predecessor finishes OK.
#
# Stages:
#   1. simulate         : SLURM array, N_TASKS tasks * SEEDS_PER_TASK seeds
#                         = N_SIMS simulations
#   2. collate          : single job, depends on simulate
#   3. train            : single GPU job, depends on collate
#   4. benchmark-sim    : SLURM array, N_BENCH_TASKS tasks * SEEDS_PER_TASK
#                         seeds = N_BENCH_SIMS test simulations
#   5. benchmark-analyze: single GPU job, depends on benchmark-sim
#
# Multi-seed-per-task pattern: each array task runs SEEDS_PER_TASK seeds
# sequentially, so 10000 sims fit in 1000 array slots (the SLURM cap on
# many clusters).
#
# Usage:
#   bash launch_pipeline.sh                       # 5000 sims, 200 bench
#   bash launch_pipeline.sh 10000                 # 10000 sims, 200 bench
#   bash launch_pipeline.sh 5000 100              # 5000 sims, 100 bench
#   bash launch_pipeline.sh 5000 200 --collate-only   # skip simulate
#   bash launch_pipeline.sh 5000 200 --train-only     # skip simulate+collate
#   bash launch_pipeline.sh 5000 200 --benchmark-only # skip everything before bench
#
# Environment overrides:
#   SEEDS_PER_TASK    default 10 (seeds per array task; lets large
#                     simulation counts fit under cluster array-job caps)
#   MAX_CONCURRENT    default 200 (max simultaneously-running array tasks)
#   KEEP_TREES        set to 1 to persist per-window tree sequences
#                     alongside the .npz tensors (~5-20 MB per window).
#                     Useful for cheap re-extraction of channels later
#                     via 01b_reprocess.sh, without rerunning SLiM.
#   KEEP_TREES_DIR    override where trees are saved (default
#                     <SIM_DIR>/trees)
#   AGGREGATOR        default "attention" (backward-compatible with v4).
#                     Set to "deepsets" or "sigmoid_attn" for K-invariant
#                     cross-window aggregation. Passed through to
#                     slurm/03_train.sh.
#   VARIABLE_K        set to 1 to enable variable-K / variable-N training
#                     (recommended with deepsets / sigmoid_attn).
#   K_MIN, K_MAX      MATRIX bounds when VARIABLE_K=1 (window count; def 5, 20).
#   N_MIN, N_MAX      SUMSTAT bounds when VARIABLE_K=1 (per-sim TE count;
#                     def 20, 500). The sumstat path reads these; matrix reads
#                     K_MIN/K_MAX. Both are forwarded, so set the pair that
#                     matches your REPRESENTATION.
#   All of the above propagate to the training job via `sbatch --export=ALL`.
#   Example (sumstat, the date-palm setup):
#     VARIABLE_K=1 AGGREGATOR=sigmoid_attn N_MIN=20 N_MAX=500 bash launch_pipeline.sh
# ============================================================================

set -e

N_SIMS=${1:-5000}
N_BENCH_SIMS=${2:-200}
STAGE=$(echo "$@" | grep -oE -- '--[a-z-]+' | head -1 || echo "")

SEEDS_PER_TASK="${SEEDS_PER_TASK:-10}"
MAX_CONCURRENT="${MAX_CONCURRENT:-200}"
# Benchmark uses the scenario/chunk model (NOT the seed loop): N_BENCH_SIMS is
# the number of test SCENARIOS (each one drawn theta = one recovery point), and
# each scenario is BENCH_WINDOWS windows split into CHUNK_SIZE-window shards.
# Default the window count to the trained K so the benchmark matches training.
BENCH_WINDOWS="${BENCH_WINDOWS:-20}"     # = N_TRAIN_WINDOWS by default
BENCH_CHUNK="${BENCH_CHUNK:-10}"
BENCH_MASTER_SEED="${BENCH_MASTER_SEED:-20240601}"
MODEL="${MODEL:-simple}"
# config.py dispatches on TE_DFE_MODEL, not MODEL. Export it so every stage
# (simulate/collate/train/benchmark) inherits it via `sbatch --export=ALL` and
# runs the SAME model. Without this, only the bench-sim stage (which passes
# --model) was context while train/collate/analyze silently defaulted to simple.
export TE_DFE_MODEL="${MODEL}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_DIR="${REPO_DIR}/slurm"
mkdir -p "${REPO_DIR}/logs" "${REPO_DIR}/simulations" \
         "${REPO_DIR}/results" "${REPO_DIR}/test_simulations" \
         "${REPO_DIR}/test_shards"

# Training array: ceil(N_SIMS / SEEDS_PER_TASK) (seed-loop model, unchanged).
N_SIM_TASKS=$(( (N_SIMS + SEEDS_PER_TASK - 1) / SEEDS_PER_TASK ))
# Benchmark array: scenario/chunk model. tasks = N_SCENARIOS * n_chunks,
# n_chunks = ceil(BENCH_WINDOWS / BENCH_CHUNK).
BENCH_N_CHUNKS=$(( (BENCH_WINDOWS + BENCH_CHUNK - 1) / BENCH_CHUNK ))
N_BENCH_TASKS=$(( N_BENCH_SIMS * BENCH_N_CHUNKS ))

echo "========================================="
echo "TE DFE + burst NPE pipeline launch"
echo "  Repo:           ${REPO_DIR}"
echo "  Training sims:  ${N_SIMS}  (${N_SIM_TASKS} tasks * ${SEEDS_PER_TASK} seeds)"
echo "  Benchmark:      ${N_BENCH_SIMS} scenarios * ${BENCH_WINDOWS} windows"
echo "                  -> ${N_BENCH_TASKS} tasks (${BENCH_N_CHUNKS} chunks/scenario, model=${MODEL})"
echo "  Max concurrent: ${MAX_CONCURRENT}"
echo "  Stage:          ${STAGE:-all}"
echo "========================================="

submit_simulate() {
    sbatch --parsable \
        --array=0-$((N_SIM_TASKS - 1))%${MAX_CONCURRENT} \
        --export=ALL,SEEDS_PER_TASK="${SEEDS_PER_TASK}" \
        "${SLURM_DIR}/01_simulate.sh"
}

submit_collate() {
    local dep_clause="$1"
    sbatch --parsable ${dep_clause} "${SLURM_DIR}/02_collate.sh"
}

submit_train() {
    local dep_clause="$1"
    sbatch --parsable ${dep_clause} \
        --export=ALL \
        "${SLURM_DIR}/03_train.sh"
}

submit_bench_sim() {
    local dep_clause="$1"
    sbatch --parsable ${dep_clause} \
        --array=0-$((N_BENCH_TASKS - 1))%${MAX_CONCURRENT} \
        --export=ALL,MODEL="${MODEL}",N_SCENARIOS="${N_BENCH_SIMS}",N_WINDOWS="${BENCH_WINDOWS}",CHUNK_SIZE="${BENCH_CHUNK}",MASTER_SEED="${BENCH_MASTER_SEED}",SHARD_DIR="${REPO_DIR}/test_shards",TEST_DIR="${REPO_DIR}/test_simulations" \
        "${SLURM_DIR}/05a_benchmark.sh"
}

submit_bench_collate() {
    # Assemble the per-chunk scenario_*.npz shards into per-scenario test_*.npz
    # (05b globs test_*.npz, so this step is mandatory — not optional).
    local dep_clause="$1"
    sbatch --parsable ${dep_clause} \
        --export=ALL,MODEL="${MODEL}",SHARD_DIR="${REPO_DIR}/test_shards",TEST_DIR="${REPO_DIR}/test_simulations" \
        "${SLURM_DIR}/05a_collate.sh"
}

submit_bench_analyze() {
    local dep_clause="$1"
    sbatch --parsable ${dep_clause} \
        --export=ALL,MODEL="${MODEL}",TEST_DIR="${REPO_DIR}/test_simulations" \
        "${SLURM_DIR}/05b_benchmark.sh"
}

case "${STAGE}" in
    --benchmark-only)
        echo "Submitting benchmark only (simulate + analyze)..."
        BSIM_ID=$(submit_bench_sim "")
        BCOL_ID=$(submit_bench_collate "--dependency=afterok:${BSIM_ID}")
        BANA_ID=$(submit_bench_analyze "--dependency=afterok:${BCOL_ID}")
        echo "  benchmark-sim array: ${BSIM_ID}"
        echo "  benchmark-collate:   ${BCOL_ID}"
        echo "  benchmark-analyze:   ${BANA_ID}"
        ;;
    --train-only)
        echo "Submitting train + benchmark..."
        TRAIN_ID=$(submit_train "")
        BSIM_ID=$(submit_bench_sim "--dependency=afterok:${TRAIN_ID}")
        BCOL_ID=$(submit_bench_collate "--dependency=afterok:${BSIM_ID}")
        BANA_ID=$(submit_bench_analyze "--dependency=afterok:${BCOL_ID}")
        echo "  train job:           ${TRAIN_ID}"
        echo "  benchmark-sim array: ${BSIM_ID}"
        echo "  benchmark-collate:   ${BCOL_ID}"
        echo "  benchmark-analyze:   ${BANA_ID}"
        ;;
    --collate-only)
        echo "Submitting collate + train + benchmark..."
        COLLATE_ID=$(submit_collate "")
        TRAIN_ID=$(submit_train "--dependency=afterok:${COLLATE_ID}")
        BSIM_ID=$(submit_bench_sim "--dependency=afterok:${TRAIN_ID}")
        BCOL_ID=$(submit_bench_collate "--dependency=afterok:${BSIM_ID}")
        BANA_ID=$(submit_bench_analyze "--dependency=afterok:${BCOL_ID}")
        echo "  collate job:         ${COLLATE_ID}"
        echo "  train job:           ${TRAIN_ID}"
        echo "  benchmark-sim array: ${BSIM_ID}"
        echo "  benchmark-collate:   ${BCOL_ID}"
        echo "  benchmark-analyze:   ${BANA_ID}"
        ;;
    *)
        echo "Submitting full pipeline (simulate -> collate -> train -> benchmark)..."
        SIM_ID=$(submit_simulate)
        # afterany on the simulate array: tolerate per-task failures
        # (the collator filters out unreadable .npz files)
        COLLATE_ID=$(submit_collate "--dependency=afterany:${SIM_ID}")
        TRAIN_ID=$(submit_train "--dependency=afterok:${COLLATE_ID}")
        BSIM_ID=$(submit_bench_sim "--dependency=afterok:${TRAIN_ID}")
        BCOL_ID=$(submit_bench_collate "--dependency=afterok:${BSIM_ID}")
        BANA_ID=$(submit_bench_analyze "--dependency=afterok:${BCOL_ID}")
        echo "  simulate array:      ${SIM_ID}"
        echo "  collate job:         ${COLLATE_ID}"
        echo "  train job:           ${TRAIN_ID}"
        echo "  benchmark-sim array: ${BSIM_ID}"
        echo "  benchmark-collate:   ${BCOL_ID}"
        echo "  benchmark-analyze:   ${BANA_ID}"
        ;;
esac

echo "========================================="
echo "Monitor:  squeue -u \$USER"
echo "Logs:     ${REPO_DIR}/logs/"
echo "Results:  ${REPO_DIR}/results/"
echo "         (benchmark plots in results/benchmark/)"
echo "========================================="
