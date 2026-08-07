#!/bin/bash
#SBATCH --job-name=te_bench_analyze
#SBATCH --cpus-per-task=4
#SBATCH --account=datepalm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/bench_analyze_%j.out
#SBATCH --error=logs/bench_analyze_%j.err

# ============================================================================
# 05b_benchmark.sh — Load trained network, run posterior inference on each
# test simulation, build SBC + recovery plots + headline metrics.
#
# Submit AFTER 05a_benchmark.sh, chained:
#   sim_id=$(sbatch --parsable slurm/05a_benchmark.sh)
#   sbatch --dependency=afterok:${sim_id} slurm/05b_benchmark.sh
#
# Or after the simulate array completes manually:
#   sbatch slurm/05b_benchmark.sh
# ============================================================================

REPO_DIR="${REPO_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-te_npe}"
TEST_DIR="${TEST_DIR:-${REPO_DIR}/test_simulations}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_DIR}/results}"

if [ -n "${CONDA_PREFIX:-}" ]; then
    :
else
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null \
        && conda activate "${CONDA_ENV}"
fi

module load slim/4.3
source /shared/projects/datepalm/test_DFE_deeplearning/.venv/bin/activate
source /shared/projects/datepalm/test_DFE_deeplearning/activate.sh
export PYTHONPATH="/shared/projects/datepalm/test_DFE_deeplearning/popgen-npe:$PYTHONPATH"

cd "${REPO_DIR}"

python 05b_benchmark_analyze.py \
    --results-dir   "${RESULTS_DIR}" \
    --test-dir      "${TEST_DIR}" \
    --out           "${RESULTS_DIR}/benchmark/" \
    --save-samples

exit $?
