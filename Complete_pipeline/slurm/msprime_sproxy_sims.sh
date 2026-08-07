#!/bin/bash
#SBATCH --job-name=msp_sproxy
#SBATCH --output=logs/msp_sproxy_%A_%a.out
#SBATCH --error=logs/msp_sproxy_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G


# Submit e.g. 10,000 sims as 1000 tasks x 10 reps:
#   mkdir -p logs
#   REP_PER_TASK=10 sbatch --array=0-999%300 slurm/msprime_sproxy_sims.sh
#
# Clusters cap array indices (~1000). For >1000 tasks, submit blocks with
# TASK_OFFSET, exactly like 01_simulate_shard.sh:
#   REP_PER_TASK=10 sbatch --array=0-999%300 --export=ALL,TASK_OFFSET=0     ...
#   REP_PER_TASK=10 sbatch --array=0-999%300 --export=ALL,TASK_OFFSET=1000  ...
# Python skips replicates whose shard already exists, so re-submitting fills gaps.
#
# PARITY: --n-windows and --window-size below must be the pair you intend to
# compare against the SLiM set. 100 x 1Mb conserves total sequence vs 20 x 5Mb;
# verify TE-count-per-replicate matches a real SLiM shard (inspect_msprime_shard.py)
# before committing to the full run.
# ============================================================================
set -eo pipefail
REPO_DIR="${REPO_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-te_npe}"
SHARD_DIR="${SHARD_DIR:-${REPO_DIR}/sim_shards_msprime_sproxy}"

# --- what to simulate -------------------------------------------------------
REP_PER_TASK="${REP_PER_TASK:-10}"      # replicates PER array task
N_WINDOWS="${N_WINDOWS:-20}"           # 100 x 1Mb ~ 20 x 5Mb (SLiM) in total bp
WINDOW_SIZE="${WINDOW_SIZE:-5000000}"   # 1 Mb; one nu per window is defensible here
MASTER_SEED="${MASTER_SEED:-12345}"
TASK_OFFSET="${TASK_OFFSET:-0}"

# --- s-proxy expansion knobs  This is only useful if approximating selection with nu and will probably be removed in future versions.
PARAM_MODE="${PARAM_MODE:-prf-prune}"    # prf-prune (per-TE removal, default)
                                        # | s-proxy (the old nu expansion)
GAMMA_SCALE="${GAMMA_SCALE:-4.0}"       # gamma = SCALE*Ne*s; match SLiM dominance
S_REF="${S_REF:-1e-3}"                  # s mapped to upsilon=1 (drift barrier at Ne=500)
NU_START="${NU_START:-1000}"            # older edge of the expansion window (gen ago)
NU_DURATION="${NU_DURATION:-1000}"      # window length; r*2Ne = log(ups)*2Ne/duration
NU_ANCHOR="${NU_ANCHOR:-present}"       # pin the young edge (selection-signed density)
BLOCK_SIZE="${BLOCK_SIZE:-5000000}"      # sub-window block size; each block gets
                                        # its own local recomb -> own nu -> own tree
                                        # (this is what gives alpha_r contrast). 
S_GAIN="${S_GAIN:-1.0}"                  # upsilon = (s/s_ref)**S_GAIN
ALPHA_GAIN="${ALPHA_GAIN:-1.0}"          # multiplier on the alpha_r contrast only
THETA_FROM="${THETA_FROM:-}"            # optional: dataset base to take theta FROM,
                                        # pairing msprime rep i with SLiM sim i

# --- environment (matches 03_train.sh / mfnpe_train.sh) --------------------
if [ -z "${CONDA_PREFIX:-}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null \
        && conda activate "${CONDA_ENV}"
fi
module load slim/4.3
source /shared/projects/datepalm/test_DFE_deeplearning/.venv/bin/activate
source /shared/projects/datepalm/test_DFE_deeplearning/activate.sh
export PYTHONPATH="/shared/projects/datepalm/test_DFE_deeplearning/popgen-npe:${PYTHONPATH:-}"

# s-proxy needs the context model (theta must contain gamma_r/log_s_global/alpha_r)
export TE_DFE_MODEL=context
export MODEL=context

cd "${REPO_DIR}"
mkdir -p logs "${SHARD_DIR}"

# This task owns replicates [base, base + REP_PER_TASK).
BASE=$(( (TASK_OFFSET + SLURM_ARRAY_TASK_ID) * REP_PER_TASK ))
echo "[step1] task $((TASK_OFFSET + SLURM_ARRAY_TASK_ID)) -> replicates ${BASE}..$((BASE + REP_PER_TASK - 1))"

for i in $(seq 0 $((REP_PER_TASK - 1))); do
    TASK_ID=$(( BASE + i ))
    python 01_simulate_msprime.py \
        --param-mode "${PARAM_MODE}" --s-ref "${S_REF}" \
        --gamma-scale "${GAMMA_SCALE}" \
        --window-size "${WINDOW_SIZE}" --n-windows "${N_WINDOWS}" \
        --nu-start-time "${NU_START}" --nu-duration "${NU_DURATION}" \
        --nu-anchor "${NU_ANCHOR}" \
        --block-size "${BLOCK_SIZE}" \
        --s-gain "${S_GAIN}" --alpha-gain "${ALPHA_GAIN}" \
        ${THETA_FROM:+--theta-from "${THETA_FROM}"} \
        --task-id "${TASK_ID}" --n-replicates 1 \
        --master-seed "${MASTER_SEED}" \
        --out-dir "${SHARD_DIR}"
done
