#!/bin/bash
#SBATCH --job-name=dfe_moments
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dfe_moments_%A_%a.out
#SBATCH --error=logs/dfe_moments_%A_%a.err

# ============================================================================
# Fit the diffusion DFE (fit_dfe_moments.py) to every test_*.npz in a folder.
#
# ONE ARRAY TASK = one shard, so the fits are independent and a single bad
# shard cannot take the run down. No GPU: this is scipy + moments on one core.
#
#   mkdir -p logs
#   TEST_DIR=simple_sumstats_full_range_recomb \
#     sbatch --array=0-199%50 slurm/dfe_moments_sweep.sh
#
# Size the array to the number of shards:
#   ls simple_sumstats_full_range_recomb/test_*.npz | wc -l
#
# Each task writes <OUT_DIR>/<shard>.txt (the full report) and appends one row
# to <OUT_DIR>/summary.tsv. The rows are written with a lock so concurrent tasks
# cannot interleave inside a line.
#
# THE FIT IS NOT SEEDED AT THE TRUTH. --true-theta is display-only, so the truth
# column is for comparison and the estimates are an independent recovery.
# ============================================================================
set -eo pipefail
set +u
export PYTHONPATH="${PYTHONPATH:-}"

REPO_DIR="${REPO_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-te_npe}"
TEST_DIR="${TEST_DIR:?set TEST_DIR to the folder holding test_*.npz}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/dfe_moments_fits}"

# --- fit options ------------------------------------------------------------
BACKEND="${BACKEND:-moments}"           # moments | analytic
N_STARTS="${N_STARTS:-8}"
N_ROUNDS="${N_ROUNDS:-3}"               # round 1 = multistart, then refine
FD_EPS="${FD_EPS:-1e-3}"                # finite-difference step (see --fd-eps)
POLISH="${POLISH:-1}"                   # derivative-free Nelder-Mead refinement
S_CONVENTION="${S_CONVENTION:-homozygote}"   # SLiM's s is the HOMOZYGOTE effect
DOMINANCE="${DOMINANCE:-0.5}"
FIX="${FIX:-}"                          # e.g. FIX=log_duration=2.3026
EXTRA="${EXTRA:-}"                      # anything else, e.g. --no-burst

if [ -z "${CONDA_PREFIX:-}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null \
        && conda activate "${CONDA_ENV}"
fi
module load slim/4.3
source /shared/projects/datepalm/test_DFE_deeplearning/.venv/bin/activate
source /shared/projects/datepalm/test_DFE_deeplearning/activate.sh
export PYTHONPATH="/shared/projects/datepalm/test_DFE_deeplearning/popgen-npe:${PYTHONPATH:-}"

export TE_DFE_MODEL=simple
export MODEL=simple

cd "${REPO_DIR}"
mkdir -p logs "${OUT_DIR}"

# --- pick this task's shard -------------------------------------------------
mapfile -t SHARDS < <(ls "${TEST_DIR}"/test_*.npz | sort)
N=${#SHARDS[@]}
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${IDX}" -ge "${N}" ]; then
    echo "[dfe] task ${IDX} beyond the ${N} shards available; nothing to do"
    exit 0
fi
SHARD="${SHARDS[$IDX]}"
BASE=$(basename "${SHARD}" .npz)
REPORT="${OUT_DIR}/${BASE}.txt"
echo "[dfe] task ${IDX}/${N}: ${SHARD}"

# --- truth, for the display-only comparison column --------------------------
T=$(python - "$SHARD" <<'PY'
import sys, numpy as np
d = np.load(sys.argv[1])
print(",".join(f"{v:.6f}" for v in np.asarray(d["theta"]).reshape(-1))
      if "theta" in d.files else "")
PY
)

# --seed is derived from the array index. With a FIXED seed every shard draws
# the SAME random starts, which (together with the zero-gradient bug) is what
# made the fitted burst parameters collapse onto a handful of repeated values.
ARGS=(--shard "${SHARD}" --backend "${BACKEND}" --n-starts "${N_STARTS}"
      --n-rounds "${N_ROUNDS}" --fd-eps "${FD_EPS}" --seed "${IDX}"
      --s-convention "${S_CONVENTION}" --dominance "${DOMINANCE}")
[ "${POLISH}" = "1" ] && ARGS+=(--polish)
[ -n "${T}"   ] && ARGS+=(--true-theta="${T}")
[ -n "${FIX}" ] && ARGS+=(--fix "${FIX}")
# seq-length is read from the shard's seq_len_bp; no need to pass it.

python fit_dfe_moments.py "${ARGS[@]}" ${EXTRA} 2>&1 | tee "${REPORT}"

# --- one row per shard in a shared TSV --------------------------------------
SUMMARY="${OUT_DIR}/summary.tsv"
python - "${BASE}" "${REPORT}" "${SUMMARY}" "${T}" <<'PY'
import sys, os, re, fcntl
base, report, summary, truth = sys.argv[1:5]
txt = open(report).read()

names = ["log_lambda", "log_gamma_shape", "log_mean_s",
         "log10_t_start", "log_duration", "log_multiplier"]
est = {}
for nm in names:                      # "  log_mean_s   -6.0371   2.389e-03  ..."
    m = re.search(rf"^\s+{nm}\s+(-?\d+\.\d+)", txt, re.M)
    if m:
        est[nm] = m.group(1)
m = re.search(r"^\s+-logL\s+(\S+)", txt, re.M)
nll = m.group(1) if m else "NA"
m = re.search(r"(\d+)/(\d+) random start\(s\) within", txt)
conv = f"{m.group(1)}/{m.group(2)}" if m else "NA"
m = re.search(r"=== all : (\d+) segregating", txt)
nte = m.group(1) if m else "NA"

tvals = truth.split(",") if truth else [""] * len(names)
row = ([base, nte, nll, conv]
       + [est.get(n, "NA") for n in names]
       + [f"{float(v):.6f}" if v else "NA" for v in tvals])
header = (["shard", "n_te", "neg_logL", "converged"]
          + [f"fit_{n}" for n in names] + [f"true_{n}" for n in names])

with open(summary, "a") as fh:                 # lock: tasks run concurrently
    fcntl.flock(fh, fcntl.LOCK_EX)
    if fh.tell() == 0:
        fh.write("\t".join(header) + "\n")
    fh.write("\t".join(row) + "\n")
    fcntl.flock(fh, fcntl.LOCK_UN)
print(f"[dfe] appended {base} to {summary}")
PY
