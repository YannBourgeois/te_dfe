#!/bin/bash
#SBATCH --job-name=dfe_aged
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dfe_aged_%A_%a.out
#SBATCH --error=logs/dfe_aged_%A_%a.err

# ============================================================================
# Tree sequences -> TE ages -> AGE-PARTITIONED DFE fit.
#
# ONE ARRAY TASK = one simulation (all its window .trees files together).
#
#   mkdir -p logs
#   TREES_ROOT=debug_dumps sbatch --array=0-49%20 slurm/dfe_aged_sweep.sh
#
# LAYOUT EXPECTED
#   TREES_ROOT/
#     sim_000000/ sim_full_window_00.trees, sim_full_window_01.trees, ...
#     sim_000001/ ...
# which is what simulator_simple.py writes when debug_dump_dir is set. The
# normal pipeline discards the trees, so re-run a handful of replicates with
# that enabled; you only need enough to validate the method.
#
# WHY AGE PARTITIONING
#   In the marginal SFS a burst and purifying selection are confounded: both
#   leave an excess of rare insertions. Once insertions are split by AGE they
#   separate, because a burst changes the COUNT in one age window while
#   selection changes the SHAPE within every window.
#
# TWO MODES
#   TRUE ages   (default) read exactly from the tree sequence. Use this first:
#               it establishes whether the method works at all.
#   JITTERED    AGE_JITTER=0.3 multiplies each age by lognormal noise, emulating
#               a real age ESTIMATOR (GEVA / Relate / tsinfer+tsdate). Sweep it
#               to find how much age error the method tolerates BEFORE choosing
#               an estimator for real data.
#
# FIXING PARAMETERS
#   FIX="log_duration=2.3026" pins a parameter at a value and optimises the
#   rest (profile-likelihood style). Space- or comma-separate several, e.g.
#   FIX="log_duration=2.3026 log_gamma_shape=-2.0". Passed straight to the fit's
#   --fix, and the fixed value shows up in the report and the summary as normal.
#
# NOTE ON MISSING ROWS
#   Every task now writes exactly one summary row, even when the fit fails or
#   the truth can't be read: the fit's exit code goes in the fit_rc column and
#   the fields it couldn't produce are NA. A non-zero fit_rc points you at the
#   matching logs/dfe_aged_*.err. Nothing disappears silently.
# ============================================================================
set -eo pipefail
set +u
export PYTHONPATH="${PYTHONPATH:-}"

REPO_DIR="${REPO_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-te_npe}"
TREES_ROOT="${TREES_ROOT:?set TREES_ROOT to the folder of per-sim trees dirs}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/dfe_aged_fits}"

# --- what to fit ------------------------------------------------------------
AGE_EDGES="${AGE_EDGES:-0,25,50,100,200,400,800}"
SEQ_LENGTH="${SEQ_LENGTH:-}"            # bp behind the TEs; required by the fit
AGE_JITTER="${AGE_JITTER:-0}"           # 0 = true ages; 0.3 emulates an estimator
SELFING="${SELFING:-0}"                 # S in [0,1); exact rescale at h=0.5
DOMINANCE="${DOMINANCE:-0.5}"
S_CONVENTION="${S_CONVENTION:-homozygote}"
N_STARTS="${N_STARTS:-8}"
N_ROUNDS="${N_ROUNDS:-3}"
MAX_TE="${MAX_TE:-}"                    # subsample, to match the network's budget
FIX="${FIX:-}"                          # e.g. FIX="log_duration=2.3026" (repeatable)
DIAGNOSTICS="${DIAGNOSTICS:-1}"         # 1 = write <sim>.sfs_fit.tsv/.png (obs vs fitted per age bin)
BOOTSTRAP="${BOOTSTRAP:-100}"           # N Godambe bootstrap replicates; 0 = FIM only
BOOT_BLOCK_COL="${BOOT_BLOCK_COL:-window}"  # "window": resample linkage blocks, not individual TEs
MODEL_KIND="${MODEL_KIND:-simple}"      # simple = fit_dfe_moments_aged.py; context = fit_dfe_moments_context.py
RECOMB="${RECOMB:-auto}"                # "auto" finds <stem>_recomb.npz; "" to skip (context model NEEDS it)
NEUTRAL="${NEUTRAL:-0}"                 # 1 = extract neutral SNPs and calibrate the fit on them
NEUTRAL_CALIBRATION="${NEUTRAL_CALIBRATION:-full}"   # full | shape
MAX_SNPS="${MAX_SNPS:-20000}"           # neutral SNPs kept per window
AGE_ERROR_SIGMA="${AGE_ERROR_SIGMA:-0}" # model age-measurement error (0 = off)
FIT_AGE_ERROR="${FIT_AGE_ERROR:-0}"     # 1 = ESTIMATE sigma instead of fixing it
RECOMB_BINS="${RECOMB_BINS:-4}"         # context model: number of recombination bins
WRITE_VCF="${WRITE_VCF:-0}"             # 1 = also emit a phased VCF for GEVA
EXTRA="${EXTRA:-}"

if [ -z "${CONDA_PREFIX:-}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null \
        && conda activate "${CONDA_ENV}"
fi
module load slim/4.3
source /shared/projects/datepalm/test_DFE_deeplearning/.venv/bin/activate
source /shared/projects/datepalm/test_DFE_deeplearning/activate.sh
export PYTHONPATH="/shared/projects/datepalm/test_DFE_deeplearning/popgen-npe:${PYTHONPATH:-}"

export TE_DFE_MODEL="${MODEL_KIND}"
export MODEL="${MODEL_KIND}"

cd "${REPO_DIR}"
mkdir -p logs "${OUT_DIR}"

# --- pick this task's simulation -------------------------------------------
mapfile -t SIMS < <(find "${TREES_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)
N=${#SIMS[@]}
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${IDX}" -ge "${N}" ]; then
    echo "[aged] task ${IDX} beyond the ${N} sims available; nothing to do"
    exit 0
fi
SIMDIR="${SIMS[$IDX]}"
BASE=$(basename "${SIMDIR}")
echo "[aged] task ${IDX}/${N}: ${SIMDIR}"

mapfile -t TREES < <(find "${SIMDIR}" -name '*.trees' | sort)
if [ "${#TREES[@]}" -eq 0 ]; then
    echo "[aged] no .trees under ${SIMDIR}; did you set debug_dump_dir when simulating?" >&2
    exit 0
fi
echo "[aged] ${#TREES[@]} window tree sequence(s)"

AGES="${OUT_DIR}/${BASE}.ages.tsv"
REPORT="${OUT_DIR}/${BASE}.fit.txt"

# --- step 1: exact ages out of the tree sequences ---------------------------
SNPS="${OUT_DIR}/${BASE}.snps.tsv"
EX_ARGS=(--trees "${TREES[@]}" --out "${AGES}")
if [ -n "${RECOMB}" ];    then EX_ARGS+=(--recomb "${RECOMB}"); fi
if [ "${NEUTRAL}" = "1" ]; then
    EX_ARGS+=(--snps-out "${SNPS}" --max-snps-per-window "${MAX_SNPS}")
fi
if [ "${WRITE_VCF}" = "1" ]; then
    EX_ARGS+=(--vcf "${OUT_DIR}/${BASE}.vcf")
fi
python extract_te_ages.py "${EX_ARGS[@]}"

# Sequence length is written into the ages file header by extract_te_ages.py,
# already summed over this simulation's windows, so the fit reads it itself.
# Set SEQ_LENGTH only to override.

# --- truth, display only ----------------------------------------------------
# The sim dirs hold metadata.npz (alongside win_NN.trees and win_NN_recomb.npz),
# not theta.txt, so read theta from there. Falls back to theta.txt if present.
# The whole body is wrapped so a missing/corrupt metadata.npz prints a note and
# yields an empty truth instead of raising -- otherwise `T=$(...)` returns
# non-zero and `set -e` would kill the task before it writes its summary row.
T=$(python - "${SIMDIR}" <<'PY'
import sys, numpy as np
from pathlib import Path
d = Path(sys.argv[1])
def emit(v):
    v = np.asarray(v, float).reshape(-1)
    print(",".join(f"{x:.6f}" for x in v)); sys.exit(0)
try:
    meta = d / "metadata.npz"
    if meta.exists():
        z = np.load(meta, allow_pickle=True)
        for key in ("theta", "params", "parameters", "true_theta"):
            if key in z.files:
                emit(z[key])
        # nothing obvious: report the keys to stderr so it is fixable, not silent
        print(f"[truth] metadata.npz has no theta-like key; saw {list(z.files)}",
              file=sys.stderr)
    elif (d / "theta.txt").exists():
        emit(np.loadtxt(d / "theta.txt"))
    else:
        print(f"[truth] no metadata.npz or theta.txt in {d}", file=sys.stderr)
except Exception as e:
    print(f"[truth] could not read truth for {d}: {type(e).__name__}: {e}",
          file=sys.stderr)
print("")
PY
)
if [ -z "${T}" ]; then
    echo "[aged] no truth found; fitting without a comparison column"
fi

# --- step 2: the age-partitioned fit ---------------------------------------
FIT_ARGS=(--ages "${AGES}" --age-edges "${AGE_EDGES}"
          --n-starts "${N_STARTS}" --n-rounds "${N_ROUNDS}" --seed "${IDX}"
          --s-convention "${S_CONVENTION}" --dominance "${DOMINANCE}"
          --selfing "${SELFING}")
if [ -n "${SEQ_LENGTH}" ]; then FIT_ARGS+=(--seq-length "${SEQ_LENGTH}"); fi
if [ -n "${T}" ];          then FIT_ARGS+=(--true-theta="${T}");        fi
if [ -n "${MAX_TE}" ];     then FIT_ARGS+=(--max-te "${MAX_TE}");       fi
if [ "${AGE_JITTER}" != "0" ]; then FIT_ARGS+=(--jitter-ages "${AGE_JITTER}"); fi
# Fixed parameters: split FIX on spaces/commas, one --fix per entry.
if [ -n "${FIX}" ]; then
    IFS=', ' read -r -a _FIX_ARR <<< "${FIX}"
    for f in "${_FIX_ARR[@]}"; do
        [ -n "${f}" ] && FIT_ARGS+=(--fix "${f}")
    done
fi

if [ "${DIAGNOSTICS}" = "1" ]; then FIT_ARGS+=(--diagnostics "${OUT_DIR}/${BASE}"); fi
# age-measurement-error model (shared by both fits)
if [ "${AGE_ERROR_SIGMA}" != "0" ]; then FIT_ARGS+=(--age-error-sigma "${AGE_ERROR_SIGMA}"); fi
if [ "${FIT_AGE_ERROR}" = "1" ];     then FIT_ARGS+=(--fit-age-error); fi
# empirical neutral baseline (shared by both fits)
if [ "${NEUTRAL}" = "1" ]; then
    FIT_ARGS+=(--neutral-ages "${SNPS}" --neutral-calibration "${NEUTRAL_CALIBRATION}")
fi

if [ "${MODEL_KIND}" = "context" ]; then
    FIT_SCRIPT=fit_dfe_moments_context.py
    export TE_DFE_MODEL=context MODEL=context
    FIT_ARGS+=(--recomb-bins "${RECOMB_BINS}")
    if [ "${BOOTSTRAP}" -gt 0 ] 2>/dev/null; then
        FIT_ARGS+=(--uncerts-bootstrap-refit "${BOOTSTRAP}")
    else
        FIT_ARGS+=(--uncerts)
    fi
    if [ -n "${BOOT_BLOCK_COL}" ]; then FIT_ARGS+=(--boot-block-col "${BOOT_BLOCK_COL}"); fi
else
    FIT_SCRIPT=fit_dfe_moments_aged.py
    if [ "${BOOTSTRAP}" -gt 0 ] 2>/dev/null; then
        FIT_ARGS+=(--uncerts-bootstrap-refit "${BOOTSTRAP}")
    else
        FIT_ARGS+=(--uncerts)
    fi
    if [ -n "${BOOT_BLOCK_COL}" ]; then FIT_ARGS+=(--boot-block-col "${BOOT_BLOCK_COL}"); fi
fi
echo "[aged] model=${MODEL_KIND} script=${FIT_SCRIPT}"

# Run the fit WITHOUT letting a non-zero exit abort the task: we still want to
# write a summary row (with NA fields) so the sim is never silently missing.
set +e
python "${FIT_SCRIPT}" "${FIT_ARGS[@]}" ${EXTRA} 2>&1 | tee "${REPORT}"
FIT_RC=${PIPESTATUS[0]}
set -e
if [ "${FIT_RC}" -ne 0 ]; then
    echo "[aged] fit exited ${FIT_RC} for ${BASE}; writing a row with NA fields " \
         "(see logs/dfe_aged_*.err for this task)" >&2
fi

# --- one row per sim in a shared TSV ---------------------------------------
# Always runs, whatever the fit did. Truth is truncated to the fitted parameter
# count so a longer metadata theta vector cannot produce ragged rows.
python - "${BASE}" "${REPORT}" "${OUT_DIR}/summary.tsv" "${T}" "${AGE_JITTER}" "${FIT_RC}" <<'PY'
import sys, re, fcntl
base, report, summary, truth, jitter, fit_rc = sys.argv[1:7]
try:
    txt = open(report).read()
except FileNotFoundError:
    txt = ""
names = ["log_lambda", "log_gamma_shape", "log_mean_s",
         "log10_t_start", "log_duration", "log_multiplier"]
est = {}
for nm in names:
    m = re.search(rf"^\s+{nm}\s+(-?\d+\.\d+)", txt, re.M)
    if m:
        est[nm] = m.group(1)
m = re.search(r"^\s+-logL\s+(\S+)", txt, re.M);      nll  = m.group(1) if m else "NA"
m = re.search(r"(\d+)/(\d+) starts within", txt)
conv = f"{m.group(1)}/{m.group(2)}" if m else "NA"
m = re.search(r"data\s+:\s+(\d+) TEs", txt);         nte  = m.group(1) if m else "NA"
# exactly len(names) truth fields, padded/truncated so rows never go ragged
tvals = (truth.split(",") if truth else [])[:len(names)]
tvals += [""] * (len(names) - len(tvals))
row = ([base, nte, jitter, nll, conv]
       + [est.get(n, "NA") for n in names]
       + [f"{float(v):.6f}" if v else "NA" for v in tvals]
       + [fit_rc])
header = (["sim", "n_te", "age_jitter", "neg_logL", "converged"]
          + [f"fit_{n}" for n in names] + [f"true_{n}" for n in names]
          + ["fit_rc"])
with open(summary, "a") as fh:
    fcntl.flock(fh, fcntl.LOCK_EX)
    if fh.tell() == 0:
        fh.write("\t".join(header) + "\n")
    fh.write("\t".join(row) + "\n")
    fcntl.flock(fh, fcntl.LOCK_UN)
print(f"[aged] appended {base} to {summary} (fit_rc={fit_rc})")
PY
