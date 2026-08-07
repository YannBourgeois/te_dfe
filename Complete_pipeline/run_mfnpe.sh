#!/bin/bash
# ============================================================================
# run_mfnpe.sh — the whole MF-NPE experiment in one submission.
#
#   Stage 1  PRETRAIN  on many CHEAP msprime sims (--param-mode s-proxy, so
#                      theta is the REAL selection parameterisation and matches
#                      SLiM's).
#   Stage 2  For each SLiM budget N, submit TWO jobs that differ only in
#            --init-from:
#              FINETUNE  = pretrained weights + N high-fidelity sims
#              SCRATCH   = random init      + N high-fidelity sims
#
# The GAP between the two curves is the measurement: how many expensive SLiM
# sims the cheap msprime pretraining is worth. Published MF-NPE results report
# up to two orders of magnitude fewer high-fidelity sims, but that is their
# simulator, not yours -- this measures YOURS.
#
# Prereqs
#   * an msprime dataset collated from --param-mode s-proxy shards
#   * a SLiM dataset (your existing one)
#   * BOTH collated with the SAME config (REPRESENTATION, PARAM_NAMES,
#     N_SUMSTAT_GLOBAL_COV). If they disagree, --init-from fails loudly.
#
# Usage
#   MSPRIME_DATASET=simulations_msprime/dataset \
#   SLIM_DATASET=results/dataset \
#   BUDGETS="100 300 1000 3000" \
#   ./run_mfnpe.sh
# ============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-te_npe}"
MSPRIME_DATASET="${MSPRIME_DATASET:-simulations_msprime/dataset}"
SLIM_DATASET="${SLIM_DATASET:-results/dataset}"
OUT_ROOT="${OUT_ROOT:-mfnpe}"
BUDGETS="${BUDGETS:-100 250 500 1000 2500}"

# Architecture — shared by every job. Pretrain and fine-tune MUST agree.
HEADS="${HEADS:-6}"
AGGREGATOR="${AGGREGATOR:-sigmoid_attn}"
VARIABLE_K="${VARIABLE_K:-1}"
N_MIN="${N_MIN:-20}"
N_MAX="${N_MAX:-1000}"

PRETRAIN_LR="${PRETRAIN_LR:-5e-4}"
# Fine-tuning uses a LOWER lr so a few high-fidelity sims CORRECT the pretrained
# map rather than overwrite it. Scratch keeps the normal lr -- it has nothing to
# preserve, and handicapping it would fake the comparison.
FINETUNE_LR="${FINETUNE_LR:-1e-4}"
SCRATCH_LR="${SCRATCH_LR:-5e-4}"

cd "${REPO_DIR}"; mkdir -p logs "${OUT_ROOT}"

# ---- Stage 1: pretrain on the cheap simulator ------------------------------
PRE_OUT="${OUT_ROOT}/pretrain_msprime"
echo "[stage1] pretrain on ${MSPRIME_DATASET} -> ${PRE_OUT}"
PRE_JOB=$(REPO_DIR="$REPO_DIR" CONDA_ENV="$CONDA_ENV" \
          DATASET="$MSPRIME_DATASET" OUT="$PRE_OUT" LR="$PRETRAIN_LR" \
          HEADS="$HEADS" AGGREGATOR="$AGGREGATOR" VARIABLE_K="$VARIABLE_K" \
          N_MIN="$N_MIN" N_MAX="$N_MAX" \
          sbatch --parsable slurm/mfnpe_train.sh)
echo "         job ${PRE_JOB}"

# ---- Stage 2: paired fine-tune / scratch at each budget --------------------
for N in ${BUDGETS}; do
  FT_OUT="${OUT_ROOT}/finetune_n${N}"
  SC_OUT="${OUT_ROOT}/scratch_n${N}"

  FT=$(REPO_DIR="$REPO_DIR" CONDA_ENV="$CONDA_ENV" \
       DATASET="$SLIM_DATASET" OUT="$FT_OUT" INIT_FROM="$PRE_OUT" \
       SUBSET_N="$N" LR="$FINETUNE_LR" \
       HEADS="$HEADS" AGGREGATOR="$AGGREGATOR" VARIABLE_K="$VARIABLE_K" \
       N_MIN="$N_MIN" N_MAX="$N_MAX" \
       sbatch --parsable --dependency=afterok:"$PRE_JOB" slurm/mfnpe_train.sh)

  # Scratch does NOT depend on the pretrain job -- it can start immediately.
  SC=$(REPO_DIR="$REPO_DIR" CONDA_ENV="$CONDA_ENV" \
       DATASET="$SLIM_DATASET" OUT="$SC_OUT" \
       SUBSET_N="$N" LR="$SCRATCH_LR" \
       HEADS="$HEADS" AGGREGATOR="$AGGREGATOR" VARIABLE_K="$VARIABLE_K" \
       N_MIN="$N_MIN" N_MAX="$N_MAX" \
       sbatch --parsable slurm/mfnpe_train.sh)

  echo "[stage2] N=${N}:  finetune ${FT} (afterok:${PRE_JOB})   scratch ${SC}"
done

cat <<EOF

Submitted. When everything finishes, compare each pair on the SAME benchmark:

  for N in ${BUDGETS}; do
    for arm in finetune scratch; do
      python 05b_benchmark_analyze.py --run-dir ${OUT_ROOT}/\${arm}_n\${N} ...
    done
  done

Read it as TWO curves, RMSE (and coverage) vs N:
  * finetune well below scratch at small N  -> the cheap sims are buying you
    high-fidelity budget; the gap in N at equal RMSE is the saving.
  * curves meeting at large N              -> expected: enough SLiM makes the
    pretraining redundant. The saving lives at the left end.
  * finetune ABOVE scratch                 -> NEGATIVE TRANSFER. The s->nu map
    is misleading (check it is monotone) or the proxy is invalid over much of
    the prior -- see the calres map from calibration_diagnostics.py.

Also check COVERAGE, not just RMSE: the risk case is confident-but-wrong in the
corner where transfer failed AND SLiM coverage was thin (low recombination,
strong selection).
EOF
