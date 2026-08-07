#!/usr/bin/env python3
"""
Export ORDER-INVARIANT per-feature summaries of the per-TE summary statistics,
one row per simulation, to a wide CSV — for classical ABC / PCA on the summary
vector.

Why summaries and not the raw per-TE matrix: the TEs in a replicate are an
unordered SET, so the j-th TE is not a stable entity across simulations.
Flattening (n_TE x n_feature) would make every column an arbitrary
order-dependent slot and a PCA on it would chase ordering noise. Summarizing
each feature ACROSS the TE set (mean, sd, quantiles, ...) is permutation-
invariant: the resulting vector has fixed width regardless of how many TEs a
replicate has, and the columns mean the same thing in every simulation. This is
the standard ABC reduction.

Per feature f, over the VALID (unpadded) TEs of a simulation, we compute:
    mean, sd, min, q10, q25, median, q75, q90, max, and the valid-TE count.
With 16 features that is 16*9 + 1 = 145 summary columns (the count is shared,
not per feature), plus the model parameters (theta) as the FIRST columns — the
layout ABC tools expect (parameters then summaries).

Reads a collated sumstat dataset (the output of 02_collate.py):
    <base>.feats.npy    (n_sims, N_TE_TRAIN_MAX, N_SUMSTATS_PER_TE) float32
    <base>.te_mask.npy  (n_sims, N_TE_TRAIN_MAX)                    float32 (1=valid)
    <base>.theta.pt     dict with "thetas" (n_sims, n_params)

Usage:
    python 07_export_sumstat_summaries.py --dataset sims/train --out summaries.csv
    python 07_export_sumstat_summaries.py --model context --dataset sims/train \
        --out summaries.csv [--quantiles 0.1,0.25,0.5,0.75,0.9] [--drop-validity]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Resolve the active model BEFORE importing te_dfe.config, so feature names and
# parameter names match the dataset's model.
from model_select import prescan_model, add_model_arg
prescan_model()

from te_dfe.config import PARAM_NAMES                       # noqa: E402
from te_dfe.sumstats import FEATURE_NAMES, subsample_valid_indices, N_SUMSTATS_PER_TE  # noqa: E402

# Features that are reliability / validity flags rather than biological signal;
# their cross-TE quantiles are rarely meaningful for ABC. --drop-validity omits
# them from the summarized set (their mean is still informative, so we keep that
# one moment when dropping the rest — see below).
_VALIDITY_FEATURES = {"valid_tajD", "valid_hapext", "n_flank_snps_norm"}


# ADDED: cross-feature correlations over the kept TEs, emitted one column each as
# order-invariant per-sim ABC summaries. dist_to_gene is a DISTANCE (gene
# proximity proxy), so a positive corr with te_density means denser TE regions
# sit FARTHER from genes (the BGS/selection asymmetry). Add pairs here freely.
_CORR_PAIRS = [
    ("te_freq",          "local_recomb"),    # frequency vs recomb        -> alpha_r
    ("local_te_density", "local_recomb"),    # TE clustering vs recomb     -> gamma_r
    ("local_te_density", "dist_to_gene"),    # TE density vs gene distance -> BGS/selection
]


def _valid_corr(valid, i, j):
    """Pearson correlation of feature columns i, j over the kept TEs of one
    simulation. NaN if <2 TEs or a column is constant."""
    if valid.shape[0] < 2:
        return np.nan
    a, b = valid[:, i], valid[:, j]
    if a.std() <= 1e-9 or b.std() <= 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _summarize_feature(values, qs):
    """Order-invariant moments + quantiles of one feature over the valid TEs.
    `values` is 1-D over the valid TEs of a single simulation."""
    if values.size == 0:
        # No valid TEs: emit NaNs so the row is identifiable downstream rather
        # than silently zero (zero is a legitimate feature value).
        return [np.nan] * (4 + len(qs))
    out = [float(values.mean()),
           float(values.std()),
           float(values.min()),
           float(values.max())]
    out.extend(float(v) for v in np.quantile(values, qs))
    return out


def _summary_columns(feature_names, qs):
    """Column names for the summary block, matching _summarize_feature order."""
    cols = []
    qlabels = [f"q{int(round(q*100)):02d}" for q in qs]
    for fname in feature_names:
        cols.append(f"{fname}__mean")
        cols.append(f"{fname}__sd")
        cols.append(f"{fname}__min")
        cols.append(f"{fname}__max")
        cols.extend(f"{fname}__{ql}" for ql in qlabels)
    return cols


def main():
    ap = argparse.ArgumentParser()
    add_model_arg(ap)
    ap.add_argument("--dataset", required=True,
                    help="collated dataset base path (without .feats.npy etc.)")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--quantiles", default="0.1,0.25,0.5,0.75,0.9",
                    help="comma-separated quantiles to include per feature")
    ap.add_argument("--drop-validity", action="store_true",
                    help="summarize only biological features (keep validity/"
                         "reliability features as a single mean each)")
    ap.add_argument("--n-te-target", type=int, default=None,
                    help="Subsample each replicate to N random valid TEs before "
                         "summarizing (TE-count sweep for ABC). Default: all.")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for --n-te-target subsampling.")
    args = ap.parse_args()

    qs = [float(x) for x in args.quantiles.split(",") if x.strip()]
    base = Path(args.dataset)
    feats_path = base.with_suffix(".feats.npy")
    mask_path  = base.with_suffix(".te_mask.npy")
    theta_path = base.with_suffix(".theta.pt")
    for p in (feats_path, mask_path, theta_path):
        if not p.exists():
            sys.exit(f"ERROR: missing {p} — is --dataset a collated sumstat set?")

    feats = np.load(feats_path, mmap_mode="r")   # (n_sims, K, F)
    mask  = np.load(mask_path,  mmap_mode="r")   # (n_sims, K)
    meta  = torch.load(theta_path, map_location="cpu", weights_only=False)
    thetas = np.asarray(meta["thetas"], dtype=np.float64)  # (n_sims, n_params)

    n_sims, K, F = feats.shape
    if F != N_SUMSTATS_PER_TE or F != len(FEATURE_NAMES):
        sys.exit(f"ERROR: dataset has {F} features but the active model expects "
                 f"{N_SUMSTATS_PER_TE} ({len(FEATURE_NAMES)} names). Wrong "
                 f"--model for this dataset?")
    if thetas.shape[0] != n_sims:
        sys.exit(f"ERROR: {n_sims} sims in feats but {thetas.shape[0]} thetas.")
    if thetas.ndim != 2 or thetas.shape[1] != len(PARAM_NAMES):
        got = thetas.shape[1] if thetas.ndim == 2 else "?"
        sys.exit(f"ERROR: dataset has {got} parameters but the active model "
                 f"expects {len(PARAM_NAMES)} ({list(PARAM_NAMES)}). The per-TE "
                 f"feature count is identical across models, so a wrong --model "
                 f"shows up here, not in the feature check — pass the --model the "
                 f"dataset was simulated under.")

    # Choose which features get the full summary vs. mean-only.
    if args.drop_validity:
        full_idx = [i for i, n in enumerate(FEATURE_NAMES)
                    if n not in _VALIDITY_FEATURES]
        mean_only_idx = [i for i, n in enumerate(FEATURE_NAMES)
                         if n in _VALIDITY_FEATURES]
    else:
        full_idx = list(range(F))
        mean_only_idx = []

    full_names = [FEATURE_NAMES[i] for i in full_idx]
    mean_only_names = [FEATURE_NAMES[i] for i in mean_only_idx]

    # Header: parameters first (ABC convention), then n_valid_te, then summaries.
    param_cols = list(PARAM_NAMES)
    summary_cols = _summary_columns(full_names, qs)
    meanonly_cols = [f"{n}__mean" for n in mean_only_names]
    header = param_cols + ["n_valid_te"] + summary_cols + meanonly_cols
    # ADDED: one column per correlation pair.
    _corr_cols = [f"corr_{a}_{b}" for (a, b) in _CORR_PAIRS]
    _corr_idx = [(FEATURE_NAMES.index(a), FEATURE_NAMES.index(b))
                 for (a, b) in _CORR_PAIRS]
    header = header + _corr_cols

    n_q = 4 + len(qs)
    print(f"Exporting {n_sims} sims | {F} features | "
          f"{len(full_names)} fully summarized x {n_q} stats "
          f"+ {len(mean_only_names)} mean-only "
          f"= {len(summary_cols) + len(meanonly_cols)} summary cols "
          f"+ {len(param_cols)} params")

    with open(args.out, "w") as fh:
        fh.write(",".join(header) + "\n")
        for s in range(n_sims):
            fmat = np.asarray(feats[s])          # (K, F)
            rng = np.random.default_rng([args.seed, s])
            idx = subsample_valid_indices(mask[s], args.n_te_target, rng)
            valid = fmat[idx]                    # (n_kept, F)
            row = [f"{v:.8g}" for v in thetas[s]]
            row.append(str(int(idx.size)))       # n_valid_te = kept count
            for i in full_idx:
                row.extend(f"{v:.6g}" for v in _summarize_feature(valid[:, i], qs))
            for i in mean_only_idx:
                col = valid[:, i]
                row.append(f"{(col.mean() if col.size else np.nan):.6g}")
            # ADDED: cross-feature correlations over the same kept set.
            for (i, j) in _corr_idx:
                row.append(f"{_valid_corr(valid, i, j):.6g}")
            fh.write(",".join(row) + "\n")
            if (s + 1) % 1000 == 0 or s == n_sims - 1:
                print(f"  wrote {s+1}/{n_sims}")

    print(f"\nWrote {args.out}  ({len(header)} columns, {n_sims} rows).")
    print("Layout: parameters | n_valid_te | per-feature summaries. "
          "Feed parameters as the targets and the summary columns as the ABC "
          "summary statistics (or run PCA on the summary columns).")


if __name__ == "__main__":
    main()
