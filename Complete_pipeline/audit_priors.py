#!/usr/bin/env python3
"""
audit_priors.py — Cross-check parameter ranges across three sources:

  1. config.py        — the prior ranges currently declared (*_RANGE constants)
  2. training data    — realized theta ranges in results/dataset.theta.pt
  3. test sims        — realized theta ranges in test_simulations/test_*.npz

Prints a per-parameter table comparing all three, and flags any source
that disagrees with the config prior. This catches the class of bug where
training data was generated under one config and benchmarked under another
(stale priors, refactored parameterization, changed normalizers).

Unlike ad-hoc diagnostic snippets, this reads ALL ranges live from
config.py — nothing is hardcoded — so it stays correct as you edit priors.

Usage:
    python audit_priors.py
    python audit_priors.py --dataset results/dataset --test-dir test_simulations/
    python audit_priors.py --tol 0.05      # tolerance for "match" (in raw theta units)

Exit code is non-zero if any mismatch is found, so it can gate a pipeline:
    python audit_priors.py && python 05b_benchmark_analyze.py ...
"""
# Model selection before te_dfe import (config dispatches at import time).
try:
    from model_select import prescan_model
    prescan_model()
except Exception:
    pass

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch


# Map each PARAM_NAMES entry to the config constant that defines its range.
# Most are "<UPPERNAME>_RANGE"; the two irregular ones are mapped explicitly.
def _range_const_name(param_name: str) -> str:
    explicit = {
        "log10_t_start": "LOG10_T_START_RANGE",
        "log_lambda":    "LOG_LAMBDA_RANGE",
        "log_gamma_shape": "LOG_GAMMA_SHAPE_RANGE",
        "log_mean_s":    "LOG_MEAN_S_RANGE",
        "log_duration":  "LOG_DURATION_RANGE",
        "log_multiplier": "LOG_MULTIPLIER_RANGE",
        # context-model params (present only in te_dfe_context configs)
        "alpha_d":       "ALPHA_D_RANGE",
        "alpha_r":       "ALPHA_R_RANGE",
        "gamma_d":       "GAMMA_D_RANGE",
        "gamma_r":       "GAMMA_R_RANGE",
        "log_s_global":  "LOG_S_GLOBAL_RANGE",
    }
    if param_name in explicit:
        return explicit[param_name]
    # Fallback: uppercase + _RANGE
    return param_name.upper() + "_RANGE"


def _load_config_ranges():
    """Return (param_names, {name: (lo, hi)}) read live from config.py."""
    from te_dfe import config
    names = list(config.PARAM_NAMES)
    ranges = {}
    missing = []
    for nm in names:
        const = _range_const_name(nm)
        if hasattr(config, const):
            lo, hi = getattr(config, const)
            ranges[nm] = (float(lo), float(hi))
        else:
            missing.append((nm, const))
            ranges[nm] = (None, None)
    return names, ranges, config.__file__, missing


def _load_training_ranges(dataset_base: Path):
    """Realized per-column theta ranges from the training memmap metadata."""
    theta_path = dataset_base.with_suffix(".theta.pt")
    if not theta_path.exists():
        return None, f"not found: {theta_path}"
    meta = torch.load(theta_path, weights_only=False)
    thetas = meta["thetas"]
    if hasattr(thetas, "numpy"):
        thetas = thetas.numpy()
    thetas = np.asarray(thetas)
    lo = thetas.min(axis=0)
    hi = thetas.max(axis=0)
    return (lo, hi), None


def _load_test_ranges(test_dir: Path, max_files: int = None):
    """Realized per-column theta ranges across test_*.npz files."""
    files = sorted(test_dir.glob("test_*.npz"))
    if not files:
        return None, f"no test_*.npz in {test_dir}", 0
    if max_files:
        files = files[:max_files]
    thetas = []
    for fp in files:
        d = np.load(fp)
        if "theta" not in d.files:
            continue
        thetas.append(np.asarray(d["theta"]).ravel())
    if not thetas:
        return None, f"no 'theta' key in any test_*.npz", 0
    arr = np.vstack(thetas)
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    return (lo, hi), None, len(thetas)


def _within(realized_lo, realized_hi, prior_lo, prior_hi, tol):
    """True if [realized_lo, realized_hi] sits within [prior_lo-tol, prior_hi+tol]."""
    if prior_lo is None:
        return None
    return (realized_lo >= prior_lo - tol) and (realized_hi <= prior_hi + tol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="results/dataset",
                    help="Training dataset basename (expects <base>.theta.pt).")
    ap.add_argument("--test-dir", type=str, default="test_simulations/",
                    help="Directory of test_*.npz benchmark sims.")
    ap.add_argument("--max-test-files", type=int, default=None,
                    help="Cap the number of test files scanned (speed).")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="Tolerance (raw theta units) for declaring a match.")
    args = ap.parse_args()

    # 1. Config (live)
    try:
        names, cfg_ranges, cfg_path, missing = _load_config_ranges()
    except Exception as e:
        print(f"ERROR: couldn't import config.py: {e}", file=sys.stderr)
        return 2
    print(f"config:        {cfg_path}")
    print(f"N_PARAMS:      {len(names)}")
    if missing:
        print(f"  WARNING: no *_RANGE constant found for: "
              f"{', '.join(f'{nm}->{c}' for nm, c in missing)}")

    # 2. Training data
    train, train_err = _load_training_ranges(Path(args.dataset))
    if train_err:
        print(f"training data: UNAVAILABLE ({train_err})")
    else:
        tr_lo, tr_hi = train
        print(f"training data: {Path(args.dataset).with_suffix('.theta.pt')} "
              f"({len(tr_lo)} params)")
        if len(tr_lo) != len(names):
            print(f"  ** PARAM-COUNT MISMATCH: config has {len(names)}, "
                  f"training data has {len(tr_lo)} **")

    # 3. Test sims
    test_res = _load_test_ranges(Path(args.test_dir), args.max_test_files)
    if test_res[1]:
        test, test_err = None, test_res[1]
        print(f"test sims:     UNAVAILABLE ({test_err})")
        n_test = 0
    else:
        test, test_err, n_test = test_res
        te_lo, te_hi = test
        print(f"test sims:     {args.test_dir}  ({n_test} files)")
        if len(te_lo) != len(names):
            print(f"  ** PARAM-COUNT MISMATCH: config has {len(names)}, "
                  f"test sims have {len(te_lo)} **")

    print()
    # Header
    hdr = (f"{'param':<16} {'config prior':>22}  "
           f"{'training realized':>22} {'':>3}  "
           f"{'test realized':>22} {'':>3}")
    print(hdr)
    print("-" * len(hdr))

    any_mismatch = False
    n = len(names)
    for i in range(n):
        nm = names[i]
        plo, phi = cfg_ranges[nm]
        # config column
        if plo is None:
            cfg_str = f"{'(no const)':>22}"
        else:
            cfg_str = f"[{plo:+8.3f},{phi:+8.3f}]"

        # training column
        if train_err or i >= len(train[0]):
            tr_str, tr_flag = f"{'--':>22}", "   "
        else:
            tlo, thi = float(train[0][i]), float(train[1][i])
            tr_str = f"[{tlo:+8.3f},{thi:+8.3f}]"
            ok = _within(tlo, thi, plo, phi, args.tol)
            tr_flag = "   " if ok or ok is None else "!!!"
            if ok is False:
                any_mismatch = True

        # test column
        if test is None or i >= len(test[0]):
            te_str, te_flag = f"{'--':>22}", "   "
        else:
            xlo, xhi = float(test[0][i]), float(test[1][i])
            te_str = f"[{xlo:+8.3f},{xhi:+8.3f}]"
            ok = _within(xlo, xhi, plo, phi, args.tol)
            te_flag = "   " if ok or ok is None else "!!!"
            if ok is False:
                any_mismatch = True

        print(f"{nm:<16} {cfg_str}  {tr_str} {tr_flag}  {te_str} {te_flag}")

    print()
    print("Legend: '!!!' = realized range falls outside the config prior "
          f"(tol={args.tol}).")
    print("        Training '!!!' => network was trained under a DIFFERENT prior")
    print("                          than config now declares (retrain or revert).")
    print("        Test     '!!!' => benchmark sims drawn under a DIFFERENT prior")
    print("                          than config (regenerate test sims).")

    # Direct train-vs-test comparison (the most actionable check for
    # benchmarking: these two MUST match for the posterior to be valid)
    if not train_err and test is not None:
        print()
        worst = 0.0
        for i in range(min(len(train[0]), len(test[0]), n)):
            d_lo = abs(float(train[0][i]) - float(test[0][i]))
            d_hi = abs(float(train[1][i]) - float(test[1][i]))
            worst = max(worst, d_lo, d_hi)
        if worst <= args.tol:
            print(f"TRAIN vs TEST: ranges agree (max edge diff {worst:.4f} "
                  f"<= tol {args.tol}).  Benchmark is valid.")
        else:
            print(f"TRAIN vs TEST: ** DISAGREE ** (max edge diff {worst:.4f} "
                  f"> tol {args.tol}).")
            print(f"               The network was trained on a different theta "
                  f"distribution than the test sims.")
            print(f"               Benchmark results will be invalid until these "
                  f"match.")
            any_mismatch = True

    print()
    if any_mismatch:
        print("RESULT: MISMATCH detected. See '!!!' rows above.")
        return 1
    print("RESULT: all sources consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
