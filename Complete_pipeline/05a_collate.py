#!/usr/bin/env python3
"""
05a_collate.py — Concatenate per-window-chunk shards into per-scenario test
files that 05b consumes.

Reads all scenario_<id>_w<start>-<end>.npz shards from --shard-dir, groups by
scenario, concatenates geno/pos/cov along the window axis in window-index order,
and writes test_<id>.npz (format: geno, pos, cov, theta) to --out-dir.

Validates per scenario:
  * all chunks share the same theta (within tolerance)
  * window indices are contiguous 0..n_windows_total-1 with no gaps/dupes
  * reports any scenario missing windows (failed shards) — these can be
    re-run by addressing the specific (scenario, chunk) before re-collating.

Usage:
    python 05a_collate.py --shard-dir test_shards/ --out-dir test_simulations/
    python 05a_collate.py --shard-dir test_shards/ --out-dir test_simulations/ \
        --require-complete        # skip scenarios with any missing windows
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Model selection before any te_dfe import (the sumstat branch reads config).
try:
    from model_select import prescan_model, add_model_arg
    prescan_model()
except Exception:
    add_model_arg = None


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shard-dir", type=str, default="test_shards/")
    p.add_argument("--out-dir",   type=str, default="test_simulations/")
    p.add_argument("--require-complete", action="store_true",
                   help="Skip (don't write) scenarios that are missing windows.")
    p.add_argument("--theta-tol", type=float, default=1e-5,
                   help="Max allowed theta disagreement across a scenario's "
                        "shards.")
    if add_model_arg is not None:
        add_model_arg(p)
    return p.parse_args()


def main():
    args = parse_args()
    shard_dir = Path(args.shard_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(shard_dir.glob("scenario_*_w*.npz"))
    if not shards:
        print(f"No shards found in {shard_dir}", file=sys.stderr)
        return 1

    # Group shard paths by scenario_id.
    by_scenario = defaultdict(list)
    for sp in shards:
        d = np.load(sp)
        sid = int(d["scenario_id"])
        by_scenario[sid].append(sp)

    print(f"Found {len(shards)} shards across {len(by_scenario)} scenarios")
    n_written = n_incomplete = n_bad = 0

    for sid in sorted(by_scenario):
        paths = by_scenario[sid]
        # Peek at the first shard to detect representation.
        is_sumstat = "feats_raw" in set(np.load(paths[0]).files)

        if is_sumstat:
            # Sumstat: pool every chunk's raw per-TE features for this scenario
            # into ONE set (the scenario's TEs span all its window-chunks), then
            # pad/subsample once to N_TE_TRAIN_MAX — the training collated format.
            from te_dfe.config import N_TE_TRAIN_MAX, N_SUMSTATS_PER_TE
            thetas, n_total, pooled, n_win_have = [], None, [], 0
            for sp in paths:
                d = np.load(sp)
                thetas.append(d["theta"])
                n_total = int(d["n_windows_total"]) if n_total is None else n_total
                n_win_have += len(d["window_indices"])
                fr = np.asarray(d["feats_raw"], dtype=np.float32)
                if fr.shape[0]:
                    pooled.append(fr)
            thetas = np.stack(thetas)
            spread = float(np.max(np.abs(thetas - thetas[0:1])))
            if spread > args.theta_tol:
                print(f"  scenario {sid}: !! THETA MISMATCH across shards "
                      f"(max diff {spread:.2e}) — skipping", file=sys.stderr)
                n_bad += 1
                continue
            theta = thetas[0]
            if n_win_have < n_total:
                msg = f"  scenario {sid}: {n_win_have}/{n_total} windows"
                if args.require_complete:
                    print(msg + " — skipped (require-complete)", file=sys.stderr)
                    n_incomplete += 1
                    continue
                print(msg + " — writing partial")
                n_incomplete += 1
            allte = (np.concatenate(pooled, axis=0) if pooled
                     else np.zeros((0, N_SUMSTATS_PER_TE), np.float32))
            feats = np.zeros((N_TE_TRAIN_MAX, N_SUMSTATS_PER_TE), dtype=np.float32)
            te_mask = np.zeros((N_TE_TRAIN_MAX,), dtype=np.float32)
            n_te = min(allte.shape[0], N_TE_TRAIN_MAX)
            if allte.shape[0] > N_TE_TRAIN_MAX:
                # deterministic subsample keyed by scenario id (reproducible)
                rng = np.random.default_rng(sid)
                sel = np.sort(rng.choice(allte.shape[0], N_TE_TRAIN_MAX,
                                         replace=False))
                feats[:] = allte[sel]; te_mask[:] = 1.0
            elif n_te:
                feats[:n_te] = allte[:n_te]; te_mask[:n_te] = 1.0
            out_path = out_dir / f"test_{sid:06d}.npz"
            from te_dfe.config import WINDOW_SIZE as _WS
            np.savez_compressed(out_path, feats=feats, te_mask=te_mask,
                                theta=theta,
                                seq_len_bp=np.float64(float(n_total) * float(_WS)))
            print(f"  scenario {sid}: wrote {out_path} feats={feats.shape} "
                  f"n_te={int(te_mask.sum())} (pooled from {n_win_have} windows)")
            n_written += 1
            continue

        # ---- matrix representation (per-window geno/pos/cov) ----
        # Load all chunks, accumulate (window_idx -> arrays).
        per_window = {}
        thetas = []
        n_total = None
        for sp in paths:
            d = np.load(sp)
            thetas.append(d["theta"])
            n_total = int(d["n_windows_total"]) if n_total is None else n_total
            widx = d["window_indices"]
            geno, pos, cov = d["geno"], d["pos"], d["cov"]
            for k, w in enumerate(widx):
                per_window[int(w)] = (geno[k], pos[k], cov[k])

        # Validate theta consistency.
        thetas = np.stack(thetas)
        spread = float(np.max(np.abs(thetas - thetas[0:1])))
        if spread > args.theta_tol:
            print(f"  scenario {sid}: !! THETA MISMATCH across shards "
                  f"(max diff {spread:.2e}) — skipping", file=sys.stderr)
            n_bad += 1
            continue
        theta = thetas[0]

        # Check completeness.
        have = sorted(per_window)
        expected = list(range(n_total))
        missing = sorted(set(expected) - set(have))
        if missing:
            msg = (f"  scenario {sid}: {len(have)}/{n_total} windows "
                   f"(missing {len(missing)}: "
                   f"{missing[:8]}{'...' if len(missing) > 8 else ''})")
            if args.require_complete:
                print(msg + " — skipped (require-complete)", file=sys.stderr)
                n_incomplete += 1
                continue
            else:
                print(msg + " — writing partial")
                n_incomplete += 1

        # Concatenate in window-index order (use the indices we actually have).
        order = have
        geno = np.stack([per_window[w][0] for w in order], axis=0).astype(np.int8)
        pos  = np.stack([per_window[w][1] for w in order], axis=0).astype(np.float32)
        cov  = np.stack([per_window[w][2] for w in order], axis=0).astype(np.float32)

        out_path = out_dir / f"test_{sid:06d}.npz"
        np.savez_compressed(out_path, geno=geno, pos=pos, cov=cov, theta=theta)
        print(f"  scenario {sid}: wrote {out_path} geno={geno.shape}")
        n_written += 1

    print(f"\nCollated {n_written} scenarios "
          f"({n_incomplete} incomplete, {n_bad} theta-mismatch).")
    if n_incomplete and not args.require_complete:
        print("Incomplete scenarios were written with the windows available. "
              "Re-run their failed shards and re-collate to fill gaps.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
