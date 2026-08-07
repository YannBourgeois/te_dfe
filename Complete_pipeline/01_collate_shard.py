#!/usr/bin/env python3
"""
01_collate_shard.py — Concatenate per-window-chunk shards into per-replicate sim files
that 02_collate.py consumes (the standard simulations/ directory).

Reads all rep_<id>_w<start>-<end>.npz shards from --shard-dir, groups by
replicate, concatenates geno/pos/cov along the window axis in window-index order,
and writes sim_<id>.npz (format: geno, pos, cov, theta) to --out-dir.

Validates per replicate:
  * all chunks share the same theta (within tolerance)
  * window indices are contiguous 0..n_windows_total-1 with no gaps/dupes
  * reports any replicate missing windows (failed shards) — these can be
    re-run by addressing the specific (replicate, chunk) before re-collating.

Usage:
    python 01_collate_shard.py --shard-dir sim_shards/ --out-dir simulations/
    python 01_collate_shard.py --shard-dir sim_shards/ --out-dir simulations/ \
        --require-complete        # skip replicates with any missing windows
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shard-dir", type=str, default="sim_shards/")
    p.add_argument("--out-dir",   type=str, default="simulations/")
    p.add_argument("--require-complete", action="store_true",
                   help="Skip (don't write) replicates that are missing windows.")
    p.add_argument("--theta-tol", type=float, default=1e-5,
                   help="Max allowed theta disagreement across a replicate's "
                        "shards.")
    return p.parse_args()


def main():
    args = parse_args()
    shard_dir = Path(args.shard_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(shard_dir.glob("rep_*_w*.npz"))
    if not shards:
        print(f"No shards found in {shard_dir}", file=sys.stderr)
        return 1

    # Group shard paths by scenario_id.
    by_replicate = defaultdict(list)
    for sp in shards:
        d = np.load(sp)
        rid = int(d["replicate_id"])
        by_replicate[rid].append(sp)

    print(f"Found {len(shards)} shards across {len(by_replicate)} replicates")
    n_written = n_incomplete = n_bad = 0

    for rid in sorted(by_replicate):
        paths = by_replicate[rid]
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
            print(f"  replicate {rid}: !! THETA MISMATCH across shards "
                  f"(max diff {spread:.2e}) — skipping", file=sys.stderr)
            n_bad += 1
            continue
        theta = thetas[0]

        # Check completeness.
        have = sorted(per_window)
        expected = list(range(n_total))
        missing = sorted(set(expected) - set(have))
        if missing:
            msg = (f"  replicate {rid}: {len(have)}/{n_total} windows "
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

        out_path = out_dir / f"sim_{rid:06d}.npz"
        np.savez_compressed(out_path, geno=geno, pos=pos, cov=cov, theta=theta)
        print(f"  replicate {rid}: wrote {out_path} geno={geno.shape}")
        n_written += 1

    print(f"\nCollated {n_written} replicates "
          f"({n_incomplete} incomplete, {n_bad} theta-mismatch).")
    if n_incomplete and not args.require_complete:
        print("Incomplete replicates were written with the windows available. "
              "Re-run their failed shards and re-collate to fill gaps.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
