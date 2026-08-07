#!/usr/bin/env python3
"""
05c_benchmark_K_sweep.py — Run 05b_benchmark_analyze.py at multiple
K_target values, then collate the resulting benchmark_summary.txt files
into a single comparison table.

Only meaningful when the trained model was variable-K (include_mask=True).

Usage (within the trained window range):
    python 05c_benchmark_K_sweep.py \\
        --results-dir results/ \\
        --test-dir    test_simulations/ \\
        --out-base    results/benchmark \\
        --k-values    5 10 15 20

Usage (EXTRAPOLATING beyond the trained window count, e.g. to 200):
    python 05c_benchmark_K_sweep.py \\
        --results-dir results/ \\
        --test-dir    test_simulations/ \\
        --out-base    results/benchmark \\
        --k-values    5 10 20 50 100 200 \\
        --extrapolate --window-chunk 20

    --extrapolate is REQUIRED for any K above the training window count: it
    forwards to 05b's chunked-embedding bypass. Without it, 05b errors out for
    K > n_train_windows.

    NOTE: K is clamped per scenario to the windows actually present in the test
    npz (n_eff = min(K, n_data)). To sweep up to 200 the test sims must have been
    generated with >=200 windows (05a_benchmark_simulate --n-windows 200);
    otherwise the high-K points silently collapse to the available count.

Produces:
    results/benchmark_K05/   (full 05b output for K=5)
    ...
    results/benchmark_K200/
    results/benchmark_K_sweep_summary.csv   (aggregated table)
"""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default="results/")
    p.add_argument("--test-dir",    type=str, default="test_simulations/")
    p.add_argument("--out-base",    type=str, default="results/benchmark",
                   help="Base path for per-K output dirs; will be suffixed "
                        "with '_K<N>' for each K.")
    p.add_argument("--k-values",    type=int, nargs="+",
                   default=[5, 10, 15, 20],
                   help="K_target values to benchmark.")
    p.add_argument("--n-post",      type=int, default=500)
    p.add_argument("--max-tests",   type=int, default=None)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip K values whose output dir already exists.")
    p.add_argument("--extrapolate", action="store_true",
                   help="Allow K values beyond the training window count. "
                        "Forwards --extrapolate to 05b so it routes through the "
                        "chunked-embedding bypass. REQUIRED for any --k-values "
                        "above n_train_windows; without it 05b errors on K>that.")
    p.add_argument("--window-chunk", type=int, default=20,
                   help="With --extrapolate, encode windows in batches of this "
                        "size to bound peak GPU memory (forwarded to 05b).")
    return p.parse_args()


def _parse_summary(summary_path: Path):
    """Parse benchmark_summary.txt into list of per-parameter dicts.

    Expected format (one row per parameter):
        param  RMSE  bias  cov_90%  SBC_p
    """
    rows = []
    if not summary_path.exists():
        return rows
    with open(summary_path) as f:
        lines = f.readlines()
    in_table = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Header row?
        if "parameter" in stripped.lower() and "rmse" in stripped.lower():
            in_table = True
            continue
        if in_table:
            # Skip dashed separator
            if set(stripped) <= {"-", "="}:
                continue
            # Stop at "Target 90%" footer
            if stripped.lower().startswith("target"):
                break
            parts = re.split(r"\s+", stripped)
            if len(parts) >= 5:
                try:
                    rows.append({
                        "param":   parts[0],
                        "rmse":    float(parts[1]),
                        "bias":    float(parts[2]),
                        "cov90":   float(parts[3]),
                        "sbc_p":   float(parts[4]),
                    })
                except ValueError:
                    continue
    return rows


def main():
    args = parse_args()
    out_base = Path(args.out_base)
    results_dir = Path(args.results_dir)
    test_dir = Path(args.test_dir)

    # Sanity: ensure the trained model is K-variable (include_mask=True)
    import pickle
    meta_path = results_dir / "training_meta.pkl"
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if not meta.get("include_mask", False):
            print(f"WARNING: training_meta.pkl shows include_mask=False — "
                  f"this model is not K-variable. K_target sweep may not "
                  f"behave as expected.", file=sys.stderr)
    else:
        print(f"WARNING: no training_meta.pkl found in {results_dir}. "
              f"Cannot verify model is K-variable.", file=sys.stderr)

    # Run benchmark at each K
    all_results = {}
    for K in args.k_values:
        out_dir = out_base.parent / (out_base.name + f"_K{K:02d}")
        if args.skip_existing and (out_dir / "benchmark_summary.txt").exists():
            print(f"== K={K}: SKIP (output exists at {out_dir})")
            all_results[K] = _parse_summary(out_dir / "benchmark_summary.txt")
            continue

        print(f"== K={K}: running 05b_benchmark_analyze.py ...")
        cmd = [
            "python", "05b_benchmark_analyze.py",
            "--results-dir", str(results_dir),
            "--test-dir",    str(test_dir),
            "--out",         str(out_base),
            "--n-post",      str(args.n_post),
            "--k-target",    str(K),
            "--save-samples",
        ]
        if args.max_tests:
            cmd += ["--max-tests", str(args.max_tests)]
        if args.extrapolate:
            # K may exceed the trained window count; 05b needs the chunked
            # bypass (and otherwise ERRORs for K > n_train_windows).
            cmd += ["--extrapolate", "--window-chunk", str(args.window_chunk)]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"   FAILED (exit {result.returncode})", file=sys.stderr)
            continue
        all_results[K] = _parse_summary(out_dir / "benchmark_summary.txt")

    # Aggregate into single table:
    #   param, K, rmse, bias, cov90, sbc_p
    out_csv = out_base.parent / f"{out_base.name}_K_sweep_summary.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["param", "K", "rmse", "bias", "cov90", "sbc_p"])
        # Stable param ordering: take from the first non-empty K's rows
        first_K = next((K for K in args.k_values if all_results.get(K)), None)
        if first_K is None:
            print("ERROR: no benchmark results to aggregate.", file=sys.stderr)
            return 1
        params = [r["param"] for r in all_results[first_K]]
        for K in args.k_values:
            rows = all_results.get(K, [])
            by_param = {r["param"]: r for r in rows}
            for p in params:
                r = by_param.get(p)
                if r is None:
                    writer.writerow([p, K, "", "", "", ""])
                else:
                    writer.writerow([
                        p, K,
                        f"{r['rmse']:.4f}", f"{r['bias']:+.4f}",
                        f"{r['cov90']:.3f}", f"{r['sbc_p']:.4f}",
                    ])
    print(f"Wrote K-sweep summary -> {out_csv}")

    # Also print a human-readable pivot to stdout
    print()
    print("=" * 70)
    print(f"K-sweep summary (RMSE per parameter per K)")
    print("=" * 70)
    header = f"{'parameter':<20}" + "".join(f"K={K:>3d}    " for K in args.k_values)
    print(header)
    print("-" * len(header))
    first_K = next((K for K in args.k_values if all_results.get(K)), None)
    if first_K is None:
        return 1
    params = [r["param"] for r in all_results[first_K]]
    for p in params:
        line = f"{p:<20}"
        for K in args.k_values:
            rows = all_results.get(K, [])
            by_param = {r["param"]: r for r in rows}
            r = by_param.get(p)
            line += f"{r['rmse']:>7.4f}  " if r else "  ---   "
        print(line)
    print()
    print(f"Lower RMSE at higher K = K-invariance gain working as expected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
