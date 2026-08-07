#!/usr/bin/env python3
"""
05d_te_count_sweep.py — TE-count analogue of 05c, for the SUMSTAT representation.

Sweeps the number of TEs the model is shown at deployment (a random subsample of
the pooled TE-set), so you get the cardinality-generalization curve for the
sumstat NPE — the sumstat counterpart of 05c's 5/10/15/20 window sweep, which is
window-based and therefore inert for sumstat models.

At each N it:
  * runs 05b_benchmark_analyze.py --n-te-target N   -> NPE recovery/SBC at N
  * (optional) runs 07_export_sumstat_summaries.py --n-te-target N on the same
    collated benchmark set -> ABC-ready per-feature summaries at N, so ABC can be
    benchmarked on the SAME increasing dataset sizes.

The NPE and ABC subsamples are each random-N (independent draws); that is fine
for the aggregate curve (performance vs N averaged over replicates). Use the same
--seed for a reproducible run.

Usage:
    python 05d_te_count_sweep.py \\
        --results-dir results/ \\
        --test-dir    test_simulations/ \\
        --dataset     test_simulations/benchmark_collated \\
        --out-base    results/benchmark \\
        --n-te-values 100 250 500 1000 2000 5000

Produces:
    results/benchmark_NTE100/  ... benchmark_NTE5000/      (per-N 05b output)
    results/benchmark_NTE<N>_abc_summaries.csv             (ABC summaries per N)
    results/benchmark_NTE_sweep_summary.csv                (aggregated NPE table)
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
    p.add_argument("--test-dir",    type=str, default="test_simulations/",
                   help="Per-scenario benchmark sims for 05b (NPE).")
    p.add_argument("--dataset",     type=str, default=None,
                   help="Collated benchmark base path (without .feats.npy) for "
                        "07 ABC-summary export. Omit to skip the ABC summaries.")
    p.add_argument("--out-base",    type=str, default="results/benchmark",
                   help="Base path for per-N output; suffixed '_NTE<N>'.")
    p.add_argument("--n-te-values", type=int, nargs="+",
                   default=[100, 250, 500, 1000, 2000],
                   help="TE-count targets. Values past the training cap probe "
                        "the extrapolation regime explicitly.")
    p.add_argument("--model",       type=str, default=None,
                   help="Model name passed through to 07 (--model).")
    p.add_argument("--quantiles",   type=str, default="0.1,0.25,0.5,0.75,0.9",
                   help="Quantiles passed through to 07.")
    p.add_argument("--n-post",      type=int, default=500)
    p.add_argument("--max-tests",   type=int, default=None)
    p.add_argument("--seed",        type=int, default=0,
                   help="Subsampling seed (shared by the NPE and ABC steps).")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--extrapolate", action="store_true",
                   help="Forward --extrapolate to 05b so the sumstat TE-set is "
                        "embedded directly via the count-invariant bypass. "
                        "REQUIRED when --n-te-values exceed the model's trained "
                        "TE width; without it 05b falls through to the flat path "
                        "and hits the sbi x_o shape check.")
    p.add_argument("--window-chunk", type=int, default=20,
                   help="Forwarded to 05b with --extrapolate (matrix-only; "
                        "harmless/ignored on the sumstat path).")
    return p.parse_args()


def _parse_summary(summary_path: Path):
    """Parse 05b's benchmark_summary.txt: one row per parameter
    (param  RMSE  bias  cov_90%  SBC_p)."""
    rows = []
    if not summary_path.exists():
        return rows
    with open(summary_path) as f:
        lines = f.readlines()
    in_table = False
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if "parameter" in s.lower() and "rmse" in s.lower():
            in_table = True
            continue
        if in_table:
            if set(s) <= {"-", "="}:
                continue
            if s.lower().startswith("target"):
                break
            parts = re.split(r"\s+", s)
            if len(parts) >= 5:
                try:
                    rows.append({"param": parts[0], "rmse": float(parts[1]),
                                 "bias": float(parts[2]), "cov90": float(parts[3]),
                                 "sbc_p": float(parts[4])})
                except ValueError:
                    continue
    return rows


def main():
    args = parse_args()
    out_base = Path(args.out_base)
    results_dir = Path(args.results_dir)
    test_dir = Path(args.test_dir)

    all_results = {}
    for N in args.n_te_values:
        out_dir = out_base.parent / (out_base.name + f"_NTE{N}")
        summary_txt = out_dir / "benchmark_summary.txt"

        if args.skip_existing and summary_txt.exists():
            print(f"== N={N}: SKIP (output exists at {out_dir})")
            all_results[N] = _parse_summary(summary_txt)
        else:
            print(f"== N={N}: NPE recovery via 05b_benchmark_analyze.py ...")
            cmd = [
                "python", "05b_benchmark_analyze.py",
                "--results-dir", str(results_dir),
                "--test-dir",    str(test_dir),
                "--out",         str(out_base),
                "--n-post",      str(args.n_post),
                "--n-te-target", str(N),
                "--n-te-seed",   str(args.seed),
                "--save-samples",
            ]
            if args.max_tests:
                cmd += ["--max-tests", str(args.max_tests)]
            if args.extrapolate:
                # Sumstat TE-counts above the trained width need 05b's
                # count-invariant bypass; without this they hit the sbi shape
                # check. window-chunk is matrix-only but harmless to pass.
                cmd += ["--extrapolate", "--window-chunk", str(args.window_chunk)]
            if subprocess.run(cmd, check=False).returncode != 0:
                print(f"   NPE step FAILED at N={N}", file=sys.stderr)
            all_results[N] = _parse_summary(summary_txt)

        # ABC summaries at the same N (optional).
        if args.dataset:
            abc_out = out_base.parent / f"{out_base.name}_NTE{N}_abc_summaries.csv"
            if args.skip_existing and abc_out.exists():
                print(f"   ABC summaries exist: {abc_out}")
            else:
                print(f"   N={N}: ABC summaries via 07_export_sumstat_summaries.py ...")
                cmd07 = [
                    "python", "07_export_sumstat_summaries.py",
                    "--dataset",     str(args.dataset),
                    "--out",         str(abc_out),
                    "--quantiles",   args.quantiles,
                    "--n-te-target", str(N),
                    "--seed",        str(args.seed),
                ]
                if args.model:
                    cmd07 += ["--model", args.model]
                if subprocess.run(cmd07, check=False).returncode != 0:
                    print(f"   ABC-summary step FAILED at N={N}", file=sys.stderr)

    # Aggregate NPE recovery: param, N, rmse, bias, cov90, sbc_p
    out_csv = out_base.parent / f"{out_base.name}_NTE_sweep_summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "n_te", "rmse", "bias", "cov90", "sbc_p"])
        first = next((N for N in args.n_te_values if all_results.get(N)), None)
        if first is None:
            print("ERROR: no NPE benchmark results to aggregate.", file=sys.stderr)
            return 1
        params = [r["param"] for r in all_results[first]]
        for N in args.n_te_values:
            by_param = {r["param"]: r for r in all_results.get(N, [])}
            for p in params:
                r = by_param.get(p)
                if r is None:
                    w.writerow([p, N, "", "", "", ""])
                else:
                    w.writerow([p, N, f"{r['rmse']:.4f}", f"{r['bias']:+.4f}",
                                f"{r['cov90']:.3f}", f"{r['sbc_p']:.4f}"])

    print(f"\nNPE sweep table: {out_csv}")
    if args.dataset:
        print(f"ABC summaries:  {out_base.name}_NTE<N>_abc_summaries.csv "
              f"(feed each to your ABC script for the matched ABC-vs-N curve).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
