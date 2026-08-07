#!/usr/bin/env python3
"""
00_sample_windows.py — pre-flight stratified WINDOW MANIFEST for sharded sims.

The old in-process stratification (WindowSampler.sample_n with
cover_covariate_space=True) sampled, per replicate, a set of windows that
spans the (gene_density, recomb_rate) covariate space — important for slope
identifiability. The sharded simulator can't do that itself, because it only
sees one window-chunk of one replicate at a time.

This script does the stratified sampling UP FRONT for every replicate and
writes a manifest of window COORDINATES (chrom, start). The sharded simulator
(01_simulate_shard.py --window-manifest) then reconstructs each window from its
coordinates via WindowSampler.window_from_coords, so the simulated windows are
exactly the stratified set chosen here — reproducible, and inspectable before
you spend cluster time.

Each replicate's n_windows windows are independently stratified across a 5x5
grid of (gene_density, recomb) quintiles, reproducing the old per-replicate
coverage. The manifest is keyed by (replicate_id, window_idx).

Outputs:
  <out>.npz   arrays: replicate_id, window_idx, chrom, start, gene_density,
              mean_recomb_scaled  (the last two for inspection / plotting)
  <out>.csv   same, human-readable
  prints a summary of the covariate distribution that WILL be simulated.

Usage:
  python 00_sample_windows.py --n-replicates 10000 --n-windows 20 \
      --out window_manifest --seed 777
  python 00_sample_windows.py ... --plot window_manifest_covariates.png
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
# Model selection before any te_dfe import (config dispatches at import time).
try:
    from model_select import prescan_model, add_model_arg
    prescan_model()
except Exception:
    add_model_arg = None



def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-replicates", type=int, required=True)
    p.add_argument("--n-windows",    type=int, default=None,
                   help="Windows per replicate (default: config N_TRAIN_WINDOWS).")
    p.add_argument("--out",          type=str, default="window_manifest")
    p.add_argument("--seed",         type=int, default=777,
                   help="Master seed for the stratified window sampling. "
                        "Recorded in the manifest; reuse it to regenerate the "
                        "identical manifest.")
    p.add_argument("--plot",         type=str, default=None,
                   help="Optional PNG of the covariate distribution.")
    # Register --model so argparse accepts it (prescan_model already
    # consumed its VALUE into TE_DFE_MODEL; this makes the flag legal).
    if add_model_arg is not None:
        add_model_arg(p)
    return p.parse_args()


def main():
    args = parse_args()
    from te_dfe.config import (
        MSMC_PATH, MU, RECOMB_MAP_PATH, ANNOTATION_PATH, ANNOTATION_FMT,
        CHROMOSOMES, SIM_LENGTH, WINDOW_SIZE, N_TRAIN_WINDOWS,
    )
    from te_dfe.demography import (
        MSMCDemography, GenomeAnnotation, WindowSampler)

    n_windows = args.n_windows or N_TRAIN_WINDOWS

    annot = (GenomeAnnotation.from_gtf(ANNOTATION_PATH)
             if ANNOTATION_FMT == "gtf"
             else GenomeAnnotation.from_gff(ANNOTATION_PATH))
    sampler = WindowSampler(
        recomb_map_path=RECOMB_MAP_PATH, annotation=annot,
        chromosomes=CHROMOSOMES, window_size=WINDOW_SIZE,
    )

    t0 = time.time()
    rep_ids, win_ids, chroms, starts, gds, rcs = [], [], [], [], [], []
    # Per-replicate stratified sampling. Each replicate gets its own RNG stream
    # derived from (seed, replicate) so the manifest is reproducible and the
    # per-replicate covariate coverage matches the old in-process behavior.
    for rid in range(args.n_replicates):
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, rid]))
        windows = sampler.sample_n(n_windows, rng, cover_covariate_space=True)
        for w_idx, gw in enumerate(windows):
            rep_ids.append(rid); win_ids.append(w_idx)
            chroms.append(str(gw.chrom)); starts.append(int(gw.start))
            gds.append(float(gw.gene_density))
            rcs.append(float(gw.mean_recomb_scaled))
        if (rid + 1) % 500 == 0:
            print(f"  sampled {rid+1}/{args.n_replicates} replicates "
                  f"({time.time()-t0:.0f}s)")

    rep_ids = np.array(rep_ids, dtype=np.int32)
    win_ids = np.array(win_ids, dtype=np.int32)
    chroms  = np.array(chroms)
    starts  = np.array(starts, dtype=np.int64)
    gds     = np.array(gds, dtype=np.float64)
    rcs     = np.array(rcs, dtype=np.float64)

    out = Path(args.out)
    np.savez_compressed(
        out.with_suffix(".npz"),
        replicate_id=rep_ids, window_idx=win_ids, chrom=chroms, start=starts,
        gene_density=gds, mean_recomb_scaled=rcs,
        n_replicates=np.int32(args.n_replicates),
        n_windows=np.int32(n_windows), master_seed=np.int64(args.seed),
    )
    with open(out.with_suffix(".csv"), "w") as f:
        f.write("replicate_id,window_idx,chrom,start,gene_density,mean_recomb_scaled\n")
        for i in range(len(rep_ids)):
            f.write(f"{rep_ids[i]},{win_ids[i]},{chroms[i]},{starts[i]},"
                    f"{gds[i]:.6g},{rcs[i]:.6g}\n")

    # Inspection summary: the covariate distribution that WILL be simulated.
    print(f"\nManifest: {len(rep_ids)} windows "
          f"({args.n_replicates} replicates x {n_windows} windows)")
    print(f"  gene_density        : min {gds.min():.4g}  median "
          f"{np.median(gds):.4g}  max {gds.max():.4g}")
    print(f"  mean_recomb_scaled  : min {rcs.min():.4g}  median "
          f"{np.median(rcs):.4g}  max {rcs.max():.4g}")
    # Per-replicate coverage check: how many of the 5x5 covariate bins does an
    # average replicate touch? (Higher = better stratification.)
    gq = np.digitize(gds, np.percentile(gds, [20, 40, 60, 80]))
    rq = np.digitize(rcs, np.percentile(rcs, [20, 40, 60, 80]))
    bins = gq * 5 + rq
    per_rep_bins = []
    for rid in range(min(args.n_replicates, 1000)):
        m = rep_ids == rid
        per_rep_bins.append(len(np.unique(bins[m])))
    print(f"  per-replicate covariate-bin coverage (of 25): "
          f"mean {np.mean(per_rep_bins):.1f} "
          f"(n_windows={n_windows} caps it at {min(n_windows,25)})")
    print(f"\nWrote {out.with_suffix('.npz')} and {out.with_suffix('.csv')} "
          f"in {time.time()-t0:.0f}s")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 5))
            sc = ax.scatter(gds, rcs, s=4, alpha=0.3)
            ax.set_xlabel("gene_density"); ax.set_ylabel("mean_recomb_scaled")
            ax.set_title("Stratified window manifest — covariate coverage")
            fig.tight_layout(); fig.savefig(args.plot, dpi=130)
            print(f"Wrote covariate plot -> {args.plot}")
        except Exception as e:
            print(f"(plot skipped: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
