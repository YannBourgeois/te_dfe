#!/usr/bin/env python3
"""
01_simulate.py — Run one DFE + burst simulation replicate.

One SLURM array task = one replicate, with SLURM_ARRAY_TASK_ID as the seed.
Resumability via out_path.exists() check.

Output: simulations/sim_{seed:06d}.npz with `obs` and `theta` (12 params).
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
# Model selection before any te_dfe import (config dispatches at import time).
try:
    from model_select import prescan_model, add_model_arg
    prescan_model()
except Exception:
    add_model_arg = None


from te_dfe.config import (
    N_TRAIN_WINDOWS, MSMC_PATH, MU, RECOMB_MAP_PATH,
    ANNOTATION_PATH, ANNOTATION_FMT, CHROMOSOMES, SIM_LENGTH,
    MAF_MIN, WINDOW_SIZE,
)
from te_dfe.demography import (
    MSMCDemography, GenomeAnnotation, WindowSampler,
)
from te_dfe.simulator import DFEBurstSimulator
from te_dfe.processor import TEDFEBurstProcessor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",       type=int, required=True)
    p.add_argument("--out-dir",    type=str, default="simulations/")
    p.add_argument("--n-windows",  type=int, default=None)
    p.add_argument("--workdir",    type=str, default=None)
    p.add_argument("--debug-dump", type=str, default=None,
                   help="If set, write per-window introspection artifacts "
                        "(schedule, surviving TEs/SNPs, summary stats, "
                        "materialized SLiM script) to <DIR>/sim_NNNNNN/")
    p.add_argument("--keep-trees", action="store_true",
                   help="Persist the per-window tree sequences alongside the "
                        ".npz tensors.")
    p.add_argument("--keep-trees-dir", type=str, default=None,
                   help="Override the default trees output location.")
    # For SNPE rounds: load a theta vector from a file rather than
    # sampling from the prior.
    p.add_argument("--theta-file", type=str, default=None,
                   help="Path to a .npy file containing a (N, N_PARAMS) array "
                        "of theta vectors. The simulation uses theta_file[row]. "
                        "If set, the simulator does NOT sample from the prior. "
                        "Used by SNPE refinement rounds.")
    p.add_argument("--theta-row",  type=int, default=None,
                   help="Row index into --theta-file. Required if --theta-file "
                        "is set.")
    # Register --model so argparse accepts it (prescan_model already
    # consumed its VALUE into TE_DFE_MODEL; this makes the flag legal).
    if add_model_arg is not None:
        add_model_arg(p)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sim_{args.seed:06d}.npz"
    if out_path.exists():
        print(f"[seed={args.seed}] Already done, skipping.")
        return 0

    n_windows = args.n_windows or N_TRAIN_WINDOWS

    t0 = time.time()
    print(f"[seed={args.seed}] Loading genomic resources...")
    demog = MSMCDemography.from_file(MSMC_PATH, mu=MU, sim_length=SIM_LENGTH)
    print(f"  {demog}")

    annot = (GenomeAnnotation.from_gtf(ANNOTATION_PATH)
             if ANNOTATION_FMT == "gtf"
             else GenomeAnnotation.from_gff(ANNOTATION_PATH))

    sampler = WindowSampler(
        recomb_map_path=RECOMB_MAP_PATH, annotation=annot,
        window_size=WINDOW_SIZE, chromosomes=CHROMOSOMES,
        workdir=args.workdir,
    )
    sim  = DFEBurstSimulator(demog, sampler, n_windows=n_windows)
    proc = TEDFEBurstProcessor(maf_min=MAF_MIN)   # uses defaults from config

    # Load theta from file if SNPE mode
    theta_provided = None
    if args.theta_file is not None:
        if args.theta_row is None:
            print("ERROR: --theta-row required when --theta-file is set",
                  file=sys.stderr)
            return 1
        theta_arr = np.load(args.theta_file)
        if args.theta_row >= len(theta_arr):
            print(f"ERROR: --theta-row {args.theta_row} >= "
                  f"len(theta_file)={len(theta_arr)}", file=sys.stderr)
            return 1
        theta_provided = theta_arr[args.theta_row].astype(np.float32)
        print(f"[seed={args.seed}] Using theta_file row {args.theta_row}: "
              f"{theta_provided}")

    print(f"[seed={args.seed}] Running {n_windows} SLiM windows...")
    try:
        windows, theta = sim(
            seed=args.seed, debug_dump_dir=args.debug_dump,
            theta_provided=theta_provided,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"[seed={args.seed}] FAILED: {e}", file=sys.stderr)
        return 1

    # Optionally persist the per-window tree sequences for later
    # re-extraction (e.g., new channels, new features) without rerunning
    # SLiM. Stored as <out>/trees/sim_NNNNNN/win_WW.trees plus a small
    # auxiliary file with the recomb map breakpoints/rates and theta.
    if args.keep_trees:
        if args.keep_trees_dir:
            trees_root = Path(args.keep_trees_dir)
        else:
            trees_root = out_dir / "trees"
        sim_trees_dir = trees_root / f"sim_{args.seed:06d}"
        sim_trees_dir.mkdir(parents=True, exist_ok=True)

        # Save each window's tree sequence and recomb map breakpoints/rates
        for w, (ts, cov, rmap, *_) in enumerate(windows):
            ts_path = sim_trees_dir / f"win_{w:02d}.trees"
            ts.dump(str(ts_path))
            if rmap is not None:
                # msprime RateMap has .position (breakpoints) and .rate
                # arrays. Save as a small .npz for cheap reload.
                rmap_path = sim_trees_dir / f"win_{w:02d}_recomb.npz"
                np.savez(
                    rmap_path,
                    position=np.asarray(rmap.position, dtype=np.float64),
                    rate    =np.asarray(rmap.rate,     dtype=np.float64),
                )

        # Save theta + per-window covariates alongside, for completeness.
        np.savez(
            sim_trees_dir / "metadata.npz",
            theta=theta,
            covariates=np.stack([cov for _, cov, *_ in windows], axis=0),
        )

    from te_dfe.config import REPRESENTATION
    if REPRESENTATION == "sumstat":
        from te_dfe.config import N_TE_TRAIN_MAX, N_SUMSTATS_PER_TE
        # Pool per-TE features across all windows into ONE set, then pad once.
        pooled = []
        for w_i, (ts, cov, rmap, *rest) in enumerate(windows):
            if getattr(sim, "last_window_coords", None):
                ch, st = sim.last_window_coords[w_i]
                proc.window_chrom, proc.window_start = ch, st
            # Thread this window's gene intervals (4th tuple element) onto the
            # processor so the per-TE dist_to_gene feature is real, not 0. The
            # simulator always emits genes from the annotation (BGS selection is
            # separate); without this they are silently dropped.
            proc.gene_intervals = rest[0] if rest else None
            raw = proc(ts, seed=args.seed, recomb_map=rmap, pad=False)  # (n_te, F)
            if raw.shape[0]:
                pooled.append(raw)
        allte = (np.concatenate(pooled, axis=0)
                 if pooled else np.zeros((0, N_SUMSTATS_PER_TE), np.float32))
        n_te = min(allte.shape[0], N_TE_TRAIN_MAX)
        feats = np.zeros((N_TE_TRAIN_MAX, N_SUMSTATS_PER_TE), dtype=np.float32)
        te_mask = np.zeros((N_TE_TRAIN_MAX,), dtype=np.float32)
        if n_te:
            # If more TEs than the cap, subsample (deterministic by seed) so the
            # set is a representative draw rather than the first-K by position.
            if allte.shape[0] > N_TE_TRAIN_MAX:
                rng = np.random.default_rng(args.seed)
                sel = rng.choice(allte.shape[0], size=N_TE_TRAIN_MAX, replace=False)
                feats[:] = allte[np.sort(sel)]
                te_mask[:] = 1.0
            else:
                feats[:n_te] = allte[:n_te]
                te_mask[:n_te] = 1.0
        from te_dfe.config import WINDOW_SIZE as _WS
        seq_len_bp = float(n_windows) * float(_WS)
        np.savez_compressed(out_path, feats=feats, te_mask=te_mask,
                            theta=theta, seq_len_bp=np.float64(seq_len_bp))
        print(f"[seed={args.seed}] Saved {out_path} "
              f"feats={feats.shape} n_te={int(te_mask.sum())} "
              f"theta={theta.shape} elapsed={time.time()-t0:.1f}s")
        return 0

    geno_list, pos_list, cov_list = [], [], []
    for w_i, (ts, cov, rmap, *rest) in enumerate(windows):
        if getattr(sim, "last_window_coords", None):
            ch, st = sim.last_window_coords[w_i]
            proc.window_chrom, proc.window_start = ch, st
        proc.gene_intervals = rest[0] if rest else None   # for dist_to_gene (context)
        geno, pos = proc(ts, seed=args.seed, recomb_map=rmap)
        geno_list.append(geno)
        pos_list.append(pos)
        cov_list.append(cov)

    # Genotypes: (n_windows, n_haps, n_sites) int8 categorical {0,1,2,3}
    # Position:  (n_windows, 2, n_sites) float32 — per-site (phys, gen) deltas
    # Covariates: (n_windows, n_cov) float32
    genotypes  = np.stack(geno_list, axis=0).astype(np.int8)
    positions  = np.stack(pos_list,  axis=0).astype(np.float32)
    covariates = np.stack(cov_list,  axis=0).astype(np.float32)

    np.savez_compressed(out_path,
                          geno=genotypes, pos=positions,
                          cov=covariates, theta=theta)
    print(f"[seed={args.seed}] Saved {out_path} "
          f"geno={genotypes.shape}({genotypes.dtype}) "
          f"pos={positions.shape} cov={covariates.shape} "
          f"theta={theta.shape} elapsed={time.time()-t0:.1f}s")
    if args.keep_trees:
        print(f"[seed={args.seed}] Tree sequences saved under "
              f"{trees_root}/sim_{args.seed:06d}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
