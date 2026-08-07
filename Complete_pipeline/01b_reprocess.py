#!/usr/bin/env python3
"""
01b_reprocess.py — Re-extract (geno, pos, cov) tensors from previously
saved tree sequences without rerunning SLiM.

Use case: you ran `01_simulate.py --keep-trees`, then later changed the
processor (e.g., new channels, new filters) and want to update your
.npz tensor files without paying the SLiM cost again.

Layout expected from `--keep-trees`:
  <trees_root>/sim_NNNNNN/win_WW.trees
  <trees_root>/sim_NNNNNN/win_WW_recomb.npz   (optional, if recomb_map saved)
  <trees_root>/sim_NNNNNN/metadata.npz        (theta + covariates)

Output: <out_dir>/sim_NNNNNN.npz, matching the format `01_simulate.py`
would have produced.

Usage:
    python 01b_reprocess.py --seed 42 --trees-dir simulations/trees \\
                              --out-dir simulations_v2/
    # Or batch with SLURM array, one seed per task.
"""
import argparse
import sys
import time
from pathlib import Path

import msprime
import numpy as np
import tskit

sys.path.insert(0, str(Path(__file__).parent))

from te_dfe.config import MAF_MIN
from te_dfe.processor import TEDFEBurstProcessor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",       type=int, required=True,
                    help="Simulation seed (used to find sim_NNNNNN/ dir)")
    p.add_argument("--trees-dir",  type=str, required=True,
                    help="Directory containing sim_NNNNNN/ subdirectory.")
    p.add_argument("--out-dir",    type=str, required=True,
                    help="Output directory for re-extracted .npz")
    return p.parse_args()


def _load_recomb_map(npz_path):
    """Reconstruct an msprime.RateMap from saved breakpoints/rates."""
    if not npz_path.exists():
        return None
    d = np.load(npz_path)
    return msprime.RateMap(
        position=d["position"], rate=d["rate"],
    )


def main():
    args = parse_args()
    trees_dir = Path(args.trees_dir) / f"sim_{args.seed:06d}"
    if not trees_dir.is_dir():
        print(f"ERROR: trees dir not found: {trees_dir}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sim_{args.seed:06d}.npz"
    if out_path.exists():
        print(f"[seed={args.seed}] Already done, skipping.")
        return 0

    # Load metadata: theta + per-window covariates
    meta_path = trees_dir / "metadata.npz"
    if not meta_path.exists():
        print(f"ERROR: metadata not found: {meta_path}", file=sys.stderr)
        return 1
    meta = np.load(meta_path)
    theta      = meta["theta"]
    covariates = meta["covariates"]
    n_windows = covariates.shape[0]

    t0 = time.time()
    proc = TEDFEBurstProcessor(maf_min=MAF_MIN)

    from te_dfe.config import REPRESENTATION
    if REPRESENTATION == "sumstat":
        from te_dfe.config import N_TE_TRAIN_MAX, N_SUMSTATS_PER_TE
        print(f"[seed={args.seed}] Re-processing {n_windows} windows (sumstat)...")
        pooled = []
        for w in range(n_windows):
            ts = tskit.load(str(trees_dir / f"win_{w:02d}.trees"))
            rmap = _load_recomb_map(trees_dir / f"win_{w:02d}_recomb.npz")
            raw = proc(ts, seed=args.seed, recomb_map=rmap, pad=False)  # (n_te, F)
            if raw.shape[0]:
                pooled.append(raw)
        allte = (np.concatenate(pooled, axis=0)
                 if pooled else np.zeros((0, N_SUMSTATS_PER_TE), np.float32))
        feats = np.zeros((N_TE_TRAIN_MAX, N_SUMSTATS_PER_TE), dtype=np.float32)
        te_mask = np.zeros((N_TE_TRAIN_MAX,), dtype=np.float32)
        if allte.shape[0] > N_TE_TRAIN_MAX:
            rng = np.random.default_rng(args.seed)
            sel = np.sort(rng.choice(allte.shape[0], N_TE_TRAIN_MAX, replace=False))
            feats[:] = allte[sel]; te_mask[:] = 1.0
        elif allte.shape[0]:
            n = allte.shape[0]; feats[:n] = allte; te_mask[:n] = 1.0
        np.savez_compressed(out_path, feats=feats, te_mask=te_mask, theta=theta)
        print(f"[seed={args.seed}] Saved {out_path} feats={feats.shape} "
              f"n_te={int(te_mask.sum())} elapsed={time.time()-t0:.1f}s")
        return 0

    print(f"[seed={args.seed}] Re-processing {n_windows} windows...")
    geno_list, pos_list = [], []
    for w in range(n_windows):
        ts_path   = trees_dir / f"win_{w:02d}.trees"
        rmap_path = trees_dir / f"win_{w:02d}_recomb.npz"
        ts = tskit.load(str(ts_path))
        rmap = _load_recomb_map(rmap_path)
        geno, pos = proc(ts, seed=args.seed, recomb_map=rmap)
        geno_list.append(geno)
        pos_list.append(pos)

    genotypes = np.stack(geno_list, axis=0).astype(np.int8)
    positions = np.stack(pos_list,  axis=0).astype(np.float32)

    np.savez_compressed(
        out_path,
        geno=genotypes, pos=positions,
        cov=covariates.astype(np.float32), theta=theta,
    )
    print(f"[seed={args.seed}] Saved {out_path} "
          f"geno={genotypes.shape}({genotypes.dtype}) "
          f"pos={positions.shape} cov={covariates.shape} "
          f"theta={theta.shape} elapsed={time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
