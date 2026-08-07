#!/usr/bin/env python3
"""
01_simulate_shard.py — SHARDED training-simulation producer for SLURM.

The original 01_simulate.py runs one full replicate per task (one theta, all
N_TRAIN_WINDOWS windows, serially). With heavy 5 Mb windows that wastes cluster
parallelism: a task only finishes after its slowest window. This sharded
version splits each replicate's windows across array tasks, so a task produces
a CHUNK of windows for one replicate; 01_collate_shard.py then assembles the
per-replicate sim_<id>.npz that 02_collate.py consumes.

A "replicate" here is one training example: ONE theta drawn from the prior,
realized over n_windows windows. Reproducibility (identical scheme to the 05a
benchmark sharding):
  * Per-replicate theta from SeedSequence([master, replicate, THETA_STREAM]) —
    identical across all chunks of a replicate.
  * Per-window SLiM seed from SeedSequence([master, replicate, WINDOW_STREAM,
    window_index]) — a pure function of the window index, so window w is
    bit-identical regardless of how windows are chunked.
  * Each window samples its genomic location independently (no joint
    covariate-space stratification). This matches the sharded benchmark sims;
    do NOT mix sharded training sims with the old stratified 20-window sims in
    the same training set.

SLURM array mapping:
  n_chunks = ceil(n_windows / chunk_size); total = n_replicates * n_chunks
  task_id -> replicate = task_id // n_chunks ; chunk = task_id % n_chunks

Usage:
  python 01_simulate_shard.py --task-id $SLURM_ARRAY_TASK_ID \
      --n-replicates 10000 --n-windows 20 --chunk-size 5 --out-dir sim_shards/
  python 01_simulate_shard.py --n-replicates 10000 --n-windows 20 \
      --chunk-size 5 --print-array-size
"""
import argparse
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
# Model selection before any te_dfe import (config dispatches at import time).
try:
    from model_select import prescan_model, add_model_arg
    prescan_model()
except Exception:
    add_model_arg = None


# Heavy imports (SLiM/tskit via the simulator) are deferred into main() so that
# --print-array-size works in any environment, including ones without SLiM.

THETA_STREAM  = 0
WINDOW_STREAM = 1


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-replicates", type=int, required=True,
                   help="Total number of training replicates (each = one theta).")
    p.add_argument("--n-windows",    type=int, default=None,
                   help="Windows per replicate (default: config N_TRAIN_WINDOWS).")
    p.add_argument("--chunk-size",   type=int, default=5,
                   help="Windows per array task.")
    p.add_argument("--master-seed",  type=int, default=12345)
    p.add_argument("--task-id",      type=int, default=None,
                   help="SLURM_ARRAY_TASK_ID -> (replicate, chunk). 0-indexed.")
    p.add_argument("--replicate-id", type=int, default=None)
    p.add_argument("--chunk-id",     type=int, default=None)
    p.add_argument("--out-dir",      type=str, default="sim_shards/")
    p.add_argument("--window-manifest", type=str, default=None,
                   help="Path to a 00_sample_windows.py manifest (.npz). When "
                        "given, this replicate's windows are taken from the "
                        "manifest's stratified coordinates instead of sampled "
                        "independently — restores per-replicate covariate-space "
                        "coverage. n-windows/master-seed should match the "
                        "manifest.")
    p.add_argument("--print-array-size", action="store_true")
    p.add_argument("--overwrite",    action="store_true")
    # Register --model so argparse accepts it (prescan_model already
    # consumed its VALUE into TE_DFE_MODEL; this makes the flag legal).
    if add_model_arg is not None:
        add_model_arg(p)
    return p.parse_args()


def replicate_theta(master_seed, replicate_id):
    """Deterministic per-replicate theta (shared across all its chunks)."""
    from te_dfe.prior import DFEBurstPrior
    ss = np.random.SeedSequence([master_seed, replicate_id, THETA_STREAM])
    return DFEBurstPrior.sample(np.random.default_rng(ss))


def window_seed(master_seed, replicate_id, window_idx):
    """Deterministic per-window SLiM seed, independent of chunking."""
    ss = np.random.SeedSequence(
        [master_seed, replicate_id, WINDOW_STREAM, window_idx])
    return int(np.random.default_rng(ss).integers(1, 2_000_000_000))


def main():
    args = parse_args()

    # Resolve the windows-per-replicate default from config (deferred so
    # --print-array-size needs no heavy imports).
    if args.n_windows is None:
        from te_dfe.config import N_TRAIN_WINDOWS
        args.n_windows = N_TRAIN_WINDOWS

    n_chunks = math.ceil(args.n_windows / args.chunk_size)
    total_tasks = args.n_replicates * n_chunks

    if args.print_array_size:
        print(total_tasks)
        return 0

    # Heavy imports only past this point.
    from te_dfe.config import (
        MSMC_PATH, MU, RECOMB_MAP_PATH, ANNOTATION_PATH, ANNOTATION_FMT,
        CHROMOSOMES, SIM_LENGTH, WINDOW_SIZE, MAF_MIN,
    )
    from te_dfe.demography import (
        MSMCDemography, GenomeAnnotation, WindowSampler)
    from te_dfe.simulator import DFEBurstSimulator
    from te_dfe.processor import TEDFEBurstProcessor

    if args.task_id is not None:
        replicate_id = args.task_id // n_chunks
        chunk_id     = args.task_id %  n_chunks
    elif args.replicate_id is not None and args.chunk_id is not None:
        replicate_id, chunk_id = args.replicate_id, args.chunk_id
    else:
        print("ERROR: give --task-id, or both --replicate-id and --chunk-id.",
              file=sys.stderr)
        return 2

    if not (0 <= replicate_id < args.n_replicates):
        print(f"ERROR: replicate {replicate_id} out of range "
              f"[0,{args.n_replicates}).", file=sys.stderr)
        return 2
    if not (0 <= chunk_id < n_chunks):
        print(f"ERROR: chunk {chunk_id} out of range [0,{n_chunks}).",
              file=sys.stderr)
        return 2

    w_start = chunk_id * args.chunk_size
    w_end   = min(w_start + args.chunk_size, args.n_windows)
    window_indices = list(range(w_start, w_end))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / (f"rep_{replicate_id:06d}"
                            f"_w{w_start:04d}-{w_end:04d}.npz")
    if shard_path.exists() and not args.overwrite:
        print(f"[rep{replicate_id} c{chunk_id}] {shard_path} exists, skipping")
        return 0

    t0 = time.time()
    print(f"[rep{replicate_id} c{chunk_id}] windows {w_start}..{w_end-1} "
          f"({len(window_indices)} of {args.n_windows})")

    demog = MSMCDemography.from_file(MSMC_PATH, mu=MU, sim_length=SIM_LENGTH)
    annot = (GenomeAnnotation.from_gtf(ANNOTATION_PATH)
             if ANNOTATION_FMT == "gtf"
             else GenomeAnnotation.from_gff(ANNOTATION_PATH))
    sampler = WindowSampler(
        recomb_map_path=RECOMB_MAP_PATH, annotation=annot,
        chromosomes=CHROMOSOMES, window_size=WINDOW_SIZE,
    )
    sim = DFEBurstSimulator(demog, sampler, n_windows=1)
    proc = TEDFEBurstProcessor(maf_min=MAF_MIN)

    theta = replicate_theta(args.master_seed, replicate_id)

    # If a stratified manifest is given, build a (window_idx -> (chrom, start))
    # lookup for THIS replicate so each window's location is the pre-sampled
    # stratified one rather than an independent draw.
    manifest_coords = None
    if args.window_manifest is not None:
        m = np.load(args.window_manifest, allow_pickle=True)
        sel = m["replicate_id"] == replicate_id
        if not np.any(sel):
            print(f"ERROR: replicate {replicate_id} not in manifest "
                  f"{args.window_manifest}", file=sys.stderr)
            return 2
        manifest_coords = {
            int(wi): (str(ch), int(st))
            for wi, ch, st in zip(m["window_idx"][sel],
                                  m["chrom"][sel], m["start"][sel])
        }

    geno_list, pos_list, cov_list, ok_idx = [], [], [], []
    for w_idx in window_indices:
        w_seed = window_seed(args.master_seed, replicate_id, w_idx)
        try:
            if manifest_coords is not None:
                chrom, start = manifest_coords[w_idx]
                gw = sampler.window_from_coords(chrom, start)
                windows, _ = sim(seed=w_seed, theta_provided=theta,
                                 n_windows_override=1, window_seeds=[w_seed],
                                 windows_in=[gw])
            else:
                # One window, shared theta, this window's own seed
                # (independent location).
                windows, _ = sim(seed=w_seed, theta_provided=theta,
                                 n_windows_override=1, window_seeds=[w_seed])
            ts, cov, rmap, *rest = windows[0]
            proc.gene_intervals = rest[0] if rest else None
            ch_st = sim.last_window_coords[0]
            proc.window_chrom, proc.window_start = ch_st[0], ch_st[1]
            geno, pos = proc(ts, seed=w_seed, recomb_map=rmap)
            geno_list.append(geno)
            pos_list.append(pos)
            cov_list.append(cov)
            ok_idx.append(w_idx)
        except Exception as e:
            traceback.print_exc()
            print(f"[rep{replicate_id} c{chunk_id}] window {w_idx} FAILED: {e}",
                  file=sys.stderr)

    if not geno_list:
        print(f"[rep{replicate_id} c{chunk_id}] no windows succeeded",
              file=sys.stderr)
        return 1

    genotypes  = np.stack(geno_list, axis=0).astype(np.int8)
    positions  = np.stack(pos_list,  axis=0).astype(np.float32)
    covariates = np.stack(cov_list,  axis=0).astype(np.float32)

    np.savez_compressed(
        shard_path,
        geno=genotypes, pos=positions, cov=covariates, theta=theta,
        window_indices=np.array(ok_idx, dtype=np.int32),
        replicate_id=np.int32(replicate_id),
        n_windows_total=np.int32(args.n_windows),
        master_seed=np.int64(args.master_seed),
    )
    print(f"[rep{replicate_id} c{chunk_id}] saved {shard_path} "
          f"geno={genotypes.shape} ({len(ok_idx)} windows) "
          f"elapsed={time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
