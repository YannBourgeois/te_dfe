#!/usr/bin/env python3
"""
05a_benchmark_simulate.py — Benchmark simulation, SHARDED for SLURM arrays.

A "scenario" is one test case: ONE shared theta, simulated over many windows
(e.g. 200). Because each window is an independent SLiM run, we shard the work:
each array task simulates a CHUNK of windows for one scenario and writes a
partial .npz. A separate collate step (05a_collate.py) concatenates the chunks
of each scenario into the final per-scenario test_<id>.npz that 05b consumes.

Reproducibility (critical):
  * Per-scenario theta is drawn from SeedSequence(master_seed, scenario_id) —
    identical across all chunks of a scenario.
  * Per-window SLiM seed is drawn from SeedSequence(master_seed, scenario_id,
    WINDOW_STREAM, window_index) — a deterministic function of the window index
    ALONE, so window w is bit-identical no matter which chunk simulates it, or
    whether it is run alone or with others. Chunk boundaries never change it.
  * Each window independently samples its own genomic location, so many windows
    sample many independent contexts (recomb, gene-density, ...).

SLURM array mapping:
  n_chunks_per_scenario = ceil(n_windows / chunk_size)
  total_tasks           = n_scenarios * n_chunks_per_scenario
  task_id -> scenario_id = task_id // n_chunks_per_scenario
             chunk_id     = task_id %  n_chunks_per_scenario
             window slice = [chunk_id*chunk_size, ... +chunk_size) (clipped)

Usage (single shard, e.g. from SLURM_ARRAY_TASK_ID):
    python 05a_benchmark_simulate.py \
        --task-id $SLURM_ARRAY_TASK_ID \
        --n-scenarios 50 --n-windows 200 --chunk-size 10 \
        --out-dir test_shards/

Address a shard explicitly:
    python 05a_benchmark_simulate.py --scenario-id 7 --chunk-id 3 \
        --n-scenarios 50 --n-windows 200 --chunk-size 10 --out-dir test_shards/

Print the array size for your sbatch script:
    python 05a_benchmark_simulate.py --n-scenarios 50 --n-windows 200 \
        --chunk-size 10 --print-array-size
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


from te_dfe.config import (
    MSMC_PATH, MU, RECOMB_MAP_PATH, ANNOTATION_PATH, ANNOTATION_FMT,
    CHROMOSOMES, SIM_LENGTH, WINDOW_SIZE, N_TRAIN_WINDOWS, MAF_MIN,
)
from te_dfe.demography import MSMCDemography, GenomeAnnotation, WindowSampler
from te_dfe.simulator import DFEBurstSimulator
from te_dfe.processor import TEDFEBurstProcessor
from te_dfe.prior import DFEBurstPrior

# Stream identifiers keep theta-draws and window-draws independent and
# reproducible. Changing these reshuffles seeds, so keep them fixed.
THETA_STREAM  = 0
WINDOW_STREAM = 1


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-scenarios", type=int, default=50,
                   help="Number of distinct test scenarios (each one theta).")
    p.add_argument("--n-windows",   type=int, default=N_TRAIN_WINDOWS,
                   help="Windows per scenario (e.g. 200).")
    p.add_argument("--chunk-size",  type=int, default=10,
                   help="Windows simulated per array task.")
    p.add_argument("--master-seed", type=int, default=20240601,
                   help="Master seed; defines the whole benchmark set.")
    p.add_argument("--task-id",     type=int, default=None,
                   help="SLURM_ARRAY_TASK_ID -> (scenario, chunk). 0-indexed.")
    p.add_argument("--scenario-id", type=int, default=None,
                   help="Explicit scenario (alternative to --task-id).")
    p.add_argument("--chunk-id",    type=int, default=None,
                   help="Explicit chunk within the scenario.")
    p.add_argument("--out-dir",     type=str, default="test_shards/")
    p.add_argument("--print-array-size", action="store_true",
                   help="Print total_tasks for the sbatch array and exit.")
    p.add_argument("--overwrite",   action="store_true",
                   help="Re-simulate even if the shard file exists.")
    # Register --model so argparse accepts it (prescan_model already
    # consumed its VALUE into TE_DFE_MODEL; this makes the flag legal).
    if add_model_arg is not None:
        add_model_arg(p)
    return p.parse_args()


def scenario_theta(master_seed, scenario_id):
    """Deterministic per-scenario theta (shared across all its chunks)."""
    ss = np.random.SeedSequence([master_seed, scenario_id, THETA_STREAM])
    return DFEBurstPrior.sample(np.random.default_rng(ss))


def window_seed(master_seed, scenario_id, window_idx):
    """Deterministic per-window SLiM seed, independent of chunking."""
    ss = np.random.SeedSequence(
        [master_seed, scenario_id, WINDOW_STREAM, window_idx])
    return int(np.random.default_rng(ss).integers(1, 2_000_000_000))


def main():
    args = parse_args()
    n_chunks = math.ceil(args.n_windows / args.chunk_size)
    total_tasks = args.n_scenarios * n_chunks

    if args.print_array_size:
        print(total_tasks)
        return 0

    if args.task_id is not None:
        scenario_id = args.task_id // n_chunks
        chunk_id    = args.task_id %  n_chunks
    elif args.scenario_id is not None and args.chunk_id is not None:
        scenario_id, chunk_id = args.scenario_id, args.chunk_id
    else:
        print("ERROR: give --task-id, or both --scenario-id and --chunk-id.",
              file=sys.stderr)
        return 2

    if not (0 <= scenario_id < args.n_scenarios):
        print(f"ERROR: scenario {scenario_id} out of range "
              f"[0,{args.n_scenarios}).", file=sys.stderr)
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
    shard_path = out_dir / (f"scenario_{scenario_id:04d}"
                            f"_w{w_start:04d}-{w_end:04d}.npz")
    if shard_path.exists() and not args.overwrite:
        print(f"[s{scenario_id} c{chunk_id}] {shard_path} exists, skipping")
        return 0

    t0 = time.time()
    print(f"[s{scenario_id} c{chunk_id}] windows {w_start}..{w_end-1} "
          f"({len(window_indices)} of {args.n_windows})")

    demog = MSMCDemography.from_file(MSMC_PATH, mu=MU, sim_length=SIM_LENGTH)
    annot = (GenomeAnnotation.from_gtf(ANNOTATION_PATH)
             if ANNOTATION_FMT == "gtf"
             else GenomeAnnotation.from_gff(ANNOTATION_PATH))
    sampler = WindowSampler(
        recomb_map_path=RECOMB_MAP_PATH, annotation=annot,
        chromosomes=CHROMOSOMES, window_size=WINDOW_SIZE,
    )
    sim = DFEBurstSimulator(demography=demog, sampler=sampler, n_windows=1)
    proc = TEDFEBurstProcessor(maf_min=MAF_MIN)

    theta = scenario_theta(args.master_seed, scenario_id)

    from te_dfe.config import REPRESENTATION
    if REPRESENTATION == "sumstat":
        # Per-TE summary statistics. A scenario's TEs are pooled across ALL its
        # windows, but those windows are split across chunk-shards, so each task
        # saves the RAW (unpadded) per-TE features for ITS windows; 05b
        # concatenates a scenario's shards and pads once. proc(..., pad=False)
        # returns (n_te_window, N_SUMSTATS_PER_TE).
        from te_dfe.config import N_SUMSTATS_PER_TE
        pooled, ok_idx = [], []
        for w_idx in window_indices:
            w_seed = window_seed(args.master_seed, scenario_id, w_idx)
            try:
                # FIX: pass window_seeds so the window is sampled INDEPENDENTLY
                # (simulator: window_seeds -> [sampler.sample(rng)]), exactly as
                # 01_simulate_shard.py does for training. Without it the call
                # fell through to sampler.sample_n(cover_covariate_space=True),
                # which oversamples 6 candidates and returns the one in the
                # LOWEST (gene_density, recomb) quintile bin -- i.e. every
                # benchmark window was the min-gene-density/min-recomb pick of 6,
                # not a random draw. That made the benchmark a biased, atypically
                # low-recombination slice of the genome (median feature_r ~ -1.5
                # vs ~ -0.1 in training), depressing flanking-SNP heterozygosity
                # ~2x via background selection, and leaving almost no
                # recombination range from which to identify alpha_r / gamma_r.
                windows, _ = sim(seed=w_seed, theta_provided=theta,
                                 n_windows_override=1, window_seeds=[w_seed])
                ts, cov, rmap, *rest = windows[0]
                proc.gene_intervals = rest[0] if rest else None
                # Optional: some simulators expose last_window_coords, others do
                # not (simulator_context defines it nowhere; simulator_simple only
                # inside __call__). GUARD it exactly as 01_simulate.py does --
                # gene_intervals are already window-relative and the sumstat path
                # does not read window_start, so this is a nice-to-have only.
                if getattr(sim, "last_window_coords", None):
                    ch, st = sim.last_window_coords[0]
                    proc.window_chrom, proc.window_start = ch, st
                raw = proc(ts, seed=w_seed, recomb_map=rmap, pad=False)
                if raw.shape[0]:
                    pooled.append(np.asarray(raw, dtype=np.float32))
                ok_idx.append(w_idx)
            except Exception as e:
                traceback.print_exc()
                print(f"[s{scenario_id} c{chunk_id}] window {w_idx} FAILED: {e}",
                      file=sys.stderr)
        feats_raw = (np.concatenate(pooled, axis=0) if pooled
                     else np.zeros((0, N_SUMSTATS_PER_TE), dtype=np.float32))
        if not ok_idx:
            print(f"[s{scenario_id} c{chunk_id}] no windows succeeded",
                  file=sys.stderr)
            return 1
        from te_dfe.config import WINDOW_SIZE as _WS
        np.savez_compressed(
            shard_path,
            feats_raw=feats_raw, theta=theta, format="sumstat",
            seq_len_bp=np.float64(float(args.n_windows) * float(_WS)),
            window_indices=np.array(ok_idx, dtype=np.int32),
            scenario_id=np.int32(scenario_id),
            n_windows_total=np.int32(args.n_windows),
            master_seed=np.int64(args.master_seed),
        )
        print(f"[s{scenario_id} c{chunk_id}] saved {shard_path} "
              f"feats_raw={feats_raw.shape} ({len(ok_idx)} windows) "
              f"elapsed={time.time()-t0:.1f}s")
        return 0

    geno_list, pos_list, cov_list, ok_idx = [], [], [], []
    for w_idx in window_indices:
        w_seed = window_seed(args.master_seed, scenario_id, w_idx)
        try:
            # FIX: same independent-window sampling as the sumstat branch above
            # and as 01_simulate_shard.py training. See the comment there.
            windows, _ = sim(seed=w_seed, theta_provided=theta,
                             n_windows_override=1, window_seeds=[w_seed])
            ts, cov, rmap, *rest = windows[0]
            proc.gene_intervals = rest[0] if rest else None
            if getattr(sim, "last_window_coords", None):      # guarded; see above
                ch, st = sim.last_window_coords[0]
                proc.window_chrom, proc.window_start = ch, st
            geno, pos = proc(ts, seed=w_seed, recomb_map=rmap)
            geno_list.append(geno)
            pos_list.append(pos)
            cov_list.append(cov)
            ok_idx.append(w_idx)
        except Exception as e:
            traceback.print_exc()
            print(f"[s{scenario_id} c{chunk_id}] window {w_idx} FAILED: {e}",
                  file=sys.stderr)

    if not geno_list:
        print(f"[s{scenario_id} c{chunk_id}] no windows succeeded",
              file=sys.stderr)
        return 1

    genotypes  = np.stack(geno_list, axis=0).astype(np.int8)
    positions  = np.stack(pos_list,  axis=0).astype(np.float32)
    covariates = np.stack(cov_list,  axis=0).astype(np.float32)

    np.savez_compressed(
        shard_path,
        geno=genotypes, pos=positions, cov=covariates, theta=theta,
        window_indices=np.array(ok_idx, dtype=np.int32),
        scenario_id=np.int32(scenario_id),
        n_windows_total=np.int32(args.n_windows),
        master_seed=np.int64(args.master_seed),
    )
    print(f"[s{scenario_id} c{chunk_id}] saved {shard_path} "
          f"geno={genotypes.shape} ({len(ok_idx)} windows) "
          f"elapsed={time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
