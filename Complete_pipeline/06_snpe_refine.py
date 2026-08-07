#!/usr/bin/env python3
"""
06_snpe_refine.py — SNPE refinement on a single observation.

Given a starting trained network (results_dir) and a target observation
(test_NNNNNN.npz), run N rounds of SNPE-C refinement:
  1. Sample theta from current posterior given the target observation
  2. Run SLiM with each sampled theta (SLURM array)
  3. Re-collate accumulated data
  4. Re-train the network
  5. Build new posterior

Outputs per-round posterior samples and a CSV summary comparing rounds
to the original amortized inference. Designed for benchmarking SNPE
sharpening on a small handful of test sims.

NOTE: v4 pure-gamma architecture. The simulator's input parameters ARE
the network's output parameters — no internal/external transform needed.
SNPE samples from the posterior in this single parameter space and feeds
directly to the simulator.

Usage:
    python 06_snpe_refine.py \\
        --results-dir results/ \\
        --observation test_simulations/test_10000005.npz \\
        --out-dir results/snpe/sim_5/ \\
        --n-rounds 3 --n-sims-per-round 500 \\
        --slurm
"""
import argparse
import json
import pickle
import subprocess
import sys
import time
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



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True,
                    help="Trained-network directory (network.pt, posterior.pkl, "
                         "dataset.*.npy/.pt)")
    p.add_argument("--observation", required=True,
                    help="Path to a test_NNNNNN.npz file (the target observation)")
    p.add_argument("--out-dir", required=True,
                    help="Output directory for per-round artifacts")
    p.add_argument("--n-rounds", type=int, default=3,
                    help="Number of SNPE refinement rounds (default 3)")
    p.add_argument("--n-sims-per-round", type=int, default=500,
                    help="Number of new sims per round (default 500)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-posterior-samples", type=int, default=2000,
                    help="Posterior samples to draw at the start of each round")
    p.add_argument("--slurm", action="store_true",
                    help="Submit sims via SLURM array (otherwise run inline)")
    p.add_argument("--slurm-script", default="slurm/06_snpe_simulate_round.sh",
                    help="SLURM template script to use for per-round sim batches")
    p.add_argument("--slurm-account", default=None,
                    help="Optional --account argument to pass to sbatch")
    p.add_argument("--training-epochs", type=int, default=200,
                    help="Max training epochs per round (default 200)")
    p.add_argument("--ci-level", type=float, default=0.9)
    # Register --model so argparse accepts it (prescan_model already
    # consumed its VALUE into TE_DFE_MODEL; this makes the flag legal).
    if add_model_arg is not None:
        add_model_arg(p)
    return p.parse_args()


def _load_results_dir(results_dir, device):
    """Load network, posterior, and existing training data."""
    results_dir = Path(results_dir)
    # Load core network
    from te_dfe.config import (
        N_HAPS, N_SITES, N_COVARIATES, N_STATE_CHANNELS,
    )
    try:
        from te_dfe.config import N_CONTINUOUS_CHANNELS as N_CONT
    except ImportError:
        try:
            from te_dfe.config import N_POSITION_CHANNELS as N_CONT
        except ImportError:
            N_CONT = 0

    from te_dfe.network import TECovariateAttentionNet

    # Read training metadata to pick up aggregator kind + mask requirement
    meta_path = results_dir / "training_meta.pkl"
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        aggregator_kind   = meta.get("aggregator_kind", "attention")
        n_attention_heads = meta.get("n_attention_heads", 1)
        include_mask      = meta.get("include_mask", False)
        use_embedding_norm = meta.get("use_embedding_norm", False)
    else:
        aggregator_kind, n_attention_heads, include_mask = "attention", 1, False
        use_embedding_norm = False

    core = TECovariateAttentionNet(
        n_individuals=N_HAPS, n_sites=N_SITES, n_covariates=N_COVARIATES,
        n_state_channels=N_STATE_CHANNELS, n_cont_channels=N_CONT,
        aggregator_kind=aggregator_kind,
        n_attention_heads=n_attention_heads,
        use_embedding_norm=use_embedding_norm,
    )
    state_dict = torch.load(
        results_dir / "network.pt", map_location="cpu", weights_only=True,
    )
    core.load_state_dict(state_dict)
    core.eval()
    core = core.to(device)

    with open(results_dir / "posterior.pkl", "rb") as f:
        posterior = pickle.load(f)
    # Move posterior internals to device (sbi quirks across versions)
    for attr in ("posterior_estimator", "_neural_net", "neural_net"):
        if hasattr(posterior, attr):
            try:
                getattr(posterior, attr).to(device)
                break
            except Exception:
                continue
    for chain in (("prior", "base_dist"), ("_prior", "base_dist")):
        try:
            obj = posterior
            for a in chain:
                obj = getattr(obj, a)
            if hasattr(obj, "low") and hasattr(obj, "high"):
                obj.low  = obj.low.to(device)
                obj.high = obj.high.to(device)
            break
        except AttributeError:
            continue

    # Load existing training data
    geno  = np.load(results_dir / "dataset.geno.npy", mmap_mode="r")
    pos   = (np.load(results_dir / "dataset.pos.npy", mmap_mode="r")
              if (results_dir / "dataset.pos.npy").exists() else None)
    cov   = np.load(results_dir / "dataset.cov.npy", mmap_mode="r")
    theta = torch.load(results_dir / "dataset.theta.pt", map_location="cpu",
                        weights_only=True).cpu().numpy()
    return core, posterior, N_CONT, geno, pos, cov, theta, include_mask


def _flatten_observation(geno, pos, cov, n_windows: int = None,
                          include_mask: bool = False):
    """Pack a single multi-window observation into the flat tensor format
    SBIWrapper expects: [geno_flat, pos_flat, cov_flat, mask_flat?].

    When `include_mask=True`, append an all-ones mask of length
    `n_windows` (signals "use all windows"). Required when the trained
    SBIWrapper has include_mask=True.
    """
    parts = [torch.from_numpy(np.asarray(geno, dtype=np.float32).reshape(-1))]
    if pos is not None and pos.size > 0:
        parts.append(torch.from_numpy(np.asarray(pos, dtype=np.float32).reshape(-1)))
    parts.append(torch.from_numpy(np.asarray(cov, dtype=np.float32).reshape(-1)))
    if include_mask:
        if n_windows is None:
            raise ValueError("include_mask=True requires n_windows")
        parts.append(torch.ones(n_windows, dtype=torch.float32))
    return torch.cat(parts)


def _sample_posterior(posterior, x_obs_flat, n, device):
    """Sample n posterior samples given the flattened observation."""
    x_t = x_obs_flat.to(device).unsqueeze(0)
    samples = posterior.sample((n,), x=x_t, show_progress_bars=False)
    return samples.cpu().numpy()


def _summarize_samples(samples, ci_level=0.9):
    lo_q = (1.0 - ci_level) / 2.0
    hi_q = 1.0 - lo_q
    return {
        "mean":   samples.mean(axis=0),
        "median": np.median(samples, axis=0),
        "ci_lo":  np.quantile(samples, lo_q, axis=0),
        "ci_hi":  np.quantile(samples, hi_q, axis=0),
        "ci_w":   np.quantile(samples, hi_q, axis=0) - np.quantile(samples, lo_q, axis=0),
    }


def _run_simulations_inline(thetas, out_dir, base_seed):
    """Run sims one by one (slow; for debugging only). Each sim is launched
    as a subprocess of 01_simulate.py with --theta-file/--theta-row."""
    theta_file = out_dir / "round_thetas.npy"
    np.save(theta_file, thetas)
    repo = Path(__file__).parent
    sim_dir = out_dir / "simulations"
    sim_dir.mkdir(parents=True, exist_ok=True)
    for i, _ in enumerate(thetas):
        seed = base_seed + i
        subprocess.run([
            sys.executable, str(repo / "01_simulate.py"),
            "--seed", str(seed),
            "--out-dir", str(sim_dir),
            "--theta-file", str(theta_file),
            "--theta-row", str(i),
        ], check=True)
    return sim_dir, theta_file


def _run_simulations_slurm(thetas, out_dir, base_seed, slurm_script,
                             slurm_account=None):
    """Submit a SLURM array job for the sims; wait for completion."""
    out_dir = Path(out_dir)
    theta_file = out_dir / "round_thetas.npy"
    np.save(theta_file, thetas)
    sim_dir = out_dir / "simulations"
    sim_dir.mkdir(parents=True, exist_ok=True)
    n_sims = len(thetas)
    # Submit
    cmd = ["sbatch", "--parsable",
           "--array", f"0-{n_sims - 1}",
           "--export", (
               f"ALL,"
               f"THETA_FILE={theta_file},"
               f"BASE_SEED={base_seed},"
               f"OUT_DIR={sim_dir},"
               f"REPO_DIR={Path(__file__).parent.resolve()}"
           )]
    if slurm_account:
        cmd += ["--account", slurm_account]
    cmd += [str(slurm_script)]
    print(f"  [slurm] submitting: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    job_id = result.stdout.strip().split(";")[0]
    print(f"  [slurm] submitted as job array {job_id}")

    # Poll for completion
    while True:
        time.sleep(30)
        squeue = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%i"],
            capture_output=True, text=True,
        )
        running = [l for l in squeue.stdout.splitlines() if l.strip()]
        if not running:
            print(f"  [slurm] job array {job_id} complete")
            break
        print(f"  [slurm] still running: {len(running)} tasks remain")

    return sim_dir, theta_file


def _collate_sims(sim_dir, n_expected):
    """Load all sim_NNNNNN.npz files from sim_dir, return arrays."""
    files = sorted(sim_dir.glob("sim_*.npz"))
    if len(files) < n_expected:
        print(f"  [collate] WARNING: expected {n_expected} sims, found {len(files)}")
    genos, poses, covs, thetas = [], [], [], []
    for fp in files:
        d = np.load(fp)
        genos.append(d["geno"])
        if "pos" in d.files:
            poses.append(d["pos"])
        covs.append(d["cov"])
        thetas.append(d["theta"])
    geno = np.stack(genos, axis=0).astype(np.int8)
    pos  = (np.stack(poses, axis=0).astype(np.float32)
             if poses else None)
    cov  = np.stack(covs, axis=0).astype(np.float32)
    theta = np.stack(thetas, axis=0).astype(np.float32)
    return geno, pos, cov, theta


def _pack_observation_tensor(geno_batch, pos_batch, cov_batch, n_cont,
                              n_windows: int = None, include_mask: bool = False):
    """Pack multi-sim arrays into flat (B, D) tensor for sbi training.

    When `include_mask=True`, append an all-ones mask column (B, n_windows).
    """
    B = geno_batch.shape[0]
    parts = [torch.from_numpy(geno_batch.astype(np.float32).reshape(B, -1))]
    if pos_batch is not None and n_cont > 0:
        parts.append(torch.from_numpy(pos_batch.astype(np.float32).reshape(B, -1)))
    parts.append(torch.from_numpy(cov_batch.astype(np.float32).reshape(B, -1)))
    if include_mask:
        if n_windows is None:
            raise ValueError("include_mask=True requires n_windows")
        parts.append(torch.ones(B, n_windows, dtype=torch.float32))
    return torch.cat(parts, dim=1)


def _retrain(core, accumulated_data, prior, device, epochs,
             include_mask: bool = False):
    """Re-train the NPE posterior on accumulated (theta, x) data."""
    from sbi.inference import SNPE
    from te_dfe.network import SBIWrapper
    from te_dfe.config import (
        N_HAPS, N_SITES, N_COVARIATES, N_TRAIN_WINDOWS, N_STATE_CHANNELS,
    )
    try:
        from te_dfe.config import N_CONTINUOUS_CHANNELS as N_CONT
    except ImportError:
        try:
            from te_dfe.config import N_POSITION_CHANNELS as N_CONT
        except ImportError:
            N_CONT = 0

    theta_t, x_t = accumulated_data
    # Build a fresh SBIWrapper around the core
    embedding = SBIWrapper(
        network=core, n_windows=N_TRAIN_WINDOWS,
        n_channels=N_STATE_CHANNELS, n_individuals=N_HAPS,
        n_sites=N_SITES, n_cov=N_COVARIATES,
        n_position_channels=N_CONT,
        include_mask=include_mask,
    )

    # Build a Neural Spline Flow with this embedding
    from sbi.neural_nets import posterior_nn
    nn_builder = posterior_nn(
        model="nsf", embedding_net=embedding,
        z_score_theta="independent", z_score_x="none",
    )
    inference = SNPE(prior=prior, density_estimator=nn_builder, device=device)
    inference.append_simulations(theta_t, x_t)
    density_est = inference.train(
        max_num_epochs=epochs, show_train_summary=True,
    )
    posterior = inference.build_posterior(density_est)
    return posterior, density_est


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading starting network + training data ...")
    core, posterior, n_cont, geno0, pos0, cov0, theta0, include_mask = _load_results_dir(
        args.results_dir, device,
    )
    print(f"  loaded {len(theta0)} initial sims (N_PARAMS={theta0.shape[1]})")
    if include_mask:
        print(f"  (model expects mask column; will append all-ones at obs time)")

    # Need n_windows for mask padding
    from te_dfe.config import N_TRAIN_WINDOWS

    # Load the target observation
    obs_npz = np.load(args.observation)
    obs_geno  = obs_npz["geno"]
    obs_pos   = obs_npz["pos"] if "pos" in obs_npz.files else None
    obs_cov   = obs_npz["cov"]
    true_theta = obs_npz.get("theta", None)
    x_obs_flat = _flatten_observation(
        obs_geno, obs_pos, obs_cov,
        n_windows=N_TRAIN_WINDOWS, include_mask=include_mask,
    )
    print(f"  target observation: {args.observation}")
    if true_theta is not None:
        print(f"  true theta: {true_theta}")

    # Sample round-0 posterior (the amortized one)
    print("\nRound 0 (amortized): sampling posterior ...")
    samples_r0 = _sample_posterior(
        posterior, x_obs_flat, args.n_posterior_samples, device,
    )
    summary_r0 = _summarize_samples(samples_r0, args.ci_level)
    np.save(out_dir / "posterior_samples_round0.npy", samples_r0)
    print(f"  posterior CI widths: {summary_r0['ci_w']}")

    # Build prior for SNPE
    from te_dfe.prior import DFEBurstPrior
    prior = DFEBurstPrior.to_sbi_prior()

    # Pack initial training data
    accum_theta = torch.from_numpy(theta0.astype(np.float32))
    accum_x     = _pack_observation_tensor(
        np.asarray(geno0), np.asarray(pos0) if pos0 is not None else None,
        np.asarray(cov0), n_cont,
        n_windows=N_TRAIN_WINDOWS, include_mask=include_mask,
    )
    print(f"  initial training tensor: {accum_x.shape}")

    round_summaries = [{"round": 0, **{
        f"{k}_{i}": float(summary_r0[k][i])
        for k in ("mean", "ci_lo", "ci_hi", "ci_w")
        for i in range(samples_r0.shape[1])
    }}]

    for r in range(1, args.n_rounds + 1):
        print(f"\n===== ROUND {r} =====")
        # 1. Sample theta from current posterior
        print(f"  sampling {args.n_sims_per_round} theta from current posterior...")
        new_theta = _sample_posterior(
            posterior, x_obs_flat, args.n_sims_per_round, device,
        )
        # Clip to prior support to be safe (sbi sometimes outputs slightly OOB)
        prior_low  = prior.base_dist.low.cpu().numpy()
        prior_high = prior.base_dist.high.cpu().numpy()
        new_theta = np.clip(new_theta, prior_low, prior_high)

        # 2. Run simulations
        base_seed = 9_000_000 + r * 1_000_000  # avoid collision with training seeds
        round_out_dir = out_dir / f"round_{r}"
        round_out_dir.mkdir(parents=True, exist_ok=True)
        if args.slurm:
            sim_dir, _ = _run_simulations_slurm(
                new_theta, round_out_dir, base_seed,
                args.slurm_script, args.slurm_account,
            )
        else:
            sim_dir, _ = _run_simulations_inline(
                new_theta, round_out_dir, base_seed,
            )

        # 3. Collate new sims
        new_geno, new_pos, new_cov, new_theta_actual = _collate_sims(
            sim_dir, args.n_sims_per_round,
        )
        print(f"  collated {new_theta_actual.shape[0]} new sims")

        # 4. Append to accumulated training data
        new_x = _pack_observation_tensor(
            new_geno, new_pos, new_cov, n_cont,
            n_windows=N_TRAIN_WINDOWS, include_mask=include_mask,
        )
        accum_x     = torch.cat([accum_x, new_x], dim=0)
        accum_theta = torch.cat([
            accum_theta, torch.from_numpy(new_theta_actual.astype(np.float32))
        ], dim=0)
        print(f"  accumulated training tensor: {accum_x.shape}")

        # 5. Re-train
        print(f"  re-training ...")
        posterior, density_est = _retrain(
            core, (accum_theta, accum_x), prior, device, args.training_epochs,
            include_mask=include_mask,
        )
        # Save round artifacts
        torch.save(
            density_est.state_dict() if hasattr(density_est, "state_dict")
            else density_est,
            round_out_dir / "density_est.pt",
        )
        with open(round_out_dir / "posterior.pkl", "wb") as f:
            pickle.dump(posterior, f)

        # 6. Sample new posterior and record
        samples_r = _sample_posterior(
            posterior, x_obs_flat, args.n_posterior_samples, device,
        )
        summary_r = _summarize_samples(samples_r, args.ci_level)
        np.save(out_dir / f"posterior_samples_round{r}.npy", samples_r)
        print(f"  round {r} posterior CI widths: {summary_r['ci_w']}")

        round_summaries.append({"round": r, **{
            f"{k}_{i}": float(summary_r[k][i])
            for k in ("mean", "ci_lo", "ci_hi", "ci_w")
            for i in range(samples_r.shape[1])
        }})

    # Write summary JSON + CSV
    summary_path = out_dir / "snpe_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "observation": str(args.observation),
            "true_theta":  true_theta.tolist() if true_theta is not None else None,
            "n_rounds":    args.n_rounds,
            "n_sims_per_round": args.n_sims_per_round,
            "rounds":      round_summaries,
        }, f, indent=2)

    # Per-parameter CI-width-over-rounds CSV
    from te_dfe.config import PARAM_NAMES
    csv_path = out_dir / "snpe_ci_widths.csv"
    with open(csv_path, "w") as f:
        f.write("round," + ",".join(f"{p}_mean,{p}_ci_lo,{p}_ci_hi,{p}_ci_w"
                                       for p in PARAM_NAMES))
        if true_theta is not None:
            f.write("," + ",".join(f"{p}_true" for p in PARAM_NAMES))
        f.write("\n")
        for s in round_summaries:
            row = [str(s["round"])]
            for i in range(len(PARAM_NAMES)):
                row += [
                    f"{s[f'mean_{i}']:.6g}",
                    f"{s[f'ci_lo_{i}']:.6g}",
                    f"{s[f'ci_hi_{i}']:.6g}",
                    f"{s[f'ci_w_{i}']:.6g}",
                ]
            if true_theta is not None:
                row += [f"{float(true_theta[i]):.6g}" for i in range(len(PARAM_NAMES))]
            f.write(",".join(row) + "\n")

    print(f"\nDone. Outputs in {out_dir}/")
    print(f"  - summary: {summary_path}")
    print(f"  - CI table: {csv_path}")
    print(f"  - per-round posterior samples: posterior_samples_round*.npy")


if __name__ == "__main__":
    main()
