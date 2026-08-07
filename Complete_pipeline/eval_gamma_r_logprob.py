#!/usr/bin/env python3
"""
Stratified log-prob by true gamma_r — diagnose whether the flow can FIT high
gamma_r (r in [1,2]) or hits a representation ceiling there.

Computes per-sim log q(theta | x) = de.log_prob(theta, x) EXACTLY as 03_train's
loss does, then bins by the true gamma_r and reports the mean log-prob per bin.

  * TRAIN fits [1,2] but VAL doesn't  -> capacity / overfitting
  * TRAIN ALSO can't fit [1,2]        -> representation ceiling (per-TE sumstats
                                         miss the empty low-recomb field)

Speed: runs the trained flow's forward pass over the data, so it is dominated by
network cost. Use a GPU (default: auto), and --max-sims to subsample — a few
thousand sims estimate the per-bin means fine.

Usage:
  python eval_gamma_r_logprob.py --model context --dataset sims/train \
      --run-dir results [--bins 0,1,2,10] [--k-eval 1000] [--max-sims 2000] \
      [--batch 512] [--device cuda]
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from model_select import prescan_model, add_model_arg
prescan_model()

from te_dfe.config import PARAM_NAMES, INFER_NAMES, INFER_INDICES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arg(ap)
    ap.add_argument("--dataset", required=True,
                    help="collated dataset base path (without .feats.npy etc.)")
    ap.add_argument("--run-dir", required=True,
                    help="training output dir with posterior.pkl + training_meta.pkl")
    ap.add_argument("--bins", default="0,1,2,10",
                    help="comma-separated gamma_r bin edges (default 0,1,2,10)")
    ap.add_argument("--k-eval", type=int, default=None,
                    help="cap each sim to this many random valid TEs (match the "
                         "variable-K training range; default: full mask)")
    ap.add_argument("--max-sims", type=int, default=None,
                    help="score only the first N sims (a random subsample, since "
                         "sim order is prior-draw order). Big speedup for a diagnostic.")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = ("cuda" if (args.device == "auto" and torch.cuda.is_available())
              else args.device if args.device != "auto" else "cpu")

    run = Path(args.run_dir)
    with open(run / "posterior.pkl", "rb") as f:
        posterior = pickle.load(f)
    de = posterior.posterior_estimator
    de.eval().to(device)                                 # <-- move flow+embedding to GPU
    meta = {}
    if (run / "training_meta.pkl").exists():
        with open(run / "training_meta.pkl", "rb") as f:
            meta = pickle.load(f)

    base = Path(args.dataset)
    feats_path = base.with_suffix(".feats.npy")
    mask_path  = base.with_suffix(".te_mask.npy")
    theta_path = base.with_suffix(".theta.pt")
    for p in (feats_path, mask_path, theta_path):
        if not p.exists():
            sys.exit(f"ERROR: missing {p} — is --dataset a collated sumstat set?")

    feats = np.load(feats_path, mmap_mode="r")           # (N, Kmax, F)
    tmask = np.load(mask_path,  mmap_mode="r")           # (N, Kmax)
    tmeta = torch.load(theta_path, map_location="cpu", weights_only=False)
    thetas_full = np.asarray(tmeta["thetas"], dtype=np.float32)
    N, Kmax, F = feats.shape
    if args.max_sims is not None:
        N = min(N, args.max_sims)

    infer_idx = list(INFER_INDICES)
    thetas = thetas_full[:N, infer_idx]                  # (N, P_inf)
    if "gamma_r" not in INFER_NAMES:
        sys.exit(f"ERROR: gamma_r not inferred ({list(INFER_NAMES)}).")
    gr = list(INFER_NAMES).index("gamma_r")
    gamma_vals = thetas[:, gr]

    use_seqlen = bool(meta.get("sumstat_seqlen_cov", False))
    seq_len = np.asarray(tmeta["seq_len_bp"]) if (use_seqlen and "seq_len_bp" in tmeta) else None
    if use_seqlen and seq_len is None:
        sys.exit("ERROR: meta sumstat_seqlen_cov=True but no seq_len_bp in theta.pt")
    if use_seqlen:
        from te_dfe.config import seqlen_covariate

    edges = np.array([float(x) for x in args.bins.split(",")], dtype=np.float64)
    rng = np.random.default_rng(args.seed)

    def build_x(b0, b1):
        """Vectorized flat batch [feats_flat | mask_flat | seqlen_cov?]."""
        fm = np.asarray(feats[b0:b1], dtype=np.float32).reshape(b1 - b0, -1)
        mk = np.asarray(tmask[b0:b1], dtype=np.float32).copy()
        if args.k_eval is not None:
            for r in range(mk.shape[0]):
                real = np.flatnonzero(mk[r] > 0)
                if real.size > args.k_eval:
                    mk[r, rng.permutation(real)[args.k_eval:]] = 0.0
        parts = [fm, mk]
        if use_seqlen:
            sc = np.array([[seqlen_covariate(float(seq_len[s]))]
                           for s in range(b0, b1)], np.float32)
            parts.append(sc)
        return torch.from_numpy(np.concatenate(parts, axis=1)).float()

    logp = np.empty(N, dtype=np.float64)
    t0 = time.time()
    with torch.no_grad():
        for b0 in range(0, N, args.batch):
            b1 = min(b0 + args.batch, N)
            th = torch.from_numpy(thetas[b0:b1]).float().to(device)
            x  = build_x(b0, b1).to(device)
            lp = de.log_prob(th, x).squeeze()
            logp[b0:b1] = lp.detach().cpu().numpy().reshape(-1)
            if (b1 % (args.batch * 10) == 0) or b1 == N:
                print(f"  scored {b1}/{N}  ({time.time()-t0:.0f}s, dev={device})")

    print(f"\nStratified log-prob by true gamma_r  (dataset={base.name}, "
          f"run={run.name}, k_eval={args.k_eval or 'full'}, n={N})")
    print(f"{'gamma_r bin':>16} {'n':>7} {'mean logp':>12} {'median':>10}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (gamma_vals >= lo) & (gamma_vals < hi)
        if not sel.any():
            print(f"{f'[{lo:g},{hi:g})':>16} {0:>7} {'--':>12} {'--':>10}")
            continue
        v = logp[sel]
        print(f"{f'[{lo:g},{hi:g})':>16} {sel.sum():>7} {v.mean():>12.4f} {np.median(v):>10.4f}")
    print(f"{'ALL':>16} {N:>7} {logp.mean():>12.4f} {np.median(logp):>10.4f}")
    print("\nRead: high-gamma_r bin far below low bins on the TRAIN set -> "
          "representation ceiling (more heads/width won't help). TRAIN fits it "
          "but held-out doesn't -> overfitting (width + regularization).")


if __name__ == "__main__":
    main()
