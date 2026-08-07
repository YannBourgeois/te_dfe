#!/usr/bin/env python
"""
eval_val_recovery.py — IN-DISTRIBUTION recovery on the held-out VALIDATION split
of the TRAINING set. A cheap, early sanity check BEFORE any benchmark:

  * true-vs-estimated scatter per parameter (identifiability of the point estimate)
  * per-parameter RMSE / bias
  * coverage of the 90% credible interval + SBC p-value (calibration)

This is NOT a benchmark. The val sims are held out from TRAINING but drawn from
the SAME prior and SAME simulator as the training data, so this answers only
"did the network learn the map, or memorise it?" -- it catches unidentifiable
parameters and miscalibration early and cheaply. It CANNOT see the fidelity gap
(msprime vs SLiM) or the deployment distribution shift (stratified vs
abundance-weighted windows). Real evaluation still requires INDEPENDENT sims
under deployment conditions (05b_benchmark_analyze.py).

For a PRETRAIN (msprime) model the coverage panel is the more informative one:
clean low-fidelity data breeds tight posteriors that a scatter flatters but
coverage exposes as over-confident -- exactly what fine-tuning must widen.

Usage
  TE_DFE_MODEL=context python eval_val_recovery.py \
      --dataset simulations_msprime/dataset \
      --run-dir mfnpe/pretrain_msprime \
      --val-frac 0.10 --seed 0 --n-post 500 [--max-sims 500] [--device auto]

--val-frac and --seed MUST match the training run (03_train.py defaults
val_frac=0.1, seed=0), or the "held-out" set is not actually held out.
"""
import argparse, pickle, sys
from pathlib import Path
import numpy as np
import torch

# Reuse the TESTED pieces from the benchmark analyzer rather than re-implement
# sampling/plotting/coverage (which is where a hand-rolled version goes subtly
# wrong -- e.g. the K-extrapolation embedding bypass).
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "bench05b", str(Path(__file__).parent / "05b_benchmark_analyze.py"))
b = importlib.util.module_from_spec(_spec)
sys.modules["bench05b"] = b
_spec.loader.exec_module(b)

from te_dfe.config import PARAM_NAMES, INFER_NAMES, INFER_INDICES, MODEL_NAME


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="TRAINING dataset base (…/dataset); the val split is "
                         "reconstructed from it.")
    ap.add_argument("--run-dir", required=True,
                    help="training output dir with posterior.pkl, network.pt, "
                         "training_meta.pkl")
    ap.add_argument("--val-frac", type=float, default=0.10,
                    help="MUST match the training run (03_train.py default 0.1).")
    ap.add_argument("--subset-n", type=int, default=None,
                    help="MUST match the training run's --subset-n (the budget). "
                         "finetune_nN / scratch_nN trained on the FIRST N sims, so "
                         "the val split must be reconstructed over range(N), not the "
                         "whole dataset -- otherwise the 'held-out' set overlaps the "
                         "training data and the numbers are optimistic. Omit only for "
                         "the pretrain run (trained on all sims).")
    ap.add_argument("--seed", type=int, default=0,
                    help="MUST match the training run (default 0).")
    ap.add_argument("--n-post", type=int, default=500,
                    help="posterior samples per val sim.")
    ap.add_argument("--max-sims", type=int, default=500,
                    help="cap on val sims scored (speed). None = all.")
    ap.add_argument("--k-eval", type=int, default=None,
                    help="cap each sim to this many random valid TEs.")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", default=None,
                    help="output dir (default <run-dir>/val_recovery).")
    args = ap.parse_args()

    device = ("cuda" if (args.device == "auto" and torch.cuda.is_available())
              else args.device if args.device != "auto" else "cpu")
    run = Path(args.run_dir)
    out = Path(args.out) if args.out else run / "val_recovery"
    out.mkdir(parents=True, exist_ok=True)

    # --- posterior + embedding core (both from the run dir) ------------------
    with open(run / "posterior.pkl", "rb") as f:
        posterior = pickle.load(f)
    posterior.posterior_estimator.eval().to(device)
    with open(run / "training_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    core = b.load_core_network(run, meta).to(device)

    # --- dataset: feats, te_mask, thetas -------------------------------------
    # 02_collate writes <base>.feats.npy / <base>.te_mask.npy / <base>.theta.pt
    # via Path.with_suffix (so results/dataset -> results/dataset.feats.npy).
    base = Path(args.dataset)
    feats = np.load(base.with_suffix(".feats.npy"), mmap_mode="r")
    tmask = np.load(base.with_suffix(".te_mask.npy"), mmap_mode="r")
    tmeta = torch.load(base.with_suffix(".theta.pt"),
                       map_location="cpu", weights_only=False)
    thetas = np.asarray(tmeta["thetas"], np.float64)
    N = len(thetas)

    if thetas.shape[1] != len(PARAM_NAMES):
        sys.exit(f"theta width {thetas.shape[1]} != len(PARAM_NAMES) "
                 f"{len(PARAM_NAMES)} for TE_DFE_MODEL={MODEL_NAME!r}. Wrong model.")

    # --- RECONSTRUCT THE EXACT VAL SPLIT (same as 03_train.py) ---------------
    # 03_train.py, in order:
    #   if subset_n: full = Subset(full, range(subset_n))     # FIRST N sims
    #   n_val = round(val_frac * len(full));  random_split(full, ..., gen(seed))
    # So the split is over range(subset_n) (or range(N) if no subset), and the
    # indices ARE dataset rows because the subset is range(subset_n). We must
    # mirror the subset FIRST, then split -- splitting over range(N) would treat
    # sims the run never saw as "held out".
    pool_n = N if args.subset_n is None else min(args.subset_n, N)
    if args.subset_n is not None:
        print(f"budget subset: first {pool_n} sims (matching the run's --subset-n)")
    n_val = max(1, int(round(args.val_frac * pool_n)))
    n_train = pool_n - n_val
    gen = torch.Generator().manual_seed(args.seed)
    _, val_subset = torch.utils.data.random_split(
        range(pool_n), [n_train, n_val], generator=gen)
    val_idx = np.array(list(val_subset), dtype=int)
    print(f"reconstructed val split: {len(val_idx)} held-out sims "
          f"(val_frac={args.val_frac}, seed={args.seed}, pool={pool_n})")
    if args.max_sims and len(val_idx) > args.max_sims:
        val_idx = np.random.default_rng(args.seed).choice(
            val_idx, args.max_sims, replace=False)
        print(f"  scoring a random {args.max_sims} of them (--max-sims)")

    infer_idx = list(INFER_INDICES)
    names = list(INFER_NAMES)
    P = len(infer_idx)

    # --- sample the posterior per val sim ------------------------------------
    rng = np.random.default_rng(args.seed)
    true_t, mean, median, lo, hi = ([] for _ in range(5))
    sbc_ranks, covered, ok_idx = [], [], []
    n_failed = 0
    for j, i in enumerate(val_idx):
        f_i = np.asarray(feats[i])                      # (Kmax, F)
        m_i = np.asarray(tmask[i]).astype(np.float32)   # (Kmax,)
        if args.k_eval is not None:
            real = np.flatnonzero(m_i > 0)
            if real.size > args.k_eval:
                m_i = m_i.copy()
                m_i[rng.permutation(real)[args.k_eval:]] = 0.0
        feats_t = torch.from_numpy(f_i[None]).float().to(device)   # (1,Kmax,F)
        mask_t  = torch.from_numpy(m_i[None]).float().to(device)   # (1,Kmax)
        try:
            samp = b.sample_via_embedding_sumstat(
                posterior, core, feats_t, mask_t, args.n_post)      # (n_kept,P)
        except Exception as e:
            n_failed += 1
            if n_failed <= 3:
                print(f"  sim {i}: sampling failed ({type(e).__name__}: {e})")
            continue
        if samp is None or len(samp) < 10:
            n_failed += 1; continue
        th_true = thetas[i, infer_idx]
        ok_idx.append(int(i))
        true_t.append(th_true)
        mean.append(samp.mean(0)); median.append(np.median(samp, 0))
        q = np.percentile(samp, [5, 95], axis=0)
        lo.append(q[0]); hi.append(q[1])
        covered.append((th_true >= q[0]) & (th_true <= q[1]))
        # SBC rank: # posterior draws below the truth, per param
        sbc_ranks.append((samp < th_true[None]).sum(0))
        if (j + 1) % 50 == 0:
            print(f"  scored {j+1}/{len(val_idx)}")

    if not true_t:
        sys.exit("No val sims scored successfully.")
    true_t = np.array(true_t); mean = np.array(mean); median = np.array(median)
    lo = np.array(lo); hi = np.array(hi)
    covered = np.array(covered); sbc_ranks = np.array(sbc_ranks)

    rmse = np.sqrt(((median - true_t) ** 2).mean(0))
    bias = (median - true_t).mean(0)
    coverage = covered.mean(0)
    # SBC p-value: uniformity of ranks (KS vs discrete uniform), reuse 05b if present
    try:
        sbc_p = b.sbc_pvalues(sbc_ranks, args.n_post)
    except Exception:
        from scipy import stats
        sbc_p = np.array([stats.kstest(sbc_ranks[:, k] / args.n_post,
                                       "uniform").pvalue for k in range(P)])

    # --- outputs: reuse 05b's writers/plotters -------------------------------
    # write_recovery_csv expects each 'used' element to be a file-like object
    # with a .name (it was written for the benchmark's per-file loop). Wrap the
    # val-sim indices so .name reports the original dataset row.
    class _ValRef:
        __slots__ = ("name",)
        def __init__(self, idx): self.name = f"val_sim_{int(idx)}"
    used = [_ValRef(i) for i in ok_idx]   # actual successful sims, aligned
    b.write_summary(out / "val_recovery_summary.txt", used, args.n_post,
                    rmse, bias, coverage, sbc_p, n_failed, names=names)
    b.write_recovery_csv(out / "val_recovery_values.csv", used, true_t,
                         mean, median, lo, hi, covered, names=names)
    try:
        b.make_plots(out, true_t, mean, lo, hi, sbc_ranks, args.n_post,
                     rmse, coverage, sbc_p, len(used), names=names)
    except Exception as e:
        print(f"  (make_plots skipped: {type(e).__name__}: {e})")

    print(f"\n=== IN-DISTRIBUTION VAL RECOVERY ({MODEL_NAME}) ===")
    print("NOT a benchmark -- real evaluation needs independent sims under "
          "deployment conditions (05b).")
    print(f"  {'param':<16}{'RMSE':>9}{'bias':>9}{'cover90':>9}{'sbc_p':>9}")
    for k, nm in enumerate(names):
        flag = "  <- under-covered" if coverage[k] < 0.8 else ""
        print(f"  {nm:<16}{rmse[k]:>9.3f}{bias[k]:>9.3f}"
              f"{coverage[k]:>9.2f}{sbc_p[k]:>9.3f}{flag}")
    print(f"\n  scored {len(true_t)} val sims ({n_failed} failed)  ->  {out}/")


if __name__ == "__main__":
    main()
