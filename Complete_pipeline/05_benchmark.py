#!/usr/bin/env python3
"""05_benchmark.py — SBC, recovery, and scaling tests for DFE+burst model."""
import argparse
import importlib.util
import os
import pickle
import sys
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
    N_TRAIN_WINDOWS, PARAM_NAMES, PARAM_LABELS, N_PARAMS,
    MSMC_PATH, MU, RECOMB_MAP_PATH, ANNOTATION_PATH, ANNOTATION_FMT,
    CHROMOSOMES, SIM_LENGTH, WINDOW_SIZE,
)
from te_dfe.demography import MSMCDemography, GenomeAnnotation, WindowSampler
from te_dfe.simulator import DFEBurstSimulator

_spec = importlib.util.spec_from_file_location(
    "infer_mod", Path(__file__).parent / "04_infer.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
DFEBurstInference    = _mod.DFEBurstInference
bypass_embedding_net = _mod.bypass_embedding_net


def _build_test_simulator(n_windows):
    demog = MSMCDemography.from_file(MSMC_PATH, mu=MU, sim_length=SIM_LENGTH)
    annot = (GenomeAnnotation.from_gtf(ANNOTATION_PATH)
             if ANNOTATION_FMT == "gtf"
             else GenomeAnnotation.from_gff(ANNOTATION_PATH))
    sampler = WindowSampler(
        recomb_map_path=RECOMB_MAP_PATH, annotation=annot,
        chromosomes=CHROMOSOMES, window_size=WINDOW_SIZE,
    )
    return DFEBurstSimulator(demog, sampler, n_windows=n_windows)


def _simulate_and_embed(infer, sim, seed):
    windows, theta = sim(seed=seed)
    ts_list = [ts for ts, *_ in windows]
    cov = np.stack([c for _, c, *_ in windows])
    emb, attn = infer.get_embedding(ts_list, cov)
    return emb, theta, attn


def run_sbc(infer, n_test, n_post=500, seed_offset=100_000):
    print(f"\n=== SBC: {n_test} reps, {n_post} post samples ===")
    sim = _build_test_simulator(N_TRAIN_WINDOWS)
    ranks = []
    for i in range(n_test):
        try:
            emb, theta, _ = _simulate_and_embed(infer, sim, seed_offset + i)
            with bypass_embedding_net(infer.posterior, embedding_dim=emb.shape[0]):
                samp = infer.posterior.sample(
                    (n_post,), x=emb.unsqueeze(0),
                ).numpy()
            ranks.append((samp < theta[None, :]).sum(axis=0))
            if (i+1) % 25 == 0: print(f"  {i+1}/{n_test} done")
        except Exception as e:
            print(f"  rep {i} failed: {e}", file=sys.stderr)
    return {"ranks": np.stack(ranks), "n_post": n_post}


def report_sbc(res, out_path=None):
    ranks, n_post = res["ranks"], res["n_post"]
    print("\nSBC coverage report:")
    for ci in [50, 90]:
        lo, hi = (100-ci)/2, 100-(100-ci)/2
        in_ci = np.mean(
            (ranks > n_post*lo/100) & (ranks < n_post*hi/100), axis=0,
        )
        print(f"\n  {ci}% CI:")
        for name, c in zip(PARAM_NAMES, in_ci):
            flag = "OK" if abs(c - ci/100) < 0.05 else "!!"
            print(f"    [{flag}] {name:<25}: {c:.3f}  (target {ci/100:.2f})")
    if out_path:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            ncols = 4
            nrows = (N_PARAMS + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3*nrows))
            axes = axes.flatten()
            n_test = ranks.shape[0]
            for i, lbl in enumerate(PARAM_LABELS):
                axes[i].hist(ranks[:, i], bins=20, range=(0, n_post),
                              edgecolor="white", alpha=0.85)
                axes[i].axhline(n_test/20, color="red", lw=1, ls="--")
                axes[i].set_title(lbl, fontsize=8)
            for j in range(N_PARAMS, len(axes)): axes[j].set_visible(False)
            plt.suptitle(f"SBC ranks (n={n_test})", fontsize=11)
            plt.tight_layout()
            fig.savefig(out_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"  Saved → {out_path}")
        except ImportError:
            pass


def run_recovery(infer, n_test, seed_offset=200_000, n_samples=1000):
    print(f"\n=== Recovery: {n_test} reps ===")
    sim = _build_test_simulator(N_TRAIN_WINDOWS)
    means, lo90, hi90, truths = [], [], [], []
    for i in range(n_test):
        try:
            emb, theta, _ = _simulate_and_embed(infer, sim, seed_offset + i)
            with bypass_embedding_net(infer.posterior, embedding_dim=emb.shape[0]):
                samp = infer.posterior.sample(
                    (n_samples,), x=emb.unsqueeze(0),
                ).numpy()
            means.append(samp.mean(0))
            lo90.append(np.percentile(samp, 5, axis=0))
            hi90.append(np.percentile(samp, 95, axis=0))
            truths.append(theta)
        except Exception as e:
            print(f"  rep {i} failed: {e}", file=sys.stderr)
    return {"means": np.stack(means), "lo90": np.stack(lo90),
            "hi90": np.stack(hi90), "truths": np.stack(truths)}


def report_recovery(res):
    print("\nRecovery:")
    print(f"  {'Parameter':<35}{'r':>8}{'RMSE':>10}{'Cov90':>10}")
    print("  " + "-"*63)
    for i, lbl in enumerate(PARAM_LABELS):
        x, y = res["truths"][:, i], res["means"][:, i]
        r = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 else 0.0
        rmse = float(np.sqrt(np.mean((x-y)**2)))
        cov90 = float(np.mean((x >= res["lo90"][:, i]) & (x <= res["hi90"][:, i])))
        print(f"  {lbl:<35}{r:>8.3f}{rmse:>10.4f}{cov90:>10.2f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=str, required=True)
    p.add_argument("--out",     type=str, default=None)
    p.add_argument("--test",    type=str, default="all",
                   choices=["all", "sbc", "recovery"])
    p.add_argument("--n-test",  type=int, default=200)
    # Register --model so argparse accepts it (prescan_model already
    # consumed its VALUE into TE_DFE_MODEL; this makes the flag legal).
    if add_model_arg is not None:
        add_model_arg(p)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out or os.path.join(args.results, "benchmark"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading model from {args.results}...")
    infer = DFEBurstInference.load(args.results)

    sbc_res = None
    rec_res = None

    if args.test in ("all", "sbc"):
        sbc_res = run_sbc(infer, args.n_test)
        with open(out_dir/"sbc.pkl", "wb") as f: pickle.dump(sbc_res, f)
        report_sbc(sbc_res, out_dir/"sbc_ranks.pdf")

    if args.test in ("all", "recovery"):
        rec_res = run_recovery(infer, args.n_test)
        with open(out_dir/"recovery.pkl", "wb") as f: pickle.dump(rec_res, f)
        report_recovery(rec_res)

    # ----- popgen-npe-style diagnostic plots -----
    try:
        from te_dfe.diagnostics import (
            plot_calibration, plot_expectation, plot_concentration,
            write_recovery_csv,
        )
        if sbc_res is not None:
            plot_calibration(sbc_res, PARAM_LABELS,
                              out_dir / "posterior_calibration.pdf")
            print(f"  Saved → posterior_calibration.pdf")
        if rec_res is not None:
            plot_expectation(rec_res, PARAM_LABELS,
                              out_dir / "posterior_expectation.pdf")
            print(f"  Saved → posterior_expectation.pdf")

            # Build prior bounds for concentration plot
            from te_dfe.prior import DFEBurstPrior
            sbi_prior = DFEBurstPrior.to_sbi_prior()
            plot_concentration(
                rec_res,
                sbi_prior.low.numpy(),
                sbi_prior.high.numpy(),
                PARAM_LABELS,
                out_dir / "posterior_concentration.pdf",
            )
            print(f"  Saved → posterior_concentration.pdf")

            write_recovery_csv(rec_res, PARAM_NAMES,
                                out_dir / "recovery_results.csv")
            print(f"  Saved → recovery_results.csv")
    except Exception as e:
        print(f"  (Diagnostic plots skipped: {e})")

    print(f"\nBenchmarks done → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
