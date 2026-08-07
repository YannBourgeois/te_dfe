#!/usr/bin/env python3
"""
inspect_prior.py — v4 prior sampling diagnostic.

Samples N draws from the 6-dim pure-gamma prior and reports each
marginal's empirical distribution, plus the induced distribution on
the derived binned proportions.

Usage:
    python inspect_prior.py
    python inspect_prior.py --n-samples 10000 --pdf prior.pdf
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from te_dfe.prior import DFEBurstPrior, compute_binned_props
from te_dfe.config import N_PARAMS, PARAM_NAMES, PARAM_LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--ne-harmonic", type=int, default=500,
                         help="Ne for computing derived bin proportions")
    parser.add_argument("--pdf", default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    samples = np.stack([DFEBurstPrior.sample(rng) for _ in range(args.n_samples)])
    print(f"Sampled {args.n_samples} thetas, shape {samples.shape}\n")

    print("=" * 70)
    print(f"{'param':<20s} {'min':>10s} {'med':>10s} {'max':>10s} {'std':>10s}")
    print("=" * 70)
    for i, n in enumerate(PARAM_NAMES):
        v = samples[:, i]
        print(f"{n:<20s} {v.min():>10.3f} {np.median(v):>10.3f} "
              f"{v.max():>10.3f} {v.std():>10.3f}")

    # Derived binned proportions
    p_nn, p_w, p_str = compute_binned_props(
        samples[:, 1], samples[:, 2], args.ne_harmonic,
    )
    print()
    print("=" * 70)
    print(f"Derived binned proportions (Ne_h = {args.ne_harmonic}):")
    print("=" * 70)
    for name, arr in [("p_nn", p_nn), ("p_w", p_w), ("p_strong", p_str)]:
        print(f"{name:<10s} min={arr.min():.3f}  med={np.median(arr):.3f}  "
              f"max={arr.max():.3f}  mean={arr.mean():.3f}")

    if args.pdf is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            with PdfPages(args.pdf) as pdf:
                fig, axes = plt.subplots(2, 3, figsize=(13, 8))
                for i, (n, l) in enumerate(zip(PARAM_NAMES, PARAM_LABELS)):
                    ax = axes[i // 3, i % 3]
                    ax.hist(samples[:, i], bins=40, color="#1976d2", alpha=0.7)
                    ax.set_title(l, fontsize=9)
                    ax.grid(alpha=0.2)
                plt.tight_layout()
                pdf.savefig(fig, dpi=120)
                plt.close(fig)

                fig, axes = plt.subplots(1, 3, figsize=(13, 4))
                for ax, name, arr in zip(axes, ("p_nn", "p_w", "p_strong"), (p_nn, p_w, p_str)):
                    ax.hist(arr, bins=40, color="#388e3c", alpha=0.7)
                    ax.set_title(f"Derived {name} (Ne_h={args.ne_harmonic})", fontsize=10)
                    ax.set_xlim(0, 1)
                    ax.grid(alpha=0.2)
                plt.tight_layout()
                pdf.savefig(fig, dpi=120)
                plt.close(fig)
            print(f"\nPDF saved: {args.pdf}")
        except ImportError:
            print("matplotlib not available; skipping PDF.")


if __name__ == "__main__":
    main()
