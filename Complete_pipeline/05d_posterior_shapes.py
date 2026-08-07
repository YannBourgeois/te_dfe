#!/usr/bin/env python3
"""
05d_posterior_shapes.py — Inspect posterior SHAPE from saved benchmark samples.

Reads the posterior_samples.npz written by 05b --save-samples and produces:

  1. corner_sim<N>.pdf      joint (pairwise) posterior for one chosen sim, with
                            true values marked — reveals parameter correlations
                            and degeneracies (e.g. t_start / duration / multiplier).
  2. marginal_shapes.pdf    per-parameter marginal posteriors overlaid across a
                            few sims, standardized to (sample - true), to see if
                            the bulk is sharp with long tails (non-Gaussian).
  3. ci_shrinkage.pdf       per-parameter CI half-width vs nominal level, averaged
                            over sims, against the Gaussian reference. A curve
                            that rises faster than Gaussian at high levels = long
                            tails: the inner CI (50%) is much tighter than the
                            outer (90/95%).
  4. regime_ci_width.pdf    per-parameter CI half-width vs TRUE value, per sim
                            (Min et al. Fig 3 style). Shows regime-dependent
                            identifiability: where the parameter is well-
                            constrained (narrow) vs structurally non-identifiable
                            (wide) across parameter space. Plus an optional 2-D
                            map (regime_2d_*.pdf) of one parameter's CI width over
                            a pair of true-value axes — e.g. the burst
                            multiplier's "narrow for recent+intense" gradient.
  5. concentration.pdf      posterior CI width / PRIOR width vs level (Min et al.
                            Fig 4 / S3). The honest way to show a wide-looking CI
                            still carries information: well below 1 = informative.

These mirror the diagnostics in Min et al. (NPE for population genetics): wide
CIs are often regime-dependent structural non-identifiability, not method
failure, and concentration-vs-prior shows the posterior is still informative.

The tail question directly: if 50% CI << 90% CI relative to Gaussian, then
RMSE / 90%-CI summaries UNDERSTATE how concentrated the posterior bulk is, and
reporting the 50% CI (or HPD) is fairer.

Usage:
    python 05d_posterior_shapes.py \
        --samples results/benchmark/posterior_samples.npz \
        --out     results/benchmark/shapes/ \
        --corner-sim 0 --n-overlay 6 \
        --regime-2d log10_t_start,log_multiplier,log_multiplier
"""
# Model selection before te_dfe import (config dispatches at import time).
try:
    from model_select import prescan_model
    prescan_model()
except Exception:
    pass

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", type=str,
                   default="results/benchmark/posterior_samples.npz")
    p.add_argument("--out", type=str, default="results/benchmark/shapes/")
    p.add_argument("--corner-sim", type=int, default=0,
                   help="Which sim index to draw the joint corner plot for.")
    p.add_argument("--n-overlay", type=int, default=6,
                   help="How many sims to overlay in the marginal-shape plot.")
    p.add_argument("--use-bins", action="store_true",
                   help="Make the corner plot over derived DFE bins instead of "
                        "the raw parameters.")
    p.add_argument("--ci-level", type=int, default=90,
                   help="Central CI level (percent) for the regime-resolved "
                        "CI-width-vs-true-value plot.")
    p.add_argument("--regime-2d", type=str, default=None,
                   help="Optional 2-D regime map as 'xname,yname,widthof' "
                        "(e.g. 'log10_t_start,log_multiplier,log_multiplier'). "
                        "Colours width_of's CI half-width over the (x,y) "
                        "true-value plane. Default tries the burst triple.")
    p.add_argument("--no-prior-bounds", action="store_true",
                   help="Skip loading DFEBurstPrior bounds for the "
                        "concentration plot (fall back to true-value spread).")
    return p.parse_args()


def load_prior_bounds(names):
    """
    Best-effort load of (lower, upper) prior bounds per parameter from
    DFEBurstPrior.to_sbi_prior(), for the concentration-ratio plot. Returns
    (lo, hi) numpy arrays aligned to `names`, or (None, None) if unavailable
    (e.g. running over DFE bins, or prior import fails).
    """
    try:
        import sys
        from pathlib import Path as _P
        sys.path.insert(0, str(_P(__file__).parent))
        from te_dfe.prior import DFEBurstPrior
        prior = DFEBurstPrior.to_sbi_prior()
        base = getattr(prior, "base_dist", prior)
        lo = np.asarray(base.low.cpu().numpy(), dtype=float)
        hi = np.asarray(base.high.cpu().numpy(), dtype=float)
        if len(lo) != len(names):
            return None, None
        return lo, hi
    except Exception as e:
        print(f"  (prior bounds unavailable: {e}; using true-value spread)")
        return None, None


def corner_plot(samples, true_vals, names, out_path, title):
    """Pairwise joint posterior for a single sim. samples: (n_post, D)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D = samples.shape[1]
    fig, axes = plt.subplots(D, D, figsize=(2.2 * D, 2.2 * D))
    for r in range(D):
        for c in range(D):
            ax = axes[r, c]
            if c > r:
                ax.axis("off")
                continue
            if r == c:
                ax.hist(samples[:, r], bins=40, color="steelblue", alpha=0.8)
                ax.axvline(true_vals[r], color="red", lw=1.5)
                ax.set_yticks([])
            else:
                ax.scatter(samples[:, c], samples[:, r], s=3, alpha=0.25,
                           color="steelblue", edgecolors="none")
                ax.axvline(true_vals[c], color="red", lw=1.0, alpha=0.7)
                ax.axhline(true_vals[r], color="red", lw=1.0, alpha=0.7)
            if c == 0 and r > 0:
                ax.set_ylabel(names[r], fontsize=8)
            if r == D - 1:
                ax.set_xlabel(names[c], fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def marginal_shapes(samples, true_t, names, out_path, n_overlay):
    """Per-parameter marginals, centered on truth, overlaid across sims."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_sims, n_post, D = samples.shape
    sel = np.linspace(0, n_sims - 1, min(n_overlay, n_sims)).astype(int)
    ncols = min(3, D)
    nrows = (D + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for j in range(D):
        ax = axes[j]
        for si in sel:
            centered = samples[si, :, j] - true_t[si, j]
            ax.hist(centered, bins=40, histtype="step", alpha=0.7, density=True)
        ax.axvline(0, color="red", lw=1.2, ls="--")
        ax.set_title(names[j], fontsize=10)
        ax.set_xlabel("sample - true")
        ax.set_yticks([])
    for j in range(D, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Marginal posterior shapes (centered on truth, "
                 f"{len(sel)} sims overlaid)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def ci_shrinkage(samples, names, out_path):
    """Mean CI half-width vs nominal level, vs Gaussian reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    n_sims, n_post, D = samples.shape
    levels = np.array([20, 50, 68, 80, 90, 95, 99])
    # mean half-width per (param, level)
    hw = np.zeros((D, len(levels)))
    for li, c in enumerate(levels):
        lo_p, hi_p = (100 - c) / 2.0, 100 - (100 - c) / 2.0
        w = (np.percentile(samples, hi_p, axis=1)
             - np.percentile(samples, lo_p, axis=1)) / 2.0   # (n_sims, D)
        hw[:, li] = w.mean(axis=0)

    z = norm.ppf(0.5 + levels / 200.0)   # Gaussian half-width in sd units

    ncols = min(3, D)
    nrows = (D + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for j in range(D):
        ax = axes[j]
        # Normalize both curves to their 90% value so SHAPE is comparable
        ref90 = hw[j, list(levels).index(90)]
        ax.plot(levels, hw[j] / max(ref90, 1e-12), "o-", label="observed",
                color="steelblue")
        zref90 = z[list(levels).index(90)]
        ax.plot(levels, z / zref90, "s--", label="Gaussian", color="gray",
                alpha=0.8)
        ax.set_title(names[j], fontsize=10)
        ax.set_xlabel("central CI level (%)")
        ax.set_ylabel("half-width / 90% half-width")
        ax.legend(fontsize=7)
    for j in range(D, len(axes)):
        axes[j].axis("off")
    fig.suptitle("CI shrinkage vs level — observed below Gaussian at low levels "
                 "=> sharp core, long tails", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def regime_ci_width(samples, true_t, names, out_path, ci_level=90):
    """
    Regime-resolved identifiability (à la Min et al. Fig 3): for each
    parameter, plot the posterior CI half-width of each sim against that sim's
    TRUE value, ranked. Reveals where the parameter is well-identified (narrow
    CI) vs structurally non-identifiable (wide CI) across parameter space.

    For the burst multiplier this exposes the "narrow for recent/intense,
    wide otherwise" gradient directly, rather than averaging it into one RMSE.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_sims, n_post, D = samples.shape
    lo_p, hi_p = (100 - ci_level) / 2.0, 100 - (100 - ci_level) / 2.0
    ncols = min(3, D)
    nrows = (D + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for j in range(D):
        ax = axes[j]
        hw = (np.percentile(samples[:, :, j], hi_p, axis=1)
              - np.percentile(samples[:, :, j], lo_p, axis=1)) / 2.0  # (n_sims,)
        tv = true_t[:, j]
        order = np.argsort(tv)
        # colour by CI width so the identifiable regions pop out
        sc = ax.scatter(tv[order], hw[order], c=hw[order], cmap="viridis_r",
                        s=16, edgecolors="none")
        ax.set_xlabel(f"true {names[j]}")
        ax.set_ylabel(f"{ci_level}% CI half-width")
        ax.set_title(names[j], fontsize=10)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    for j in range(D, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Regime-resolved identifiability: {ci_level}% CI width vs "
                 "true value\n(low = well-identified region, high = "
                 "structurally non-identifiable region)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def regime_ci_width_2d(samples, true_t, names, out_path, x_name, y_name,
                       width_of):
    """
    2-D regime map: CI half-width of parameter `width_of` shown as colour over
    the (x_name, y_name) true-value plane. This is the figure that makes the
    multiplier's "narrow for recent + intense" gradient explicit — e.g.
    x=log10_t_start, y=log_multiplier, colour = multiplier CI width.
    Skipped silently if any of the three names aren't present.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        jx, jy, jw = names.index(x_name), names.index(y_name), names.index(width_of)
    except ValueError:
        return False

    hw = (np.percentile(samples[:, :, jw], 95, axis=1)
          - np.percentile(samples[:, :, jw], 5, axis=1)) / 2.0
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sc = ax.scatter(true_t[:, jx], true_t[:, jy], c=hw, cmap="viridis_r",
                    s=40, edgecolors="k", linewidths=0.3)
    ax.set_xlabel(f"true {x_name}")
    ax.set_ylabel(f"true {y_name}")
    ax.set_title(f"{width_of} 90% CI half-width across ({x_name}, {y_name})")
    plt.colorbar(sc, ax=ax, label=f"{width_of} CI half-width")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def concentration_ratio(samples, names, out_path, prior_lo=None, prior_hi=None):
    """
    Posterior-to-prior CI-width ratio vs confidence level (Min et al. Fig 4
    right / Fig S3). Values well below 1 mean the posterior is much tighter
    than the prior — the honest way to show a "wide-looking" CI still carries
    real information. Prior widths are taken from BoxUniform bounds if given,
    else estimated from the spread of true values (a lower bound on prior
    width, so ratios are conservative/over-large).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_sims, n_post, D = samples.shape
    levels = np.array([20, 50, 68, 80, 90, 95, 99])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for j in range(D):
        if prior_lo is not None and prior_hi is not None:
            prior_w = float(prior_hi[j] - prior_lo[j])
        else:
            prior_w = float(np.ptp(samples[:, :, j]))  # fallback
        ratios = []
        for c in levels:
            lo_p, hi_p = (100 - c) / 2.0, 100 - (100 - c) / 2.0
            w = (np.percentile(samples[:, :, j], hi_p, axis=1)
                 - np.percentile(samples[:, :, j], lo_p, axis=1)).mean()
            ratios.append(w / max(prior_w, 1e-12))
        ax.plot(levels, ratios, "o-", label=names[j], markersize=4)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("central CI level (%)")
    ax.set_ylabel("posterior CI width / prior width")
    ax.set_title("Posterior concentration vs prior\n(well below 1 = "
                 "informative, even where absolute CI looks wide)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    d = np.load(args.samples, allow_pickle=False)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.use_bins and "bin_samples" in d.files:
        samples = d["bin_samples"]
        true_t  = d["true_bins"]
        names   = [str(x) for x in d["bin_names"]]
        tag = "bins"
    else:
        samples = d["samples"]        # (n_sims, n_post, D)
        true_t  = d["true_theta"]
        names   = [str(x) for x in d["param_names"]]
        tag = "params"

    n_sims, n_post, D = samples.shape
    print(f"Loaded {n_sims} sims x {n_post} samples x {D} {tag}")

    ci = args.corner_sim
    if not (0 <= ci < n_sims):
        print(f"corner-sim {ci} out of range [0,{n_sims-1}], using 0")
        ci = 0
    corner_plot(samples[ci], true_t[ci], names,
                out_dir / f"corner_sim{ci}_{tag}.pdf",
                f"Joint posterior — sim {ci} ({tag}); red = truth")
    print(f"Wrote corner plot -> {out_dir}/corner_sim{ci}_{tag}.pdf")

    marginal_shapes(samples, true_t, names,
                    out_dir / f"marginal_shapes_{tag}.pdf", args.n_overlay)
    print(f"Wrote marginal shapes -> {out_dir}/marginal_shapes_{tag}.pdf")

    ci_shrinkage(samples, names, out_dir / f"ci_shrinkage_{tag}.pdf")
    print(f"Wrote CI-shrinkage plot -> {out_dir}/ci_shrinkage_{tag}.pdf")

    # Regime-resolved identifiability: CI width vs true value (Min et al. Fig 3).
    regime_ci_width(samples, true_t, names,
                    out_dir / f"regime_ci_width_{tag}.pdf",
                    ci_level=args.ci_level)
    print(f"Wrote regime CI-width plot -> {out_dir}/regime_ci_width_{tag}.pdf")

    # Optional 2-D regime map (default: the burst triple, if present).
    if tag == "params":
        if args.regime_2d:
            try:
                xn, yn, wn = [s.strip() for s in args.regime_2d.split(",")]
            except ValueError:
                print(f"  (--regime-2d needs 'xname,yname,widthof'; got "
                      f"{args.regime_2d!r}; skipping)")
                xn = yn = wn = None
        else:
            xn, yn, wn = "log10_t_start", "log_multiplier", "log_multiplier"
        if xn is not None:
            ok = regime_ci_width_2d(samples, true_t, names,
                                    out_dir / f"regime_2d_{wn}.pdf",
                                    xn, yn, wn)
            if ok:
                print(f"Wrote 2-D regime map -> {out_dir}/regime_2d_{wn}.pdf")
            else:
                print(f"  (2-D regime map skipped: one of "
                      f"{xn},{yn},{wn} not in params)")

    # Concentration ratio vs prior (Min et al. Fig 4 / S3).
    if tag == "params" and not args.no_prior_bounds:
        prior_lo, prior_hi = load_prior_bounds(names)
    else:
        prior_lo, prior_hi = None, None
    concentration_ratio(samples, names,
                        out_dir / f"concentration_{tag}.pdf",
                        prior_lo=prior_lo, prior_hi=prior_hi)
    print(f"Wrote concentration plot -> {out_dir}/concentration_{tag}.pdf")

    # Numeric tail summary to stdout
    from scipy.stats import norm
    print("\nTail-shape numeric summary (half-width ratios, mean over sims):")
    print(f"  {'param':<18}{'50/90':>9}{'gauss':>8}{'80/90':>9}{'gauss':>8}")
    for j in range(D):
        w50 = (np.percentile(samples[:, :, j], 75, axis=1)
               - np.percentile(samples[:, :, j], 25, axis=1)).mean() / 2
        w80 = (np.percentile(samples[:, :, j], 90, axis=1)
               - np.percentile(samples[:, :, j], 10, axis=1)).mean() / 2
        w90 = (np.percentile(samples[:, :, j], 95, axis=1)
               - np.percentile(samples[:, :, j], 5, axis=1)).mean() / 2
        g50 = norm.ppf(0.75) / norm.ppf(0.95)
        g80 = norm.ppf(0.90) / norm.ppf(0.95)
        print(f"  {names[j]:<18}{w50/max(w90,1e-12):>9.3f}{g50:>8.3f}"
              f"{w80/max(w90,1e-12):>9.3f}{g80:>8.3f}")
    print("\nIf observed 50/90 < Gaussian 50/90, the posterior bulk is sharper "
          "than its tails suggest (long-tailed); report 50% CI alongside 90%.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
