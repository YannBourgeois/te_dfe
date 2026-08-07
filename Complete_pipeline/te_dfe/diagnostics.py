"""
diagnostics.py — Post-hoc diagnostic plots for the te_dfe_burst pipeline.

Produces plots in the style of popgen-npe's tutorial:

  posterior_calibration.pdf   PP plot of SBC ranks vs. uniform CDF
  posterior_expectation.pdf   true vs. posterior mean scatter (with 90% CI bars)
  posterior_concentration.pdf posterior std / prior std, per parameter
  recovery_results.csv        per-replicate true / mean / 5% / 95% per parameter

These are computed from the SBC and recovery data already produced by
05_benchmark.py. The diagnostic functions take the result dicts as input,
so they decouple from how the data was produced (sequential or parallelized).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Plot helpers (matplotlib only loaded when needed)
# ---------------------------------------------------------------------------

def _plt():
    """Lazy import — matplotlib is optional."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise RuntimeError(
            "matplotlib is required for diagnostics; install it via "
            "`uv pip install matplotlib`."
        ) from e


# ---------------------------------------------------------------------------
# 1. Calibration: PP-plot of SBC ranks
# ---------------------------------------------------------------------------

def plot_calibration(sbc_results: Dict, param_labels: List[str],
                      out_path: Path) -> None:
    """
    PP plot: empirical CDF of normalised SBC ranks vs. the uniform reference
    diagonal. Well-calibrated posteriors give curves on the diagonal.

    sbc_results : dict with key "ranks" of shape (n_replicates, n_params)
                  and key "n_post" giving total posterior samples per replicate.
    """
    plt = _plt()
    ranks  = np.asarray(sbc_results["ranks"], dtype=float)
    n_post = int(sbc_results["n_post"])
    n_reps, n_params = ranks.shape

    # Normalise ranks to [0, 1]
    norm_ranks = ranks / max(n_post, 1)

    n_cols = 4
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4*n_cols, 3.0*n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, lbl in enumerate(param_labels):
        ax = axes[i]
        sorted_ranks = np.sort(norm_ranks[:, i])
        ecdf_x = sorted_ranks
        ecdf_y = np.arange(1, n_reps + 1) / n_reps
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7, label="ideal")
        ax.plot(ecdf_x, ecdf_y, lw=1.5, label="empirical")
        # 95% confidence band for n_reps samples from uniform
        eps = 1.36 / np.sqrt(max(n_reps, 1))
        ax.fill_between([0, 1], [-eps, 1-eps], [eps, 1+eps],
                         color="gray", alpha=0.15)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("nominal CDF"); ax.set_ylabel("empirical CDF")
        ax.set_title(lbl, fontsize=9)
        ax.tick_params(labelsize=8)

    for j in range(n_params, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Posterior calibration (n_reps={n_reps})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Expectation: true vs. posterior mean scatter
# ---------------------------------------------------------------------------

def plot_expectation(recovery: Dict, param_labels: List[str],
                      out_path: Path) -> None:
    """
    True vs. posterior mean scatter, one panel per parameter, with the 90%
    posterior CI as vertical error bars and the y=x diagonal as reference.

    recovery : dict with "truths", "means", "lo90", "hi90"
                (each shape (n_replicates, n_params)).
    """
    plt = _plt()
    truths = np.asarray(recovery["truths"], dtype=float)
    means  = np.asarray(recovery["means"],  dtype=float)
    lo90   = np.asarray(recovery["lo90"],   dtype=float)
    hi90   = np.asarray(recovery["hi90"],   dtype=float)
    n_reps, n_params = truths.shape

    n_cols = 4
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4*n_cols, 3.0*n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, lbl in enumerate(param_labels):
        ax = axes[i]
        x = truths[:, i]
        y = means[:, i]
        yerr = np.array([y - lo90[:, i], hi90[:, i] - y])
        # Use only valid finite entries
        mask = np.isfinite(x) & np.isfinite(y)
        ax.errorbar(x[mask], y[mask], yerr=yerr[:, mask],
                    fmt="o", ms=4, capsize=2, alpha=0.7,
                    elinewidth=0.7, zorder=3)
        # Diagonal across observed range
        if mask.any():
            lo = float(min(x[mask].min(), y[mask].min()))
            hi = float(max(x[mask].max(), y[mask].max()))
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, zorder=1)
        # Pearson r
        if mask.sum() >= 3 and np.std(x[mask]) > 0 and np.std(y[mask]) > 0:
            r = float(np.corrcoef(x[mask], y[mask])[0, 1])
            ax.text(0.05, 0.95, f"r = {r:+.2f}", transform=ax.transAxes,
                    fontsize=8, va="top",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        ax.set_xlabel("true"); ax.set_ylabel("posterior mean")
        ax.set_title(lbl, fontsize=9)
        ax.tick_params(labelsize=8)

    for j in range(n_params, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Parameter recovery (n_reps={n_reps}, error bars = 90% CI)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Concentration: posterior std / prior std per parameter
# ---------------------------------------------------------------------------

def plot_concentration(recovery: Dict, prior_low, prior_high,
                        param_labels: List[str],
                        out_path: Path) -> None:
    """
    Bar plot showing how much the posterior shrinks vs. the prior.

    Concentration ratio = (posterior_std) / (prior_std).
      ratio = 1   : posterior matches prior (no learning)
      ratio < 1   : posterior is tighter (informative data)
      ratio < 0.5 : strong posterior shrinkage
    """
    plt = _plt()
    n_reps, n_params = np.asarray(recovery["means"]).shape

    # Posterior spread: average per-rep std across reps
    # Approximate posterior std from CI: ~ (hi - lo) / (2 * 1.645) for 90%
    lo90 = np.asarray(recovery["lo90"])
    hi90 = np.asarray(recovery["hi90"])
    posterior_std = (hi90 - lo90) / (2 * 1.645)
    mean_post_std = np.nanmean(posterior_std, axis=0)

    # Prior std for a uniform on [low, high] is (high - low) / sqrt(12)
    prior_low  = np.asarray(prior_low,  dtype=float)
    prior_high = np.asarray(prior_high, dtype=float)
    prior_std  = (prior_high - prior_low) / np.sqrt(12)

    ratio = mean_post_std / np.where(prior_std > 0, prior_std, np.nan)

    fig, ax = plt.subplots(figsize=(8, 0.4 * n_params + 1.5))
    y = np.arange(n_params)
    ax.barh(y, ratio, color="steelblue", alpha=0.8)
    ax.axvline(1.0, color="black", lw=0.8, ls="--", alpha=0.6, label="prior std")
    ax.axvline(0.5, color="green", lw=0.8, ls=":",  alpha=0.6, label="2× shrinkage")
    ax.set_yticks(y)
    ax.set_yticklabels(param_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("posterior std / prior std  (lower = more informative)")
    ax.set_title(f"Posterior concentration (n_reps={n_reps})")
    ax.set_xlim(0, max(1.2, ratio[np.isfinite(ratio)].max() if np.any(np.isfinite(ratio)) else 1.2))
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. CSV export of recovery results
# ---------------------------------------------------------------------------

def write_recovery_csv(recovery: Dict, param_names: List[str],
                        out_path: Path) -> None:
    """One row per (replicate, parameter) with true / mean / 5% / 95%."""
    truths = np.asarray(recovery["truths"])
    means  = np.asarray(recovery["means"])
    lo90   = np.asarray(recovery["lo90"])
    hi90   = np.asarray(recovery["hi90"])
    n_reps = truths.shape[0]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["replicate", "param_name", "param_index",
                         "true", "post_mean", "post_5pct", "post_95pct",
                         "in_90pct_CI"])
        for r in range(n_reps):
            for i, name in enumerate(param_names):
                t  = float(truths[r, i])
                m  = float(means[r, i])
                lo = float(lo90[r, i])
                hi = float(hi90[r, i])
                in_ci = (lo <= t <= hi)
                writer.writerow([r, name, i, t, m, lo, hi, int(in_ci)])
