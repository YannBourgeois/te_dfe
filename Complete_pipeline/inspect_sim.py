#!/usr/bin/env python3
"""
inspect_sim.py — Quick visual inspection of one simulation's tensor + theta.

Updated for pass-3 layout:
  - .npz contains keys: geno (int8), cov (float32), theta (float32)
  - geno shape:  (n_windows, n_haps, n_sites) with values {0=anc, 1=SNP, 2=TE, 3=pad}
  - cov shape:   (n_windows, n_covariates)
  - theta shape: (N_PARAMS,) = (log_lambda, p_nn, p_w, log_mean_s_overall,
                                log10_t_start, log_duration, log_multiplier)

Usage:
    python inspect_sim.py simulations/sim_000042.npz
    python inspect_sim.py simulations/sim_000042.npz --pdf out.pdf
    python inspect_sim.py simulations/sim_000042.npz --windows 0 5 10
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from te_dfe.config import (
    N_HAPS, N_SITES, N_TRAIN_WINDOWS, N_COVARIATES,
    N_TE_SITES, N_TE_FLANK_SITES, N_RANDOM_BLOCK_SITES,
    PARAM_NAMES,
)
from te_dfe.prior import DFEBurstPrior


# ---------------------------------------------------------------------------
# Legacy auto-detection: if file uses pre-pass-3 'obs' flat array, fall back.
# ---------------------------------------------------------------------------
def _channel_titles(n_ch):
    """Map continuous-channel count -> human-readable per-channel titles.
    v0/v1/v2 (legacy): 2 channels (anchor-relative phys + gen)
    v3:                4 channels (phys + gen + local TE density + local SNP density)
    """
    if n_ch == 2:
        return [
            "ch0: phys / FLANK_WINDOW_BP (anchor-relative)",
            "ch1: genetic / CM_NORMALIZER (anchor-relative)",
        ]
    if n_ch == 4:
        return [
            "ch0: phys / FLANK_WINDOW_BP",
            "ch1: genetic / CM_NORMALIZER",
            "ch2: log1p(local TE density) / log1p(NORM)",
            "ch3: log1p(local SNP density) / log1p(NORM)",
        ]
    return [f"ch{i}" for i in range(n_ch)]


def load_sim(path):
    """
    Load any of three formats:
      - newer:  geno + pos + cov + theta
      - new:    geno + cov + theta
      - legacy: obs + theta
    Returns (geno, pos, cov, theta) tuple. pos is None for older formats.
    """
    d = np.load(path)
    keys = set(d.files)

    if "geno" in keys and "pos" in keys and "cov" in keys:
        # newer
        geno  = np.asarray(d["geno"],  dtype=np.int8)
        pos   = np.asarray(d["pos"],   dtype=np.float32)
        cov   = np.asarray(d["cov"],   dtype=np.float32)
        theta = np.asarray(d["theta"], dtype=np.float32)
    elif "geno" in keys and "cov" in keys:
        # new (no positions)
        geno  = np.asarray(d["geno"],  dtype=np.int8)
        pos   = None
        cov   = np.asarray(d["cov"],   dtype=np.float32)
        theta = np.asarray(d["theta"], dtype=np.float32)
    elif "obs" in keys:
        # Legacy: obs is a flat float32 vector of one-hot (W, 4, H, S) + (W, C_cov)
        obs   = np.asarray(d["obs"],   dtype=np.float32)
        theta = np.asarray(d["theta"], dtype=np.float32)
        n_geno_4ch = N_TRAIN_WINDOWS * 4 * N_HAPS * N_SITES
        n_cov      = N_TRAIN_WINDOWS * N_COVARIATES
        assert obs.size == n_geno_4ch + n_cov, (
            f"Legacy obs size {obs.size} != expected {n_geno_4ch + n_cov}"
        )
        one_hot = obs[:n_geno_4ch].reshape(
            N_TRAIN_WINDOWS, 4, N_HAPS, N_SITES,
        )
        # Collapse to categorical via argmax
        # all-zero cells become 0 = ancestral, which is fine for visualization).
        geno = np.argmax(one_hot, axis=1).astype(np.int8)  # (W, H, S)
        pos  = None
        cov  = obs[n_geno_4ch:].reshape(N_TRAIN_WINDOWS, N_COVARIATES)
    else:
        raise ValueError(f"Unrecognized .npz keys: {keys}")

    return geno, pos, cov, theta


def print_theta_summary(theta):
    print("=" * 64)
    print("True parameters (external theta, what the network sees):")
    print("-" * 64)
    for n, v in zip(PARAM_NAMES, theta):
        print(f"  {n:<25} {float(v):+.4f}")

    lam, shape, ms, t_start, dur, mult = DFEBurstPrior.unpack(theta)
    # Derived bins
    try:
        from te_dfe.prior import compute_binned_props
        log_shape = float(theta[1])
        log_ms    = float(theta[2])
        # Use a default Ne_harmonic=500 for derived bins; if known
        # demography, can pass that instead. Reporting is informational.
        ne_h = 500
        p_nn, p_w, p_str = compute_binned_props(log_shape, log_ms, ne_h)
        derived_str = (
            f"  derived (Ne_h={ne_h}): p_nn={float(p_nn):.3f}  "
            f"p_w={float(p_w):.3f}  p_strong={float(p_str):.3f}\n"
        )
    except Exception:
        derived_str = ""
    print()
    print("Interpreted:")
    print(f"  lambda            = {lam:.3e}  per-bp per-gen insertion rate")
    print(f"  gamma_shape       = {shape:.3f}")
    print(f"  mean_s            = {ms:.4e}  (mean of pure-gamma DFE on |s|)")
    if derived_str:
        print(derived_str.rstrip())
    print(f"  burst t_start     = {t_start} gen ago")
    print(f"  burst duration    = {dur} gen")
    print(f"  burst multiplier  = {mult:.2f}x baseline")


def per_window_table(geno, cov):
    """Per-window summary: TE/SNP/padding counts in each region."""
    te_end     = N_TE_SITES
    flank_end  = te_end + N_TE_FLANK_SITES
    n_windows  = geno.shape[0]
    n_haps     = geno.shape[1]

    print()
    print("=" * 92)
    print(f"Genotype tensor: shape={tuple(geno.shape)} dtype={geno.dtype} "
          f"({geno.nbytes / 1e6:.1f} MB)")
    print(f"Regions: TE anchors [0..{te_end})  flanks [{te_end}..{flank_end})  "
          f"random [{flank_end}..{geno.shape[2]})")
    print("-" * 92)
    print(f"{'Win':<4}{'TE-real':>8}{'SNP-real':>10}{'PAD-tot':>9}"
          f"{'TE-anc-pad':>12}{'SNP-flank-pad':>15}{'RND-pad':>9}"
          f"{'gene_dens':>11}{'recomb':>9}{'te_dens':>9}{'snp_dens':>10}")
    print("-" * 92)
    for w in range(n_windows):
        gw = geno[w]
        te_region    = gw[:, :te_end]
        flank_region = gw[:, te_end:flank_end]
        rand_region  = gw[:, flank_end:]

        # Cells of each state, summed over haplotypes and sites
        n_te_in_te    = int((te_region    == 2).sum())
        n_pad_in_te   = int((te_region    == 3).sum())
        n_snp_in_flnk = int((flank_region == 1).sum())
        n_pad_in_flnk = int((flank_region == 3).sum())
        n_pad_in_rnd  = int((rand_region  == 3).sum())

        # Real-TE count: each TE row has all haplotypes filled (state 0 or 2)
        # → count columns with at least one cell == TE
        te_cols_real  = int(((te_region == 2).any(axis=0)).sum())
        # Real-SNP count in flanks: columns with at least one state-1 cell
        snp_cols_real = int(((flank_region == 1).any(axis=0)).sum())
        # Total padding across the window
        pad_total     = int((gw == 3).sum())

        gd, rm, td, sd = cov[w]
        print(f"{w:<4}{te_cols_real:>8}{snp_cols_real:>10}{pad_total:>9}"
              f"{n_pad_in_te:>12}{n_pad_in_flnk:>15}{n_pad_in_rnd:>9}"
              f"{gd:>11.4f}{rm:>9.3f}{td:>9.1f}{sd:>10.1f}")
    print("=" * 92)


def plot_windows(geno, pos, cov, theta, windows, pdf_path):
    """Render one heatmap per requested window. Saves a single multi-page PDF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.backends.backend_pdf import PdfPages

    # 4-color discrete colormap for the 4 categorical states
    state_colors = [
        "#ffffff",   # 0 ancestral → white
        "#1976d2",   # 1 SNP-derived → blue
        "#d32f2f",   # 2 TE-present → red
        "#bdbdbd",   # 3 padding → light gray
    ]
    cmap = ListedColormap(state_colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=4)

    te_end    = N_TE_SITES
    flank_end = te_end + N_TE_FLANK_SITES

    has_pos = pos is not None
    print(f"\nWriting {len(windows)} window plots to {pdf_path} "
          f"(position channels: {'yes' if has_pos else 'no'})...")
    with PdfPages(pdf_path) as pdf:
        for w in windows:
            if w < 0 or w >= geno.shape[0]:
                print(f"  skipping window {w}: out of range")
                continue

            gw = geno[w]  # (n_haps, n_sites)
            gd, rm, td, sd = cov[w]

            # One genotype panel + one panel per continuous channel.
            if has_pos:
                n_ch_pos = pos[w].shape[0]
                n_panels = 1 + n_ch_pos
            else:
                n_panels = 1

            fig, axes = plt.subplots(
                n_panels, 1, figsize=(14, 3.5 * n_panels),
                sharex=True,
            )
            if n_panels == 1:
                axes = [axes]

            # Panel 1: categorical state heatmap
            ax = axes[0]
            im = ax.imshow(gw, aspect="auto", cmap=cmap, norm=norm,
                           interpolation="nearest")
            ax.axvline(te_end - 0.5,    color="black", lw=0.6, alpha=0.7)
            ax.axvline(flank_end - 0.5, color="black", lw=0.6, alpha=0.7)
            ax.text(te_end / 2, -3, "TE anchors",
                    ha="center", va="bottom", fontsize=9)
            ax.text((te_end + flank_end) / 2, -3, "TE flanks",
                    ha="center", va="bottom", fontsize=9)
            ax.text((flank_end + geno.shape[2]) / 2, -3,
                    "Random anchors",
                    ha="center", va="bottom", fontsize=9)
            ax.set_ylabel("Haplotype")
            ax.set_title(
                f"Window {w}: gene_dens={gd:.3f}, recomb={rm:.3f}, "
                f"te_dens={td:.1f}/Mb, snp_dens={sd:.1f}/Mb"
            )
            cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], shrink=0.8)
            cbar.set_ticklabels(["anc (0)", "SNP (1)", "TE (2)", "pad (3)"])

            if has_pos:
                # pos shape: (W, n_continuous, n_sites)
                # Layout (v3):
                #   ch 0: phys position / FLANK_WINDOW_BP
                #   ch 1: genetic position / CM_NORMALIZER
                #   ch 2: log1p(local TE density) / log1p(NORM)
                #   ch 3: log1p(local SNP density) / log1p(NORM)
                pos_w = pos[w]              # (n_ch, n_sites)
                n_ch  = pos_w.shape[0]

                titles = _channel_titles(n_ch)
                colors = ["#1976d2", "#388e3c",
                          "#d32f2f", "#7b1fa2", "#f57c00"][:n_ch]

                for ch_i in range(n_ch):
                    ax_i = axes[1 + ch_i]
                    vals = pos_w[ch_i]
                    ax_i.plot(np.arange(len(vals)), vals,
                             color=colors[ch_i], lw=0.7)
                    ax_i.axhline(0.0, color="black", lw=0.4, alpha=0.5)
                    ax_i.axvline(te_end - 0.5,
                                color="black", lw=0.6, alpha=0.7)
                    ax_i.axvline(flank_end - 0.5,
                                color="black", lw=0.6, alpha=0.7)
                    ax_i.set_ylabel(titles[ch_i].split(":")[0]
                                     if ":" in titles[ch_i] else titles[ch_i])
                    ax_i.set_title(titles[ch_i])
                    ax_i.grid(alpha=0.2)
                axes[-1].set_xlabel("Site index")
            else:
                axes[0].set_xlabel("Site index")

            fig.tight_layout()
            pdf.savefig(fig, dpi=140)
            plt.close(fig)

        # Final page: theta summary
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        lam, shape, ms, t_start, dur, mult = DFEBurstPrior.unpack(theta)
        from te_dfe.prior import compute_binned_props
        p_nn, p_w, p_str = compute_binned_props(float(theta[1]), float(theta[2]), 500)
        text = (
            f"Simulation summary (v4 pure-gamma)\n"
            f"==================================\n"
            f"lambda      = {lam:.3e}\n"
            f"gamma_shape = {shape:.3f}\n"
            f"mean_s      = {ms:.4e}\n"
            f"derived (Ne_h=500):\n"
            f"  p_nn      = {float(p_nn):.3f}\n"
            f"  p_w       = {float(p_w):.3f}\n"
            f"  p_strong  = {float(p_str):.3f}\n"
            f"t_start     = {t_start} gen ago\n"
            f"duration    = {dur} gen\n"
            f"multiplier  = {mult:.2f}x\n"
        )
        ax.text(0.05, 0.95, text, family="monospace", fontsize=11,
                verticalalignment="top")
        pdf.savefig(fig, dpi=120)
        plt.close(fig)

    print(f"  → {pdf_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="Path to sim_NNNNNN.npz")
    p.add_argument("--pdf", default=None,
                   help="Output PDF for per-window heatmaps. If unset, "
                        "no plotting is done.")
    p.add_argument("--windows", nargs="+", type=int, default=None,
                   help="Window indices to plot. Default: first 5 windows.")
    return p.parse_args()


def main():
    args = parse_args()
    geno, pos, cov, theta = load_sim(args.path)

    print()
    print(f"File: {args.path}")
    print_theta_summary(theta)
    per_window_table(geno, cov)

    if pos is not None:
        print()
        print(f"Continuous channels present: pos shape = {pos.shape}")
        n_ch = pos.shape[1]
        titles = _channel_titles(n_ch)
        for ch_i in range(n_ch):
            vals = pos[:, ch_i]
            print(f"  {titles[ch_i]}: "
                  f"min={vals.min():+.3f} "
                  f"max={vals.max():+.3f} "
                  f"mean={vals.mean():+.3f} "
                  f"std={vals.std():.3f}")
    else:
        print("\nNo position channels in this file (older format).")

    if args.pdf is not None:
        windows = args.windows if args.windows is not None \
                                else list(range(min(5, geno.shape[0])))
        plot_windows(geno, pos, cov, theta, windows, args.pdf)


if __name__ == "__main__":
    main()
