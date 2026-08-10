#!/usr/bin/env python
"""
fit_dfe_moments_aged.py — fit the simple TE model to an AGE-PARTITIONED SFS.

The marginal SFS confounds a transposition burst with purifying selection: both
leave an excess of rare insertions. Partitioning the insertions by AGE separates
them, because the two act on different axes:

    a BURST     changes the total COUNT in one age window
    SELECTION   changes the SFS SHAPE within every window

moments.Demes.SFS supports exactly this via `mutation_time_windows`, which
returns one spectrum per window containing the mutations that ORIGINATED in it.

Two things follow, and the second is what makes this practical:

  * under infinite sites the spectrum is LINEAR in the influx, so if the influx
    during window w is theta * m_w, that window contributes m_w * S_w with S_w
    the unit-influx spectrum. The expensive moments integration therefore
    depends only on (gamma, demography) and NOT on the burst;
  * so the three burst parameters are a linear reweighting of precomputed rows.
    They are almost free to move, and the objective is smooth in them by
    construction, which also removes the zero-gradient failure that the
    marginal-SFS version had to work around.

OPTIMISATION
    log_lambda is theta0: a pure SCALE on the expected spectrum. For a Poisson
    likelihood its MLE is closed form (theta_te = obs.sum() / u.sum(), exactly
    moments/dadi's optimal_sfs_scaling), so it is PROFILED OUT by default rather
    than searched. That drops a dimension and removes the lambda<->multiplier
    scale trade-off that makes the surface rugged. The remaining parameters are
    fit with differential_evolution (gradient-free, robust to the kinks that
    burst-window overlaps introduce), polished with one L-BFGS-B step. Pass
    --optimizer multistart for the old random-restart behaviour, or
    --no-profile-lambda / --fix log_lambda=... to fit the scale explicitly.

WHERE THE AGES COME FROM
    validation : extract_te_ages.py reads them exactly out of a SLiM tree
                 sequence. Use this first, to check the method works before
                 trusting an estimator.
    real data  : an allele-age estimator on a phased VCF (GEVA, Relate,
                 tsinfer+tsdate). extract_te_ages.py --vcf writes the input.
                 Age ERROR will smear insertions across windows and blunt the
                 separation, so validate with --jitter-ages to see how much
                 error the method tolerates before committing to an estimator.

    LIMITATION: allele-age estimators assume outcrossing and recombination.
    Highly selfing species are out of scope for the real-data route.

    TE_DFE_MODEL=simple python fit_dfe_moments_aged.py \\
        --ages ages.tsv --seq-length 1e9 \\
        --age-edges 0,25,50,100,200,400,800
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from te_dfe.moments_dfe import (demes_graph_from_msmc, age_window_spectra,
                                age_smearing_matrix,
                                selfing_correction,
                                burst_influx_per_window, observed_age_sfs,
                                s_bin_midpoints, gamma_dfe_bins, poisson_ll,
                                dfe_categories, drift_barrier_fraction,
                                GAMMA_SCALE_DEFAULT)


def load_ages(path, te_only=True, max_te=None, seed=0, block_col=None):
    """Read the TSV from extract_te_ages.py.

    Returns (freqs, ages, n_haps, seq_length_bp, n_windows, block_ids). block_ids
    is the per-TE value of `block_col` (e.g. a genomic-window id) used for the
    BLOCK bootstrap, or None when the column is absent -- linked TEs must be
    resampled together, so this column is what makes the CI honest. The
    sequence length
    is taken from the '# seq_length_bp=' header that extract_te_ages.py writes,
    which is the SUM over a simulation's windows. A simulation is typically 20
    windows of 5 Mb, and getting the total wrong shifts log_lambda silently
    without disturbing anything else.
    """
    seq_len, n_win, n_comment = None, None, 0
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            n_comment += 1
            for tok in line.lstrip("#").split():
                if tok.startswith("seq_length_bp="):
                    seq_len = float(tok.split("=", 1)[1])
                elif tok.startswith("n_windows="):
                    n_win = int(tok.split("=", 1)[1])
    # skip_header past the '#' lines and take the NEXT line as the column names.
    # Using comments="#" with names=True instead makes genfromtxt strip the
    # comment line to nothing and then read that empty line as the names.
    d = np.genfromtxt(path, names=True, delimiter="\t",
                      skip_header=n_comment, comments=None)
    if te_only and "is_te" in d.dtype.names:
        d = d[d["is_te"] == 1]
    n_haps = int(np.median(d["n_haps"]))
    freqs = d["derived_count"] / n_haps
    ages = d["age_gen"]
    blocks = None
    if block_col:
        if block_col in d.dtype.names:
            blocks = np.asarray(d[block_col])
        else:
            print(f"  !! --boot-block-col {block_col!r} not in the ages file "
                  f"(columns: {', '.join(d.dtype.names)}); the block bootstrap "
                  f"will fall back to resampling individual TEs, which treats "
                  f"linked TEs as independent and leaves the CI optimistic",
                  file=sys.stderr)
    if max_te and len(freqs) > max_te:
        idx = np.random.default_rng(seed).choice(len(freqs), max_te, replace=False)
        freqs, ages = freqs[idx], ages[idx]
        if blocks is not None:
            blocks = blocks[idx]
        # NOTE subsampling TEs does NOT change the sequence they came from, so
        # seq_length is left alone; log_lambda then reports the rate implied by
        # the subsample, which is what you want when matching the network's budget.
    return freqs, ages, n_haps, seq_len, n_win, blocks


def _hessian(f, x, eps=None):
    x = np.asarray(x, float); n = x.size
    if eps is None:
        eps = np.maximum(1e-4 * np.maximum(np.abs(x), 1.0), 1e-6)
    eps = np.broadcast_to(np.asarray(eps, float), (n,)).copy()
    H = np.zeros((n, n)); f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                xp = x.copy(); xp[i] += eps[i]
                xm = x.copy(); xm[i] -= eps[i]
                H[i, i] = (f(xp) - 2*f0 + f(xm)) / eps[i]**2
            else:
                xpp=x.copy(); xpp[i]+=eps[i]; xpp[j]+=eps[j]
                xpm=x.copy(); xpm[i]+=eps[i]; xpm[j]-=eps[j]
                xmp=x.copy(); xmp[i]-=eps[i]; xmp[j]+=eps[j]
                xmm=x.copy(); xmm[i]-=eps[i]; xmm[j]-=eps[j]
                H[i,j]=H[j,i]=(f(xpp)-f(xpm)-f(xmp)+f(xmm))/(4*eps[i]*eps[j])
    return H

def _grad(f, x, eps=None):
    x=np.asarray(x,float); n=x.size
    if eps is None: eps=np.maximum(1e-5*np.maximum(np.abs(x),1.0),1e-7)
    eps=np.broadcast_to(np.asarray(eps,float),(n,)).copy()
    g=np.zeros(n)
    for i in range(n):
        xp=x.copy(); xp[i]+=eps[i]; xm=x.copy(); xm[i]-=eps[i]
        g[i]=(f(xp)-f(xm))/(2*eps[i])
    return g

def uncertainties(nll, x, boot_nlls=None):
    """SE of the free parameters at the optimum.

    nll        : negative log-likelihood of the fitted data
    boot_nlls  : list of nll functions built on bootstrap datasets. If given,
                 the GODAMBE sandwich cov = H^-1 J H^-1 is used, with J the
                 covariance of the score across bootstraps. This corrects the
                 COMPOSITE likelihood's overconfidence; the plain FIM (H^-1)
                 assumes every SFS cell is independent and understates the SE.
    """
    H = _hessian(nll, x)
    H = 0.5*(H+H.T)
    try: Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError: Hinv = np.linalg.pinv(H)
    if not boot_nlls:
        cov = Hinv; method="FIM"
    else:
        scores = np.array([-_grad(nb, x) for nb in boot_nlls])
        J = np.cov(scores, rowvar=False)
        if J.ndim==0: J=J.reshape(1,1)
        cov = Hinv @ J @ Hinv; method="Godambe"
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return se, cov, H, method


def write_fit_diagnostics(prefix, obs, exp, age_edges, ages=None, plot=True):
    """Observed vs fitted SFS per age window, Poisson residuals, age distribution.

    obs, exp : (n_windows, n_haps - 1) observed counts and fitted expectations.
    Writes <prefix>.sfs_fit.tsv always, and <prefix>.sfs_fit.png if matplotlib
    is available.

    Residuals are the dadi/moments Poisson residual (exp - obs)/sqrt(exp): under
    a correct model they are ~N(0,1), so |resid| > 2-3 marks a real misfit rather
    than a fluctuation. Sign convention matches moments: positive = model over-
    predicts that cell.
    """
    obs = np.asarray(obs, float)
    exp = np.asarray(exp, float)
    n_win, n_cls = obs.shape
    counts = np.arange(1, n_cls + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        resid = np.where(exp > 0, (exp - obs) / np.sqrt(np.maximum(exp, 1e-300)),
                         np.nan)

    lo_edges = list(age_edges)
    hi_edges = list(age_edges[1:]) + [np.inf]

    # ---- TSV (always) --------------------------------------------------
    tsv = f"{prefix}.sfs_fit.tsv"
    with open(tsv, "w") as fh:
        fh.write("window\tage_lo\tage_hi\tderived_count\tobs\texp\tpoisson_resid\n")
        for w in range(n_win):
            for j, i in enumerate(counts):
                fh.write(f"{w}\t{lo_edges[w]:g}\t{hi_edges[w]:g}\t{i}\t"
                         f"{obs[w, j]:.6g}\t{exp[w, j]:.6g}\t{resid[w, j]:.6g}\n")
        o_p, e_p = obs.sum(axis=0), exp.sum(axis=0)
        r_p = (e_p - o_p) / np.sqrt(np.maximum(e_p, 1e-300))
        for j, i in enumerate(counts):
            fh.write(f"pooled\tNA\tNA\t{i}\t{o_p[j]:.6g}\t{e_p[j]:.6g}\t{r_p[j]:.6g}\n")

    if not plot:
        return tsv, None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return tsv, None

    ncol = n_win
    fig = plt.figure(figsize=(2.3 * ncol + 1.5, 8.6))
    gs = fig.add_gridspec(3, ncol, height_ratios=[3.0, 1.5, 2.4],
                          hspace=0.42, wspace=0.32)

    vmax = np.nanmax(np.abs(resid)) if np.isfinite(resid).any() else 1.0
    vmax = max(vmax, 1.0)

    for w in range(n_win):
        hi = hi_edges[w]
        title = (f"[{lo_edges[w]:g}, {'inf' if np.isinf(hi) else f'{hi:g}'})")

        ax = fig.add_subplot(gs[0, w])
        # mask empty cells rather than flooring them: a floored zero drags the
        # log axis down by orders of magnitude and hides the real spectrum
        o_plot = np.where(obs[w] > 0, obs[w], np.nan)
        e_plot = np.where(exp[w] > 0, exp[w], np.nan)
        ax.step(counts, o_plot, where="mid", lw=1.2, color="k", label="observed")
        ax.plot(counts, e_plot, lw=1.6, color="#c0392b", label="fitted")
        n_zero = int((obs[w] == 0).sum())
        if n_zero:
            ax.plot(counts[obs[w] == 0],
                    np.full(n_zero, np.nanmin(o_plot) if np.isfinite(o_plot).any() else 1.0),
                    "v", ms=3, color="grey", clip_on=False)
        ax.set_yscale("log")
        ax.set_ylim(bottom=1.0)   # <1 expected = absent allele; don't show 1e-9
        ax.set_title(f"{title}\n{int(obs[w].sum())} TEs", fontsize=8)
        ax.tick_params(labelsize=7)
        if w == 0:
            ax.set_ylabel("count", fontsize=8)
            ax.legend(fontsize=6, frameon=False)

        axr = fig.add_subplot(gs[1, w])
        axr.axhline(0, color="grey", lw=0.8)
        axr.axhline(2, color="grey", lw=0.5, ls=":")
        axr.axhline(-2, color="grey", lw=0.5, ls=":")
        axr.bar(counts, np.nan_to_num(resid[w]), width=0.85, color="#2c6fbb")
        axr.set_ylim(-1.15 * vmax, 1.15 * vmax)
        axr.tick_params(labelsize=7)
        axr.set_xlabel("derived count", fontsize=7)
        if w == 0:
            axr.set_ylabel("Poisson\nresidual", fontsize=8)

    # ---- bottom row: age distribution + residual histogram --------------
    half = max(1, ncol // 2)
    axa = fig.add_subplot(gs[2, :half])
    xw = np.arange(n_win)
    axa.bar(xw - 0.2, obs.sum(axis=1), width=0.4, color="k", label="observed")
    axa.bar(xw + 0.2, exp.sum(axis=1), width=0.4, color="#c0392b", label="fitted")
    axa.set_xticks(xw)
    axa.set_xticklabels([f"{lo_edges[w]:g}" for w in range(n_win)], fontsize=7)
    axa.set_xlabel("age window start (gen ago)", fontsize=8)
    axa.set_ylabel("TEs in window", fontsize=8)
    axa.set_title("age distribution: this is where the burst lives", fontsize=8)
    axa.legend(fontsize=7, frameon=False)
    axa.tick_params(labelsize=7)

    axh = fig.add_subplot(gs[2, half:])
    r = resid[np.isfinite(resid)]
    if r.size:
        axh.hist(r, bins=30, color="#2c6fbb", alpha=0.85)
        axh.axvline(0, color="k", lw=0.8)
        axh.set_title(f"residuals: mean {r.mean():+.2f}, sd {r.std():.2f}",
                      fontsize=8)
    axh.set_xlabel("Poisson residual", fontsize=8)
    axh.tick_params(labelsize=7)

    fig.suptitle("observed vs fitted SFS by age window", fontsize=11)
    png = f"{prefix}.sfs_fit.png"
    fig.savefig(png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return tsv, png


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ages", required=True,
                    help="TSV from extract_te_ages.py (or any file with "
                         "derived_count, age_gen, n_haps columns)")
    ap.add_argument("--age-edges", default="0,25,50,100,200,400,800",
                    help="comma-separated window starts in generations ago. Must "
                         "begin at 0; the last window runs to infinity. Windows "
                         "narrower than the burst give the sharpest separation, "
                         "but each needs enough TEs to be informative.")
    ap.add_argument("--seq-length", type=float, default=None,
                    help="total sequence in bp behind these TEs; sets the scale "
                         "log_lambda is read against. Default: read from the "
                         "'# seq_length_bp=' header written by extract_te_ages.py, "
                         "which already sums over a simulation's windows.")
    ap.add_argument("--jitter-ages", type=float, default=0.0,
                    help="multiply each age by lognormal(0, this) before fitting, "
                         "to emulate the error of a real age ESTIMATOR. 0.3 is a "
                         "plausible order for GEVA/tsdate. Use it to find how much "
                         "age error the method tolerates.")
    ap.add_argument("--max-te", type=int, default=None)
    ap.add_argument("--s-convention", choices=["homozygote", "per-allele"],
                    default="homozygote")
    ap.add_argument("--gamma-scale", type=float, default=GAMMA_SCALE_DEFAULT)
    ap.add_argument("--dominance", type=float, default=0.5)
    ap.add_argument("--selfing", type=float, default=0.0,
                    help="selfing rate S in [0,1). At h=0.5 a selfing population "
                         "is EXACTLY a randomly mating one with Ne/(1+F), "
                         "F=S/(2-S), so this simply rescales Ne and needs no "
                         "selfing support in moments (whose SFS path has none). "
                         "At h<0.5 selfing also raises selection efficacy and the "
                         "equivalence becomes approximate.")
    ap.add_argument("--ne-for-gamma", type=float, default=None)
    ap.add_argument("--n-s-bins", type=int, default=25)
    ap.add_argument("--s-min", type=float, default=1e-8)
    ap.add_argument("--s-max", type=float, default=1.0)
    # optimiser controls
    ap.add_argument("--optimizer", choices=["de", "multistart"],
                    default="multistart",
                    help="'multistart' = random-restart L-BFGS-B with theta0 "
                         "profiled out (default, matches the marginal-SFS fit); "
                         "'de' = differential_evolution + one L-BFGS-B polish.")
    ap.add_argument("--de-maxiter", type=int, default=200)
    ap.add_argument("--de-popsize", type=int, default=15)
    ap.add_argument("--no-profile-lambda", action="store_true",
                    help="optimise log_lambda explicitly instead of profiling it "
                         "out via the closed-form Poisson scale.")
    ap.add_argument("--n-starts", type=int, default=8,
                    help="multistart: number of random starts. Under 'de' it only "
                         "sets the number of local basin-probe restarts.")
    ap.add_argument("--n-rounds", type=int, default=3,
                    help="rounds of local polish/basin sampling around the best.")
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fix", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--bound", action="append", default=[], metavar="NAME=LO,HI",
                    help="override a parameter's SEARCH range in model log-units, "
                         "decoupling the fit from the NPE training prior. "
                         "Repeatable, e.g. --bound log_mean_s=-6.9,-2.3")
    ap.add_argument("--dfe-coord", choices=["scale", "mean_s"], default="scale",
                    help="Gamma DFE search coordinate. 'scale' fits (shape, "
                         "scale=mean_s/shape): scale is the direction the age-SFS "
                         "identifies, shape the loose nuisance, so the ridge is "
                         "axis-aligned (default). 'mean_s' fits (shape, mean_s), "
                         "the old parameterisation. Report and summary stay in "
                         "(shape, mean_s) either way.")
    ap.add_argument("--uncerts", action="store_true",
                    help="report standard errors from the numerical Hessian "
                         "(FIM). Cheap, but assumes every SFS cell is "
                         "independent, so it UNDERSTATES the true error.")
    ap.add_argument("--uncerts-bootstrap", type=int, default=0, metavar="N",
                    help="Godambe standard errors from N bootstrap datasets "
                         "(100 is usually plenty). Corrects the composite "
                         "likelihood's overconfidence. Only the SCORE is "
                         "evaluated per replicate, never a refit, so this is "
                         "fast -- unlike a full bootstrap.")
    ap.add_argument("--boot-block-col", default=None, metavar="COLUMN",
                    help="resample whole blocks of this column (e.g. a genomic "
                         "window id) instead of individual TEs, so linked TEs "
                         "stay together. Strongly preferred when available.")
    ap.add_argument("--uncerts-bootstrap-refit", type=int, default=0, metavar="N",
                    help="REFIT the model on each of N bootstrap datasets and take "
                         "percentile CIs of the estimates. Slower than the Godambe "
                         "score bootstrap, but the intervals respect the bounds "
                         "and are asymmetric where the optimum sits against one, "
                         "so they never cross a bound the way a Godambe interval "
                         "can. Each refit warm-starts from the point estimate.")
    ap.add_argument("--boot-cold-starts", type=int, default=1, metavar="K",
                    help="extra random starts per refit replicate, on top of the "
                         "warm start, as a hedge against the warm start locking "
                         "onto the point estimate's basin (default 1).")
    ap.add_argument("--age-error-sigma", type=float, default=0.0, metavar="SIGMA",
                    help="model AGE MEASUREMENT ERROR: ages are assumed observed "
                         "as a_true * lognormal(0, SIGMA), and the per-window "
                         "spectra are mixed accordingly before the likelihood. "
                         "This is the correct way to handle an allele-age "
                         "estimator's error: without it, leakage across bins "
                         "attenuates the fit and biases selection toward "
                         "neutrality. Use the same value you would pass to "
                         "--jitter-ages. Windows narrower than exp(2*SIGMA) "
                         "exchange most of their TEs with neighbours.")
    ap.add_argument("--fit-age-error", action="store_true",
                    help="ESTIMATE the age-error sigma jointly with the rest "
                         "instead of fixing it at --age-error-sigma (which then "
                         "acts as the starting value). Use when you have no "
                         "external estimate of your age estimator's uncertainty. "
                         "NOTE sigma trades off against log_duration: both blur "
                         "the age structure, so fixing duration (--fix) is "
                         "strongly advised when fitting sigma.")
    ap.add_argument("--age-error-bounds", default="0.01,1.0", metavar="LO,HI",
                    help="search range for the fitted age-error sigma.")
    ap.add_argument("--neutral-ages", default=None, metavar="FILE",
                    help="TSV of NEUTRAL SNP ages (extract_te_ages.py "
                         "--snps-out). Their age-partitioned SFS is used as an "
                         "EMPIRICAL neutral baseline in place of the purely "
                         "demographic expectation: because the SNPs pass through "
                         "the same demography AND the same age estimator as the "
                         "TEs, this absorbs demographic misspecification and "
                         "age-estimation distortion together. Same logic as "
                         "calibrating a DFE on synonymous sites.")
    ap.add_argument("--neutral-calibration", choices=["full", "shape"],
                    default="full",
                    help="'full' calibrates BOTH the across-window age profile "
                         "and the within-window SFS shape (SNPs have no burst, "
                         "so they give the burst-free baseline the burst then "
                         "multiplies). 'shape' calibrates only the within-window "
                         "shape and leaves the age profile theoretical -- safer "
                         "if you suspect TEs and SNPs are dated differently.")
    ap.add_argument("--diagnostics", default=None, metavar="PREFIX",
                    help="write observed-vs-fitted SFS per age window: "
                         "<PREFIX>.sfs_fit.tsv (window, derived_count, obs, exp, "
                         "Poisson residual) and <PREFIX>.sfs_fit.png (spectra, "
                         "residuals, age distribution).")
    ap.add_argument("--true-theta", default=None,
                    help="display only; never used as a starting point")
    args = ap.parse_args()

    from te_dfe.config import PARAM_NAMES, MODEL_NAME
    import te_dfe.config_base as CB
    if MODEL_NAME != "simple":
        sys.exit(f"this fits the SIMPLE model; TE_DFE_MODEL is {MODEL_NAME!r}")
    names = list(PARAM_NAMES)

    # --- demography -------------------------------------------------------
    from te_dfe.demography import MSMCDemography
    from te_dfe.config_base import MSMC_PATH, MU, SIM_LENGTH
    baseline = (MSMCDemography.from_file(MSMC_PATH, mu=MU, sim_length=SIM_LENGTH)
                if Path(MSMC_PATH).exists() else None)
    graph = demes_graph_from_msmc(baseline, verbose=True)
    N_ref = float(graph["pop"].epochs[0].start_size)
    print(f"demography : N_ref={N_ref:.1f} from {'MSMC' if baseline else 'fallback'}")

    # --- selfing: rescale Ne, which is exact at h = 0.5 -------------------
    if args.selfing > 0:
        import demes
        ne_fac, sel_fac, F = selfing_correction(args.selfing, args.dominance)
        b = demes.Builder(time_units="generations")
        # keep end_size too: an exponential epoch would otherwise be silently
        # flattened to a constant one. Resolved demes epochs are already
        # ancient-first with decreasing end_time, so the order is preserved.
        b.add_deme("pop", epochs=[dict(start_size=e.start_size * ne_fac,
                                       end_size=e.end_size * ne_fac,
                                       end_time=e.end_time)
                                  for e in graph["pop"].epochs])
        graph = b.resolve()
        N_ref *= ne_fac
        print(f"selfing    : S={args.selfing:.2f}, F={F:.3f} -> Ne x {ne_fac:.3f} "
              f"(N_ref now {N_ref:.1f})")
        if abs(args.dominance - 0.5) > 1e-9:
            print(f"  !! h={args.dominance} != 0.5: selfing also scales selection "
                  f"efficacy by {sel_fac:.2f}x, which this rescale does NOT apply.\n"
                  f"     Multiply --gamma-scale by {sel_fac:.3f} to include it, and "
                  f"treat the result as approximate.", file=sys.stderr)
        print(f"  NOTE selfing also cuts effective recombination by "
              f"(1-S)/(1+S) = {(1-args.selfing)/(1+args.selfing):.3f}, which is what "
              f"degrades allele-AGE estimators on real data.")
    if args.ne_for_gamma is None:
        args.ne_for_gamma = N_ref

    # --- data -------------------------------------------------------------
    freqs, ages, n_haps, seq_hdr, n_win, block_ids = load_ages(
        args.ages, max_te=args.max_te, seed=args.seed,
        block_col=args.boot_block_col)
    if args.seq_length is None:
        if seq_hdr is None:
            sys.exit("--seq-length not given and the file has no "
                     "'# seq_length_bp=' header; regenerate with the current "
                     "extract_te_ages.py or pass it explicitly.")
        args.seq_length = seq_hdr
        print(f"seq length : {args.seq_length/1e6:.1f} Mb over {n_win} window(s) "
              f"(from the file header)")
    elif seq_hdr and abs(seq_hdr - args.seq_length) / seq_hdr > 0.01:
        print(f"  !! --seq-length {args.seq_length/1e6:.1f} Mb disagrees with the "
              f"{seq_hdr/1e6:.1f} Mb recorded in the file", file=sys.stderr)
    if args.jitter_ages > 0:
        rng0 = np.random.default_rng(args.seed)
        ages = ages * rng0.lognormal(0.0, args.jitter_ages, len(ages))
        print(f"ages       : jittered by lognormal(0, {args.jitter_ages}) to "
              f"emulate estimator error")
    age_edges = [float(x) for x in args.age_edges.split(",")]
    if age_edges[0] != 0:
        sys.exit("--age-edges must start at 0")
    obs = observed_age_sfs(freqs, ages, n_haps, age_edges)
    print(f"data       : {len(freqs)} TEs, {args.seq_length/1e6:.1f} Mb, "
          f"{len(age_edges)} age windows")
    for w, c in enumerate(obs.sum(axis=1)):
        hi = age_edges[w + 1] if w + 1 < len(age_edges) else "inf"
        print(f"   [{age_edges[w]:>6.0f}, {str(hi):>6}) gen ago : {int(c):>7} TEs")

    # --- empirical neutral baseline ---------------------------------------
    NEUTRAL_CAL = None          # (n_win, n_freq) multiplicative correction
    if args.neutral_ages:
        nf, na, n_haps_n, _, _, _ = load_ages(args.neutral_ages, te_only=False)
        if n_haps_n != n_haps:
            print(f"  !! neutral SNPs have n_haps={n_haps_n} but TEs have "
                  f"{n_haps}; the baseline cannot be compared cell-by-cell",
                  file=sys.stderr)
        if args.jitter_ages > 0:
            # the SNPs must go through the SAME age distortion as the TEs,
            # otherwise the calibration removes an error the TEs still carry
            rng_n = np.random.default_rng(args.seed + 7)
            na = na * rng_n.lognormal(0.0, args.jitter_ages, len(na))
        NEUT_OBS = observed_age_sfs(nf, na, n_haps, age_edges)
        print(f"neutral    : {len(nf)} SNPs, "
              f"{int(NEUT_OBS.sum())} segregating, used as the empirical "
              f"baseline ({args.neutral_calibration} calibration)")
        for w, c in enumerate(NEUT_OBS.sum(axis=1)):
            if c < 20:
                print(f"  !! only {int(c)} neutral SNPs in age window {w}; the "
                      f"baseline there will be noisy", file=sys.stderr)

    # --- spectrum grid: one set of age-window spectra per s bin ------------
    s_edges = np.concatenate([[0.0], np.geomspace(args.s_min, args.s_max,
                                                  args.n_s_bins)])
    s_mid = s_bin_midpoints(s_edges)
    gam_grid = args.gamma_scale * args.ne_for_gamma * s_mid
    # Fail early and legibly if this moments predates mutation_time_windows,
    # rather than deep inside the grid loop.
    import inspect
    import moments
    if "mutation_time_windows" not in inspect.signature(moments.Demes.SFS).parameters:
        sys.exit("this moments build has no `mutation_time_windows` in "
                 "moments.Demes.SFS, which the age-partitioned fit requires. "
                 "Upgrade moments-popgen, or use fit_dfe_moments.py (marginal "
                 "SFS) instead.")

    print(f"building {len(gam_grid)} x {len(age_edges)} age-window spectra "
          f"(done ONCE; the burst is a reweight of these) ...")
    SPEC = np.array([
        age_window_spectra(graph, n_haps, g, age_edges,
                           gamma_scale=args.gamma_scale,
                           ne_for_gamma=args.ne_for_gamma, h=args.dominance,
                           s_convention=args.s_convention)
        for g in gam_grid])                       # (n_s, n_win, n_haps-1)
    print(f"  grid shape {SPEC.shape}")

    # Correction factor: empirical neutral / theoretical neutral. SPEC[0] is the
    # lowest s bin, below the drift barrier and therefore neutral in every
    # respect the SFS can see. Applied AFTER the model's own age mixing, so the
    # distortion the SNPs carry (demography error, and the age estimator's own
    # smearing if sigma is not modelled separately) is absorbed here.
    if args.neutral_ages:
        theo = np.asarray(SPEC[0], float)                 # (n_win, n_freq)
        emp = np.asarray(NEUT_OBS, float)
        if args.neutral_calibration == "full":
            # normalise globally: the SNPs set the age profile too. They have no
            # burst, so this is the burst-FREE baseline that m_w multiplies.
            a = emp / max(emp.sum(), 1e-300)
            b = theo / max(theo.sum(), 1e-300)
        else:
            # per-window: only the within-window shape is calibrated
            a = emp / np.maximum(emp.sum(axis=1, keepdims=True), 1e-300)
            b = theo / np.maximum(theo.sum(axis=1, keepdims=True), 1e-300)
        with np.errstate(divide="ignore", invalid="ignore"):
            C = np.where(b > 0, a / b, 1.0)
        # guard sparse cells: a window with no SNPs must not zero the model, and
        # a huge ratio from one lucky SNP must not dominate
        C = np.clip(np.nan_to_num(C, nan=1.0, posinf=1.0), 0.05, 20.0)
        NEUTRAL_CAL = C
        print(f"  neutral calibration: correction in "
              f"[{C.min():.2f}, {C.max():.2f}], median {np.median(C):.2f}")

    # exactly what moments is fed: the s grid -> gamma_moments, and how many bins
    # hit the stability clip. This is the ground truth of the selection strengths
    # the fit can actually resolve.
    from te_dfe.moments_dfe import (GAMMA_MOMENTS_MAX as _GMM,
                                    S_CONVENTION_FACTOR as _SCF)
    _gm = np.clip(-2.0 * N_ref * (s_mid * _SCF[args.s_convention]), -_GMM, _GMM)
    _ncl = int(np.sum(np.abs(_gm) >= _GMM))
    print(f"  moments input: {len(s_mid)} s-bins, s in [{s_mid[0]:.2e}, "
          f"{s_mid[-1]:.2e}]  ->  gamma_moments in [{_gm.min():.1f}, "
          f"{_gm.max():.1f}]  (clip +/-{_GMM:g}; {_ncl} bin(s) clipped)")

    # --- what is fixed / profiled ----------------------------------------
    from te_dfe.config import (LOG_LAMBDA_RANGE, LOG_GAMMA_SHAPE_RANGE,
                               LOG_MEAN_S_RANGE, LOG10_T_START_RANGE,
                               LOG_DURATION_RANGE, LOG_MULTIPLIER_RANGE)
    bounds_all = [LOG_LAMBDA_RANGE, LOG_GAMMA_SHAPE_RANGE, LOG_MEAN_S_RANGE,
                  LOG10_T_START_RANGE, LOG_DURATION_RANGE, LOG_MULTIPLIER_RANGE]
    I_LAM = names.index("log_lambda")
    I_SHAPE = names.index("log_gamma_shape")
    I_MEANS = names.index("log_mean_s")
    I_T = names.index("log10_t_start")
    I_DUR = names.index("log_duration")
    I_MULT = names.index("log_multiplier")

    # --- what is fixed / profiled ----------------------------------------
    fixed = {}
    for spec_str in args.fix:
        k, _, v = spec_str.partition("=")
        if k not in names:
            sys.exit(f"--fix: unknown parameter {k!r}; choose from {names}")
        fixed[names.index(k)] = float(v)
    profile_lambda = (I_LAM not in fixed) and not args.no_profile_lambda
    # the optimiser searches everything not fixed and (if profiling) not lambda
    free = [i for i in range(len(names))
            if i not in fixed and not (profile_lambda and i == I_LAM)]

    # --- DFE search coordinate: (shape, scale) or (shape, mean_s) ----------
    # The age-SFS constrains the Gamma SCALE (= mean_s/shape), not shape and
    # mean_s separately -- they slide along a rate-conserving ridge. Fitting the
    # mean_s slot as log_scale puts that identified direction on its own axis
    # (shape becomes the explicit loose nuisance), which conditions the optimiser
    # and gives a clean profile CI. expand() maps it back via
    # log_mean_s = log_shape + log_scale, so model/report/summary stay in
    # (shape, mean_s). --dfe-coord mean_s restores the old parameterisation.
    scale_slot = (args.dfe_coord == "scale") and (I_MEANS in free)
    if scale_slot:
        a_m, b_m = LOG_MEAN_S_RANGE
        a_s, b_s = bounds_all[I_SHAPE]
        bounds_all[I_MEANS] = (a_m - b_s, b_m - a_s)   # log_scale box-implied range
    else:
        # mean_s mode only: cap the MEAN at s = 0.1 (saturation makes higher
        # means unidentifiable). Meaningless when fitting scale.
        MEAN_S_HI = 0.1
        lo_ms, hi_ms = bounds_all[I_MEANS][0], float(np.log(MEAN_S_HI))
        if not hi_ms > lo_ms:
            sys.exit(f"mean_s cap s={MEAN_S_HI} (log {hi_ms:.4f}) is below the "
                     f"prior lower bound {lo_ms:.4f}; adjust the cap.")
        bounds_all[I_MEANS] = (lo_ms, hi_ms)

    # explicit per-parameter search-range overrides (win over the above). In
    # scale mode, --bound log_mean_s sets the log_scale range.
    for spec_str in args.bound:
        k, _, rng_s = spec_str.partition("=")
        if k not in names:
            sys.exit(f"--bound: unknown parameter {k!r}; choose from {names}")
        lo, hi = (float(t) for t in rng_s.split(","))
        if not hi > lo:
            sys.exit(f"--bound {k}: need LO<HI, got {lo},{hi}")
        bounds_all[names.index(k)] = (lo, hi)

    bounds = [bounds_all[i] for i in free]
    if args.fit_age_error:
        bounds = bounds + [tuple(float(v) for v in args.age_error_bounds.split(","))]

    # the mean_s slot is searched as log_scale in scale mode; label it as such
    def _coord_name(i):
        return "log_scale" if (scale_slot and i == I_MEANS) else names[i]

    from te_dfe.moments_dfe import GAMMA_MOMENTS_MAX, S_CONVENTION_FACTOR
    s_sat = GAMMA_MOMENTS_MAX / (2.0 * N_ref * S_CONVENTION_FACTOR[args.s_convention])
    print(f"identifiab.: selection saturates near s ~ {s_sat:.3g} "
          f"(|gamma| clipped at {GAMMA_MOMENTS_MAX:g}, N_ref={N_ref:.0f}); "
          f"the age-SFS constrains the Gamma SCALE, not shape/mean_s separately.")
    if fixed:
        print("  fixed: " + ", ".join(f"{names[i]}={v:.4f}"
                                      for i, v in sorted(fixed.items())))
    if profile_lambda:
        print("  log_lambda profiled out (theta0 = obs.sum()/u.sum(), "
              "closed-form Poisson MLE)")
    print(f"  DFE search coordinate: "
          f"{'(shape, scale)' if scale_slot else '(shape, mean_s)'}")

    # the exact search box, fit log-units AND natural units, so it is obvious
    # what the optimiser searches and what a bound-pinned estimate means.
    print("  search bounds (fit log-units | natural):")
    for i in range(len(names)):
        lo, hi = bounds_all[i]
        cn = _coord_name(i)
        if i in fixed:
            tag = f"FIXED {fixed[i]:.4f}"
        elif profile_lambda and i == I_LAM:
            tag = "profiled"
        else:
            tag = "free"
        if cn == "log_scale":
            nat = f"scale in [{np.exp(lo):.2e}, {np.exp(hi):.2e}]"
        elif cn == "log_mean_s":
            nat = f"s in [{np.exp(lo):.2e}, {np.exp(hi):.2e}]"
        elif cn == "log_gamma_shape":
            nat = f"shape in [{np.exp(lo):.3g}, {np.exp(hi):.3g}]"
        elif cn == "log_multiplier":
            nat = f"mult in [{np.exp(lo):.3g}, {np.exp(hi):.3g}]"
        elif cn == "log_duration":
            nat = f"dur in [{np.exp(lo):.3g}, {np.exp(hi):.3g}] gen"
        elif cn == "log10_t_start":
            nat = f"t in [{10**lo:.3g}, {10**hi:.3g}] gen"
        elif cn == "log_lambda":
            nat = f"lambda in [{np.exp(lo):.2e}, {np.exp(hi):.2e}]"
        else:
            nat = ""
        print(f"    {cn:<18}[{lo:9.4f}, {hi:9.4f}]  {tag:<12}{nat}")

    def expand(xf):
        full = np.empty(len(names))
        for i, v in fixed.items():
            full[i] = v
        if profile_lambda:
            full[I_LAM] = 0.0            # placeholder; model_per_unit ignores it
        full[free] = xf
        if scale_slot:                  # fit coord log_scale -> model log_mean_s
            full[I_MEANS] = full[I_SHAPE] + full[I_MEANS]
        return full

    # --- objective --------------------------------------------------------
    t_max = float(getattr(CB, "SIM_LENGTH", 5000))

    # Age-measurement-error mixing. Built ONCE: it depends only on the age edges
    # and sigma, not on the parameters. u_observed = M.T @ u_true.
    FIT_SIGMA = bool(args.fit_age_error)
    # cache the mixing matrix by sigma: when sigma is FITTED the optimiser asks
    # for many nearby values, and rebuilding is the only per-call cost.
    _mix_cache = {}
    _mix_quad = 120 if FIT_SIGMA else 400

    def age_mix(sigma):
        if not (sigma and sigma > 0):
            return None
        key = round(float(sigma), 4)
        M = _mix_cache.get(key)
        if M is None:
            M = age_smearing_matrix(age_edges, key, t_max=t_max,
                                    n_quad=_mix_quad).T
            _mix_cache[key] = M
        return M

    if args.age_error_sigma and args.age_error_sigma > 0:
        _M0 = age_mix(args.age_error_sigma)
        stay = float(np.mean(np.diag(_M0)))
        print(f"age error  : sigma={args.age_error_sigma:g} "
              f"{'(starting value, FITTED)' if FIT_SIGMA else 'modelled'}; mean "
              f"P(TE stays in its own window) = {stay:.2f}")
        if stay < 0.5:
            print(f"  !! windows are narrow relative to the error kernel "
                  f"(one-sigma span x{np.exp(-args.age_error_sigma):.2f} to "
                  f"x{np.exp(args.age_error_sigma):.2f}); consider coarser "
                  f"--age-edges", file=sys.stderr)
    elif FIT_SIGMA:
        print("age error  : sigma FITTED (starting from the bound midpoint)")

    def model_per_unit(full, sigma=None):
        """Expected (age x freq) spectrum at unit influx; log_lambda is ignored."""
        if not np.isfinite(full).all():
            return None
        shape, mean_s = np.exp(full[I_SHAPE]), np.exp(full[I_MEANS])
        if not (shape > 0 and mean_s > 0):
            return None
        w_s, _ = gamma_dfe_bins(s_edges, shape, mean_s)          # (n_s,)
        m_w = burst_influx_per_window(age_edges, 10.0 ** full[I_T],
                                      np.exp(full[I_DUR]), np.exp(full[I_MULT]),
                                      t_max=t_max)               # (n_win,)
        u = m_w[:, None] * np.tensordot(w_s, SPEC, axes=(0, 0))  # (n_win, n_f)
        M = age_mix(args.age_error_sigma if sigma is None else sigma)
        if M is not None:
            u = M @ u                # mix ORIGIN windows into OBSERVED windows
        if NEUTRAL_CAL is not None:
            u = u * NEUTRAL_CAL      # empirical neutral baseline
        if not np.isfinite(u).all() or (u < 0).any() or u.sum() <= 0:
            return None
        return u

    def theta0_of(full, u):
        if profile_lambda:
            return obs.sum() / u.sum()                          # closed-form MLE
        return 4.0 * N_ref * np.exp(full[I_LAM]) * args.seq_length

    def nll_with_obs(full, obs_use, sigma=None):
        u = model_per_unit(full, sigma)
        if u is None:
            return 1e12
        theta_te = (obs_use.sum() / u.sum() if profile_lambda
                    else 4.0 * N_ref * np.exp(full[I_LAM]) * args.seq_length)
        if not np.isfinite(theta_te) or theta_te <= 0:
            return 1e12
        return -poisson_ll(obs_use.ravel(), (theta_te * u).ravel())

    def nll(full, sigma=None):
        return nll_with_obs(full, obs, sigma)

    # When sigma is fitted it is an EXTRA trailing coordinate of the optimiser
    # vector, kept outside `free` (which indexes model parameters only).
    n_model_free = len(free)

    def split_x(x):
        x = np.asarray(x, float)
        if FIT_SIGMA:
            return x[:n_model_free], float(x[n_model_free])
        return x, None

    def obj(x):
        xm, sg = split_x(x)
        return nll(expand(xm), sg)

    # --- fit --------------------------------------------------------------
    from scipy.optimize import minimize, differential_evolution
    rng = np.random.default_rng(args.seed)
    best, runs = None, []

    if not free:
        # everything fixed and/or profiled: nothing to search
        class _Result:
            pass
        best = _Result()
        best.x = np.array([])
        best.fun = obj(best.x)
        runs.append(best)
        print(f"  no free parameters; -logL = {best.fun:.2f}")
    elif args.optimizer == "de":
        de = differential_evolution(
            obj, bounds, seed=args.seed, maxiter=args.de_maxiter,
            popsize=args.de_popsize, tol=1e-7, mutation=(0.5, 1.0),
            recombination=0.7, init="latinhypercube", polish=False,
            updating="deferred")
        pol = minimize(obj, de.x, method="L-BFGS-B", bounds=bounds,
                       options=dict(maxiter=args.maxiter))
        best = pol if pol.fun <= de.fun else de
        runs.append(best)
        print(f"  DE: -logL = {de.fun:.2f}  ->  polished {best.fun:.2f}")
    else:
        for k in range(args.n_starts):
            st = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
            try:
                r = minimize(obj, st, method="L-BFGS-B", bounds=bounds,
                             options=dict(maxiter=args.maxiter))
            except Exception as e:
                print(f"  start {k}: failed ({type(e).__name__}: {e})",
                      file=sys.stderr)
                continue
            runs.append(r)
            if best is None or r.fun < best.fun:
                best = r
            print(f"  start {k+1}/{args.n_starts}: -logL = {r.fun:.2f}"
                  f"{'   <-- best' if best is r else ''}")
        if best is None:
            sys.exit("all starts failed")

    # local polish + basin sampling around the best (shared by both optimisers)
    if free:
        lo_b = np.array([lo for lo, _ in bounds])
        hi_b = np.array([hi for _, hi in bounds])
        span = hi_b - lo_b
        for rnd in range(1, args.n_rounds):
            for _ in range(max(2, args.n_starts // 2)):
                st = np.clip(best.x + rng.normal(0, 0.1 / rnd * span), lo_b, hi_b)
                r = minimize(obj, st, method="L-BFGS-B", bounds=bounds,
                             options=dict(maxiter=args.maxiter))
                runs.append(r)
                if r.fun < best.fun:
                    best = r
            print(f"  round {rnd+1}: -logL = {best.fun:.2f}")
    near = sum(1 for r in runs if r.fun < best.fun + 1.0)
    # wording kept ("starts within 1 log-unit") so existing summary parsers match
    print(f"  {near}/{len(runs)} starts within 1 log-unit of the best")

    # --- report -----------------------------------------------------------
    x_model, SIGMA_HAT = split_x(best.x)
    x = expand(x_model)
    if profile_lambda:
        u_best = model_per_unit(x, SIGMA_HAT)
        if u_best is not None and u_best.sum() > 0:
            theta_te = obs.sum() / u_best.sum()
            x[I_LAM] = np.log(theta_te / (4.0 * N_ref * args.seq_length))
        else:
            x[I_LAM] = np.nan
    truth = ([float(v) for v in args.true_theta.split(",")]
             if args.true_theta else None)
    print(f"\n  {'parameter':<18}{'estimate':>12}{'natural':>14}"
          + (f"{'truth':>12}" if truth else ""))
    for k, nm in enumerate(names):
        nat = (f"{10**x[k]:.0f} gen" if nm == "log10_t_start"
               else f"{np.exp(x[k]):.3e}")
        line = f"  {nm:<18}{x[k]:>12.4f}{nat:>14}"
        if truth:
            line += f"{truth[k]:>12.4f}"
        print(line)
    if FIT_SIGMA:
        print(f"  {'age_error_sigma':<18}{SIGMA_HAT:>12.4f}   "
              f"(x{np.exp(-SIGMA_HAT):.2f} to x{np.exp(SIGMA_HAT):.2f} at 1 sigma)")
    print(f"  {'-logL':<18}{best.fun:>12.2f}")

    # the Gamma in (shape, scale): scale is the identifiable direction; mean_s and
    # shape are only jointly constrained, sliding along a scale-conserving ridge.
    sh_r, ms_r = np.exp(x[I_SHAPE]), np.exp(x[I_MEANS])
    print(f"  {'Gamma scale':<18}{ms_r/sh_r:>12.4e}   "
          f"(rate {sh_r/ms_r:.4e}; the identifiable DFE quantity)")

    # flag any free parameter sitting on a search bound: its estimate is a bound,
    # not an optimum. Reported in the fit coordinate (log_scale in scale mode).
    for j, i in enumerate(free):
        lo, hi = bounds_all[i]
        xi = float(x_model[j]); w = hi - lo
        if w <= 0:
            continue
        cn = _coord_name(i)
        if xi - lo < 1e-3 * w:
            print(f"  !! {cn} pinned at LOWER bound {lo:.4f} "
                  f"(natural {np.exp(lo):.2e})", file=sys.stderr)
        elif hi - xi < 1e-3 * w:
            print(f"  !! {cn} pinned at UPPER bound {hi:.4f} "
                  f"(natural {np.exp(hi):.2e})", file=sys.stderr)

    sh, ms = np.exp(x[names.index("log_gamma_shape")]), np.exp(x[names.index("log_mean_s")])
    fr, s_e, g_e = dfe_categories(sh, ms, args.ne_for_gamma, args.gamma_scale)
    lab = ["effectively neutral", "weakly deleterious",
           "moderately deleterious", "strongly deleterious"]
    print(f"\n  DFE composition (gamma = {args.gamma_scale:g}*Ne*s):")
    for nm, f_ in zip(lab, fr):
        print(f"    {nm:<24}{f_:>8.1%}")
    print(f"    {'below drift barrier':<24}"
          f"{drift_barrier_fraction(sh, ms, args.ne_for_gamma):>8.1%}   "
          f"(unidentifiable by construction)")

    # --- uncertainties ----------------------------------------------------
    # shared bootstrap resampler: block (linked TEs together) or TE-level
    def _boot_index(rng_b, blocks, n_te):
        if blocks is not None:
            uniq = np.unique(blocks)
            pick = rng_b.choice(uniq, uniq.size, replace=True)
            return np.concatenate([np.flatnonzero(blocks == b) for b in pick])
        return rng_b.integers(0, n_te, n_te)

    def _resolve_blocks(n_te):
        if not args.boot_block_col or block_ids is None:
            return None
        blocks = np.asarray(block_ids)
        if blocks.size != n_te:
            print(f"  !! block column has {blocks.size} entries for {n_te} TEs; "
                  f"resampling individual TEs instead", file=sys.stderr)
            return None
        return blocks

    # REFIT bootstrap: re-optimise on each resampled dataset, percentile CIs.
    # Preferred when the optimum sits on a bound, where the Godambe interval is
    # unreliable (its quadratic sandwich draws a symmetric interval straight
    # through the bound). A refit interval piles up at the bound honestly.
    if args.uncerts_bootstrap_refit and free:
        from scipy.optimize import minimize as _min
        rng_b = np.random.default_rng(args.seed + 4242)
        n_te = len(freqs)
        blocks = _resolve_blocks(n_te)
        lo_b = np.array([lo for lo, _ in bounds])
        hi_b = np.array([hi for _, hi in bounds])
        print(f"\n  refit bootstrap: {args.uncerts_bootstrap_refit} replicate(s), "
              f"warm-started from the point estimate + {args.boot_cold_starts} "
              f"cold start(s) each "
              f"({'block' if blocks is not None else 'TE-level'} resampling)")
        boot_x, n_fail = [], 0
        for b in range(args.uncerts_bootstrap_refit):
            idx = _boot_index(rng_b, blocks, n_te)
            ob = observed_age_sfs(freqs[idx], ages[idx], n_haps, age_edges)
            def objb(xf, _o=ob):
                xm, sg = split_x(xf)
                return nll_with_obs(expand(xm), _o, sg)
            cand = []
            starts = [np.asarray(best.x, float)]
            for _ in range(max(0, args.boot_cold_starts)):
                starts.append(rng_b.uniform(lo_b, hi_b))
            for st in starts:
                try:
                    cand.append(_min(objb, st, method="L-BFGS-B", bounds=bounds,
                                     options=dict(maxiter=args.maxiter)))
                except Exception:
                    pass
            cand = [r for r in cand if np.isfinite(r.fun)]
            if cand:
                boot_x.append(min(cand, key=lambda r: r.fun).x)
            else:
                n_fail += 1
        boot_x = np.asarray(boot_x, float)
        if boot_x.shape[0] < 2:
            print("    !! too few successful refits for an interval", file=sys.stderr)
        else:
            se = boot_x.std(axis=0, ddof=1)
            lo = np.percentile(boot_x, 2.5, axis=0)
            hi = np.percentile(boot_x, 97.5, axis=0)
            if n_fail:
                print(f"    ({n_fail} replicate(s) had no finite refit and were "
                      f"skipped)", file=sys.stderr)
            print(f"\n  standard errors (bootstrap-refit; fit coordinates, "
                  f"95% CI = percentile):")
            for j, i in enumerate(free):
                cn = _coord_name(i)
                print(f"    {cn:<18}{float(best.x[j]):>10.4f} +/- {se[j]:<9.4f}"
                      f"[{lo[j]:>9.4f}, {hi[j]:>9.4f}]")
            # Dump the per-replicate (shape, mean_s) draws in MODEL coordinates,
            # so downstream CIs on DERIVED quantities (class proportions, which are
            # non-monotone in mean_s and correlated with shape) are percentiles of
            # the joint bootstrap rather than an endpoint propagation. Written next
            # to the report when --diagnostics gives a prefix.
            if args.diagnostics:
                # strip the fitted-sigma coordinate: expand() takes MODEL params only
                draws = np.array([expand(split_x(bx)[0]) for bx in boot_x])
                sig_draws = (np.array([split_x(bx)[1] for bx in boot_x])
                             if FIT_SIGMA else None)
                dpath = f"{args.diagnostics}.boot.tsv"
                with open(dpath, "w") as fh:
                    fh.write("log_gamma_shape\tlog_mean_s"
                             + ("\tage_error_sigma\n" if sig_draws is not None else "\n"))
                    for k_d, row in enumerate(draws):
                        extra = (f"\t{sig_draws[k_d]:.6g}"
                                 if sig_draws is not None else "")
                        fh.write(f"{row[I_SHAPE]:.6g}\t{row[I_MEANS]:.6g}{extra}\n")
                print(f"  bootstrap draws: {dpath} ({draws.shape[0]} replicates)")

    # Godambe (score bootstrap) / FIM: fast, but the interval is a local
    # quadratic and can cross a bound.
    elif (args.uncerts or args.uncerts_bootstrap) and free:
        boot_nlls = None
        if args.uncerts_bootstrap:
            rng_b = np.random.default_rng(args.seed + 991)
            n_te = len(freqs)
            blocks = _resolve_blocks(n_te)
            boot_nlls = []
            for _ in range(args.uncerts_bootstrap):
                idx = _boot_index(rng_b, blocks, n_te)
                ob = observed_age_sfs(freqs[idx], ages[idx], n_haps, age_edges)
                def _bnll(xf, _o=ob):
                    xm, sg = split_x(xf)
                    return nll_with_obs(expand(xm), _o, sg)
                boot_nlls.append(_bnll)
            print(f"\n  {args.uncerts_bootstrap} bootstrap replicate(s) for the "
                  f"Godambe correction "
                  f"({'block' if blocks is not None else 'TE-level'} resampling)")
        se, cov, H, method = uncertainties(obj, np.asarray(best.x, float),
                                           boot_nlls=boot_nlls)
        print(f"\n  standard errors ({method}; fit coordinates, 95% CI):")
        for j, i in enumerate(free):
            cn = _coord_name(i)
            xi = float(best.x[j])
            print(f"    {cn:<18}{xi:>10.4f} +/- {se[j]:<9.4f}"
                  f"[{xi-1.96*se[j]:>9.4f}, {xi+1.96*se[j]:>9.4f}]")
        if method == "FIM":
            print("    NOTE the FIM assumes independent SFS cells, so these are "
                  "LOWER bounds on the true error.\n"
                  "         Add --uncerts-bootstrap 100 for the Godambe "
                  "correction.")
        if not np.all(np.linalg.eigvals(H) > 0):
            print("    !! the Hessian is not positive definite: the optimum is on "
                  "a flat/ridged surface,\n       so these SEs are unreliable and "
                  "the parameter is effectively unidentified.", file=sys.stderr)

    # --- observed vs fitted SFS per age window ----------------------------
    if args.diagnostics:
        u_fit = model_per_unit(x, SIGMA_HAT)
        if u_fit is None:
            print("  !! could not build the fitted spectrum; no diagnostics written",
                  file=sys.stderr)
        else:
            exp_fit = theta0_of(x, u_fit) * u_fit
            tsv, png = write_fit_diagnostics(args.diagnostics, obs, exp_fit,
                                             age_edges, ages=ages)
            print(f"\n  diagnostics: {tsv}" + (f", {png}" if png else
                  " (no matplotlib; TSV only)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
