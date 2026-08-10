#!/usr/bin/env python
"""
fit_dfe_moments.py — fit the SIMPLE TE model by maximum likelihood on the SFS.

The diffusion counterpart of the NPE pipeline: no simulation, no network. The
expected TE site-frequency spectrum is computed from diffusion theory under the
MSMC demography, a transposition burst (a time-varying mutation influx) and a
Gamma DFE, then fitted to the observed spectrum by Poisson likelihood.

Independence from the NPE route is the point. The two share almost no
assumptions, so agreement between them is far stronger evidence than either
alone, and disagreement localises a problem.

    TE_DFE_MODEL=simple python fit_dfe_moments.py \\
        --dataset results/dataset            # or --shard test_000000.npz
        [--backend moments|analytic] [--no-burst] [--recomb-bins 3]
        [--true-theta "-20.7,-1.6,-4.6,3.0,4.0,2.3"]

WHAT IT CAN AND CANNOT DO
    fits : log_lambda, log_gamma_shape, log_mean_s,
           log10_t_start, log_duration, log_multiplier
    not  : alpha_r, gamma_r. Diffusion has no spatial structure, so recombination
           dependence is only reachable by stratifying TEs into recombination
           bins and fitting each separately (--recomb-bins), then reading the
           trend in mean_s across bins. That is a two-stage estimate, not joint
           inference, and it is why the context model does not port.

ASSUMPTION WORTH STATING
    Diffusion/PRF treats the mutation influx as EXOGENOUS. Classical TE dynamics
    are endogenous, with transposition proportional to current copy number, so a
    real burst is a branching process with feedback rather than a prescribed rate
    curve. This inherits the simulator's own choice of an externally imposed
    multiplier; if that assumption is ever relaxed on the simulation side, it has
    to be relaxed here too.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from te_dfe.moments_dfe import (epochs_from_msmc, burst_segments,
                                build_spectrum_grid, gamma_dfe_bins,
                                s_bin_midpoints, dfe_spectrum,
                                poisson_ll, observed_sfs, dfe_categories,
                                drift_barrier_fraction, check_gamma_convention,
                                CONVENTIONS, GAMMA_SCALE_DEFAULT)


# ---------------------------------------------------------------------------
def load_te_freqs(args, n_haps):
    """Return (te_freq, local_recomb, seq_len_bp or None).

    The sequence length is READ FROM THE DATA when the producer recorded it
    (01_simulate.py and 05a_benchmark_simulate.py both save `seq_len_bp`, and
    02_collate carries it into <base>.theta.pt). Guessing it from
    N_TRAIN_WINDOWS * WINDOW_SIZE is wrong whenever the shard was made with a
    different window count, and since seq_len_bp sets the scale that log_lambda
    is read against, a wrong value drives the DFE parameters onto their bounds
    to compensate.
    """
    from te_dfe.sumstats import FEATURE_NAMES
    fi, ri = FEATURE_NAMES.index("te_freq"), FEATURE_NAMES.index("local_recomb")
    if args.shard:
        d = np.load(args.shard)
        feats, mask = np.asarray(d["feats"]), np.asarray(d["te_mask"]) > 0
        L = float(d["seq_len_bp"]) if "seq_len_bp" in d.files else None
        if L is None and "n_windows_total" in d.files:
            import te_dfe.config_base as _CB
            L = float(d["n_windows_total"]) * float(getattr(_CB, "WINDOW_SIZE", 5e6))
        return feats[mask][:, fi], feats[mask][:, ri], L
    base = Path(args.dataset)
    feats = np.load(base.with_suffix(".feats.npy"), mmap_mode="r")
    mask = np.load(base.with_suffix(".te_mask.npy"), mmap_mode="r")
    lim = min(len(feats), args.max_sims) if args.max_sims else len(feats)
    L = None
    try:
        import torch
        meta = torch.load(base.with_suffix(".theta.pt"), map_location="cpu",
                          weights_only=False)
        sl = np.asarray(meta.get("seq_len_bp", []), float)
        sl = sl[:lim][sl[:lim] > 0]
        if sl.size:
            L = float(sl.sum())          # TEs are pooled, so lengths add
    except Exception:
        pass
    fr, rc = [], []
    for i in range(lim):
        m = np.asarray(mask[i]) > 0
        if m.any():
            f = np.asarray(feats[i])[m]
            fr.append(f[:, fi]); rc.append(f[:, ri])
    return np.concatenate(fr), np.concatenate(rc), L


def make_objective(obs, n_haps, N_ref, epochs, T_total, s_edges, args):
    """Negative log-likelihood as a function of the 6 free parameters.

    The spectra depend on the burst through the integration segments, so the
    grid has to be rebuilt whenever a burst parameter moves. When --no-burst is
    set (or the demography is flat and the analytic backend is used) the grid is
    computed once and the fit is far faster.
    """
    cache = {}
    # The s at which SPECTRA are evaluated is FIXED (geometric bin midpoints).
    # Using gamma_dfe_bins' conditional means here was a bug: those move with
    # shape and mean_s, so the weights ended up referring to different s values
    # than the spectra they multiplied, and the grid was rebuilt every step.
    s_mid = s_bin_midpoints(s_edges)
    gam_grid = args.gamma_scale * args.ne_for_gamma * s_mid
    bail = {"n": 0, "why": None}

    def spectra_for(t_start, duration, multiplier):
        # EXACT key. Rounding here was a silent killer: L-BFGS-B probes the
        # gradient with a step of ~1e-8, which never changed a key rounded to
        # 3 decimals, so the probe returned the IDENTICAL cached spectrum, the
        # likelihood was bit-identical, and the gradient of all three burst
        # parameters was exactly 0. L-BFGS-B then declared them converged
        # without moving them, so with a fixed --seed every shard reported the
        # same handful of starting values: a discrete "fitted" t_start.
        key = (float(t_start), float(duration), float(multiplier))
        if args.backend == "analytic":
            key = "flat"                       # analytic ignores burst/demography
        if key not in cache:
            segs = burst_segments(epochs, T_total, t_start, duration, multiplier)
            if len(cache) > 4:
                cache.clear()                   # keep memory bounded
            cache[key] = build_spectrum_grid(
                n_haps, gam_grid, segs, N_ref, backend=args.backend,
                gamma_scale=args.gamma_scale, ne_for_gamma=args.ne_for_gamma,
                h=args.dominance,
                s_convention=args.s_convention)
        return cache[key]

    def nll(theta):
        log_lambda, log_shape, log_mean_s, log10_t, log_dur, log_mult = theta
        shape, mean_s = np.exp(log_shape), np.exp(log_mean_s)
        t_start, duration = 10.0 ** log10_t, np.exp(log_dur)
        multiplier = np.exp(log_mult)
        if not (np.isfinite(theta).all() and shape > 0 and mean_s > 0):
            bail["n"] += 1; bail["why"] = "non-finite theta or shape/mean_s <= 0"
            return 1e12
        w, _ = gamma_dfe_bins(s_edges, shape, mean_s)
        spec = spectra_for(t_start, duration, multiplier)
        # theta_te carries log_lambda: total influx = 4 * Ne * lambda * L
        theta_te = 4.0 * N_ref * np.exp(log_lambda) * args.seq_length
        exp = dfe_spectrum(spec, w, theta_te)
        if not np.isfinite(exp).all() or (exp < 0).any():
            bail["n"] += 1
            bail["why"] = (f"expected SFS had "
                           f"{int((~np.isfinite(exp)).sum())} non-finite and "
                           f"{int((exp < 0).sum())} negative entries "
                           f"(spectrum grid max |gamma| = {gam_grid.max():.0f})")
            return 1e12
        if exp.sum() <= 0:
            bail["n"] += 1; bail["why"] = "expected SFS summed to zero"
            return 1e12
        return -poisson_ll(obs, exp)

    nll.bail = bail
    nll.gam_grid = gam_grid
    return nll


def fit(obs, n_haps, N_ref, epochs, T_total, s_edges, args, x0=None):
    from scipy.optimize import minimize
    from te_dfe.config import (LOG_LAMBDA_RANGE, LOG_GAMMA_SHAPE_RANGE,
                               LOG_MEAN_S_RANGE, LOG10_T_START_RANGE,
                               LOG_DURATION_RANGE, LOG_MULTIPLIER_RANGE)
    bounds = [LOG_LAMBDA_RANGE, LOG_GAMMA_SHAPE_RANGE, LOG_MEAN_S_RANGE,
              LOG10_T_START_RANGE, LOG_DURATION_RANGE, LOG_MULTIPLIER_RANGE]
    if args.no_burst:                       # pin the burst off: multiplier = 1
        bounds[5] = (0.0, 1e-9)
    full_nll = make_objective(obs, n_haps, N_ref, epochs, T_total, s_edges, args)

    # --fix: optimise only the free coordinates, splicing the fixed ones back in.
    fixed = getattr(args, "_fixed", {})            # {index: value}
    free = [i for i in range(len(bounds)) if i not in fixed]

    def expand(x_free):
        full = np.empty(len(bounds))
        for j, i in enumerate(free):
            full[i] = x_free[j]
        for i, v in fixed.items():
            full[i] = v
        return full

    def nll(x_free):
        return full_nll(expand(x_free))
    nll.bail, nll.gam_grid = full_nll.bail, full_nll.gam_grid
    bounds_free = [bounds[i] for i in free]
    # Evaluate once up front so a systematically broken objective is reported
    # instead of showing up as eight identical 1e12 "fits".
    probe = nll(np.array([np.mean(b) for b in bounds_free]))
    if probe >= 1e11:
        print(f"  !! the objective bails at every evaluation: {nll.bail['why']}",
              file=sys.stderr)
        print(f"     gamma grid spans [{nll.gam_grid.min():.2g}, "
              f"{nll.gam_grid.max():.0f}]; lower --s-max if the top is extreme.",
              file=sys.stderr)

    rng = np.random.default_rng(args.seed)
    best, runs = None, []
    for k in range(args.n_starts):
        start = (np.array([x0[i] for i in free]) if (x0 is not None and k == 0)
                 else np.array([rng.uniform(lo, hi) for lo, hi in bounds_free]))
        try:
            r = minimize(nll, start, method="L-BFGS-B", bounds=bounds_free,
                         options=dict(maxiter=args.maxiter, eps=args.fd_eps))
            if args.polish:
                # Nelder-Mead uses no gradients at all, so it cannot suffer the
                # zero-gradient failure above. Cheap as a final refinement.
                r = minimize(nll, r.x, method="Nelder-Mead", bounds=bounds_free,
                             options=dict(maxiter=args.maxiter, fatol=1e-4))
        except Exception as e:
            print(f"  start {k}: failed ({type(e).__name__}: {e})", file=sys.stderr)
            continue
        r.x = expand(r.x)                       # report in FULL parameter space
        if best is None or r.fun < best.fun:
            best = r
        seeded = (x0 is not None and k == 0)
        runs.append(r)
        print(f"  start {k + 1}/{args.n_starts}: -logL = {r.fun:.2f}"
              f"{'  [seeded at truth]' if seeded else ''}"
              f"{'   <-- best so far' if best is r else ''}")
    # Extra rounds: restart from the best point with a shrinking perturbation.
    # One round of random multistart often lands near, but not in, the optimum.
    for rnd in range(1, max(1, args.n_rounds)):
        span = np.array([hi - lo for lo, hi in bounds_free])
        lo_v = np.array([lo for lo, _ in bounds_free])
        hi_v = np.array([hi for _, hi in bounds_free])
        scale = 0.1 / rnd                       # 10%, 5%, 3.3% ... of the box
        improved = False
        for _ in range(max(2, args.n_starts // 2)):
            incumbent = np.array([best.x[i] for i in free])
            start = np.clip(incumbent + rng.normal(0.0, scale * span), lo_v, hi_v)
            try:
                r = minimize(nll, start, method="L-BFGS-B", bounds=bounds_free,
                             options=dict(maxiter=args.maxiter, eps=args.fd_eps))
            except Exception:
                continue
            r.x = expand(r.x)
            runs.append(r)
            if r.fun < best.fun - 1e-9:
                best, improved = r, True
        print(f"  round {rnd + 1}: -logL = {best.fun:.2f}"
              f"{'  (improved)' if improved else '  (no improvement)'}")

    # How many independent starts landed on the same optimum? A single start
    # reaching a good value proves nothing; several random starts agreeing is
    # what makes the estimate credible.
    if best is not None and runs:
        rand = [r.fun for i, r in enumerate(runs) if not (x0 is not None and i == 0)]
        near = sum(1 for f in rand if f < best.fun + 1.0)
        print(f"  {near}/{len(rand)} random start(s) within 1 log-unit of the best"
              f"{'  (converged)' if near >= 2 else '  -- weak agreement, raise --n-starts'}")
    return best, runs


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="collated dataset base (…/dataset)")
    src.add_argument("--shard", help="a single sim_*.npz / test_*.npz")
    ap.add_argument("--backend", choices=["moments", "analytic"], default="moments",
                    help="'moments' integrates the real demography AND the burst. "
                         "'analytic' is the closed-form constant-Ne equilibrium: "
                         "exact only for a flat history with no burst, and is "
                         "there to validate the moments path.")
    ap.add_argument("--no-burst", action="store_true",
                    help="pin the multiplier to 1; fits the DFE alone.")
    ap.add_argument("--recomb-bins", type=int, default=0,
                    help="fit separately within this many local_recomb bins. The "
                         "trend of mean_s across bins is the diffusion analogue "
                         "of alpha_r (two-stage, not joint).")
    ap.add_argument("--s-convention", choices=["homozygote", "per-allele"],
                    default="homozygote",
                    help="what the prior's s MEANS. 'homozygote' (default) is "
                         "SLiM's convention AA=1, Aa=1+h*s, aa=1+s, which is what "
                         "te_dfe_burst.slim uses via setSelectionCoeff. "
                         "'per-allele' is dadi/moments native, AA=1, Aa=1+2*h*s, "
                         "aa=1+2*s. They differ by a factor 2 in selection "
                         "strength; choosing wrong shifts mean_s by 2x with no "
                         "other symptom.")
    ap.add_argument("--gamma-scale", type=float, default=GAMMA_SCALE_DEFAULT,
                    help="reporting scale gamma = SCALE*Ne*s for the category "
                         "table. Does not change the fit.")
    ap.add_argument("--ne-for-gamma", type=float, default=None,
                    help="Ne in gamma = SCALE*Ne*s. Default: the MSMC ancestral size.")
    ap.add_argument("--dominance", type=float, default=0.5)
    ap.add_argument("--seq-length", type=float, default=None,
                    help="total sequence in bp behind the observed SFS; sets the "
                         "scale that log_lambda is read against. Default: read "
                         "from the data (seq_len_bp), falling back to "
                         "N_TRAIN_WINDOWS * WINDOW_SIZE with a warning.")
    ap.add_argument("--n-s-bins", type=int, default=40)
    ap.add_argument("--s-min", type=float, default=1e-8)
    ap.add_argument("--s-max", type=float, default=1.0,
                    help="top of the s grid. At Ne=500 the spectrum is already "
                         "saturated (all singletons) by s ~ 0.1, and moments goes "
                         "numerically unstable well before s=1, so raising this "
                         "buys nothing and can break the fit. The Gamma mass above "
                         "the top is renormalised away.")
    ap.add_argument("--n-starts", type=int, default=5)
    ap.add_argument("--n-rounds", type=int, default=1,
                    help="rounds of optimisation. Round 1 is random multistart; "
                         "each later round restarts from the best point so far "
                         "with a shrinking perturbation.")
    ap.add_argument("--fd-eps", type=float, default=1e-3,
                    help="finite-difference step for L-BFGS-B. scipy's default "
                         "1e-8 is far too small here: the burst parameters only "
                         "enter through a rebuilt spectrum grid, so a 1e-8 probe "
                         "can return a numerically identical likelihood and a "
                         "zero gradient. 1e-3 in log space moves t_start by a "
                         "few generations, which is a real change.")
    ap.add_argument("--polish", action="store_true",
                    help="follow each L-BFGS-B fit with derivative-free "
                         "Nelder-Mead, immune to the zero-gradient failure.")
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--max-sims", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--true-theta", default=None,
                    help="comma-separated truth, for validation on simulated data. "
                         "DISPLAY ONLY by default: it is printed next to the "
                         "estimates but never used as a starting point, so the fit "
                         "cannot be flattered by being handed the answer.")
    ap.add_argument("--fix", default=None,
                    help="hold parameters at fixed values and optimise only the "
                         "rest, e.g. --fix log_duration=2.3026 or "
                         "--fix log_duration=2.3026,log_multiplier=0. Useful when "
                         "a parameter is unidentifiable in the data at hand (an "
                         "old burst leaves almost no trace, so duration and "
                         "multiplier trade off freely); fixing one turns a ridge "
                         "into a point and makes the rest interpretable.")
    ap.add_argument("--gamma-edges", default="1,10,100",
                    help="category boundaries on gamma = GAMMA_SCALE * Ne * s, for "
                         "the reported proportions of new insertions.")
    ap.add_argument("--check-conventions", action="store_true",
                    help="MEASURE the scaling moments actually uses by matching it "
                         "against the closed-form PRF spectrum, print the "
                         "conventions note, and exit. Run this once per setup.")
    ap.add_argument("--seed-at-truth", action="store_true",
                    help="additionally use --true-theta as the first optimiser "
                         "start. Useful to check whether a random-start fit has "
                         "found the same optimum, but the result is then NOT an "
                         "independent recovery test.")
    args = ap.parse_args()

    from te_dfe.config import PARAM_NAMES, MODEL_NAME
    import te_dfe.config_base as CB
    if MODEL_NAME != "simple":
        sys.exit(f"This fits the SIMPLE model; TE_DFE_MODEL is {MODEL_NAME!r}. "
                 f"alpha_r / gamma_r have no diffusion counterpart -- use "
                 f"--recomb-bins on the simple model instead.")

    n_haps = getattr(CB, "N_HAPS", getattr(CB, "N_SAMPLES", 100))

    # --fix name=value,...  -> {param index: value}
    args._fixed = {}
    if args.fix:
        names = list(PARAM_NAMES)
        for item in args.fix.split(","):
            if "=" not in item:
                sys.exit(f"--fix expects name=value, got {item!r}")
            k, v = item.split("=", 1)
            k = k.strip()
            if k not in names:
                sys.exit(f"--fix: unknown parameter {k!r}. Choose from {names}")
            args._fixed[names.index(k)] = float(v)
        print("fixed      : " + ", ".join(f"{names[i]}={v:g}"
                                          for i, v in sorted(args._fixed.items())))

    # --- demography -------------------------------------------------------
    from te_dfe.demography import MSMCDemography
    from te_dfe.config_base import MSMC_PATH, MU, SIM_LENGTH
    baseline = (MSMCDemography.from_file(MSMC_PATH, mu=MU, sim_length=SIM_LENGTH)
                if Path(MSMC_PATH).exists() else None)
    N_ref, epochs, T_total = epochs_from_msmc(baseline)
    if args.ne_for_gamma is None:
        args.ne_for_gamma = N_ref
    flat = len({round(ne, 6) for ne, _ in epochs}) <= 1
    if args.check_conventions:
        print(CONVENTIONS)
        if not flat:
            print("  (the check assumes a constant-Ne history; yours is not flat, "
                  "so run it once against a flat MSMC file)\n")
        check_gamma_convention(n_haps, args.ne_for_gamma or N_ref,
                               gamma_scale=args.gamma_scale, h=args.dominance)
        return 0
    print(f"demography : N_ref={N_ref:.1f}, {len(epochs)} epoch(s), "
          f"T_total={T_total:.0f} gen{'  (constant Ne)' if flat else ''}")
    print(f"gamma      : {args.gamma_scale:g} * Ne({args.ne_for_gamma:.0f}) * s, h={args.dominance}")
    print(f"backend    : {args.backend}"
          + ("   [exact here: flat history]" if (args.backend == "analytic" and flat
                                                 and args.no_burst) else ""))

    # --- data -------------------------------------------------------------
    freqs, recomb, L_data = load_te_freqs(args, n_haps)
    if args.seq_length is None:
        if L_data:
            args.seq_length = L_data
            src = "read from the data (seq_len_bp)"
        else:
            args.seq_length = (getattr(CB, "N_TRAIN_WINDOWS", 20)
                               * getattr(CB, "WINDOW_SIZE", 5_000_000))
            src = "GUESSED from config"
            print("  !! the data did not record seq_len_bp, so the sequence length "
                  "is a guess.\n     It sets the scale log_lambda is read against; "
                  "if it is wrong the DFE\n     parameters run onto their bounds to "
                  "compensate. Pass --seq-length.", file=sys.stderr)
    else:
        src = "given on the command line"
        if L_data and abs(L_data - args.seq_length) / L_data > 0.01:
            print(f"  !! --seq-length {args.seq_length/1e6:.1f} Mb disagrees with the "
                  f"{L_data/1e6:.1f} Mb recorded in the data.", file=sys.stderr)
    print(f"data       : {len(freqs)} TEs, {args.seq_length/1e6:.1f} Mb  ({src})")

    s_edges = np.concatenate([[0.0], np.geomspace(args.s_min, args.s_max,
                                                  args.n_s_bins)])
    truth = ([float(v) for v in args.true_theta.split(",")]
             if args.true_theta else None)
    x0 = truth if (truth is not None and args.seed_at_truth) else None
    if truth is not None:
        print(f"truth      : shown for comparison"
              + (", AND used as start 0 (not an independent test)"
                 if args.seed_at_truth else ", not used as a starting point"))

    groups = [("all", np.ones(len(freqs), bool))]
    if args.recomb_bins > 1:
        q = np.quantile(recomb, np.linspace(0, 1, args.recomb_bins + 1))
        groups = [(f"recomb bin {k+1}/{args.recomb_bins}",
                   (recomb >= q[k]) & (recomb <= q[k + 1]))
                  for k in range(args.recomb_bins)]

    results = []
    for label, sel in groups:
        obs = observed_sfs(freqs[sel], n_haps)
        print(f"\n=== {label} : {int(obs.sum())} segregating TEs ===")
        if obs.sum() < 20:
            print("  too few TEs to fit"); continue
        best, _runs = fit(obs, n_haps, N_ref, epochs, T_total, s_edges, args, x0=x0)
        if best is None:
            print("  all starts failed"); continue
        print(f"\n  {'parameter':<18}{'estimate':>12}{'natural':>14}"
              + (f"{'truth':>12}" if truth else ""))
        nat = lambda nm, v: (f"{np.exp(v):.3e}" if nm.startswith("log_") and
                             nm != "log10_t_start" else
                             f"{10**v:.0f} gen" if nm == "log10_t_start" else f"{v:.3f}")
        for k, nm in enumerate(PARAM_NAMES):
            line = f"  {nm:<18}{best.x[k]:>12.4f}{nat(nm, best.x[k]):>14}"
            if truth:
                line += f"{truth[k]:>12.4f}"
            print(line)
        print(f"  {'-logL':<18}{best.fun:>12.2f}")

        # --- expected proportion of NEW insertions per selection class -------
        i_sh = list(PARAM_NAMES).index("log_gamma_shape")
        i_ms = list(PARAM_NAMES).index("log_mean_s")
        sh, ms = np.exp(best.x[i_sh]), np.exp(best.x[i_ms])
        fr, s_e, g_e = dfe_categories(sh, ms, args.ne_for_gamma, args.gamma_scale)
        lab = ["effectively neutral", "weakly deleterious",
               "moderately deleterious", "strongly deleterious"]
        print(f"\n  DFE composition   (gamma = {args.gamma_scale:g}*Ne*s, "
              f"Ne={args.ne_for_gamma:.0f})")
        for k, (f_, nm) in enumerate(zip(fr, lab)):
            rng_txt = (f"gamma < {g_e[1]:g}" if k == 0 else
                       f"gamma > {g_e[-2]:g}" if k == len(fr) - 1 else
                       f"{g_e[k]:g} < gamma < {g_e[k+1]:g}")
            s_txt = (f"s < {s_e[1]:.1e}" if k == 0 else
                     f"s > {s_e[-2]:.1e}" if k == len(fr) - 1 else
                     f"{s_e[k]:.1e} - {s_e[k+1]:.1e}")
            print(f"    {nm:<24}{f_:>8.1%}   ({rng_txt}, {s_txt})")
        db = drift_barrier_fraction(sh, ms, args.ne_for_gamma)
        print(f"    {'below drift barrier':<24}{db:>8.1%}   "
              f"(s < 1/(2Ne) = {1/(2*args.ne_for_gamma):.1e}; unidentifiable)")

        # --- proportion of NEW insertions per effective-selection class -----
        i_sh = list(PARAM_NAMES).index("log_gamma_shape")
        i_ms = list(PARAM_NAMES).index("log_mean_s")
        sh, ms = np.exp(best.x[i_sh]), np.exp(best.x[i_ms])
        g_edges = tuple(float(v) for v in args.gamma_edges.split(","))
        fr, s_e, g_full = dfe_categories(sh, ms, args.ne_for_gamma,
                                         args.gamma_scale, g_edges)
        labels = ([f"gamma < {g_full[1]:g}"]
                  + [f"{g_full[k]:g} - {g_full[k+1]:g}"
                     for k in range(1, len(g_full) - 2)]
                  + [f"gamma > {g_full[-2]:g}"])
        print(f"\n  DFE composition (gamma = {args.gamma_scale:g} * "
              f"Ne({args.ne_for_gamma:.0f}) * s)")
        print(f"  {'class':<18}{'s range':>22}{'fraction':>11}")
        for k, lab in enumerate(labels):
            lo = s_e[k]; hi = s_e[k + 1]
            rng_txt = (f"< {hi:.1e}" if k == 0 else
                       f"> {lo:.1e}" if k == len(labels) - 1 else
                       f"{lo:.1e} - {hi:.1e}")
            print(f"  {lab:<18}{rng_txt:>22}{fr[k]:>11.3f}")
        db = drift_barrier_fraction(sh, ms, args.ne_for_gamma)
        print(f"  {'below drift barrier':<18}{'< ' + f'{1/(2*args.ne_for_gamma):.1e}':>22}"
              f"{db:>11.3f}")
        print(f"    -> {db:.0%} of insertions are effectively neutral and carry no "
              f"information about s")
        results.append((label, best, fr, db))

    if len(results) > 1:
        print(f"\n=== mean_s across recombination bins "
              f"(the diffusion analogue of alpha_r) ===")
        i = list(PARAM_NAMES).index("log_mean_s")
        for label, b, fr, db in results:
            print(f"  {label:<22}mean_s = {np.exp(b.x[i]):.3e}   "
                  f"strongly selected = {fr[-1]:.3f}   neutral = {db:.3f}")
        print("  A monotone trend here is what alpha_r encodes directly in the "
              "context model. The strongly-selected fraction is often the more "
              "stable readout, since mean_s is dominated by the unidentifiable "
              "near-neutral mass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
