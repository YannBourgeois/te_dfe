#!/usr/bin/env python
"""
fit_dfe_moments_context.py — diffusion fit of the CONTEXT model on a TE SFS
partitioned jointly by AGE and by LOCAL RECOMBINATION.

The simple model asks how strong selection is on average. The context model asks
how selection and insertion VARY ALONG THE GENOME with local recombination:

    s_local(p)      = exp(log_s_global + alpha_r * phi_r(p))
    lambda_local(p) = lambda_global * exp(gamma_r * phi_r(p))

with phi_r the normalised log local recombination rate. Both slopes are
identifiable from a 2-D partition of the TEs:

  * gamma_r moves the COUNT of TEs between recombination bins (more insertions
    where recombination is high, if positive);
  * alpha_r moves the SFS SHAPE between recombination bins (more skew toward
    rare, if selection is stronger there);
  * the burst moves the COUNT between AGE bins;
  * the DFE / s_global sets the shape everywhere.

so binning on both axes at once separates all four. This is the same linearity
trick as the age-only fit: the expected spectrum is linear in the per-cell
influx, so the expensive moments integration depends only on (gamma, demography)
and is done ONCE on a grid of s; each recombination bin then just needs the
spectrum at ITS OWN s_local, obtained by interpolating that grid in log s. No
moments call is ever made inside the optimiser.

    TE_DFE_MODEL=context python fit_dfe_moments_context.py \\
        --ages ages.tsv --age-edges 0,25,50,100,200,400 --recomb-bins 4
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fit_dfe_moments_aged import uncertainties
from te_dfe.moments_dfe import (demes_graph_from_msmc, age_window_spectra,
                                age_smearing_matrix, selfing_correction,
                                burst_influx_per_window, s_bin_midpoints,
                                poisson_ll, GAMMA_SCALE_DEFAULT)


def load_ages_recomb(path, max_te=None, seed=0, block_col=None):
    """Read the ages TSV, requiring a local-recombination column.

    Returns (freqs, ages, recomb, n_haps, seq_len_bp, n_windows, blocks).
    `blocks` is the per-TE linkage-block id (e.g. the genomic window) used by the
    block bootstrap, or None when the column is absent.
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
    d = np.genfromtxt(path, names=True, delimiter="\t",
                      skip_header=n_comment, comments=None)
    if "is_te" in d.dtype.names:
        d = d[d["is_te"] == 1]
    rcol = next((c for c in ("local_recomb", "recomb", "recomb_rate")
                 if c in d.dtype.names), None)
    if rcol is None:
        sys.exit("the ages file has no local recombination column "
                 "(looked for local_recomb / recomb / recomb_rate). The context "
                 "model needs it; re-run extract_te_ages.py with recombination.")
    n_haps = int(np.median(d["n_haps"]))
    freqs = d["derived_count"] / n_haps
    ages = d["age_gen"]
    rec = d[rcol]
    blocks = None
    if block_col:
        if block_col in d.dtype.names:
            blocks = np.asarray(d[block_col])
        else:
            print(f"  !! --boot-block-col {block_col!r} not in the ages file "
                  f"(columns: {', '.join(d.dtype.names)}); falling back to "
                  f"resampling individual TEs", file=sys.stderr)
    if max_te and len(freqs) > max_te:
        idx = np.random.default_rng(seed).choice(len(freqs), max_te, replace=False)
        freqs, ages, rec = freqs[idx], ages[idx], rec[idx]
        if blocks is not None:
            blocks = blocks[idx]
    return freqs, ages, rec, n_haps, seq_len, n_win, blocks


def phi_r(rec, typical=1e-7, log_scale=1.0, clip=2.0):
    """Normalised log local recombination, matching te_dfe/config_base.py:
    phi_r = clip( log10(r / RECOMB_TYPICAL) / RECOMB_LOG_SCALE, -clip, clip )."""
    rec = np.asarray(rec, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.log10(np.where(rec > 0, rec, np.nan) / typical) / log_scale
    v = np.nan_to_num(v, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(v, -clip, clip)


def observed_age_recomb_sfs(freqs, ages, rbin, n_haps, age_edges, n_rbins):
    """(n_age, n_rbin, n_haps-1) counts: derived-count SFS per (age, recomb) cell."""
    f = np.asarray(freqs, float); a = np.asarray(ages, float)
    rb = np.asarray(rbin, int)
    ok = np.isfinite(f) & np.isfinite(a)
    f, a, rb = f[ok], a[ok], rb[ok]
    counts = np.rint(f * n_haps).astype(int)
    keep = (counts > 0) & (counts < n_haps)
    counts, a, rb = counts[keep], a[keep], rb[keep]
    hi = list(age_edges[1:]) + [np.inf]
    obs = np.zeros((len(age_edges), n_rbins, n_haps - 1))
    for w in range(len(age_edges)):
        in_w = (a >= age_edges[w]) & (a < hi[w])
        for r in range(n_rbins):
            sel = in_w & (rb == r)
            if sel.any():
                obs[w, r] = np.bincount(counts[sel], minlength=n_haps)[1:n_haps]
    return obs


def interp_spectrum(SPEC, log_s_grid, log_s_query):
    """Spectrum at an arbitrary s, by linear interpolation of the precomputed
    grid in LOG s. SPEC is (n_s, n_age, n_freq); returns (n_age, n_freq).

    This is what keeps the context model cheap: s_local differs per recombination
    bin and moves with every optimiser step, but no new moments integration is
    needed -- the grid was built once and each bin just reads off it.
    """
    ls = float(np.clip(log_s_query, log_s_grid[0], log_s_grid[-1]))
    j = int(np.searchsorted(log_s_grid, ls, side="right") - 1)
    j = max(0, min(j, len(log_s_grid) - 2))
    lo, hi = log_s_grid[j], log_s_grid[j + 1]
    t = 0.0 if hi <= lo else (ls - lo) / (hi - lo)
    return (1.0 - t) * SPEC[j] + t * SPEC[j + 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ages", required=True)
    ap.add_argument("--age-edges", default="0,25,50,100,200,400")
    ap.add_argument("--recomb-bins", type=int, default=4,
                    help="number of local-recombination bins (equal-count "
                         "quantiles). More bins resolve the slopes better but "
                         "leave fewer TEs per cell; 3-5 is usually the sweet spot.")
    ap.add_argument("--seq-length", type=float, default=None)
    ap.add_argument("--jitter-ages", type=float, default=0.0)
    ap.add_argument("--age-error-sigma", type=float, default=0.0)
    ap.add_argument("--fit-age-error", action="store_true")
    ap.add_argument("--age-error-bounds", default="0.01,1.0")
    ap.add_argument("--max-te", type=int, default=None)
    ap.add_argument("--s-convention", choices=["homozygote", "per-allele"],
                    default="homozygote")
    ap.add_argument("--gamma-scale", type=float, default=GAMMA_SCALE_DEFAULT)
    ap.add_argument("--dominance", type=float, default=0.5)
    ap.add_argument("--selfing", type=float, default=0.0)
    ap.add_argument("--ne-for-gamma", type=float, default=None)
    ap.add_argument("--n-s-bins", type=int, default=25)
    ap.add_argument("--s-min", type=float, default=1e-8)
    ap.add_argument("--s-max", type=float, default=1.0)
    ap.add_argument("--recomb-typical", type=float, default=1e-7)
    ap.add_argument("--recomb-log-scale", type=float, default=1.0)
    ap.add_argument("--recomb-clip", type=float, default=2.0,
                    help="phi_r is clipped to +/- this. Too small a value piles "
                         "a point mass on the boundary, which ties the quantile "
                         "edges and collapses recombination bins.")
    ap.add_argument("--n-starts", type=int, default=20)
    ap.add_argument("--n-rounds", type=int, default=3)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fix", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--bound", action="append", default=[], metavar="NAME=LO,HI")
    ap.add_argument("--true-theta", default=None)
    ap.add_argument("--uncerts", action="store_true",
                    help="standard errors from the numerical Hessian (FIM). "
                         "Cheap, but assumes independent SFS cells and so "
                         "understates the true error.")
    ap.add_argument("--uncerts-bootstrap", type=int, default=0, metavar="N",
                    help="Godambe standard errors from N bootstrap datasets "
                         "(score only, no refit: fast).")
    ap.add_argument("--uncerts-bootstrap-refit", type=int, default=0, metavar="N",
                    help="REFIT on each of N bootstrap datasets and take "
                         "percentile CIs. Preferred when a parameter sits on a "
                         "bound, where the Godambe interval can cross it.")
    ap.add_argument("--boot-cold-starts", type=int, default=1, metavar="K")
    ap.add_argument("--boot-block-col", default=None, metavar="COLUMN",
                    help="resample whole blocks of this column (e.g. 'window') "
                         "so linked TEs stay together.")
    ap.add_argument("--recomb-map", default=None, metavar="FILE",
                    help="interval table from extract_te_ages.py "
                         "--recomb-map-out (window, start, end, rate). Gives the "
                         "EXACT sequence length behind each recombination bin, "
                         "which is what log_lambda is scaled against. Strongly "
                         "preferred over the neutral-SNP proxy, and the natural "
                         "place to apply callability masks later.")
    ap.add_argument("--neutral-ages", default=None, metavar="FILE",
                    help="TSV of NEUTRAL SNP ages+recombination "
                         "(extract_te_ages.py --snps-out --recomb auto). Used as "
                         "an EMPIRICAL baseline per (age, recombination) cell. "
                         "Removes the component of the recombination signal that "
                         "is NOT TE-specific (e.g. the mutation-recombination "
                         "correlation), leaving gamma_r to describe genuine "
                         "insertion preference.")
    ap.add_argument("--neutral-calibration", choices=["full", "shape"],
                    default="full")
    ap.add_argument("--diagnostics", default=None, metavar="PREFIX")
    args = ap.parse_args()

    from te_dfe.config import PARAM_NAMES, MODEL_NAME
    import te_dfe.config_base as CB
    if MODEL_NAME != "context":
        sys.exit(f"this fits the CONTEXT model; TE_DFE_MODEL is {MODEL_NAME!r}")
    names = list(PARAM_NAMES)
    for req in ("log_lambda", "log_s_global", "alpha_r", "gamma_r",
                "log10_t_start", "log_duration", "log_multiplier"):
        if req not in names:
            sys.exit(f"context PARAM_NAMES is missing {req!r}: {names}")
    I = {nm: names.index(nm) for nm in names}

    # --- demography -------------------------------------------------------
    from te_dfe.demography import MSMCDemography
    from te_dfe.config_base import MSMC_PATH, MU, SIM_LENGTH
    baseline = (MSMCDemography.from_file(MSMC_PATH, mu=MU, sim_length=SIM_LENGTH)
                if Path(MSMC_PATH).exists() else None)
    graph = demes_graph_from_msmc(baseline)
    N_ref = float(graph["pop"].epochs[0].start_size)
    if args.selfing > 0:
        import demes
        ne_fac, _, F = selfing_correction(args.selfing, args.dominance)
        b = demes.Builder(time_units="generations")
        b.add_deme("pop", epochs=[dict(start_size=e.start_size * ne_fac,
                                       end_size=e.end_size * ne_fac,
                                       end_time=e.end_time)
                                  for e in graph["pop"].epochs])
        graph = b.resolve(); N_ref *= ne_fac
    ne_g = args.ne_for_gamma if args.ne_for_gamma is not None else N_ref
    t_max = float(getattr(CB, "SIM_LENGTH", 5000))
    print(f"demography : N_ref={N_ref:.1f}")

    # --- data -------------------------------------------------------------
    freqs, ages, rec, n_haps, seq_hdr, n_win, block_ids = load_ages_recomb(
        args.ages, max_te=args.max_te, seed=args.seed,
        block_col=args.boot_block_col)
    if args.seq_length is None:
        if seq_hdr is None:
            sys.exit("--seq-length not given and no '# seq_length_bp=' header")
        args.seq_length = seq_hdr
    if args.jitter_ages > 0:
        rng0 = np.random.default_rng(args.seed)
        ages = ages * rng0.lognormal(0.0, args.jitter_ages, len(ages))

    age_edges = [float(x) for x in args.age_edges.split(",")]
    if age_edges[0] != 0:
        sys.exit("--age-edges must start at 0")

    # equal-count recombination bins, so every cell carries comparable weight
    ph = phi_r(rec, args.recomb_typical, args.recomb_log_scale, args.recomb_clip)
    n_rb = int(args.recomb_bins)
    qs = np.quantile(ph, np.linspace(0, 1, n_rb + 1))
    # phi_r is CLIPPED at +/-clip, so a wide recombination distribution piles a
    # point mass on the boundary. Tied values make np.quantile return duplicate
    # edges, which silently yields EMPTY bins (n=0, phi_rep=0) -- and an empty
    # bin then makes the exact sequence-share check fail and fall back. Collapse
    # duplicate edges and reduce the bin count to what the data can actually
    # support, rather than carrying dead bins through the whole fit.
    # Deduplicate BEFORE widening the outer edges. Doing it the other way round
    # leaves the -1e-9 as a real edge, so bin 0 becomes [v-1e-9, v) and is empty
    # by construction while the tied mass lands in bin 1 -- which then fails the
    # exact sequence-share check and silently falls back to TE counts.
    qs = np.unique(np.round(qs, 12))
    if len(qs) - 1 < n_rb:
        n_rb_new = max(1, len(qs) - 1)
        vals, cnts = np.unique(np.round(ph, 12), return_counts=True)
        top = vals[np.argmax(cnts)]
        print(f"  !! only {n_rb_new} distinct recombination quantiles available "
              f"(asked for {n_rb}): {cnts.max()/len(ph):.0%} of TEs share "
              f"phi_r={top:.3f}, so the quantile edges tie. Using {n_rb_new} "
              f"bins. (A point mass usually means zero-rate intervals in the "
              f"map, or clipping -- see --recomb-clip.)", file=sys.stderr)
        n_rb = n_rb_new
    qs = qs.astype(float)
    qs[0] -= 1e-9; qs[-1] += 1e-9
    rbin = np.clip(np.searchsorted(qs, ph, side="right") - 1, 0, n_rb - 1)
    # representative phi per bin (mean of the TEs in it) and the sequence
    # FRACTION each bin represents (equal-count bins -> equal TE counts, but the
    # underlying sequence share is what scales the influx)
    phi_rep = np.array([ph[rbin == r].mean() if (rbin == r).any() else 0.0
                        for r in range(n_rb)])
    # SEQUENCE share of each recombination bin -- NOT the TE-count share.
    # With equal-count bins the TE share is 1/n_rb by construction, but bins
    # where TEs are dense span LESS sequence. Using TE counts inflates u.sum()
    # and, because theta = obs.sum()/u.sum(), biases log_lambda DOWNWARD by
    # roughly log(AM/HM) of exp(gamma_r*phi): about -0.27 at |gamma_r|=0.6 and
    # -0.65 at 1.0, and it does NOT shrink with more bins.
    #
    # Neutral SNPs accumulate at a near-constant rate per bp, so their count per
    # recombination bin estimates the sequence share directly. Without them we
    # fall back on the TE share and warn, since lambda is then biased.
    frac_L = None
    # BEST: exact sequence length per recombination bin, from the map itself.
    if args.recomb_map:
        try:
            mp = np.genfromtxt(args.recomb_map, names=True, delimiter="\t")
            m_len = np.atleast_1d(mp["end"]) - np.atleast_1d(mp["start"])
            m_phi = phi_r(np.atleast_1d(mp["rate"]), args.recomb_typical,
                          args.recomb_log_scale, args.recomb_clip)
            m_bin = np.clip(np.searchsorted(qs, m_phi, side="right") - 1,
                            0, n_rb - 1)
            bp = np.array([m_len[m_bin == r].sum() for r in range(n_rb)], float)
            if bp.sum() > 0 and (bp > 0).all():   # every kept bin has sequence
                frac_L = bp / bp.sum()
                print("  sequence share per recombination bin from the "
                      "recombination map (exact): "
                      + ", ".join(f"{v:.3f}" for v in frac_L)
                      + f"  [{bp.sum()/1e6:.1f} Mb covered]")
            else:
                print("  !! recombination map leaves a bin with no sequence; "
                      "falling back", file=sys.stderr)
        except Exception as e:
            print(f"  !! could not read {args.recomb_map}: {type(e).__name__}: {e}",
                  file=sys.stderr)
    if frac_L is None and args.neutral_ages:
        try:
            _nf, _na, _nrec, _nh, _sl, _nw, _blk = load_ages_recomb(args.neutral_ages)
            _nph = phi_r(_nrec, args.recomb_typical, args.recomb_log_scale,
                         args.recomb_clip)
            _nrb = np.clip(np.searchsorted(qs, _nph, side="right") - 1, 0, n_rb - 1)
            cnt = np.array([(_nrb == r).sum() for r in range(n_rb)], float)
            if cnt.sum() > 0 and (cnt > 0).all():
                frac_L = cnt / cnt.sum()
                print("  sequence share per recombination bin from neutral SNPs: "
                      + ", ".join(f"{v:.3f}" for v in frac_L))
        except Exception as e:
            print(f"  !! could not derive sequence share from {args.neutral_ages}: "
                  f"{type(e).__name__}", file=sys.stderr)
    if frac_L is None:
        frac_L = np.array([(rbin == r).sum() for r in range(n_rb)], float)
        frac_L = frac_L / frac_L.sum()
        print("  !! sequence share per recombination bin taken from TE counts "
              "(no --neutral-ages): log_lambda will be biased LOW, by ~0.3 log "
              "units at |gamma_r|=0.6. alpha_r / gamma_r are unaffected.",
              file=sys.stderr)

    obs = observed_age_recomb_sfs(freqs, ages, rbin, n_haps, age_edges, n_rb)
    print(f"data       : {len(freqs)} TEs, {args.seq_length/1e6:.1f} Mb, "
          f"{len(age_edges)} age x {n_rb} recomb cells")
    print(f"  recomb bins (phi_r): " +
          ", ".join(f"[{r}] {phi_rep[r]:+.2f} n={int((rbin==r).sum())}"
                    for r in range(n_rb)))
    for w in range(len(age_edges)):
        hi = age_edges[w + 1] if w + 1 < len(age_edges) else np.inf
        row = " ".join(f"{int(obs[w, r].sum()):>6}" for r in range(n_rb))
        print(f"   [{age_edges[w]:>6.0f}, {('inf' if np.isinf(hi) else f'{hi:.0f}'):>6}) "
              f"gen ago : {row}")

    # --- spectrum grid: age-window spectra per s (ONCE) --------------------
    s_edges = np.concatenate([[0.0], np.geomspace(args.s_min, args.s_max,
                                                  args.n_s_bins)])
    s_mid = s_bin_midpoints(s_edges)
    log_s_grid = np.log(s_mid)
    import inspect, moments
    if "mutation_time_windows" not in inspect.signature(moments.Demes.SFS).parameters:
        sys.exit("this moments build has no mutation_time_windows")
    print(f"building {len(s_mid)} x {len(age_edges)} age-window spectra "
          f"(done ONCE; recombination bins interpolate this grid) ...")
    SPEC = np.array([
        age_window_spectra(graph, n_haps, args.gamma_scale * ne_g * s, age_edges,
                           gamma_scale=args.gamma_scale, ne_for_gamma=ne_g,
                           h=args.dominance, s_convention=args.s_convention)
        for s in s_mid])                                  # (n_s, n_age, n_freq)
    print(f"  grid shape {SPEC.shape}")

    # --- empirical neutral baseline (age x recomb) -------------------------
    NEUTRAL_CAL = None
    if args.neutral_ages:
        nf, na, nrec, nh_n, _, _, _ = load_ages_recomb(args.neutral_ages)
        if args.jitter_ages > 0:
            rngn = np.random.default_rng(args.seed + 7)
            na = na * rngn.lognormal(0.0, args.jitter_ages, len(na))
        # bin the SNPs with the SAME recombination edges as the TEs
        nph = phi_r(nrec, args.recomb_typical, args.recomb_log_scale)
        nrb_idx = np.clip(np.searchsorted(qs, nph, side="right") - 1, 0, n_rb - 1)
        NEUT = observed_age_recomb_sfs(nf, na, nrb_idx, n_haps, age_edges, n_rb)
        theo = np.stack([frac_L[r] * SPEC[0] for r in range(n_rb)], axis=1)
        if args.neutral_calibration == "full":
            a = NEUT / max(NEUT.sum(), 1e-300)
            b = theo / max(theo.sum(), 1e-300)
        else:
            a = NEUT / np.maximum(NEUT.sum(axis=2, keepdims=True), 1e-300)
            b = theo / np.maximum(theo.sum(axis=2, keepdims=True), 1e-300)
        with np.errstate(divide="ignore", invalid="ignore"):
            C = np.where(b > 0, a / b, 1.0)
        n_clip = float(np.mean((C < 0.05) | (C > 20.0) | ~np.isfinite(C)))
        C = np.clip(np.nan_to_num(C, nan=1.0, posinf=1.0), 0.05, 20.0)
        # see the note in fit_dfe_moments_aged.py: clipping breaks the
        # total-preserving property of C and biases log_lambda through theta0.
        renorm = (theo * C).sum() / max(theo.sum(), 1e-300)
        if renorm > 0:
            C = C / renorm
        NEUTRAL_CAL = C
        print(f"neutral    : {len(nf)} SNPs -> empirical baseline "
              f"({args.neutral_calibration}); correction in "
              f"[{C.min():.2f}, {C.max():.2f}]; {n_clip:.1%} clipped, "
              f"renormalised by {renorm:.4f} "
              f"(would shift log_lambda {-np.log(renorm):+.3f})")

    # --- age-error mixing --------------------------------------------------
    FIT_SIGMA = bool(args.fit_age_error)
    _cache = {}

    def age_mix(sigma):
        if not (sigma and sigma > 0):
            return None
        key = round(float(sigma), 4)
        if key not in _cache:
            _cache[key] = age_smearing_matrix(
                age_edges, key, t_max=t_max,
                n_quad=(120 if FIT_SIGMA else 400)).T
        return _cache[key]

    # --- bounds / fixed ----------------------------------------------------
    import te_dfe.config as CFG
    bounds_all = []
    for nm in names:
        rng_nm = f"{nm.upper()}_RANGE"
        bounds_all.append(tuple(getattr(CFG, rng_nm)))
    for spec_str in args.bound:
        k, _, rs = spec_str.partition("=")
        if k not in names:
            sys.exit(f"--bound: unknown parameter {k!r}")
        lo, hi = (float(t) for t in rs.split(","))
        bounds_all[names.index(k)] = (lo, hi)
    fixed = {}
    for spec_str in args.fix:
        k, _, v = spec_str.partition("=")
        if k not in names:
            sys.exit(f"--fix: unknown parameter {k!r}")
        fixed[names.index(k)] = float(v)
    profile_lambda = I["log_lambda"] not in fixed
    free = [i for i in range(len(names))
            if i not in fixed and not (profile_lambda and i == I["log_lambda"])]
    bounds = [bounds_all[i] for i in free]
    if FIT_SIGMA:
        bounds = bounds + [tuple(float(v) for v in args.age_error_bounds.split(","))]
    n_model_free = len(free)
    if fixed:
        print("  fixed: " + ", ".join(f"{names[i]}={v:.4f}"
                                      for i, v in sorted(fixed.items())))
    if profile_lambda:
        print("  log_lambda profiled out (closed-form Poisson scale)")
    print("  search bounds:")
    for i in range(len(names)):
        lo, hi = bounds_all[i]
        tag = (f"FIXED {fixed[i]:.4f}" if i in fixed else
               ("profiled" if profile_lambda and i == I["log_lambda"] else "free"))
        print(f"    {names[i]:<16}[{lo:9.4f}, {hi:9.4f}]  {tag}")

    def expand(xf):
        full = np.empty(len(names))
        for i, v in fixed.items():
            full[i] = v
        if profile_lambda:
            full[I["log_lambda"]] = 0.0
        full[free] = xf
        return full

    def split_x(x):
        x = np.asarray(x, float)
        if FIT_SIGMA:
            return x[:n_model_free], float(x[n_model_free])
        return x, args.age_error_sigma

    # --- model -------------------------------------------------------------
    def model_per_unit(full, sigma):
        if not np.isfinite(full).all():
            return None
        ls_g = full[I["log_s_global"]]
        a_r = full[I["alpha_r"]]
        g_r = full[I["gamma_r"]]
        m_w = burst_influx_per_window(age_edges, 10.0 ** full[I["log10_t_start"]],
                                      np.exp(full[I["log_duration"]]),
                                      np.exp(full[I["log_multiplier"]]),
                                      t_max=t_max)                # (n_age,)
        M = age_mix(sigma)
        u = np.empty((len(age_edges), n_rb, n_haps - 1))
        for r in range(n_rb):
            # selection in this recombination bin
            S_r = interp_spectrum(SPEC, log_s_grid, ls_g + a_r * phi_rep[r])
            # insertion intensity in this bin (gamma_r) x its sequence share
            w_r = np.exp(g_r * phi_rep[r]) * frac_L[r]
            cell = w_r * (m_w[:, None] * S_r)                     # (n_age, n_freq)
            if M is not None:
                cell = M @ cell
            u[:, r, :] = cell
        if NEUTRAL_CAL is not None:
            u = u * NEUTRAL_CAL
        if not np.isfinite(u).all() or (u < 0).any() or u.sum() <= 0:
            return None
        return u

    def nll_with_obs(full, obs_use, sigma):
        u = model_per_unit(full, sigma)
        if u is None:
            return 1e12
        theta = (obs_use.sum() / u.sum() if profile_lambda
                 else 4.0 * N_ref * np.exp(full[I["log_lambda"]]) * args.seq_length)
        if not np.isfinite(theta) or theta <= 0:
            return 1e12
        return -poisson_ll(obs_use.ravel(), (theta * u).ravel())

    def obj(x):
        xm, sg = split_x(x)
        return nll_with_obs(expand(xm), obs, sg)

    # --- fit ---------------------------------------------------------------
    from scipy.optimize import minimize
    rng = np.random.default_rng(args.seed)
    lo_b = np.array([lo for lo, _ in bounds]); hi_b = np.array([hi for _, hi in bounds])
    best, runs = None, []
    for k in range(args.n_starts):
        st = rng.uniform(lo_b, hi_b)
        try:
            r = minimize(obj, st, method="L-BFGS-B", bounds=bounds,
                         options=dict(maxiter=args.maxiter))
        except Exception as e:
            print(f"  start {k}: failed ({type(e).__name__})", file=sys.stderr)
            continue
        runs.append(r)
        if best is None or r.fun < best.fun:
            best = r
        print(f"  start {k+1}/{args.n_starts}: -logL = {r.fun:.2f}"
              f"{'   <-- best' if best is r else ''}")
    if best is None:
        sys.exit("all starts failed")
    for rnd in range(1, args.n_rounds):
        span = hi_b - lo_b
        for _ in range(max(2, args.n_starts // 2)):
            st = np.clip(best.x + rng.normal(0, 0.1 / rnd * span), lo_b, hi_b)
            r = minimize(obj, st, method="L-BFGS-B", bounds=bounds,
                         options=dict(maxiter=args.maxiter))
            runs.append(r)
            if r.fun < best.fun:
                best = r
        print(f"  round {rnd+1}: -logL = {best.fun:.2f}")
    near = sum(1 for r in runs if r.fun < best.fun + 1.0)
    print(f"  {near}/{len(runs)} starts within 1 log-unit of the best")

    # --- report -------------------------------------------------------------
    x_model, sigma_hat = split_x(best.x)
    x = expand(x_model)
    if profile_lambda:
        u_b = model_per_unit(x, sigma_hat)
        if u_b is not None and u_b.sum() > 0:
            x[I["log_lambda"]] = np.log((obs.sum() / u_b.sum())
                                        / (4.0 * N_ref * args.seq_length))
    truth = ([float(v) for v in args.true_theta.split(",")]
             if args.true_theta else None)
    print(f"\n  {'parameter':<18}{'estimate':>12}" + (f"{'truth':>12}" if truth else ""))
    for k, nm in enumerate(names):
        line = f"  {nm:<18}{x[k]:>12.4f}"
        if truth:
            line += f"{truth[k]:>12.4f}"
        print(line)
    if FIT_SIGMA:
        print(f"  {'age_error_sigma':<18}{sigma_hat:>12.4f}")
    print(f"  {'-logL':<18}{best.fun:>12.2f}")

    # s_local implied in each recombination bin: the interpretable output
    # --- uncertainties ------------------------------------------------------
    def _boot_idx(rng_b, blocks, n_te):
        if blocks is not None:
            uq = np.unique(blocks)
            pick = rng_b.choice(uq, uq.size, replace=True)
            return np.concatenate([np.flatnonzero(blocks == b) for b in pick])
        return rng_b.integers(0, n_te, n_te)

    def _obs_from(idx):
        """Rebuild the (age x recomb x freq) table for a resampled TE set.
        The recombination BIN EDGES are kept fixed at the ones fitted on the full
        data: re-deriving quantiles per replicate would resample the design as
        well as the data and would understate the uncertainty."""
        return observed_age_recomb_sfs(freqs[idx], ages[idx], rbin[idx],
                                       n_haps, age_edges, n_rb)

    n_te = len(freqs)
    blocks = None
    if args.boot_block_col and block_ids is not None:
        blocks = np.asarray(block_ids)
        if blocks.size != n_te:
            blocks = None

    if args.uncerts_bootstrap_refit and free:
        rng_b = np.random.default_rng(args.seed + 4242)
        print(f"\n  refit bootstrap: {args.uncerts_bootstrap_refit} replicate(s) "
              f"({'block' if blocks is not None else 'TE-level'} resampling)")
        bx = []
        for _ in range(args.uncerts_bootstrap_refit):
            ob = _obs_from(_boot_idx(rng_b, blocks, n_te))
            def objb(xf, _o=ob):
                xm, sg = split_x(xf)
                return nll_with_obs(expand(xm), _o, sg)
            cand = []
            for st in [np.asarray(best.x, float)] + \
                      [rng_b.uniform(lo_b, hi_b) for _ in range(max(0, args.boot_cold_starts))]:
                try:
                    cand.append(minimize(objb, st, method="L-BFGS-B", bounds=bounds,
                                         options=dict(maxiter=args.maxiter)))
                except Exception:
                    pass
            cand = [r for r in cand if np.isfinite(r.fun)]
            if cand:
                bx.append(min(cand, key=lambda r: r.fun).x)
        bx = np.asarray(bx, float)
        if bx.shape[0] >= 2:
            se = bx.std(axis=0, ddof=1)
            lo_p = np.percentile(bx, 2.5, axis=0); hi_p = np.percentile(bx, 97.5, axis=0)
            print("\n  standard errors (bootstrap-refit; 95% CI = percentile):")
            for j, i in enumerate(free):
                print(f"    {names[i]:<16}{float(x_model[j]):>10.4f} +/- {se[j]:<9.4f}"
                      f"[{lo_p[j]:>9.4f}, {hi_p[j]:>9.4f}]")
            if FIT_SIGMA and bx.shape[1] > n_model_free:
                k = n_model_free
                print(f"    {'age_error_sigma':<16}{sigma_hat:>10.4f} +/- {se[k]:<9.4f}"
                      f"[{lo_p[k]:>9.4f}, {hi_p[k]:>9.4f}]")
            # per-replicate draws, so CIs on DERIVED quantities (s_local per
            # recombination bin) are percentiles of the joint bootstrap
            if args.diagnostics:
                dp = f"{args.diagnostics}.context_boot.tsv"
                with open(dp, "w") as fh:
                    fh.write("log_s_global\talpha_r\tgamma_r\n")
                    for row in bx:
                        full_b = expand(row[:n_model_free])
                        fh.write(f"{full_b[I['log_s_global']]:.6g}\t"
                                 f"{full_b[I['alpha_r']]:.6g}\t"
                                 f"{full_b[I['gamma_r']]:.6g}\n")
                print(f"  bootstrap draws: {dp} ({bx.shape[0]} replicates)")
        else:
            print("    !! too few successful refits for an interval", file=sys.stderr)

    elif (args.uncerts or args.uncerts_bootstrap) and free:
        boot_nlls = None
        if args.uncerts_bootstrap:
            rng_b = np.random.default_rng(args.seed + 991)
            boot_nlls = []
            for _ in range(args.uncerts_bootstrap):
                ob = _obs_from(_boot_idx(rng_b, blocks, n_te))
                def _bn(xf, _o=ob):
                    xm, sg = split_x(xf)
                    return nll_with_obs(expand(xm), _o, sg)
                boot_nlls.append(_bn)
            print(f"\n  {args.uncerts_bootstrap} bootstrap replicate(s) for the "
                  f"Godambe correction")
        se, cov, H, method = uncertainties(obj, np.asarray(best.x, float),
                                           boot_nlls=boot_nlls)
        print(f"\n  standard errors ({method}; 95% CI):")
        for j, i in enumerate(free):
            xi = float(x_model[j])
            print(f"    {names[i]:<16}{xi:>10.4f} +/- {se[j]:<9.4f}"
                  f"[{xi-1.96*se[j]:>9.4f}, {xi+1.96*se[j]:>9.4f}]")
        if method == "FIM":
            print("    NOTE FIM assumes independent SFS cells: LOWER bounds on "
                  "the error. Add --uncerts-bootstrap-refit 100.")

    print("\n  implied selection by recombination bin:")
    for r in range(n_rb):
        s_r = np.exp(x[I["log_s_global"]] + x[I["alpha_r"]] * phi_rep[r])
        lam_r = np.exp(x[I["gamma_r"]] * phi_rep[r])
        print(f"    bin {r} (phi_r={phi_rep[r]:+.2f}): s_local={s_r:.3e}  "
              f"2Ne*s={2*ne_g*s_r:.1f}  insertion x{lam_r:.2f}")

    for j, i in enumerate(free):
        lo, hi = bounds_all[i]; xi = float(x_model[j]); w = hi - lo
        if w > 0 and (xi - lo < 1e-3 * w or hi - xi < 1e-3 * w):
            side = "LOWER" if xi - lo < 1e-3 * w else "UPPER"
            print(f"  !! {names[i]} pinned at {side} bound", file=sys.stderr)

    if args.diagnostics:
        u_f = model_per_unit(x, sigma_hat)
        if u_f is not None:
            theta = (obs.sum() / u_f.sum() if profile_lambda
                     else 4.0 * N_ref * np.exp(x[I["log_lambda"]]) * args.seq_length)
            e = theta * u_f
            path = f"{args.diagnostics}.context_sfs_fit.tsv"
            with open(path, "w") as fh:
                fh.write("age_window\tage_lo\trecomb_bin\tphi_r\tderived_count\t"
                         "obs\texp\tpoisson_resid\n")
                for w in range(len(age_edges)):
                    for r in range(n_rb):
                        for j in range(n_haps - 1):
                            ev = e[w, r, j]
                            res = (ev - obs[w, r, j]) / np.sqrt(max(ev, 1e-300))
                            fh.write(f"{w}\t{age_edges[w]:g}\t{r}\t{phi_rep[r]:.4f}\t"
                                     f"{j+1}\t{obs[w,r,j]:.6g}\t{ev:.6g}\t{res:.6g}\n")
            print(f"\n  diagnostics: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
