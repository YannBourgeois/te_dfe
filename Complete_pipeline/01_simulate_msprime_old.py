#!/usr/bin/env python3
"""
01_simulate_msprime.py — FAST neutral-coalescent TE simulator (msprime), a drop-in
alternative to the SLiM producer for the INSERTION parameters + a coalescent
EXPANSION proxy for purifying selection.

Idea (after Xue, Ruggiero, Hickerson & Boissinot 2018, GBE 10.1093/gbe/evy083):
purifying selection holds deleterious TEs rare, distorting the TE frequency
spectrum like a population EXPANSION. We replace forward selection with a
coalescent expansion of rate nu ON TOP OF the real MSMC baseline demography; an
expansion-distorted (star-like) genealogy makes branch-placed TEs rare. A
per-recomb-segment mosaic makes nu recombination-dependent -> the alpha_r analogue.

nu IS THE PARAMETER (Stage 2): we draw log_nu_global and nu_r directly and put
them in the log_s_global / alpha_r theta slots, so the 7-vector and 02_collate /
03_train are unchanged, but there is NO hidden selection assumption in the prior.
The nu -> true-selection mapping is fixed AFTERWARD by calibrate_msprime_selection.py
(run FIRST to choose --nu-range). The expansion is parameterised by the
total size RATIO upsilon over a BOUNDED window (Xue et al. 2023-style), not
by an unbounded rate: see expansion_log_upsilon / build_demography.

Windows, recomb map, genes, and baseline demography come from the SAME
WindowSampler / MSMCDemography the SLiM path uses, so sims are directly comparable.

msprime/tskit are cluster-only; the pure-numpy helpers are unit-testable.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Pure-numpy helpers (unit-testable without msprime/tskit)
# ---------------------------------------------------------------------------
def feature_r(recomb_rate, recomb_typical, log_scale, clip=2.0):
    """Normalized log local recombination, matching processor.feature_r."""
    with np.errstate(divide="ignore"):
        v = np.log10(np.asarray(recomb_rate, float) / recomb_typical) / log_scale
    return np.clip(v, -clip, clip)


# ---------------------------------------------------------------------------
# PURIFYING SELECTION BY PER-TE PRUNING (replaces the nu/expansion proxy)
#
# The expansion proxy mimicked selection by distorting the GENEALOGY, which has
# two structural faults the diagnostics exposed: (i) it also suppresses neutral
# SNP diversity, so local_snp_density / n_flank_snps_norm drift 2-4 SDs from
# SLiM and get worse the harder you push it; (ii) it only RAREFIES TEs, so mean
# frequency bottoms out at the all-singleton floor (~1/n) and cannot reach the
# strong-selection regime -- the observed ceiling around SLiM log_mean_s ~ -6.
#
# Instead: simulate TEs NEUTRALLY on one undistorted tree, then PRUNE each TE
# with a probability derived from Wright-Fisher / Poisson-Random-Field theory,
# using that TE's OWN local selection coefficient. Removal has no floor and no
# SNP side effect.
#
# Equilibrium density of segregating sites at frequency x (Sawyer & Hartl 1992),
# with gamma = 4*Ne*s and s > 0 deleterious:
#
#     f(x | gamma) = (1 - e^{gamma(1-x)}) / ((1 - e^{gamma}) * x(1-x))
#
# gamma -> 0 gives f = 1/x (neutral); gamma large gives f ~ e^{-gamma x}/(x(1-x)).
# The keep-probability is the ratio of the SELECTED to the NEUTRAL expected
# SAMPLE spectrum, so thinning a neutral sample reproduces the selected one:
#
#     p_keep(i | gamma) = E[S_i | gamma] / E[S_i | 0],   E[S_i | 0] = 1/i
#
# p_keep(i|0) = 1 for all i (exactly nested null), and p_keep decreases with
# frequency, so common TEs are removed preferentially -- purifying selection.
# The TOTAL count also drops, which restores the density channel alpha_r and
# gamma_r need and which the expansion proxy could not produce.
# ---------------------------------------------------------------------------
def _prf_pop_density(x, gamma):
    """f(x | gamma), the PRF equilibrium frequency density (vectorised in x)."""
    x = np.asarray(x, float)
    if abs(gamma) < 1e-9:
        return 1.0 / x
    if gamma < 500.0:                       # expm1 form: stable for small gamma
        return np.expm1(gamma * (1.0 - x)) / (np.expm1(gamma) * x * (1.0 - x))
    return np.exp(-gamma * x) / (x * (1.0 - x))     # asymptote, avoids overflow


def build_prf_keep_table(n_haps, gamma_grid, n_quad=4000):
    """p[g, i] = E[S_i | gamma_grid[g]] / E[S_i | 0], for i = 1 .. n_haps-1.

    One vectorised quadrature over a shared x grid; the binomial sampling matrix
    is built once. Costs well under a second and is done at startup.
    """
    from scipy.special import gammaln
    n = int(n_haps)
    # x grid concentrated near the boundaries where the density is steep
    u = np.linspace(0.0, 1.0, n_quad + 2)[1:-1]
    x = u ** 3 / (u ** 3 + (1 - u) ** 3)            # symmetric boundary clustering
    x = np.clip(x, 1e-9, 1 - 1e-9)
    dx = np.gradient(x)
    i_idx = np.arange(1, n)
    # B[i, x] = C(n,i) x^i (1-x)^(n-i)   in logs for stability
    lb = (gammaln(n + 1) - gammaln(i_idx + 1) - gammaln(n - i_idx + 1))[:, None]
    B = np.exp(lb + i_idx[:, None] * np.log(x)[None, :]
               + (n - i_idx)[:, None] * np.log1p(-x)[None, :])
    neutral = 1.0 / i_idx                            # E[S_i | 0]
    tab = np.empty((len(gamma_grid), n - 1))
    for gi, g in enumerate(gamma_grid):
        f = _prf_pop_density(x, float(g))
        tab[gi] = (B * (f * dx)[None, :]).sum(axis=1) / neutral
    return np.clip(tab, 0.0, 1.0)


def build_prf_keep_table_moments(n_haps, gamma_grid, baseline_msmc, ne_ref,
                                 gamma_scale, h=0.5, verbose=True):
    """RIGOROUS backend: p_keep from spectra computed under the ACTUAL demography.

        p_keep(i | gamma) = E[S_i | gamma, demography] / E[S_i | 0, demography]

    Both numerator and denominator are integrated through the real MSMC history
    with `moments`, so the separability approximation of the analytic backend
    (selection's RELATIVE distortion is demography-independent) is dropped
    entirely. Under a constant history this must reduce to the analytic table --
    that is what --prf-selftest checks.

    CONVENTIONS (the easy thing to get wrong):
      * moments/dadi use gamma_m = 2*N_ref*s with gamma_m POSITIVE = BENEFICIAL.
        Our gamma is gamma_scale*Ne*s with gamma POSITIVE = DELETERIOUS. So
        gamma_m = -2*N_ref*gamma / (gamma_scale*ne_ref).
      * `h` is the dominance coefficient; 0.5 = additive. Match the SLiM sims.
      * Time in moments is in units of 2*N_ref generations; sizes are relative
        to N_ref. N_ref is taken as the ancestral size (where the steady state
        is computed).
    """
    import moments
    n = int(n_haps)

    # --- forward-time epoch list: (relative size, scaled duration) ------------
    epochs = []
    N_ref = float(ne_ref)
    if baseline_msmc is not None and getattr(baseline_msmc, "epochs", None):
        eps = sorted(baseline_msmc.epochs, key=lambda e: e.slim_gen_start)
        N_ref = float(getattr(baseline_msmc, "ancestral_ne", eps[0].ne))
        bounds = [e.slim_gen_start for e in eps] + [float(baseline_msmc.sim_length)]
        for k, e in enumerate(eps):
            dur = float(bounds[k + 1] - bounds[k])
            if dur > 0:
                epochs.append((float(e.ne) / N_ref, dur / (2.0 * N_ref)))
    if verbose:
        print(f"[prf-moments] N_ref={N_ref:.1f}, {len(epochs)} epoch(s) integrated, "
              f"h={h}")

    def _spectrum(gamma_ours):
        gm = -2.0 * N_ref * float(gamma_ours) / (gamma_scale * float(ne_ref))
        fs = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(n, gamma=gm, h=h))
        for nu_rel, T in epochs:
            if T > 0:
                fs.integrate([nu_rel], T, gamma=gm, h=h)
        return np.asarray(fs)[1:n]          # segregating classes i = 1 .. n-1

    neut = _spectrum(0.0)
    neut = np.where(neut > 0, neut, np.nan)
    tab = np.empty((len(gamma_grid), n - 1))
    for gi, g in enumerate(gamma_grid):
        tab[gi] = _spectrum(float(g)) / neut
    tab = np.nan_to_num(tab, nan=0.0, posinf=0.0, neginf=0.0)
    # p_keep is a THINNING probability: it can only remove. If some class comes
    # out >1 (possible off-equilibrium), rescale the whole table so max = 1 --
    # this preserves the SHAPE and pushes the lost count into a global factor,
    # which --lambda-boost can compensate if the absolute count matters.
    mx = float(np.nanmax(tab))
    if mx > 1.0:
        if verbose:
            print(f"[prf-moments] max p_keep={mx:.3f} > 1 -> rescaling table "
                  f"(thinning cannot ADD sites; shape preserved)")
        tab = tab / mx
    return np.clip(tab, 0.0, 1.0)


def prf_selftest(n_haps, gamma_grid, baseline_msmc, ne_ref, gamma_scale, h=0.5):
    """Compare the moments backend to the analytic one. Under a CONSTANT-Ne
    history they must agree; a mismatch means the gamma convention or the
    dominance is wrong, which would silently mis-scale every prune probability."""
    an = build_prf_keep_table(n_haps, gamma_grid)
    mo = build_prf_keep_table_moments(n_haps, gamma_grid, baseline_msmc, ne_ref,
                                      gamma_scale, h=h, verbose=True)
    ne_set = {float(e.ne) for e in getattr(baseline_msmc, "epochs", [])} or {ne_ref}
    const = len(ne_set) <= 1
    d = np.abs(an - mo)
    print("=" * 66)
    print("PRF BACKEND SELF-TEST" + ("  (constant Ne: backends MUST agree)"
                                     if const else
                                     "  (varying Ne: differences are EXPECTED)"))
    print(f"  max |analytic - moments| = {d.max():.4f}   mean = {d.mean():.4f}")
    for g in (2.0, 20.0, 200.0):
        gi = int(np.argmin(np.abs(np.asarray(gamma_grid) - g)))
        print(f"  gamma={g:6.1f}  i=1: an={an[gi,0]:.3f} mo={mo[gi,0]:.3f}   "
              f"i=10: an={an[gi,9]:.4f} mo={mo[gi,9]:.4f}")
    if const:
        print("  VERDICT:", "PASS" if d.max() < 0.05 else
              "FAIL - check --gamma-scale and -h against the SLiM convention")
    else:
        print("  (constant-Ne comparison unavailable; run once with a flat MSMC "
              "file to validate the convention, then trust moments here)")
    print("=" * 66)
    return an, mo


def prf_keep_prob(gamma, i, gamma_grid, table):
    """Interpolate p_keep for scaled coefficient `gamma` at carrier count `i`."""
    j = int(np.clip(i - 1, 0, table.shape[1] - 1))
    return float(np.interp(gamma, gamma_grid, table[:, j]))


def harmonic_mean_ne(baseline_msmc, fallback):
    """Harmonic-mean Ne over the MSMC epochs -- the size that governs coalescence
    (and hence gamma) when the demography varies. Constant history returns Ne."""
    if baseline_msmc is None:
        return float(fallback)
    eps = sorted(baseline_msmc.epochs, key=lambda e: e.slim_gen_start)
    if len(eps) <= 1:
        return float(eps[0].ne) if eps else float(fallback)
    bounds = [e.slim_gen_start for e in eps] + [float(baseline_msmc.sim_length)]
    dur = np.diff(bounds)
    ne = np.array([e.ne for e in eps], float)
    ok = (dur > 0) & (ne > 0)
    if not ok.any():
        return float(fallback)
    return float(dur[ok].sum() / (dur[ok] / ne[ok]).sum())


def insertion_intensity(recomb_rate, log_lambda, gamma_r, recomb_typical, log_scale):
    """lambda_local(p) = lambda_global * exp(gamma_r * feature_r). gamma_r>0 ->
    more insertions where recombination is high."""
    fr = feature_r(recomb_rate, recomb_typical, log_scale)
    return np.exp(log_lambda) * np.exp(gamma_r * fr)


def expansion_log_upsilon(a, b, recomb_rate, recomb_typical, log_scale,
                          log_offset=0.0, clip_at_neutral=False,
                          s_gain=1.0, alpha_gain=1.0):
    """log(upsilon) for a segment, where upsilon is the TOTAL size RATIO of the
    exponential size change (Xue et al.'s parameterisation), NOT a rate:

        log upsilon = a + b * feature_r(recomb) + log_offset

    Parameterising by the RATIO rather than the rate is what keeps this bounded:
    upsilon = 1 -> r = 0 -> EXACTLY the null demography (a clean nested anchor for
    "no selection"), and a prior on log upsilon caps total size change by
    construction. Parameterising by rate with no end time (the previous version)
    let N(t) = N0*exp(-r*t) collapse to <1 individual, forcing a star genealogy
    and destroying monotonicity in s.

    upsilon > 1 (growth)      mimics purifying selection  (excess rare alleles)
    upsilon < 1 (contraction) mimics positive selection    (intermediate-freq skew)

    nu-direct : a = log upsilon_global, b = nu_r, log_offset = 0. Both signs are
                meaningful and the calibration grid maps upsilon <-> s post hoc.
    s-proxy   : a = log_s_global, b = alpha_r (the REAL params), and
                log_offset = -log(s_ref) so that upsilon = s_local / s_ref.
                clip_at_neutral floors log upsilon at 0: below s_ref selection is
                weaker than drift and genuinely CANNOT act, so a contraction there
                would be wrong-signed for a purifying-selection model. The flat
                region is honest -- SLiM cannot distinguish those s values either.
    """
    fr = feature_r(recomb_rate, recomb_typical, log_scale)
    # GAINS (s_gain, alpha_gain) make the s -> upsilon map a POWER LAW instead of
    # strictly proportional:
    #     upsilon = (s_local / s_ref) ** s_gain          [global term]
    # so s=1e-2 with s_ref=1e-3 gives upsilon = 10 at gain 1, 100 at gain 2, 1000
    # at gain 3, ... The gain multiplies log upsilon, and log upsilon = 0 at
    # s = s_ref is preserved for ANY gain -- the nested "no selection" anchor
    # survives, which is what keeps the null exact.
    # alpha_gain scales the RECOMB-CONTRAST term independently of the global one,
    # so the alpha_r signal can be accentuated WITHOUT also inflating overall
    # selection strength. (nu_duration scales both terms together and therefore
    # cannot separate them; these gains can.)
    lu = s_gain * (a + log_offset) + alpha_gain * (b * fr)
    if clip_at_neutral:
        lu = max(lu, 0.0)
    return lu


def growth_rate_from_log_upsilon(log_upsilon, duration):
    """msprime growth_rate r achieving total ratio upsilon over `duration`
    generations: N(t) = N(t_young)*exp(-r*(t - t_young)), so r = ln(upsilon)/D.

    The SFS distortion scales as r*2Ne = ln(upsilon)*2Ne/duration -- DURATION is
    the denominator, and therefore the lever on how strong the proxy can get.
    Xue et al. used duration ~125,000 generations, which is why their exponential
    "mimics WEAK purifying selection": at 2Ne~20,000, even upsilon=10 buys
    r*2Ne~0.4. Short duration -> strong reachable distortion."""
    return float(log_upsilon) / float(duration)



def recomb_segments(seg_edges, ratemap_positions, ratemap_rates):
    """Given segment edges and a step recomb map (positions, rates), return each
    segment's (start, end, mean_recomb). Pure numpy so it's testable; on-cluster
    positions/rates come from gw.recomb_map (an msprime RateMap)."""
    rp = np.asarray(ratemap_positions, float)
    rr = np.asarray(ratemap_rates, float)
    out = []
    for a, b in zip(seg_edges[:-1], seg_edges[1:]):
        lo = np.searchsorted(rp, a, "right") - 1
        hi = np.searchsorted(rp, b, "right") - 1
        idx = range(max(lo, 0), max(hi, 0) + 1)
        segs = [rr[min(i, len(rr) - 1)] for i in idx]
        out.append((float(a), float(b),
                    float(np.mean(segs)) if segs else float(rr.mean())))
    return out


# DEPRECATED (unused): the population-count + branch-pick TE model. Replaced
# by the two-pass sim_mutations model in simulate_window. Kept only for
# reference; safe to delete.
def place_te_positions(seg_start, seg_end, seg_recomb, log_lambda, gamma_r,
                       recomb_typical, log_scale, mean_2ne, rng):
    """Poisson number of TEs in a segment. Expected count uses the population-level
    conversion E[n] = lambda_local * seg_len * integrated_2Ne(), where mean_2ne IS
    integrated_2Ne() (NOT divided by window size). Positions uniform within the recomb-homogeneous
    segment."""
    lam = insertion_intensity(seg_recomb, log_lambda, gamma_r, recomb_typical, log_scale)
    expected = lam * (seg_end - seg_start) * mean_2ne
    n = rng.poisson(max(expected, 0.0))
    if n == 0:
        return np.empty(0, dtype=np.float64)
    return np.sort(rng.uniform(seg_start, seg_end, size=n))


# DEPRECATED (unused): see note on place_te_positions.
def _branch_weighted_carriers(tree, n_samples, rng):
    """Pick a branch with prob proportional to its length; return carrier bool over
    samples. Expansion-distorted trees -> long tip-ward branches -> rare TEs.
    Requires tskit at call time (cluster only)."""
    nodes, weights = [], []
    for u in tree.nodes():
        if u == tree.root:
            continue
        bl = tree.branch_length(u)
        if bl > 0:
            nodes.append(u); weights.append(bl)
    if not nodes:
        return np.zeros(n_samples, dtype=bool)
    w = np.asarray(weights, float); w /= w.sum()
    u = nodes[rng.choice(len(nodes), p=w)]
    carriers = np.zeros(n_samples, dtype=bool)
    for s in tree.samples(u):
        carriers[s] = True
    return carriers


# ---------------------------------------------------------------------------
# Demography: MSMC baseline + present-day expansion nu
# ---------------------------------------------------------------------------
def _msmc_size_at(baseline_msmc, t_ago, Ne_fallback):
    """Baseline Ne at t_ago generations before the present.

    MSMCDemography stores `epochs` as DemographicEpoch(slim_gen_start, ne) in
    FORWARD (SLiM) time over [1, sim_length]. Time-ago maps to forward time as
    g_fwd = sim_length - t_ago, and the size at that point is the ne of the epoch
    with the largest slim_gen_start <= g_fwd. t_ago beyond the oldest epoch ->
    ancestral_ne.
    """
    if baseline_msmc is None:
        return float(Ne_fallback)
    eps = sorted(baseline_msmc.epochs, key=lambda e: e.slim_gen_start)
    g_fwd = baseline_msmc.sim_length - float(t_ago)
    if g_fwd <= eps[0].slim_gen_start:
        return float(baseline_msmc.ancestral_ne)
    ne = eps[0].ne
    for ep in eps:
        if ep.slim_gen_start <= g_fwd:
            ne = ep.ne
        else:
            break
    return float(ne)


def build_demography(baseline_msmc, log_upsilon, Ne_fallback,
                     nu_start_time, nu_duration, anchor="present"):
    """MSMC baseline with the expansion SPLICED INTO a bounded window.

    The window spans [t_young, t_old] generations ago, with
        t_old   = nu_start_time                 (older edge)
        t_young = nu_start_time - nu_duration   (younger edge; 0 = the present)
    MSMC events INSIDE the window are dropped (the history there is replaced by
    the exponential); outside, the baseline is untouched. Growth reverts to 0 at
    t_old, which is what BOUNDS the whole thing -- note msprime leaves growth_rate
    UNCHANGED on a parameters_change unless you pass growth_rate explicitly, so
    the reversion event must set it to 0.0 by hand.

    anchor="present"  : pin the YOUNG edge to the baseline; old edge = N_young/ups.
        High-upsilon segments then have FEWER SNPs and a rare-skewed SFS -- both
        agreeing in SIGN with real selection, and present-day diversity stays on
        the baseline so upsilon does not leak into the neutral parameters.
    anchor="ancestral": pin the OLD edge; young edge = ups * N_old. Continuous at
        t_old, but high-upsilon segments end up with MORE present-day diversity --
        the OPPOSITE sign to selection in local_snp_density. Available for
        experiments; not the default, and a likely source of negative transfer.
    """
    import msprime
    t_old   = float(nu_start_time)
    t_young = float(nu_start_time) - float(nu_duration)
    if t_young < 0:
        raise ValueError(f"--nu-duration ({nu_duration}) exceeds --nu-start-time "
                         f"({nu_start_time}): the window would start in the future.")
    if nu_duration <= 0:
        raise ValueError("--nu-duration must be > 0.")

    r   = growth_rate_from_log_upsilon(log_upsilon, nu_duration)
    ups = float(np.exp(log_upsilon))

    N_present  = _msmc_size_at(baseline_msmc, 0.0,     Ne_fallback)
    N_young_bl = _msmc_size_at(baseline_msmc, t_young, Ne_fallback)
    N_old_bl   = _msmc_size_at(baseline_msmc, t_old,   Ne_fallback)
    N_young    = N_young_bl if anchor == "present" else ups * N_old_bl

    dem = msprime.Demography()
    if t_young <= 0.0:
        # Expansion reaches the present: growth is set on the population itself.
        dem.add_population(name="pop", initial_size=N_young, growth_rate=r)
    else:
        dem.add_population(name="pop", initial_size=N_present, growth_rate=0.0)

    # to_msprime_events() returns msprime.PopulationParametersChange OBJECTS
    # (time-ago), NOT dicts -- read .time, and re-attach by copying fields.
    events = baseline_msmc.to_msprime_events() if baseline_msmc is not None else []

    def _add(ev):
        dem.add_population_parameters_change(
            time=float(ev.time), initial_size=float(ev.initial_size),
            population="pop")

    for ev in events:                                # baseline YOUNGER than window
        if float(ev.time) < t_young:
            _add(ev)

    if t_young > 0.0:                                # open the window
        dem.add_population_parameters_change(
            time=t_young, initial_size=N_young, growth_rate=r, population="pop")

    dem.add_population_parameters_change(            # CLOSE it -- the bound
        time=t_old, initial_size=N_old_bl, growth_rate=0.0, population="pop")

    for ev in events:                                # baseline OLDER than window
        if float(ev.time) > t_old:
            _add(ev)

    # ANCESTRAL FLOOR -- the fix for the SNP-density divergence. SLiM recapitation
    # (simulator_context.py:201) creates its population at initial_size=
    # ancestral_ne and to_msprime_events() DELIBERATELY SKIPS the ancestral epoch
    # because recapitation sets it via initial_size. build_demography, using the
    # same to_msprime_events(), therefore inherited NO ancestral size: its deep
    # history settled at N_old_bl (the ~t_old MSMC size) instead of ancestral_ne.
    # Since E[SNPs] = mu * total_branch_length and most branch length lives in the
    # deep coalescent, a wrong ancestral size shifts SNP density systematically --
    # exactly the msprime-vs-SLiM boxplot divergence. Anchor it to ancestral_ne at
    # the oldest epoch boundary so both simulators coalesce under the same size.
    if baseline_msmc is not None:
        anc_ne = float(getattr(baseline_msmc, "ancestral_ne", N_old_bl))
        # oldest epoch boundary in time-ago (sim_length - oldest slim_gen_start);
        # place the floor just past t_old if the epochs are all younger.
        _eps = sorted(baseline_msmc.epochs, key=lambda e: e.slim_gen_start)
        t_anc = max(t_old, float(baseline_msmc.sim_length) - _eps[0].slim_gen_start)
        dem.add_population_parameters_change(
            time=t_anc, initial_size=anc_ne, growth_rate=0.0, population="pop")

    dem.sort_events()
    return dem


def _te_mutations_with_burst(ts, rate, theta, param_idx, seed):
    """Lay TE markers at `rate` (scalar or RateMap), with a transposition BURST.

    log10_t_start / log_duration / log_multiplier define a time window in which
    the insertion rate is elevated. msprime places mutations on branches within
    [start_time, end_time] and requires OLDER mutations to be added FIRST, so the
    layering is baseline-old -> burst -> baseline-recent.
    """
    import msprime
    t_start = 10.0 ** theta[param_idx["log10_t_start"]]
    dur     = float(np.exp(theta[param_idx["log_duration"]]))
    mult    = float(np.exp(theta[param_idx["log_multiplier"]]))
    t_old, t_young = float(t_start), max(0.0, float(t_start) - dur)
    if not (mult > 1.0 and dur > 0 and t_old > t_young):
        return msprime.sim_mutations(ts, rate=rate, random_seed=seed)
    m = msprime.sim_mutations(ts, rate=rate, start_time=t_old, random_seed=seed)
    try:
        boosted = rate * mult
    except TypeError:                       # RateMap: rebuild scaled
        boosted = msprime.RateMap(position=np.asarray(rate.position),
                                  rate=np.asarray(rate.rate) * mult)
    m = msprime.sim_mutations(m, rate=boosted, start_time=t_young,
                              end_time=t_old, random_seed=seed, keep=True)
    return msprime.sim_mutations(m, rate=rate, end_time=t_young,
                                 random_seed=seed, keep=True)


def simulate_window(theta, gw, n_haps, param_idx, cfg, n_segments, baseline_msmc,
                    mean_2ne, rng, nu_log_offset=0.0, nu_clip_neutral=False,
                    nu_start_time=1000.0, nu_duration=1000.0, nu_anchor="present"):
    """One window simulated WHOLE, under a SINGLE nu-distorted demography.

    Design (see the discussion in MFNPE.md, "coarse expansion proxy"):

      * The window is simulated as ONE recombining tree sequence, NOT split into
        independent per-TE segments. This is REQUIRED for the density features:
        _local_te_density / _local_snp_density count within +/-LOCAL_DENSITY_BP
        (250 kb) of each TE, so the TEs and SNPs in a neighbourhood must live on
        SHARED geography / a shared ARG. Independent per-TE segments would make
        TE density structurally ~0 and decorrelate neighbouring flanks.

      * ONE expansion nu for the whole window, from the window's MEAN recomb
        (the alpha_r signal then rides on BETWEEN-window recomb variation, which
        is how the per-TE local_recomb feature is read anyway). msprime cannot
        vary Ne ALONG the sequence within one sim_ancestry call, so within-window
        nu variation would require breaking the ARG -- which breaks density. The
        window is kept short (~1 Mb, set via --window ... upstream) so one nu is
        defensible.

      * The real VARYING recombination map (gw.recomb_map) is still passed to
        sim_ancestry, so linked-diversity structure along the window is realistic
        even though Ne is flat.

      * TWO mutation passes on the SAME tree sequence:
          - SNPs at rate MU               -> neutral flanking variation
          - TE markers at rate lambda_local -> TEs, frequency read from genotypes
        Both inherit the nu-distorted genealogy (the linked-selection analogue:
        in SLiM the flanking SNPs hitchhike on the selected background, so they
        SHOULD carry the distorted signal too). This replaces the old
        per-TE Poisson-count + branch-pick loop, which mis-scaled the count
        (population vs sample) and was O(TEs x nodes).

    KNOWN LIMITATION (coarse by design): real selection gives a diversity DIP,
    deepest at the TE and recovering toward the window edges; the flat-Ne
    expansion gives a PLATEAU of depressed diversity. The proxy captures the
    magnitude via nu, not the spatial gradient. Fine-tuning on SLiM restores the
    shape. Do not "fix" this here -- it is the fidelity gap MF-NPE exists to
    absorb.

    Returns (te_rows, te_pos, snp_rows, snp_pos), positions window-relative.
    gw.recomb_map is an msprime RateMap (cluster only).
    """
    import msprime

    log_lambda = theta[param_idx["log_lambda"]]
    gamma_r    = theta[param_idx["gamma_r"]]
    # nu-direct: these slots carry (log upsilon_global, nu_r).
    # s-proxy  : they carry the REAL (log_s_global, alpha_r); nu_log_offset
    #            = -log(s_ref) converts to log upsilon = log(s_local/s_ref).
    nu_a = theta[param_idx["log_s_global"]]
    nu_b = theta[param_idx["alpha_r"]]

    rtyp, lscale = cfg["recomb_typical"], cfg["recomb_log_scale"]
    mu_snp   = cfg["mu_snp"]
    ratemap  = gw.recomb_map                            # msprime RateMap
    seq_len  = float(gw.end - gw.start)
    block_bp = float(cfg.get("block_size_bp", 100_000))

    rp = np.asarray(ratemap.position, float)
    rr = np.asarray(ratemap.rate, float)
    seg_bp = np.diff(rp)

    def _mean_recomb(lo, hi):
        """bp-weighted mean recombination rate over [lo, hi) of the window."""
        # clip each rate segment to [lo,hi] and weight by overlap length
        seg_lo = np.maximum(rp[:-1], lo); seg_hi = np.minimum(rp[1:], hi)
        w = np.clip(seg_hi - seg_lo, 0, None)
        if w.sum() > 0:
            return float(np.average(rr[:len(w)], weights=w))
        return float(np.mean(rr)) if len(rr) else rtyp

    # --- BLOCK SPLIT: partition the window into block_bp segments, each with its
    # OWN local recombination -> its OWN nu -> its OWN demography/tree. This is
    # what gives alpha_r contrast: alpha_r is the recomb-dependence of SELECTION,
    # and in the proxy selection IS nu (a genealogy distortion), so different
    # local recomb must produce a DIFFERENT genealogy per block -- which a single
    # tree with a mutation-rate map cannot do (that only varies insertion density
    # = gamma_r). We simulate one tree per block and stitch markers in genomic
    # order. LD across block seams is broken; acceptable because the sims are
    # neutral coalescent and the proxy need not be exact. gamma_r still enters via
    # insertion_intensity at each block's local recomb (finer than window-mean).
    # ======================================================================
    # PRF-PRUNE MODE (default): purifying selection by per-TE removal.
    #
    # ONE tree over the whole window under the UNDISTORTED MSMC demography, so
    # the neutral SNP features match SLiM instead of drifting under an expansion.
    # TEs are laid neutrally, then each is pruned with the PRF keep-probability
    # evaluated at ITS OWN local selection coefficient
    #     s_local = exp(log_s_global + alpha_r * feature_r(local recomb))
    # read at that TE's own position. alpha_r therefore acts at per-TE spatial
    # resolution -- finer than the 100 kb blocks, with NO block seams and NO LD
    # breakage. gamma_r acts through a TE-rate map on the same tree.
    #
    # ACCEPTED APPROXIMATION: the genealogy stays neutral, so there is no
    # background-selection dip in the flanks of surviving TEs. he_carrier,
    # tajD_carrier and especially hapext_carrier will therefore lack the local
    # diversity reduction SLiM has. This is a smooth, monotone shortfall -- the
    # kind fine-tuning absorbs -- and is far milder than the expansion proxy's
    # 2-4 SD displacement of every SNP-diversity feature.
    # ======================================================================
    if cfg.get("prune_mode", False):
        dem = build_demography(baseline_msmc, 0.0, cfg["Ne"],
                               nu_start_time, nu_duration, nu_anchor)
        if n_haps % 2 != 0:
            raise ValueError(f"n_haps must be even; got {n_haps}")
        seed = int(rng.integers(1, 2**31 - 1))
        ts = msprime.sim_ancestry(
            samples=n_haps // 2, demography=dem, ploidy=2,
            sequence_length=seq_len, recombination_rate=ratemap,
            random_seed=seed)

        snp_rows, snp_pos = [], []
        ts_snp = msprime.sim_mutations(ts, rate=mu_snp, random_seed=seed)
        for var in ts_snp.variants():
            g = var.genotypes.astype(np.int8)
            c = int(g.sum())
            if 0 < c < n_haps:
                snp_rows.append(g); snp_pos.append(float(var.site.position))

        # TE insertion rate varies along the window with local recomb (gamma_r)
        rate_te = np.array([
            insertion_intensity(rr[k], log_lambda, gamma_r, rtyp, lscale)
            for k in range(len(seg_bp))], float)
        te_rows, te_pos = [], []
        n_pruned = 0
        if np.any(rate_te > 0):
            te_rm = msprime.RateMap(position=rp, rate=np.nan_to_num(rate_te))
            ts_te = _te_mutations_with_burst(
                ts, te_rm, theta, param_idx,
                (seed ^ 0x5DEECE66) & 0x7FFFFFFF)
            g_scale = float(cfg.get("gamma_scale", 4.0))
            ne_eff  = float(cfg.get("prune_ne", 500.0))
            gg, ptab = cfg["prf_gamma_grid"], cfg["prf_table"]
            for var in ts_te.variants():
                g = var.genotypes.astype(np.int8)
                c = int(g.sum())
                if not (0 < c < n_haps):
                    continue
                pos = float(var.site.position)
                k = int(np.clip(np.searchsorted(rp, pos) - 1, 0, len(rr) - 1))
                s_loc = float(np.exp(nu_a + nu_b * feature_r(rr[k], rtyp, lscale)))
                gam = g_scale * ne_eff * s_loc
                if rng.random() < prf_keep_prob(gam, c, gg, ptab):
                    te_rows.append(np.where(g > 0, cfg["TE_STATE"], 0).astype(np.int8))
                    te_pos.append(pos)
                else:
                    n_pruned += 1
        cfg["_last_n_pruned"] = n_pruned
        return te_rows, np.asarray(te_pos), snp_rows, np.asarray(snp_pos)

    n_blocks = max(1, int(np.ceil(seq_len / block_bp)))
    seed0 = int(rng.integers(1, 2**31 - 1))

    if n_haps % 2 != 0:
        raise ValueError(f"n_haps must be even for diploid sampling; got {n_haps}")

    snp_rows, snp_pos = [], []
    te_rows,  te_pos  = [], []
    for bi in range(n_blocks):
        b_lo = bi * block_bp
        b_hi = min((bi + 1) * block_bp, seq_len)
        b_len = b_hi - b_lo
        if b_len <= 0:
            continue
        loc_r = _mean_recomb(b_lo, b_hi)                 # local recomb for this block

        # local nu / demography from THIS block's recomb (alpha_r acts here)
        log_ups = float(expansion_log_upsilon(
            nu_a, nu_b, loc_r, rtyp, lscale,
            log_offset=nu_log_offset, clip_at_neutral=nu_clip_neutral,
            s_gain=cfg.get("nu_s_gain", 1.0),
            alpha_gain=cfg.get("nu_alpha_gain", 1.0)))
        dem = build_demography(baseline_msmc, log_ups, cfg["Ne"],
                               nu_start_time, nu_duration, nu_anchor)

        b_seed = (seed0 + bi * 2_654_435_761) & 0x7FFFFFFF
        # constant recomb within the block (its local mean); tree is per-block
        ts = msprime.sim_ancestry(
            samples=n_haps // 2, demography=dem, ploidy=2,
            sequence_length=b_len, recombination_rate=max(loc_r, 0.0),
            random_seed=max(1, b_seed))

        # SNPs at MU
        ts_snp = msprime.sim_mutations(ts, rate=mu_snp, random_seed=max(1, b_seed))
        for var in ts_snp.variants():
            g = var.genotypes.astype(np.int8)
            s_ = int(g.sum())
            if 0 < s_ < n_haps:
                snp_rows.append(g)
                snp_pos.append(b_lo + float(var.site.position))   # -> genomic order

        # TE markers at lambda_local (gamma_r via local recomb), same block tree.
        # TRANSPOSITION BURST: log10_t_start / log_duration / log_multiplier
        # define a time window in which the insertion rate is elevated. msprime
        # places mutations on branches within [start_time, end_time], so a burst
        # is baseline lambda everywhere EXCEPT an elevated rate during the burst
        # interval. Multiple sim_mutations calls with start_time/end_time layer
        # the rates; msprime requires OLDEST mutations added FIRST, so we go
        # baseline-old -> burst -> baseline-recent. This is what makes
        # log10_t_start / log_multiplier recoverable (previously flat: the sim
        # ignored them, so the network had no signal to invert).
        lam_te = float(insertion_intensity(loc_r, log_lambda, gamma_r, rtyp, lscale))
        if lam_te > 0:
            t_burst_start = 10.0 ** theta[param_idx["log10_t_start"]]   # gen ago (old edge)
            burst_dur     = np.exp(theta[param_idx["log_duration"]])    # gen
            mult          = np.exp(theta[param_idx["log_multiplier"]])  # rate x
            t_burst_old   = t_burst_start                # older boundary (gen ago)
            t_burst_young = max(0.0, t_burst_start - burst_dur)  # younger boundary
            te_seed = max(1, (b_seed ^ 0x5DEECE66) & 0x7FFFFFFF)
            if mult > 1.0 and burst_dur > 0 and t_burst_old > t_burst_young:
                # layer OLDEST first: baseline older than the burst ...
                mts = msprime.sim_mutations(ts, rate=lam_te,
                                            start_time=t_burst_old, random_seed=te_seed)
                # ... elevated rate DURING the burst ...
                mts = msprime.sim_mutations(mts, rate=lam_te * mult,
                                            start_time=t_burst_young, end_time=t_burst_old,
                                            random_seed=te_seed, keep=True)
                # ... baseline more recent than the burst.
                ts_te = msprime.sim_mutations(mts, rate=lam_te,
                                              end_time=t_burst_young,
                                              random_seed=te_seed, keep=True)
            else:
                # no burst (multiplier 1 or degenerate window): constant lambda
                ts_te = msprime.sim_mutations(ts, rate=lam_te, random_seed=te_seed)
            for var in ts_te.variants():
                g = var.genotypes.astype(np.int8)
                s_ = int(g.sum())
                if 0 < s_ < n_haps:
                    te_rows.append(np.where(g > 0, cfg["TE_STATE"], 0).astype(np.int8))
                    te_pos.append(b_lo + float(var.site.position))  # -> genomic order

    # markers are appended block-by-block, so already in genomic order
    return te_rows, np.asarray(te_pos), snp_rows, np.asarray(snp_pos)


def debug_window(theta, gw, n_haps, param_idx, cfg, baseline_msmc, mean_2ne,
                 nu_log_offset, nu_clip_neutral, nu_start_time, nu_duration,
                 nu_anchor):
    """Print raw diagnostics for ONE window vs analytic expectation, so a run
    can be sanity-checked instead of inferred. Compares the realised segregating-
    site count to the Watterson expectation E[S] = theta * L * sum(1/i), which is
    the reference that breaks the msprime-vs-SLiM symmetry: whichever matches
    theory is correct. Also dumps the demography and the nu actually applied.
    """
    import msprime
    rng = np.random.default_rng(12345)
    te_rows, te_pos, snp_rows, snp_pos = simulate_window(
        theta, gw, n_haps, param_idx, cfg, 1, baseline_msmc, mean_2ne, rng,
        nu_log_offset, nu_clip_neutral, nu_start_time, nu_duration, nu_anchor)

    # recompute the pieces the function used, for reporting
    rtyp, lscale = cfg["recomb_typical"], cfg["recomb_log_scale"]
    rp = np.asarray(gw.recomb_map.position, float)
    rr = np.asarray(gw.recomb_map.rate, float)
    seg_bp = np.diff(rp)
    win_mean_r = (float(np.average(rr[:len(seg_bp)], weights=seg_bp))
                  if seg_bp.sum() > 0 else float(np.mean(rr)))
    log_ups = float(expansion_log_upsilon(
        theta[param_idx["log_s_global"]], theta[param_idx["alpha_r"]],
        win_mean_r, rtyp, lscale,
        log_offset=nu_log_offset, clip_at_neutral=nu_clip_neutral,
        s_gain=cfg.get("nu_s_gain", 1.0), alpha_gain=cfg.get("nu_alpha_gain", 1.0)))
    dem = build_demography(baseline_msmc, log_ups, cfg["Ne"],
                           nu_start_time, nu_duration, nu_anchor)
    L = float(gw.end - gw.start)
    mu = cfg["mu_snp"]

    # analytic neutral expectation (constant-Ne Watterson); only exact when the
    # demography is flat, but a useful order-of-magnitude anchor otherwise.
    Ne_eff = _msmc_size_at(baseline_msmc, 0.0, cfg["Ne"])
    harmonic = float(np.sum(1.0 / np.arange(1, n_haps)))
    theta_wat = 4.0 * Ne_eff * mu                      # per bp, DIPLOID convention
    exp_S = theta_wat * L * harmonic
    exp_S_hap = 2.0 * Ne_eff * mu * L * harmonic       # if msprime ploidy=1 halves it

    print("=" * 70)
    print("DEBUG WINDOW")
    print("  --- parameters (theta) ---")
    for _nm, _v in zip(list(cfg.get("param_names", [])) or
                       [f"theta[{k}]" for k in range(len(theta))], theta):
        _ex = ""
        if _nm in ("log_lambda","log_s_global","log_mean_s"): _ex = f"  -> {np.exp(_v):.3e}"
        if _nm == "log_gamma_shape": _ex = f"  -> shape={np.exp(_v):.3f}"
        if _nm.startswith("log10_t"): _ex = f"  -> {10**_v:.0f} gen"
        if _nm == "log_duration": _ex = f"  -> {np.exp(_v):.1f} gen"
        if _nm == "log_multiplier": _ex = f"  -> {np.exp(_v):.2f}x"
        print(f"      {_nm:16s} = {_v:+.4f}{_ex}")
    print(f"  chrom/start/end : {gw.chrom}:{gw.start}-{gw.end}  (L={L:.0f} bp)")
    print(f"  n_haps (samples): {n_haps}")
    print(f"  win_mean_recomb : {win_mean_r:.3e}")
    print(f"  Ne (present)    : {Ne_eff:.1f}   mu={mu:.3e}")
    print(f"  gains           : s_gain={cfg.get('nu_s_gain',1.0):.2f}  "
          f"alpha_gain={cfg.get('nu_alpha_gain',1.0):.2f}  "
          f"(upsilon = (s/s_ref)**s_gain)")
    print(f"  log_upsilon     : {log_ups:+.3f}  (upsilon={np.exp(log_ups):.3f}, "
          f"r*2Ne={np.log(np.exp(log_ups))*2*Ne_eff/nu_duration:+.2f})")
    print(f"  demography epochs (msprime view):")
    try:
        base_N = dem.populations[0].initial_size
        base_r = dem.populations[0].growth_rate
        print(f"      base pop: initial_size={base_N} (present-day) "
              f"growth_rate={base_r:.6g}")
        # The growth lives on the base population and acts BACKWARD from the
        # present, so the epoch events alone look flat. Show the actual size
        # trajectory N(t) = N_present * exp(-r*t) over the expansion window,
        # which is what the coalescent actually sees.
        if abs(base_r) > 1e-12:
            print(f"      size trajectory (present-anchored growth):")
            for _t in (0, nu_duration * 0.25, nu_duration * 0.5,
                       nu_duration * 0.75, nu_duration):
                _Nt = base_N * np.exp(-base_r * _t)
                print(f"          t={_t:8.1f} gen ago  ->  N={_Nt:8.1f}")
        for pc in dem.events[:8]:
            print(f"      event t={getattr(pc,'time','?'):>10.1f}  "
                  f"N={getattr(pc,'initial_size','?')}  "
                  f"growth={getattr(pc,'growth_rate','?')}")
    except Exception as e:
        print(f"      (could not introspect demography: {e})")
    # per-block nu gradient (the alpha_r signal) + burst window (the t_start signal)
    try:
        block_bp = float(cfg.get("block_size_bp", 100_000))
        n_blk = max(1, int(np.ceil(L / block_bp)))
        print(f"  --- BLOCK STRUCTURE ({n_blk} blocks of {block_bp/1e3:.0f}kb) ---")
        rp2 = np.asarray(gw.recomb_map.position, float)
        rr2 = np.asarray(gw.recomb_map.rate, float)
        def _mr(lo, hi):
            slo = np.maximum(rp2[:-1], lo); shi = np.minimum(rp2[1:], hi)
            w = np.clip(shi - slo, 0, None)
            return float(np.average(rr2[:len(w)], weights=w)) if w.sum() > 0 else rtyp
        lus = []
        for bi in range(min(n_blk, 5)):
            blo = bi*block_bp; bhi = min((bi+1)*block_bp, L)
            lr = _mr(blo, bhi)
            lub = float(expansion_log_upsilon(
                theta[param_idx["log_s_global"]], theta[param_idx["alpha_r"]],
                lr, rtyp, lscale, log_offset=nu_log_offset, clip_at_neutral=nu_clip_neutral,
                s_gain=cfg.get("nu_s_gain", 1.0), alpha_gain=cfg.get("nu_alpha_gain", 1.0)))
            lus.append(lub)
            print(f"      block {bi}: local_recomb={lr:.2e} -> log_ups={lub:+.3f} "
                  f"(ups={np.exp(lub):.2f})")
        if n_blk > 5: print(f"      ... ({n_blk-5} more blocks)")
        spread = (max(lus)-min(lus)) if len(lus) > 1 else 0.0
        print(f"      nu spread across blocks: {spread:.3f} in log_ups "
              f"<- alpha_r CONTRAST (0 = no within-window variation)")
        # burst window
        t_start = 10.0**theta[param_idx["log10_t_start"]]
        dur = np.exp(theta[param_idx["log_duration"]])
        mult = np.exp(theta[param_idx["log_multiplier"]])
        if mult > 1.0 and dur > 0:
            print(f"  --- TE BURST --- window [{max(0,t_start-dur):.0f}, {t_start:.0f}] "
                  f"gen ago, rate x{mult:.2f} (log10_t_start/duration/multiplier)")
        else:
            print(f"  --- TE BURST --- none (multiplier={mult:.2f}<=1); constant lambda")
    except Exception as e:
        print(f"  (block/burst introspection failed: {e})")
    if cfg.get("prune_mode", False):
        gsc = float(cfg.get("gamma_scale", 4.0)); nef = float(cfg.get("prune_ne", 500.0))
        s_g = float(np.exp(theta[param_idx["log_s_global"]]))
        a_r = float(theta[param_idx["alpha_r"]])
        print(f"  --- PRF PRUNE (selection by per-TE removal; no nu distortion) ---")
        print(f"      gamma = {gsc:g} * Ne({nef:.0f}) * s_local")
        print(f"      s_global={s_g:.3e} -> gamma_global={gsc*nef*s_g:.2f}")
        gg, ptab = cfg["prf_gamma_grid"], cfg["prf_table"]
        print(f"      keep-prob at gamma_global: p(singleton)="
              f"{prf_keep_prob(gsc*nef*s_g,1,gg,ptab):.3f}  "
              f"p(i=10)={prf_keep_prob(gsc*nef*s_g,10,gg,ptab):.3f}  "
              f"p(i=50)={prf_keep_prob(gsc*nef*s_g,50,gg,ptab):.4f}")
        for _r in (1e-8, 1e-7, 1e-6):
            _s = s_g * np.exp(a_r * feature_r(_r, rtyp, lscale))
            print(f"      recomb={_r:.0e} -> s_local={_s:.2e} "
                  f"gamma={gsc*nef*_s:7.2f} p(i=10)="
                  f"{prf_keep_prob(gsc*nef*_s,10,gg,ptab):.3f}")
        print(f"      -> alpha_r contrast now acts PER TE at its own position")
        print(f"      TEs pruned this window: {cfg.get('_last_n_pruned','?')}")
    print(f"  --- SNPs ---")
    print(f"    realised segregating sites : {len(snp_pos)}")
    print(f"    Watterson E[S] (diploid 4Ne): {exp_S:.0f}")
    print(f"    Watterson E[S] (haploid 2Ne): {exp_S_hap:.0f}")
    if len(snp_pos):
        ratio_d = len(snp_pos)/exp_S if exp_S else float('nan')
        ratio_h = len(snp_pos)/exp_S_hap if exp_S_hap else float('nan')
        # --- SFS SHAPE: Tajima's D, pi, singleton fraction ---
        # This is the diagnostic that matters for the anchor/duration choice.
        # Level (SNP count) can stay near neutral while SHAPE skews toward rare:
        # a short recent expansion piles up singletons (D << 0) without inflating
        # diversity, which is the purifying-selection SFS mimic (Xue et al.).
        G = np.asarray(snp_rows)                       # (S, n_haps) 0/1
        S = G.shape[0]
        ac = G.sum(1)                                  # derived-allele count per site
        n = n_haps
        # pi = mean pairwise differences = sum 2*p*q*n/(n-1) over sites, per site
        pi_persite = np.mean(2.0 * ac * (n - ac) / (n * (n - 1)))
        pi_total = pi_persite * S
        a1 = float(np.sum(1.0 / np.arange(1, n)))
        a2 = float(np.sum(1.0 / np.arange(1, n) ** 2))
        theta_w = S / a1
        # Tajima's D
        b1 = (n + 1) / (3 * (n - 1)); b2 = 2 * (n*n + n + 3) / (9 * n * (n - 1))
        c1 = b1 - 1/a1; c2 = b2 - (n + 2)/(a1*n) + a2/(a1*a1)
        e1 = c1/a1; e2 = c2/(a1*a1 + a2)
        varD = e1*S + e2*S*(S-1)
        tajD = (pi_total - theta_w)/np.sqrt(varD) if varD > 0 else float('nan')
        singleton_frac = np.mean((ac == 1) | (ac == n - 1))
        print(f"    realised / E[S] diploid    : {ratio_d:.3f}  "
              f"<- LEVEL: ~1.0 = matches 4Ne theory (only exact for FLAT Ne)")
        print(f"    realised / E[S] haploid    : {ratio_h:.3f}")
        print(f"    --- SFS SHAPE ---")
        print(f"    pi (per bp)                : {pi_total/L:.3e}  "
              f"(neutral expect ~{4*Ne_eff*mu:.3e})")
        print(f"    Tajima's D                 : {tajD:+.3f}  "
              f"<- SHAPE: strongly NEGATIVE = singleton excess = selection mimic")
        print(f"    singleton fraction         : {singleton_frac:.3f}  "
              f"(neutral ~{2.0/(n*a1):.3f})")
        print(f"    ==> DECOUPLING CHECK: want LEVEL ratio ~1 (diversity preserved)")
        print(f"        AND Tajima's D << 0 (shape skewed). Short recent expansion")
        print(f"        (ancestral anchor, small --nu-duration) targets exactly this.")
    print(f"  --- TEs ---")
    print(f"    realised segregating TEs   : {len(te_pos)}")
    # TE frequency spectrum -- the thing selection ACTUALLY distorts (the SNP
    # stats above are the linked/flanking context). Guarded for the low-TE case:
    # a strong expansion + low lambda can leave only a handful of TEs, so every
    # statistic checks the count first and reports "n/a (too few TEs)" rather
    # than dividing by zero or reporting a meaningless D from 2 sites.
    n_te = len(te_rows)
    if n_te == 0:
        print(f"    (no segregating TEs -- cannot compute TE-SFS. Expected when the "
              f"expansion is strong and lambda low; not an error.)")
    else:
        Gte = np.asarray(te_rows)                       # (n_te, n_haps), TE_STATE/0
        ac_te = (Gte > 0).sum(1)                        # carrier count per TE
        n = n_haps
        mean_freq = float(np.mean(ac_te)) / n
        singleton_frac_te = float(np.mean(ac_te == 1))
        print(f"    mean TE frequency          : {mean_freq:.3f}  "
              f"(lower = rarer = stronger purifying-selection mimic)")
        print(f"    TE singleton fraction      : {singleton_frac_te:.3f}"
              + ("" if n_te >= 20 else
                 f"   [n={n_te}: noisy, treat as indicative only]"))
        # Tajima's-D-style skew on the TE spectrum, only when enough sites.
        if n_te >= 10:
            S = n_te
            pi = float(np.mean(2.0 * ac_te * (n - ac_te) / (n * (n - 1)))) * S
            a1 = float(np.sum(1.0 / np.arange(1, n)))
            a2 = float(np.sum(1.0 / np.arange(1, n) ** 2))
            tw = S / a1
            b1 = (n + 1) / (3 * (n - 1)); b2 = 2 * (n*n + n + 3) / (9 * n * (n - 1))
            c1 = b1 - 1/a1; c2 = b2 - (n + 2)/(a1*n) + a2/(a1*a1)
            e1 = c1/a1; e2 = c2/(a1*a1 + a2)
            var = e1*S + e2*S*(S - 1)
            tajD_te = (pi - tw)/np.sqrt(var) if var > 0 else float('nan')
            print(f"    TE Tajima's D              : {tajD_te:+.3f}  "
                  f"<- negative = TE spectrum skewed rare (selection mimic working)")
        else:
            print(f"    TE Tajima's D              : n/a (need >=10 TEs, have {n_te})")

    # --- REACHABLE-RANGE CHECK: does the s prior still span a useful distortion
    # range at this --nu-duration? At short duration the s->distortion map can
    # SATURATE (every s gives an extreme r*2Ne), collapsing the selection signal.
    # Sweep s across its prior and report the resulting r*2Ne spread.
    try:
        from te_dfe.config import LOG_S_GLOBAL_RANGE
        s_lo, s_hi = np.exp(LOG_S_GLOBAL_RANGE[0]), np.exp(LOG_S_GLOBAL_RANGE[1])
    except Exception:
        s_lo, s_hi = 3e-4, 0.1
    print(f"  --- REACHABLE-RANGE CHECK (nu_duration={nu_duration:.0f}) ---")
    print(f"    s prior [{s_lo:.1e}, {s_hi:.1e}] -> r*2Ne (the SFS distortion):")
    s_ref = np.exp(-nu_log_offset) if nu_log_offset != 0 else 1e-3
    prev = None; saturated = 0
    for s_val in np.geomspace(s_lo, s_hi, 6):
        lu = expansion_log_upsilon(np.log(s_val), 0.0, win_mean_r, rtyp, lscale,
                                   log_offset=nu_log_offset,
                                   clip_at_neutral=nu_clip_neutral,
                                   s_gain=cfg.get("nu_s_gain", 1.0),
                                   alpha_gain=cfg.get("nu_alpha_gain", 1.0))
        r2ne = lu * 2 * Ne_eff / nu_duration
        tag = ""
        if abs(r2ne) > 20: tag = " (SATURATED - too strong to distinguish)"; saturated += 1
        if prev is not None and abs(r2ne - prev) < 0.1: tag += " (flat vs previous)"
        print(f"      s={s_val:.2e} -> r*2Ne={r2ne:+7.2f}{tag}")
        prev = r2ne
    if saturated >= 3:
        print(f"    !! {saturated}/6 s values saturate: --nu-duration too SHORT for "
              f"this s range. Either lengthen duration or narrow the s prior so the "
              f"map spans a usable range instead of pinning everything at 'strong'.")
    else:
        print(f"    OK: the s prior maps to a spread of distortion strengths "
              f"(good - the selection signal is resolvable).")
    print("=" * 70)
    return len(snp_pos), len(te_pos)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-replicates", type=int, default=None,
                   help="Replicates to simulate. Required unless --debug "
                        "(debug runs a single window and exits).")
    p.add_argument("--n-windows",    type=int, default=20)
    p.add_argument("--block-size", type=int, default=100_000,
                   help="Sub-window block size in bp (default 100kb). Each block gets its own local recomb -> own nu -> own tree, giving alpha_r (the recomb-dependence of selection) within-window contrast. Set >= window-size to recover the old single-tree behaviour.")
    p.add_argument("--prf-backend", choices=["analytic", "moments"],
                   default="analytic",
                   help="How p_keep is computed. 'analytic' uses the constant-Ne "
                        "Wright-Fisher/PRF ratio -- exact for a flat history, and an "
                        "approximation otherwise (it assumes selection's RELATIVE "
                        "distortion is demography-independent). 'moments' integrates "
                        "both the selected and neutral spectra through the ACTUAL "
                        "MSMC history and takes their ratio, dropping that "
                        "assumption; needs the `moments` package.")
    p.add_argument("--prf-dominance", type=float, default=0.5,
                   help="Dominance h for the moments backend (0.5 = additive). "
                        "Match the SLiM sims.")
    p.add_argument("--prf-selftest", action="store_true",
                   help="Build BOTH backends and compare, then exit. Under a "
                        "constant-Ne history they must agree; a mismatch means the "
                        "gamma/dominance convention is wrong. Run this ONCE before "
                        "trusting the moments backend.")
    p.add_argument("--gamma-scale", type=float, default=4.0,
                   help="gamma = GAMMA_SCALE * Ne * s_local, the scaled selection "
                        "coefficient driving the PRF prune probability. 4.0 is the "
                        "standard 4*Ne*s for genic/additive selection where s is the "
                        "HETEROZYGOTE effect. MUST match the dominance convention of "
                        "the SLiM sims being paired with (if SLiM uses homozygous s "
                        "with h=0.5, pass 2.0).")
    p.add_argument("--prune-ne", type=float, default=None,
                   help="Ne used in gamma. Default: harmonic mean of the MSMC epochs "
                        "(the size that governs coalescence when the history varies).")
    p.add_argument("--s-gain", type=float, default=1.0,
                   help="Exponent on the s -> upsilon map: upsilon = (s/s_ref)**s_gain. "
                        "1.0 = strictly proportional (the old behaviour). 2.0 means "
                        "s=1e-2 with s_ref=1e-3 gives upsilon=100 instead of 10. "
                        "upsilon=1 at s=s_ref is preserved for any gain, so the null "
                        "stays exactly nested. Raises TOTAL size change (unlike "
                        "--nu-duration, which changes the window at fixed size change).")
    p.add_argument("--alpha-gain", type=float, default=1.0,
                   help="Multiplier on the alpha_r * feature_r term ONLY. Accentuates "
                        "the recombination contrast without inflating overall selection "
                        "strength -- the one thing --nu-duration cannot do, since it "
                        "scales both terms together.")
    p.add_argument("--theta-from", type=str, default=None,
                   help="Draw theta from an EXISTING dataset instead of the prior, "
                        "so msprime replicate i uses the SAME theta as SLiM sim i. "
                        "Pass a dataset base (…/dataset -> reads <base>.theta.pt), a "
                        ".theta.pt directly, or a .npy of theta vectors. Paired theta "
                        "makes the comparison apples-to-apples: a feature difference "
                        "is then the SIMULATOR, not a different draw.")
    p.add_argument("--theta", type=str, default=None,
                   help="Comma-separated theta to use INSTEAD of a random "
                        "draw, in PARAM_NAMES order. Lets you test a specific "
                        "point (e.g. the neutral corner: log_s at floor, "
                        "alpha_r=0, gamma_r=0). Length must equal len(PARAM_NAMES). "
                        "Most useful with --debug.")
    p.add_argument("--debug", action="store_true",
                   help="Run ONE window with verbose diagnostics (realised "
                        "SNP/TE counts vs Watterson expectation, the applied "
                        "nu, and the msprime demography) then exit. Use to "
                        "check the sim matches theory before a full run.")
    p.add_argument("--task-id",      type=int, default=None,
                   help="SLURM array index -> replicate (sharded).")
    p.add_argument("--master-seed",  type=int, default=777)
    p.add_argument("--window-size", type=int, default=None,
                   help="Window length in bp for THIS msprime run, overriding "
                        "config WINDOW_SIZE. Use ~1e6: short enough that one nu "
                        "per window is defensible, >=500kb so the +/-250kb "
                        "density window fits. Threaded to BOTH the WindowSampler "
                        "(sampled span) and the processor (density denominator). "
                        "Leaves the config (and your 5Mb SLiM set) untouched.")
    p.add_argument("--n-segments",   type=int, default=8,  # DEPRECATED, ignored
                   help="DEPRECATED / IGNORED. The window is now simulated whole under a single nu (one ARG), required by the +/-250kb density features. alpha_r rides on between-window recomb variation."
                        "keep each segment >= 2*LOCAL_DENSITY_BP).")
    p.add_argument("--window-sampling", choices=["stratified", "independent"],
                   default="stratified",
                   help="How windows are drawn when no --window-manifest is given. "
                        "MUST MATCH the script that produced your SLiM set, or "
                        "pretrain and fine-tune see different genomic landscapes. "
                        "stratified (default) = sampler.sample_n(cover_covariate_"
                        "space=True), what 01_simulate.py (non-sharded) does. "
                        "independent = sampler.sample() per window, what "
                        "01_simulate_shard.py does via window_seeds. Ignored when "
                        "a manifest is supplied.")
    p.add_argument("--window-manifest", type=str, default=None,
                   help="stratified window manifest from 00_sample_windows.py "
                        "(REQUIRED for sharded runs so windows are reproducible).")
    p.add_argument("--param-mode", choices=["prf-prune", "nu-direct", "s-proxy"],
                   default="prf-prune",
                   help="nu-direct: draw log_nu_global/nu_r over --nu-range into "
                        "the log_s_global/alpha_r slots; nu IS the parameter and "
                        "is mapped to selection post hoc by the SLiM calibration "
                        "grid. s-proxy: keep the prior's REAL theta and convert "
                        "s -> nu internally with --s-to-growth; theta stays "
                        "comparable to SLiM, which is what MF-NPE pretraining "
                        "needs (see MFNPE.md).")
    p.add_argument("--s-ref", type=float, default=1e-3,
                   help="s-proxy only: the selection coefficient mapped to "
                        "upsilon = 1, i.e. to NO distortion beyond the null. "
                        "Default 1e-3 = the drift barrier 1/(2Ne) at the rescaled "
                        "Ne=500, which is the principled choice: below it "
                        "selection is weaker than drift and genuinely cannot act, "
                        "so upsilon is floored at 1 there. Note the s prior is "
                        "[3e-4, 0.1], so its bottom sits BELOW the barrier and is "
                        "correctly flat/unidentifiable.")
    p.add_argument("--nu-start-time", type=float, default=1000.0,
                   help="Older edge of the expansion window, generations ago. "
                        "Must be <~ 2Ne or lineages have coalesced before reaching "
                        "it (that is what limited Xue et al. to weak selection). "
                        "Default 1000 = 2Ne at the rescaled Ne=500.")
    p.add_argument("--nu-duration", type=float, default=1000.0,
                   help="Length of the expansion window in generations. THIS is "
                        "the strength lever: r*2Ne = log(upsilon)*2Ne/duration. "
                        "Default 1000 (= 2Ne, window reaching the present) gives "
                        "r*2Ne = log(upsilon), so --nu-range [-2,4] spans "
                        "r*2Ne in [-2,4]. MSMC events inside the window are "
                        "dropped and replaced by the exponential.")
    p.add_argument("--nu-anchor", choices=["present", "ancestral"],
                   default="present",
                   help="Which edge of the window is pinned to the MSMC baseline. "
                        "present (default): young edge pinned, so high-upsilon "
                        "segments have FEWER SNPs -- the same sign as selection. "
                        "ancestral: old edge pinned, giving high-upsilon segments "
                        "MORE present-day diversity, the OPPOSITE sign to "
                        "selection (a likely negative-transfer source).")
    p.add_argument("--nu-range", default="-2,4",
                   help="Uniform prior bounds for log_nu_global. SET FROM the "
                        "calibration diagnostics (the useful nu span).")
    p.add_argument("--nu-r-range", default="-1,1",
                   help="Uniform prior bounds for nu_r (recomb slope of nu).")
    p.add_argument("--out-dir",      type=str, default="sim_shards_msprime/")
    p.add_argument("--overwrite",    action="store_true")
    args = p.parse_args()
    if args.debug:
        args.n_replicates = args.n_replicates or 1
    elif args.n_replicates is None:
        p.error("--n-replicates is required (unless --debug).")
    return args


def main():
    args = parse_args()
    from model_select import prescan_model
    prescan_model()
    from te_dfe.config import (PARAM_NAMES, RECOMB_MAP_PATH, ANNOTATION_PATH,
                               ANNOTATION_FMT, CHROMOSOMES, WINDOW_SIZE, MSMC_PATH, MU,
                               SIM_LENGTH)
    from te_dfe import config_base as CB
    from te_dfe.prior import DFEBurstPrior
    from te_dfe.processor import TEDFEBurstProcessor, TE_STATE
    from te_dfe.demography import (MSMCDemography, GenomeAnnotation, WindowSampler)

    param_idx = {n: i for i, n in enumerate(PARAM_NAMES)}
    for need in ("log_lambda", "gamma_r", "log_s_global", "alpha_r"):
        if need not in param_idx:
            sys.exit(f"ERROR: PARAM_NAMES lacks {need}; needs the context model.")

    annot = (GenomeAnnotation.from_gtf(ANNOTATION_PATH)
             if ANNOTATION_FMT == "gtf" else GenomeAnnotation.from_gff(ANNOTATION_PATH))
    sampler = WindowSampler(recomb_map_path=RECOMB_MAP_PATH, annotation=annot,
                            chromosomes=CHROMOSOMES, window_size=(int(args.window_size)
                                if args.window_size else WINDOW_SIZE))
    baseline_msmc = (MSMCDemography.from_file(MSMC_PATH, mu=MU, sim_length=SIM_LENGTH)
                     if MSMC_PATH else None)
    # E[n_te] = lambda * seg_len * integrated_2Ne()  (the population-level
    # conversion documented on MSMCDemography.integrated_2Ne and used verbatim by
    # simulator_context.expected_n_te_baseline). integrated_2Ne() ALREADY returns
    # the full integral in haploid-generations -- do NOT divide by WINDOW_SIZE.
    # A previous version did, under-counting TEs by ~WINDOW_SIZE/seg_len and
    # producing "no TEs" replicates. Fallback for the no-MSMC case integrates a
    # constant 2*Ne over the simulation duration, same units.
    if baseline_msmc is not None:
        mean_2ne = baseline_msmc.integrated_2Ne()
    else:
        mean_2ne = 2.0 * getattr(CB, "NE", 10_000) * float(SIM_LENGTH)

    # paired-theta table (--theta-from): load once, index by replicate_id
    _theta_table = None
    if args.theta_from is not None:
        _tp = Path(args.theta_from)
        if _tp.suffix == ".npy":
            _theta_table = np.asarray(np.load(_tp), dtype=np.float64)
        else:
            import torch as _torch
            _cand = _tp if _tp.suffix == ".pt" else _tp.with_suffix(".theta.pt")
            _m = _torch.load(_cand, map_location="cpu", weights_only=False)
            _theta_table = np.asarray(_m["thetas"], dtype=np.float64)
        if _theta_table.ndim != 2 or _theta_table.shape[1] != len(PARAM_NAMES):
            sys.exit(f"--theta-from: expected (N,{len(PARAM_NAMES)}) thetas, got "
                     f"{_theta_table.shape}. Wrong model or wrong dataset?")
        print(f"[theta-from] {_theta_table.shape[0]} paired theta vectors "
              f"from {args.theta_from}")

    n_haps = getattr(CB, "N_HAPS", getattr(CB, "N_SAMPLES", 100))
    # ---- PRF prune setup (default mode) ----
    _prune = (args.param_mode == "prf-prune")
    _prune_ne = (args.prune_ne if args.prune_ne is not None
                 else harmonic_mean_ne(baseline_msmc, getattr(CB, "NE", 10_000)))
    _gam_grid = np.concatenate([[0.0], np.geomspace(0.02, 2000.0, 96)])
    if args.prf_selftest:
        prf_selftest(n_haps, _gam_grid, baseline_msmc, _prune_ne,
                     args.gamma_scale, h=args.prf_dominance)
        return 0
    if not _prune:
        _prf_tab = None
    elif args.prf_backend == "moments":
        try:
            _prf_tab = build_prf_keep_table_moments(
                n_haps, _gam_grid, baseline_msmc, _prune_ne,
                args.gamma_scale, h=args.prf_dominance)
        except ImportError:
            print("[prf] `moments` not installed -- falling back to the analytic "
                  "backend (exact only for a constant-Ne history).", file=sys.stderr)
            _prf_tab = build_prf_keep_table(n_haps, _gam_grid)
    else:
        _prf_tab = build_prf_keep_table(n_haps, _gam_grid)
    if _prune:
        print(f"[prf-prune] gamma = {args.gamma_scale:g} * Ne * s_local, "
              f"Ne={_prune_ne:.1f} (harmonic mean)  -> "
              f"gamma spans [{args.gamma_scale*_prune_ne*3e-4:.2f}, "
              f"{args.gamma_scale*_prune_ne*0.1:.0f}] across the s prior")

    cfg = dict(param_names=list(PARAM_NAMES), block_size_bp=args.block_size,
               prune_mode=_prune, gamma_scale=args.gamma_scale,
               prune_ne=_prune_ne, prf_gamma_grid=_gam_grid, prf_table=_prf_tab,
               nu_s_gain=args.s_gain, nu_alpha_gain=args.alpha_gain,
               Ne=getattr(CB, "NE", 10_000), mu_snp=MU,
               recomb_typical=CB.RECOMB_TYPICAL, recomb_log_scale=CB.RECOMB_LOG_SCALE,
               TE_STATE=TE_STATE)
    nu_lo, nu_hi   = [float(x) for x in args.nu_range.split(",")]
    nur_lo, nur_hi = [float(x) for x in args.nu_r_range.split(",")]

    win_size = int(args.window_size) if args.window_size else int(WINDOW_SIZE)
    if win_size < 2 * 250_000:
        print(f"WARNING: --window-size {win_size} < 2*LOCAL_DENSITY_BP (500kb); "
              "every TE is an edge TE and density features rely entirely on the "
              "edge-rescale.", file=sys.stderr)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    prior = DFEBurstPrior()
    # Processor density denominator MUST match the simulated window length,
    # else _edge_rescale / the +/-250kb bound use the wrong denominator.
    proc = TEDFEBurstProcessor(maf_min=getattr(CB, "MAF_MIN", 0.0),
                               window_size=win_size)

    manifest = (np.load(args.window_manifest, allow_pickle=True)
                if args.window_manifest is not None else None)

    reps = ([args.task_id] if args.task_id is not None
            else list(range(args.n_replicates)))

    for replicate_id in reps:
        t0 = time.time()
        # Name to match 02_collate.py's glob ("sim_*.npz"); the _msprime tag keeps
        # these distinguishable from SLiM shards in a shared directory.
        shard_path = out_dir / f"sim_{replicate_id:06d}_msprime.npz"
        if shard_path.exists() and not args.overwrite:
            print(f"[rep{replicate_id}] exists, skip"); continue

        # DFEBurstPrior.sample is a STATIC method taking an rng positionally
        # (te_dfe/prior_context.py:82: sample(rng, ne_harmonic=None)), NOT seed=.
        # Match 01_simulate_shard.py: build a default_rng from a SeedSequence so
        # the draw is reproducible and independent per replicate.
        if args.theta_from is not None:
            if replicate_id >= len(_theta_table):
                print(f"[rep{replicate_id}] beyond theta table "
                      f"({len(_theta_table)} rows) - skipping")
                continue
            theta = _theta_table[replicate_id].copy()
        elif args.theta is not None:
            theta = np.asarray([float(x) for x in args.theta.split(",")],
                               dtype=np.float64).reshape(-1)
            if len(theta) != len(PARAM_NAMES):
                sys.exit(f"--theta has {len(theta)} values, need {len(PARAM_NAMES)} "
                         f"({PARAM_NAMES})")
        else:
            _theta_ss = np.random.SeedSequence([int(args.master_seed), replicate_id])
            theta = np.asarray(prior.sample(np.random.default_rng(_theta_ss)),
                               dtype=np.float64).reshape(-1)
        rng = np.random.default_rng([args.master_seed, replicate_id])
        if args.param_mode == "nu-direct":
            # nu IS the parameter: overwrite the two selection slots with draws
            # over the calibrated range. theta is NOT comparable to SLiM theta.
            theta[param_idx["log_s_global"]] = rng.uniform(nu_lo, nu_hi)
            theta[param_idx["alpha_r"]]      = rng.uniform(nur_lo, nur_hi)
        # s-proxy: leave theta exactly as the prior drew it, so these sims are
        # (theta, x) pairs in the SAME theta space as SLiM -> pretraining transfers.

        if manifest is not None:
            sel = manifest["replicate_id"] == replicate_id
            coords = sorted(zip(manifest["window_idx"][sel],
                                manifest["chrom"][sel], manifest["start"][sel]))
        else:
            coords = [(w, None, None) for w in range(args.n_windows)]

        # ADDED: recomb-decile O/E enrichment histograms, summed over this sim's
        # windows (per-sim DENOMINATOR; genome-wide FIXED edges). O = TE count per
        # decile from RAW recomb at TE positions; E = bp per decile from the window
        # rate map. 02_collate carries these to theta_meta; the Dataset forms
        # log(O/E). See te_dfe/enrichment.py. No-op unless ENRICH_OE_DECILES.
        use_oe = getattr(CB, "ENRICH_OE_DECILES", False)
        oe_edges = getattr(CB, "RECOMB_DECILE_EDGES", None)
        if use_oe and oe_edges is None:
            raise ValueError("ENRICH_OE_DECILES is on but RECOMB_DECILE_EDGES is "
                             "unset - run compute_recomb_deciles.py and paste the "
                             "edges into config_base.py (the axis must be FIXED "
                             "and shared across sims, not per-window).")
        if use_oe:
            from te_dfe.enrichment import n_effective_bins
            _nb = n_effective_bins(oe_edges)
        oe_O = np.zeros(_nb) if use_oe else None
        oe_E = np.zeros(_nb) if use_oe else None

        # WINDOW-SAMPLING PARITY. The pretrain and fine-tune sets must see the
        # SAME genomic landscape, or MF-NPE spends its capacity on a covariate
        # shift in x rather than on the fidelity gap in theta.
        #
        #   01_simulate.py (the NON-SHARDED SLiM training path) calls
        #   sim(seed=...) with NO window_seeds, so the simulator takes the
        #   sampler.sample_n(n_win, rng, cover_covariate_space=True) branch:
        #   windows JOINTLY STRATIFIED over the (gene_density, recomb) quintile
        #   grid, one per bin. That deliberately SPANS the covariate space.
        #
        #   01_simulate_shard.py passes window_seeds -> independent draws, which
        #   follow the genome-wide distribution (dominated by the low-recomb
        #   pericentromeric bulk).
        #
        # These are materially different landscapes, so mirror whichever produced
        # the SLiM set. --window-sampling stratified is the default because the
        # non-sharded path is what the existing training data came from.
        if coords and coords[0][1] is None and args.window_sampling == "stratified":
            gws = sampler.sample_n(min(args.n_windows, len(coords)), rng,
                                   cover_covariate_space=True)
        else:
            gws = None

        all_feats = []
        if args.debug:
            # one window, verbose, then exit
            if coords and coords[0][1] is not None:
                gw_dbg = sampler.window_from_coords(str(coords[0][1]), int(coords[0][2]))
            elif gws is not None:
                gw_dbg = gws[0]
            else:
                gw_dbg = sampler.sample(np.random.default_rng(12345))
            proc.window_chrom, proc.window_start = gw_dbg.chrom, gw_dbg.start
            debug_window(
                theta, gw_dbg, n_haps, param_idx, cfg, baseline_msmc, mean_2ne,
                nu_log_offset=(-float(np.log(args.s_ref))
                               if args.param_mode == "s-proxy" else 0.0),
                nu_clip_neutral=(args.param_mode == "s-proxy"),
                nu_start_time=args.nu_start_time, nu_duration=args.nu_duration,
                nu_anchor=args.nu_anchor)
            return

        for _i, (w_idx, chrom, start) in enumerate(coords[:args.n_windows]):
            w_seed = int(rng.integers(1, 2**31 - 1))
            if chrom is not None:
                gw = sampler.window_from_coords(str(chrom), int(start))
            elif gws is not None:
                gw = gws[_i]
            else:
                gw = sampler.sample(np.random.default_rng(w_seed))
            proc.gene_intervals = gw.gene_intervals
            proc.window_chrom, proc.window_start = gw.chrom, gw.start
            te_rows, te_pos, snp_rows, snp_pos = simulate_window(
                theta, gw, n_haps, param_idx, cfg, args.n_segments,
                baseline_msmc, mean_2ne, np.random.default_rng(w_seed),
                nu_log_offset=(-float(np.log(args.s_ref))
                               if args.param_mode == "s-proxy" else 0.0),
                nu_clip_neutral=(args.param_mode == "s-proxy"),
                nu_start_time=args.nu_start_time,
                nu_duration=args.nu_duration,
                nu_anchor=args.nu_anchor)
            if not te_rows:
                continue
            # pad=False returns a SINGLE (n_te, F) array (raw per-TE features for
            # cross-window pooling), NOT a (feats, mask) tuple -- that 2-tuple is
            # only returned when pad=True. Same idiom as 01b_reprocess.py.
            feats = proc.build_sumstats_from_variant_table(
                te_rows, te_pos, snp_rows, snp_pos, n_haps,
                recomb_map=gw.recomb_map, window_chrom=gw.chrom,
                window_start=gw.start, pad=False)
            all_feats.append(feats)
            if use_oe:
                # feature_r space, same axis as the stored local_recomb column.
                from te_dfe.enrichment import (bp_per_bin, te_count_per_bin,
                                               feature_r as _fr)
                oe_E += bp_per_bin(np.asarray(gw.recomb_map.position),
                                   np.asarray(gw.recomb_map.rate), oe_edges,
                                   cfg["recomb_typical"], cfg["recomb_log_scale"])
                te_rates = gw.recomb_map.get_rate(np.asarray(te_pos, float))
                oe_O += te_count_per_bin(
                    _fr(te_rates, cfg["recomb_typical"], cfg["recomb_log_scale"]),
                    oe_edges)

        if not all_feats:
            print(f"[rep{replicate_id}] no TEs", file=sys.stderr); continue

        all_feats = np.concatenate(all_feats, axis=0)
        NMAX = getattr(CB, "N_TE_TRAIN_MAX", 1000)
        n_te = min(all_feats.shape[0], NMAX); F = all_feats.shape[1]
        keep = rng.permutation(all_feats.shape[0])[:n_te]
        feats_pad = np.zeros((NMAX, F), np.float32); te_mask = np.zeros((NMAX,), np.float32)
        feats_pad[:n_te] = all_feats[keep]; te_mask[:n_te] = 1.0

        oe_fields = ({"oe_O_hist": oe_O.astype(np.float64),
                      "oe_E_hist": oe_E.astype(np.float64)} if use_oe else {})
        np.savez_compressed(
            shard_path, feats=feats_pad, te_mask=te_mask,
            theta=theta.astype(np.float32), replicate_id=np.int32(replicate_id),
            n_windows_total=np.int32(args.n_windows),
            master_seed=np.int64(args.master_seed), format=np.str_("sumstat"),
            **oe_fields)
        print(f"[rep{replicate_id}] saved {shard_path.name} n_te={n_te} "
              f"elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

