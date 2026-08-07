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


def insertion_intensity(recomb_rate, log_lambda, gamma_r, recomb_typical, log_scale):
    """lambda_local(p) = lambda_global * exp(gamma_r * feature_r). gamma_r>0 ->
    more insertions where recombination is high."""
    fr = feature_r(recomb_rate, recomb_typical, log_scale)
    return np.exp(log_lambda) * np.exp(gamma_r * fr)


def expansion_log_upsilon(a, b, recomb_rate, recomb_typical, log_scale,
                          log_offset=0.0, clip_at_neutral=False):
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
    lu = a + b * fr + log_offset
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
            log_offset=nu_log_offset, clip_at_neutral=nu_clip_neutral))
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
        log_offset=nu_log_offset, clip_at_neutral=nu_clip_neutral))
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
                lr, rtyp, lscale, log_offset=nu_log_offset, clip_at_neutral=nu_clip_neutral))
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
                                   clip_at_neutral=nu_clip_neutral)
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
    p.add_argument("--param-mode", choices=["nu-direct", "s-proxy"],
                   default="nu-direct",
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

    cfg = dict(param_names=list(PARAM_NAMES), Ne=getattr(CB, "NE", 10_000), mu_snp=MU,
               recomb_typical=CB.RECOMB_TYPICAL, recomb_log_scale=CB.RECOMB_LOG_SCALE,
               TE_STATE=TE_STATE)
    n_haps = getattr(CB, "N_HAPS", getattr(CB, "N_SAMPLES", 100))
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
        if args.theta is not None:
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
