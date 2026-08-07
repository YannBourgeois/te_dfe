"""
moments_dfe.py — diffusion-based DFE estimation for the SIMPLE TE model.

The NPE pipeline infers the DFE by simulating and learning a summary. This is the
complementary route: compute the EXPECTED TE site-frequency spectrum directly
from diffusion theory and fit it by maximum likelihood. No simulation at all.

Why a burst is visible in an SFS at all
---------------------------------------
A transposition burst does not create correlations between TE frequencies. Under
the Poisson Random Field assumption the insertions remain independent given the
parameters. What a burst changes is the AGE DISTRIBUTION of the segregating
insertions, and age maps onto frequency: a recent burst piles up young, rare
insertions, an older burst leaves an excess at intermediate frequency. So the
whole burst signature lives in the marginal SFS -- exactly the object dadi and
moments compute. In diffusion terms a burst is simply a TIME-VARYING MUTATION
INFLUX, which both packages expose as the `theta` argument of the integration.

Mapping of the six simple-model parameters
------------------------------------------
    log_lambda        -> overall influx scale; sets the TOTAL expected TE count
    log_gamma_shape   -> shape of the Gamma DFE over s
    log_mean_s        -> mean of the Gamma DFE over s
    log10_t_start     -> older edge of the burst window (generations ago)
    log_duration      -> burst length (generations)
    log_multiplier    -> influx multiplier during the burst window

All six have diffusion counterparts, which is why the simple model ports cleanly.
alpha_r and gamma_r do NOT: diffusion has no spatial structure, so recombination
dependence can only be recovered by stratifying TEs into recombination bins and
fitting separately per bin (see fit_dfe_moments.py --recomb-bins).

Conventions (the easy thing to get wrong)
-----------------------------------------
* ours     : s > 0 is DELETERIOUS, gamma_ours = GAMMA_SCALE * Ne * s
* moments  : gamma_m = 2 * N_ref * s with POSITIVE meaning BENEFICIAL
  hence      gamma_m = -2 * N_ref * s
* `h` is the dominance coefficient; 0.5 is additive. Match the SLiM sims.
* moments time is in units of 2 * N_ref generations; sizes are relative to N_ref.

Backends
--------
`moments` is the real one. An `analytic` fallback implements the constant-Ne
Wright/PRF equilibrium in closed form, which is exact when the history is flat
and there is no burst. It exists so the machinery (segmentation, DFE weighting,
likelihood, optimiser) can be exercised and self-tested without moments
installed, and so the moments backend has something to be validated against.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln


GAMMA_SCALE_DEFAULT = 4.0
# Largest |gamma| handed to moments. Above this the diffusion solver goes
# unstable, and the spectrum is saturated anyway (all singletons).
GAMMA_MOMENTS_MAX = 200.0

# Separate, MUCH lower cap for the age-partitioned path. moments.Demes.SFS with
# mutation_time_windows integrates each window over a long stretch, and the
# diffusion solver goes unstable far earlier than on the short marginal path:
# measured on a constant-Ne=500 graph with edges 0..800, the unit-influx row sums
# stay ~0.15 up to |gamma| ~ 38 and then explode (1.4 at 40, 7e3 at 45, 2e160 at
# 200). The threshold falls as the oldest edge grows (~35 at 1600, ~30 at 3200).
# Those blown-up spectra are FINITE, so they slip past a nan/inf guard and poison
# the likelihood: they are the source of the -logL ~ 1e171 starts, and they push
# the optimiser away from strong selection, biasing the strongly-deleterious
# fraction DOWNWARD. Note the spectra are NOT saturated at this cap (normalised
# spectra still differ by 0.28 between |gamma| 20 and 30), so this is a genuine
# ceiling on resolvable selection, not a free clip.
AGE_WINDOW_GAMMA_MAX = 30.0
# a unit-influx window spectrum should sum to O(1); anything past this is solver
# blow-up, not biology
_AGE_WINDOW_SANE_SUM = 10.0
_AGE_WINDOW_WARNED = set()


# ---------------------------------------------------------------------------
# moments compatibility patch: gamma + mutation_time_windows
# ---------------------------------------------------------------------------
# In moments (<= 1.6.0, main included) moments.Demes.SFS crashes when SELECTION
# (`gamma`) and `mutation_time_windows` are used together. Inside _compute_sfs
# the spectrum is wrapped in a LIST when time windows are requested:
#
#     if mutation_time_windows is not None:
#         fs = [fs]
#
# but the per-epoch selection setup still treats fs as a single Spectrum:
#
#     gamma_int = [gamma_dict[pid] ... for pid in fs.pop_ids]   # fs is a list
#
# so the first integration step with T > 0 and gamma set dies with
#     AttributeError: 'list' object has no attribute 'pop_ids'
# The neutral path (gamma is None) never touches fs.pop_ids, so it works.
#
# We grab moments' own _compute_sfs, fix ONLY that access (all list entries share
# the same pop_ids, so fs[0].pop_ids is correct), and reinstall it. This tracks
# whatever version is installed and is a no-op if moments ever fixes it upstream.
_MOMENTS_MTW_PATCHED = False


def _patch_moments_mtw_gamma():
    """Faithfully fix moments' gamma + mutation_time_windows crash. Idempotent."""
    global _MOMENTS_MTW_PATCHED
    if _MOMENTS_MTW_PATCHED:
        return
    import inspect
    import moments.Demes.Demes as _md
    src = inspect.getsource(_md._compute_sfs)
    if "for pid in fs.pop_ids" in src:
        fixed = src.replace(
            "for pid in fs.pop_ids",
            "for pid in (fs[0].pop_ids if isinstance(fs, list) else fs.pop_ids)",
        )
        ns = _md.__dict__
        exec(compile(fixed, _md.__file__, "exec"), ns)
        _md._compute_sfs = ns["_compute_sfs"]
    _MOMENTS_MTW_PATCHED = True


# ---------------------------------------------------------------------------
# Demography -> moments epoch list
# ---------------------------------------------------------------------------
def epochs_from_msmc(baseline_msmc, ne_fallback=500.0):
    """Return (N_ref, epochs, T_total_gen).

    epochs is a list of (ne_absolute, duration_generations) in FORWARD time,
    oldest first. N_ref is the ancestral size, the one moments computes its
    steady state at and scales everything else against.
    """
    if baseline_msmc is None or not getattr(baseline_msmc, "epochs", None):
        return float(ne_fallback), [], 0.0
    eps = sorted(baseline_msmc.epochs, key=lambda e: e.slim_gen_start)
    N_ref = float(getattr(baseline_msmc, "ancestral_ne", eps[0].ne))
    T_total = float(baseline_msmc.sim_length)
    bounds = [float(e.slim_gen_start) for e in eps] + [T_total]
    out = []
    for k, e in enumerate(eps):
        dur = bounds[k + 1] - bounds[k]
        if dur > 0:
            out.append((float(e.ne), dur))
    return N_ref, out, T_total


def burst_segments(epochs, T_total, t_start_ago, duration, multiplier):
    """Split the forward-time integration at BOTH demographic and burst edges.

    Returns [(ne, duration_gen, influx_multiplier), ...] oldest first.

    The burst window is [t_start_ago - duration, t_start_ago] in generations AGO,
    matching the simulator's convention (t_old = t_start, t_young = t_start -
    duration). In forward time that is [T_total - t_old, T_total - t_young].
    """
    if not epochs:
        return []
    t_old_ago = float(t_start_ago)
    t_young_ago = max(0.0, t_old_ago - float(duration))
    b_lo = T_total - t_old_ago          # forward-time start of the burst
    b_hi = T_total - t_young_ago        # forward-time end
    active = (multiplier > 1.0) and (b_hi > b_lo)

    # demographic breakpoints in forward time
    edges, t = [0.0], 0.0
    for _, dur in epochs:
        t += dur
        edges.append(t)
    if active:
        for e in (b_lo, b_hi):
            if 0.0 < e < T_total:
                edges.append(e)
    edges = sorted(set(round(e, 9) for e in edges))

    def _ne_at(mid):
        acc = 0.0
        for ne, dur in epochs:
            acc += dur
            if mid <= acc:
                return ne
        return epochs[-1][0]

    segs = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        dur = hi - lo
        if dur <= 0:
            continue
        mid = 0.5 * (lo + hi)
        mult = float(multiplier) if (active and b_lo <= mid <= b_hi) else 1.0
        segs.append((_ne_at(mid), dur, mult))
    return segs


# ---------------------------------------------------------------------------
# Analytic backend (constant Ne, no burst) -- exact, and the self-test reference
# ---------------------------------------------------------------------------
def _wright_density(x, gamma_ours):
    """Sawyer & Hartl / Wright equilibrium density, gamma_ours > 0 deleterious."""
    x = np.asarray(x, float)
    g = float(gamma_ours)
    if abs(g) < 1e-9:
        return 1.0 / x
    if g < 500.0:
        return np.expm1(g * (1.0 - x)) / (np.expm1(g) * x * (1.0 - x))
    return np.exp(-g * x) / (x * (1.0 - x))


def sfs_analytic(n, gamma_ours, n_quad=4000):
    """E[S_i | gamma] for i = 1..n-1 at unit influx, constant Ne, no burst."""
    u = np.linspace(0.0, 1.0, n_quad + 2)[1:-1]
    x = np.clip(u ** 3 / (u ** 3 + (1 - u) ** 3), 1e-9, 1 - 1e-9)
    dx = np.gradient(x)
    i = np.arange(1, n)
    lb = (gammaln(n + 1) - gammaln(i + 1) - gammaln(n - i + 1))[:, None]
    B = np.exp(lb + i[:, None] * np.log(x)[None, :]
               + (n - i)[:, None] * np.log1p(-x)[None, :])
    return (B * (_wright_density(x, gamma_ours) * dx)[None, :]).sum(axis=1)


# ---------------------------------------------------------------------------
# moments backend
# ---------------------------------------------------------------------------
# How our s maps onto the diffusion s. SLiM's setSelectionCoeff(s) means
#     AA = 1,  Aa = 1 + h*s,  aa = 1 + s          (s is the HOMOZYGOTE effect)
# dadi/moments use
#     AA = 1,  Aa = 1 + 2*h*s,  aa = 1 + 2*s      (s is the PER-ALLELE effect)
# Matching the heterozygotes gives h*s_slim = 2*h*s_dadi, so s_dadi = s_slim/2,
# and the h cancels so the factor holds for any dominance. Getting this wrong is
# a silent factor-2 error in selection strength, which shows up as a shifted
# mean_s rather than as a failure.
S_CONVENTION_FACTOR = {"homozygote": 0.5,   # SLiM's convention: our default
                       "per-allele": 1.0}   # dadi/moments native


def sfs_moments(n, gamma_ours, segs, N_ref, gamma_scale=GAMMA_SCALE_DEFAULT,
                ne_for_gamma=None, h=0.5, s_convention="homozygote"):
    """E[S_i] under the real demography AND the burst, via moments.

    The burst enters as the per-segment `theta` of the integration: influx is
    theta_base during normal segments and theta_base * multiplier inside the
    burst window. That is precisely what a transposition burst is in diffusion
    terms.
    """
    import moments
    ne_g = float(ne_for_gamma if ne_for_gamma is not None else N_ref)
    # gamma_ours = gamma_scale * ne * s, so recover s, apply the convention
    # factor, then convert to the moments scaling. Note that gamma_scale used to
    # cancel out of this expression, which meant --gamma-scale silently affected
    # only the reporting categories and not what moments was actually told.
    s_ours = float(gamma_ours) / (gamma_scale * ne_g)
    s_diff = s_ours * S_CONVENTION_FACTOR[s_convention]
    # ours (positive = deleterious) -> moments (positive = beneficial)
    gm = -2.0 * N_ref * s_diff
    # moments integrates a diffusion PDE and is only stable for moderate |gamma|.
    # Beyond gm ~ -200 the spectrum is already essentially all-singleton, so
    # clipping loses nothing physically but avoids the NaN / negative entries
    # that otherwise poison every likelihood evaluation.
    gm = float(np.clip(gm, -GAMMA_MOMENTS_MAX, GAMMA_MOMENTS_MAX))
    fs = moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(n, gamma=gm, h=h))
    for ne, dur, mult in segs:
        T = dur / (2.0 * N_ref)                 # moments time units
        if T <= 0:
            continue
        fs.integrate([ne / N_ref], T, gamma=gm, h=h, theta=float(mult))
    out = np.asarray(fs, dtype=float)[1:n]
    # moments returns a masked array and can emit small negative entries from
    # numerical error; both would trip the caller's finiteness guard.
    return np.nan_to_num(np.clip(out, 0.0, None), nan=0.0, posinf=0.0)


def build_spectrum_grid(n, gamma_grid, segs, N_ref, backend="moments",
                        gamma_scale=GAMMA_SCALE_DEFAULT, ne_for_gamma=None,
                        h=0.5, s_convention="homozygote"):
    """(len(gamma_grid), n-1) array of unit-influx spectra, one row per gamma."""
    if backend == "analytic":
        # The analytic PRF density is written in OUR gamma directly, so the
        # convention factor is already inside gamma_grid and must not be applied
        # twice. Only the moments path needs the conversion.
        return np.array([sfs_analytic(n, g) for g in gamma_grid])
    return np.array([sfs_moments(n, g, segs, N_ref, gamma_scale, ne_for_gamma,
                                 h, s_convention) for g in gamma_grid])


# ---------------------------------------------------------------------------
# DFE integration
# ---------------------------------------------------------------------------
def gamma_dfe_bins(s_edges, shape, mean_s):
    """Exact Gamma(shape, scale = mean_s / shape) mass per bin, plus the
    conditional mean of s inside each bin.

    Returns (weights, s_representative), both length len(s_edges) - 1.

    Trapezoid quadrature on the DENSITY fails badly here: the prior allows
    shape down to 0.1, and a Gamma with shape < 1 has an integrable singularity
    at s = 0, so a log-spaced grid misplaces most of the mass (46% error at
    shape 0.2, 195% at shape 0.1). Using the regularised incomplete gamma
    function P(a, x) instead gives the bin masses in closed form:

        mass_k       = P(shape,     b/scale) - P(shape,     a/scale)
        E[s | bin k] = scale * [P(shape+1, b/scale) - P(shape+1, a/scale)] / mass_k

    so both the total mass and E[s] are exact by construction, for any shape.
    The lowest bin collects the near-zero mass and is represented by a spectrum
    that is indistinguishable from neutral -- which is the correct physics, since
    those insertions are below the drift barrier.
    """
    from scipy.special import gammainc
    scale = float(mean_s) / float(shape)
    e = np.asarray(s_edges, float) / scale
    lo, hi = e[:-1], e[1:]
    mass = gammainc(shape, hi) - gammainc(shape, lo)
    # Conditional mean via the shape+1 incomplete gamma. The partial expectation
    # of a Gamma(k, theta) over [a, b] is k*theta*[P(k+1, b/theta) - P(k+1, a/theta)]
    # -- the leading k is easy to drop, and doing so returns `scale` instead of
    # `shape*scale`, i.e. E[s] off by exactly a factor 1/shape.
    num = shape * scale * (gammainc(shape + 1.0, hi) - gammainc(shape + 1.0, lo))
    with np.errstate(invalid="ignore", divide="ignore"):
        s_rep = np.where(mass > 0, num / np.where(mass > 0, mass, 1.0),
                         0.5 * (s_edges[:-1] + s_edges[1:]))
    tot = mass.sum()
    if tot > 0:
        mass = mass / tot          # renormalise over the covered range
    else:
        mass = np.full_like(mass, 1.0 / len(mass))
    return mass, s_rep


def s_bin_midpoints(s_edges):
    """Geometric midpoint of each bin: the s at which the SPECTRA are evaluated.

    These must NOT depend on shape or mean_s. If they did, every optimiser step
    would move the grid and force a full spectrum rebuild, and the weights would
    then refer to different s values than the spectra they multiply.
    """
    e = np.asarray(s_edges, float)
    lo = np.where(e[:-1] > 0, e[:-1], e[1] * 1e-3)
    return np.sqrt(lo * e[1:])


def gamma_dfe_weights(s_grid, shape, mean_s):
    """Backwards-compatible wrapper: weights at the given grid POINTS.

    Kept so existing callers keep working; prefer gamma_dfe_bins, which is exact.
    """
    edges = np.concatenate([[0.0], np.sqrt(s_grid[:-1] * s_grid[1:]),
                            [s_grid[-1] * 1e3]])
    w, _ = gamma_dfe_bins(edges, shape, mean_s)
    return w


def dfe_spectrum(spectra, weights, theta_te):
    """Weighted mixture of the precomputed spectra, scaled by the total influx.

    `spectra` is (n_bins, n_haps - 1) from build_spectrum_grid evaluated at the
    representative s of each bin; `weights` is the exact bin mass from
    gamma_dfe_bins. theta_te carries log_lambda: it sets the TOTAL expected
    number of segregating TEs and is the only parameter that scales the whole
    spectrum without changing its shape.
    """
    w = np.asarray(weights, float)
    return float(theta_te) * (w[:, None] * np.asarray(spectra)).sum(axis=0)


# ---------------------------------------------------------------------------
# Parameterisation conventions -- the classic source of silent factor-of-2 errors
# ---------------------------------------------------------------------------
CONVENTIONS = """
GAMMA DISTRIBUTION
    Here the DFE is Gamma over the UNSCALED coefficient s, parameterised by
    (shape, mean_s), with

        scale = mean_s / shape          rate = 1 / scale = shape / mean_s
        E[s]  = shape * scale = mean_s  Var[s] = shape * scale^2 = mean_s^2 / shape

    scipy equivalent: scipy.stats.gamma(a=shape, scale=mean_s/shape).
    NOTE the two traps: `scale` is NOT the mean (it is mean/shape), and some
    packages parameterise by RATE = 1/scale instead.

    Other methods put the Gamma on the SCALED coefficient instead:
        fitdadi / dadi      Gamma over |gamma|
        polyDFE             Gamma over S_d = 4*Ne*s, negative for deleterious
        DFE-alpha           Gamma over s, shape reported as beta
    Since gamma = c*s implies Gamma(shape, scale) -> Gamma(shape, c*scale), the
    SHAPE is directly comparable across all of these; only the mean needs the
    factor c. So a published shape can be compared to ours as-is.

SIGN
    ours     s > 0 is DELETERIOUS
    moments  gamma > 0 is BENEFICIAL, so deleterious is NEGATIVE
    polyDFE  S_d < 0 for deleterious
    The conversion in sfs_moments carries the minus sign explicitly.

SCALING (the factor most often wrong)
    ours     gamma_ours = GAMMA_SCALE * Ne * s, GAMMA_SCALE = 4 by default,
             i.e. the standard diploid 4*Ne*s with s the HOMOZYGOUS effect
    dadi     documents gamma = 2*N_ref*s, but whether that s is the homozygous
             or heterozygous effect interacts with h, and the effective factor
             therefore depends on the dominance convention in use.
    Rather than trusting documentation, check_gamma_convention() MEASURES the
    factor: it matches a moments spectrum against the closed-form Wright/PRF
    spectrum at constant Ne and reports the conversion that actually holds.
"""


def check_gamma_convention(n, ne, gamma_scale=GAMMA_SCALE_DEFAULT, h=0.5,
                           s_probe=(1e-3, 3e-3, 1e-2), verbose=True):
    """Determine EMPIRICALLY what scaling moments is using.

    Under a constant-Ne history with no burst, the moments spectrum must equal
    the closed-form Wright/PRF spectrum at the same scaled coefficient. So for
    each probe s we ask which analytic gamma reproduces the moments output, and
    report the implied factor gamma / (Ne * s).

    A factor of 4 means our GAMMA_SCALE default is right. A factor of 2 means
    every s in the fit is being interpreted at half its intended strength, which
    would bias mean_s by exactly 2x without any other visible symptom.
    """
    from scipy.optimize import minimize_scalar
    segs = [(float(ne), 1e-9, 1.0)]          # degenerate: steady state only
    rows = []
    for s_val in s_probe:
        target = sfs_moments(n, gamma_scale * ne * s_val, segs, float(ne),
                             gamma_scale=gamma_scale, ne_for_gamma=ne, h=h)
        t = target / max(target.sum(), 1e-300)

        def mismatch(g_ours):
            a = sfs_analytic(n, max(g_ours, 1e-9))
            a = a / max(a.sum(), 1e-300)
            return float(np.sum((a - t) ** 2))

        r = minimize_scalar(mismatch, bounds=(1e-6, 4.0 * gamma_scale * ne * s_val
                                              + 50.0), method="bounded")
        rows.append((s_val, r.x, r.x / (ne * s_val)))
    if verbose:
        print("=" * 68)
        print("GAMMA CONVENTION CHECK (moments vs the closed-form PRF spectrum)")
        print(f"  {'s':>10}{'gamma matched':>16}{'implied factor':>17}")
        for s_val, g, f in rows:
            print(f"  {s_val:>10.1e}{g:>16.2f}{f:>17.2f}")
        fac = np.median([f for _, _, f in rows])
        print(f"  median implied factor = {fac:.2f}   "
              f"(--gamma-scale is set to {gamma_scale:g})")
        if abs(fac - gamma_scale) / gamma_scale > 0.15:
            print(f"  !! MISMATCH: pass --gamma-scale {fac:.0f} instead, or the DFE "
                  f"mean is off by {gamma_scale/fac:.2g}x")
        else:
            print("  OK: the scaling in use matches --gamma-scale")
        print("=" * 68)
    return rows


# ---------------------------------------------------------------------------
# AGE-PARTITIONED spectra (moments.Demes.SFS mutation_time_windows)
# ---------------------------------------------------------------------------
def selfing_correction(selfing_rate, h=0.5):
    """Return (ne_factor, sel_factor, F) for a partially selfing population.

    Selfing at rate S gives inbreeding coefficient F = S/(2-S). Two things
    change in the diffusion:

      * drift variance scales as (1+F), so   Ne_eff = N / (1+F)
      * the selection term near q=0 scales as [2h(1-F) + F] / (2h)

    At h = 0.5 the selection factor is EXACTLY 1 for every S, because additive
    fitness is linear in allele copies and genotype frequencies drop out. So for
    additive mutations a partially selfing population's SFS is exactly that of a
    randomly mating one with Ne' = N/(1+F), and NO selfing support is needed in
    the SFS machinery: rescaling Ne is sufficient and exact.

    (moments does carry selfing through its LD path, but `_compute_sfs` has no
    selfing argument, which is why moments.Demes.SFS cannot take it directly.)

    At h < 0.5 selfing exposes homozygotes and RAISES the efficacy of selection;
    fully recessive mutations go from nearly invisible to fully selected. The
    factor is returned so the caller can apply it, but the equivalence is then
    approximate rather than exact.
    """
    S = float(selfing_rate)
    if not (0.0 <= S < 1.0):
        raise ValueError(f"selfing rate must be in [0, 1), got {S}")
    F = S / (2.0 - S)
    ne_factor = 1.0 / (1.0 + F)
    h = float(h)
    sel_factor = (2.0 * h * (1.0 - F) + F) / (2.0 * h) if h > 0 else np.inf
    return ne_factor, sel_factor, F


def demes_graph_from_msmc(baseline_msmc, ne_fallback=500.0, selfing_rate=0.0,
                          verbose=False):
    """Build a one-deme `demes` graph from the MSMC epochs.

    ORDERING, which demes is strict about: epochs are listed OLDEST FIRST and
    their `end_time` (generations AGO) must be strictly DECREASING, with the
    last epoch ending at 0 and the first starting at infinity.

    MSMCDemography stores epochs in FORWARD (SLiM) time via
    slim_gen_start = sim_length - t_gen_ago, sorted ascending, so ascending
    slim_gen_start is already ancient-first. Epoch k spans forward
    [slim_gen_start[k], slim_gen_start[k+1]), i.e. it ENDS at
    T_total - slim_gen_start[k+1] generations ago.

    Any duplicated present-day epoch or zero-length epoch is removed before
    constructing the demes graph.
    """

    import demes

    b = demes.Builder(time_units="generations")

    if baseline_msmc is None or not getattr(baseline_msmc, "epochs", None):
        b.add_deme(
            "pop",
            epochs=[dict(start_size=float(ne_fallback), end_time=0)],
        )
        return b.resolve()

    # Ancient -> recent in forward time
    eps = sorted(
        baseline_msmc.epochs,
        key=lambda e: e.slim_gen_start,
    )

    T_total = float(baseline_msmc.sim_length)

    # Forward-time boundaries
    bounds = [float(e.slim_gen_start) for e in eps] + [T_total]

    ep = []
    seen_present = False

    for k, e in enumerate(eps):

        start_ago = T_total - bounds[k]
        end_ago = max(0.0, T_total - bounds[k + 1])

        # Drop zero-length epochs
        if start_ago <= end_ago + 1e-9:
            if verbose:
                print(
                    f"[demes] dropping zero-length epoch "
                    f"(start={start_ago:.3f}, end={end_ago:.3f}, Ne={e.ne})"
                )
            continue

        # Keep only one epoch ending at the present
        if abs(end_ago) < 1e-9:
            if seen_present:
                if verbose:
                    print(
                        f"[demes] dropping duplicate present-day epoch "
                        f"(Ne={e.ne})"
                    )
                continue
            seen_present = True
            end_ago = 0.0

        ep.append(
            dict(
                start_size=float(e.ne),
                end_time=float(end_ago),
            )
        )

    if not ep:
        raise ValueError("No valid epochs remain after filtering.")

    # Validate ordering
    ends = [d["end_time"] for d in ep]

    if len(ends) > 1:
        if not all(a > b + 1e-9 for a, b in zip(ends[:-1], ends[1:])):
            raise ValueError(
                "demes epochs are not strictly decreasing in end_time "
                f"(generations ago): {ends}"
            )

    if abs(ends[-1]) > 1e-9:
        raise ValueError(
            f"youngest epoch must end at 0 generations ago, got {ends[-1]}"
        )

    b.add_deme("pop", epochs=ep)
    g = b.resolve()

    if verbose:
        print("[demes]")
        for d in ep:
            print(
                f"  end={d['end_time']:10.1f}  "
                f"Ne={d['start_size']:10.1f}"
            )

    return g
    


def age_window_spectra(graph, n_haps, gamma_ours, age_edges, gamma_scale=GAMMA_SCALE_DEFAULT,
                       ne_for_gamma=None, h=0.5, s_convention="homozygote"):
    """Unit-influx spectra PARTITIONED BY MUTATION AGE.

    Returns (n_windows, n_haps - 1): row w is the expected spectrum of insertions
    that ORIGINATED in age window w, at unit influx.

    This is what makes the burst cheap AND identifiable. Under infinite sites,
    each mutation's fate is independent given the demography, so the spectrum is
    LINEAR in the influx: if the influx during window w is theta * m_w, that
    window contributes m_w * S_w. So the expensive integration depends only on
    (gamma, demography) and NOT on the burst, and the burst parameters become a
    linear reweighting of precomputed rows.

    It also separates what the marginal SFS confounds: a burst changes the total
    COUNT in one age window, while selection changes the SHAPE within every
    window.

    age_edges must start at 0 and increase; the final window runs to infinity, so
    len(age_edges) windows are returned.
    """
    import moments
    # moments (<= 1.6.0) crashes on gamma + mutation_time_windows; patch it.
    _patch_moments_mtw_gamma()
    ne_g = float(ne_for_gamma if ne_for_gamma is not None else 500.0)
    s_ours = float(gamma_ours) / (gamma_scale * ne_g)
    s_diff = s_ours * S_CONVENTION_FACTOR[s_convention]
    N_ref = float(graph["pop"].epochs[0].start_size)
    gm_want = -2.0 * N_ref * s_diff
    gm = float(np.clip(gm_want, -AGE_WINDOW_GAMMA_MAX, AGE_WINDOW_GAMMA_MAX))
    # moments already runs the final window from the last edge to infinity, so
    # pass the finite edges as-is. Appending np.inf is redundant and risks an
    # off-by-one window count against the observed age-partitioned SFS.
    edges = np.asarray(age_edges, float)
    if edges.size and np.isinf(edges[-1]):
        edges = edges[:-1]

    def _call(g_use):
        fs_list = moments.Demes.SFS(graph, sampled_demes=["pop"],
                                    sample_sizes=[int(n_haps)],
                                    gamma=g_use, h=h, theta=1.0,
                                    mutation_time_windows=edges)
        out = np.array([np.asarray(f, dtype=float)[1:n_haps] for f in fs_list])
        return np.nan_to_num(np.clip(out, 0.0, None), nan=0.0, posinf=0.0)

    # Stability guard: back off |gamma| until the spectra are physically sane.
    # The blow-up is finite-valued, so it must be caught by MAGNITUDE, not by a
    # nan/inf check. The threshold depends on the window span, so probe rather
    # than trust a fixed constant.
    out = _call(gm)
    tries = 0
    while (not np.isfinite(out).all() or out.sum(axis=1).max() > _AGE_WINDOW_SANE_SUM) \
            and tries < 15 and abs(gm) > 1e-6:
        gm *= 0.8
        out = _call(gm)
        tries += 1
    if tries:
        key = round(gm, 3)
        if key not in _AGE_WINDOW_WARNED:
            _AGE_WINDOW_WARNED.add(key)
            import sys as _sys
            print(f"  !! age-window solver unstable at gamma={gm_want:.1f}; "
                  f"backed off to {gm:.1f}. Selection stronger than this is NOT "
                  f"resolvable on the age-partitioned path, and the strongly-"
                  f"deleterious fraction will be biased low.", file=_sys.stderr)
    return out


def burst_influx_per_window(age_edges, t_start_ago, duration, multiplier,
                            t_max=None):
    """Mean influx multiplier in each age window.

    The burst covers [t_start_ago - duration, t_start_ago] generations ago. A
    window only partly covered by the burst gets the overlap-weighted average,
    so the likelihood stays smooth as the burst edges slide across a boundary
    (a hard in/out test would make it piecewise constant and stall the optimiser).
    """
    edges = list(age_edges) + [float(t_max if t_max is not None
                                     else max(age_edges) * 10 + 1.0)]
    b_hi = float(t_start_ago)
    b_lo = max(0.0, b_hi - float(duration))
    m = float(multiplier)
    out = np.ones(len(age_edges))
    if not (m > 1.0 and b_hi > b_lo):
        return out
    for w in range(len(age_edges)):
        lo, hi = edges[w], edges[w + 1]
        span = hi - lo
        if span <= 0:
            continue
        ov = max(0.0, min(hi, b_hi) - max(lo, b_lo))
        out[w] = 1.0 + (m - 1.0) * (ov / span)
    return out


def observed_age_sfs(te_freqs, ages, n_haps, age_edges):
    """(n_windows, n_haps - 1) counts, binned by age then by derived count."""
    f = np.asarray(te_freqs, float)
    a = np.asarray(ages, float)
    ok = np.isfinite(f) & np.isfinite(a)
    f, a = f[ok], a[ok]
    counts = np.rint(f * n_haps).astype(int)
    keep = (counts > 0) & (counts < n_haps)
    counts, a = counts[keep], a[keep]
    edges = list(age_edges) + [np.inf]
    obs = np.zeros((len(age_edges), n_haps - 1))
    for w in range(len(age_edges)):
        sel = (a >= edges[w]) & (a < edges[w + 1])
        if sel.any():
            obs[w] = np.bincount(counts[sel], minlength=n_haps)[1:n_haps]
    return obs


# ---------------------------------------------------------------------------
# Reporting: proportion of new insertions per effective-selection category
# ---------------------------------------------------------------------------
def dfe_categories(shape, mean_s, ne, gamma_scale=GAMMA_SCALE_DEFAULT,
                   gamma_edges=(1.0, 10.0, 100.0)):
    """Fraction of NEW insertions falling in each effective-selection class.

    Categories are defined on the scaled coefficient

        gamma = gamma_scale * ne * s          (gamma_scale = 4 by default)

    so the corresponding s edges are gamma_edges / (gamma_scale * ne). Because
    the DFE is a Gamma, each fraction is a difference of regularised incomplete
    gamma functions: exact, no quadrature, no grid, and independent of whatever
    s grid the fit happened to use.

    Returns (fractions, s_edges, gamma_edges_full) with len(fractions) =
    len(gamma_edges) + 1.

    The lowest class is the one that matters for interpretation: selection
    cannot act below the drift barrier s ~ 1/(2*Ne), i.e. gamma ~ gamma_scale/2,
    so insertions there are effectively neutral no matter how much data there
    is. Reporting that fraction makes the identifiable part of the DFE explicit
    rather than hiding it inside a mean.
    """
    from scipy.special import gammainc
    shape, mean_s, ne = float(shape), float(mean_s), float(ne)
    scale = mean_s / shape
    g_full = np.concatenate([[0.0], np.asarray(gamma_edges, float), [np.inf]])
    s_edges = g_full / (gamma_scale * ne)
    cdf = np.array([1.0 if np.isinf(e) else gammainc(shape, e / scale)
                    for e in s_edges])
    return np.diff(cdf), s_edges, g_full


def drift_barrier_fraction(shape, mean_s, ne):
    """Fraction of the DFE below the drift barrier s = 1/(2*Ne): unidentifiable
    by construction, since those insertions behave exactly as neutral."""
    from scipy.special import gammainc
    return float(gammainc(float(shape), (1.0 / (2.0 * float(ne)))
                          / (float(mean_s) / float(shape))))


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------
def poisson_ll(obs, exp):
    """Composite Poisson log-likelihood over the SFS bins."""
    obs = np.asarray(obs, float)
    exp = np.clip(np.asarray(exp, float), 1e-300, None)
    return float(np.sum(obs * np.log(exp) - exp - gammaln(obs + 1.0)))


def observed_sfs(te_freqs, n_haps):
    """Fold a vector of TE frequencies into counts per derived-allele class.

    Returns an (n_haps - 1,) array for i = 1 .. n_haps - 1. Frequencies of 0 or
    1 are not segregating and are dropped.
    """
    f = np.asarray(te_freqs, float)
    f = f[np.isfinite(f)]
    counts = np.rint(f * n_haps).astype(int)
    counts = counts[(counts > 0) & (counts < n_haps)]
    return np.bincount(counts, minlength=n_haps)[1:n_haps].astype(float)
