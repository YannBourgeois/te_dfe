"""
simulator.py — Rate-driven TE simulator with a single rectangular burst.

SLiM generates m1 mutations natively at a per-bp per-generation rate:
    rate(t) = lambda * multiplier   for t_start <= age(t) < t_start + duration
    rate(t) = lambda                otherwise

Per-mutation selection coefficients are drawn from the DFE mixture
(pi_neutral + gamma) inside SLiM by a late() callback.

No more Python-side schedule of insertion positions/times — SLiM does
that natively from the time-varying rate.

Expected total insertion count over the simulation (constant Ne case):
    E[n_te] = lambda * window_length * 2 * Ne *
              [ (sim_length - duration) + multiplier * duration ]
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import warnings
from typing import Optional

import msprime
import numpy as np
import pyslim
import tskit

# Harmless warning: SLiM produces tree sequences with time_units="ticks";
# 1 tick = 1 generation in our model.
warnings.filterwarnings(
    "ignore", category=msprime.TimeUnitsMismatchWarning,
)

from .config import (
    SIM_LENGTH, NEUTRAL_MU, N_INDIVIDUALS,
    MAX_INSERTIONS_PER_WINDOW,
    N_TRAIN_WINDOWS, MAF_MIN, SLIM_Q,
    BG_SELECTION, BG_DEL_SHAPE, BG_DEL_MEAN_S, BG_DEL_MU, BG_DEL_DOMINANCE,
)
from .demography import make_tmpdir, MSMCDemography, GenomicWindow, WindowSampler

from .prior import DFEBurstPrior


SLIM_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "slim", "te_dfe_burst.slim",
)


def expected_n_te_baseline(lambda_val, window_length, demography):
    """
    Expected count if there were no burst (multiplier = 1 everywhere).
    Used for prior calibration and as the basic Poisson expectation when
    no burst is active. Note this is computed externally to SLiM; SLiM
    actually generates mutations stochastically from the time-varying
    rate.
    """
    return (
        float(lambda_val) * float(window_length) * demography.integrated_2Ne()
    )


def expected_n_te_with_burst(lambda_val, window_length, demography,
                              t_start, duration, multiplier):
    """
    Expected count under the single-burst model. Approximates the burst
    contribution using constant-Ne integration over the burst window.
    Used only for diagnostics / sanity prints; SLiM does the real Poisson
    draw.

    Approximation: assume Ne over the burst interval ~ ancestral_ne
    (valid for constant-Ne; for changing demography this is rough).
    """
    base = expected_n_te_baseline(lambda_val, window_length, demography)
    # Burst contribution: (multiplier - 1) * lambda * L * 2 * Ne * duration
    burst_extra = (
        (float(multiplier) - 1.0) * float(lambda_val) * float(window_length)
        * 2.0 * float(demography.ancestral_ne) * float(duration)
    )
    return base + burst_extra


def burst_interval_to_ticks(t_start_gens_ago, duration, sim_length):
    """
    Convert a (t_start, duration) pair from generations-ago coordinates
    to (burst_tick_start, burst_tick_end) in SLiM tick coordinates.

    SLiM cycle 1 is the start of the simulation; SLiM cycle SIM_END is
    the present.  Age in generations ago = SIM_END - cycle.

    Returns (burst_tick_start, burst_tick_end), where the burst is
    active on ticks in [burst_tick_start, burst_tick_end).

    Clamped to [1, SIM_END+1].
    """
    burst_tick_end   = sim_length - int(t_start_gens_ago) + 1   # exclusive end
    burst_tick_start = burst_tick_end - int(duration)
    # Clamp
    burst_tick_start = max(1, burst_tick_start)
    burst_tick_end   = min(sim_length + 1, max(burst_tick_start + 1, burst_tick_end))
    return burst_tick_start, burst_tick_end


def _normalize_slim_alleles(ts):
    """
    Coerce SLiM mutation encoding so msprime's BinaryMutationModel can
    overlay neutral SNPs on top. See pass-2 README for details.
    """
    tables = ts.dump_tables()

    n_sites = len(tables.sites)
    if n_sites > 0:
        new_anc = ["0"] * n_sites
        packed_anc, off_anc = tskit.pack_strings(new_anc)
        tables.sites.set_columns(
            position=tables.sites.position,
            ancestral_state=packed_anc,
            ancestral_state_offset=off_anc,
            metadata=tables.sites.metadata,
            metadata_offset=tables.sites.metadata_offset,
        )

    mt = tables.mutations
    new_states = ["1" if m.metadata else (m.derived_state or "1") for m in mt]
    packed, offsets = tskit.pack_strings(new_states)

    times = np.array(mt.time, dtype=np.float64)
    unknown_mask = tskit.is_unknown_time(times)
    if unknown_mask.any():
        node_times = ts.tables.nodes.time
        for i in np.where(unknown_mask)[0]:
            node_id = int(mt.node[i])
            child_t = float(node_times[node_id])
            parent_t = child_t + 1.0
            for e in ts.edges():
                if e.child == node_id:
                    parent_t = float(node_times[e.parent])
                    break
            times[i] = (child_t + parent_t) / 2.0

    mt.set_columns(
        site=mt.site, node=mt.node, parent=mt.parent,
        derived_state=packed, derived_state_offset=offsets,
        metadata=mt.metadata, metadata_offset=mt.metadata_offset,
        time=times,
    )
    return tables.tree_sequence()


def _build_no_collision_rate_map(ts, base_rate, sequence_length):
    """
    Build an msprime rate map that has rate = base_rate everywhere except
    at integer positions occupied by SLiM-origin TEs, where rate = 0.

    This guarantees msprime's neutral SNP overlay cannot collide with
    SLiM TEs at the same integer position — eliminating the source of
    the "stacked mutation" / alphabet-incompatibility errors.

    Approach: for each SLiM TE position p, insert a tiny zero-rate window
    [p, p+1).  Outside these windows, rate = base_rate.
    """
    te_positions = sorted({
        int(ts.site(m.site).position)
        for m in ts.mutations()
        if m.metadata  # broad test OK here pre-overlay (only SLiM muts present)
    })

    if not te_positions:
        return msprime.RateMap.uniform(
            sequence_length=sequence_length, rate=base_rate,
        )

    positions = [0.0]
    rates = []
    for p in te_positions:
        # Background (base_rate) up to p
        if p > positions[-1]:
            positions.append(float(p))
            rates.append(base_rate)
        # Zero-rate at the integer site p
        positions.append(float(p + 1))
        rates.append(0.0)
    if positions[-1] < sequence_length:
        positions.append(float(sequence_length))
        rates.append(base_rate)

    return msprime.RateMap(position=positions, rate=rates)


def generate_synthetic_exons(seq_len, rng,
                             mean_exon, sd_exon, mean_gap, sd_gap,
                             min_exon=30):
    """
    Generate a synthetic exon landscape across [0, seq_len) when no real
    annotation is available. Exon lengths and inter-exon gaps are drawn from
    normal distributions (clamped positive), tiling the window. Returns a list
    of (start, end) window-relative coding intervals.

    Deterministic given rng; used only for background selection when no GTF
    coding intervals exist for the window.
    """
    intervals = []
    pos = int(max(0, rng.normal(mean_gap, sd_gap)))  # start after an initial gap
    while pos < seq_len:
        ex_len = int(round(rng.normal(mean_exon, sd_exon)))
        ex_len = max(min_exon, ex_len)
        start = pos
        end = min(seq_len, start + ex_len)
        if end > start:
            intervals.append((start, end))
        gap = int(round(rng.normal(mean_gap, sd_gap)))
        gap = max(min_exon, gap)
        pos = end + gap
    return intervals


def build_coding_intervals(window, seq_len, rng):
    """
    Coding intervals (window-relative, [0, seq_len)) for background selection.

    Uses the window's real annotation (gene_intervals) when BG_USE_GTF is set
    and intervals exist; otherwise generates a synthetic exon landscape from
    the config parameters. Real intervals are clipped to the window and shifted
    to window-relative coordinates.
    """
    from .config import (
        BG_USE_GTF, BG_MEAN_EXON_LEN, BG_SD_EXON_LEN,
        BG_MEAN_INTEREXON, BG_SD_INTEREXON,
    )
    if BG_USE_GTF and getattr(window, "gene_intervals", None):
        out = []
        for (g0, g1) in window.gene_intervals:
            # gene_intervals are absolute genome coords; make window-relative.
            a = max(0, int(g0) - window.start)
            b = min(seq_len, int(g1) - window.start)
            if b > a:
                out.append((a, b))
        if out:
            # CDS from multi-transcript GTFs overlap; SLiM forbids overlapping
            # genomic elements, so flatten to a disjoint sorted set first.
            return flatten_intervals(out)
    # Fallback: synthetic exon landscape (already disjoint, flattened for safety).
    return flatten_intervals(generate_synthetic_exons(
        seq_len, rng,
        mean_exon=BG_MEAN_EXON_LEN, sd_exon=BG_SD_EXON_LEN,
        mean_gap=BG_MEAN_INTEREXON, sd_gap=BG_SD_INTEREXON,
    ))


def flatten_intervals(intervals):
    """Sort and merge overlapping/touching intervals into a disjoint, sorted
    list. Required because CDS features from multi-transcript GTFs overlap, and
    SLiM's initializeGenomicElement() rejects overlapping genomic elements."""
    iv = sorted((int(a), int(b)) for (a, b) in intervals if int(b) > int(a))
    if not iv:
        return []
    merged = [list(iv[0])]
    for a, b in iv[1:]:
        if a <= merged[-1][1]:           # overlap or touch -> extend
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def write_coding_file(intervals, path):
    """Write coding intervals as 'start\\tend' lines (one per interval)."""
    with open(path, "w") as f:
        for (a, b) in intervals:
            f.write(f"{int(a)}\t{int(b)}\n")
    return path


def postprocess_ts(trees_path, window: GenomicWindow,
                    demography: MSMCDemography, seed: int, rng):
    """Recapitate, simplify, overlay neutral SNPs (collision-free).

    Q-rescaling: the SLiM tree sequence is in SCALED time units (ticks =
    generations / Q), so the ancestral coalescent and the neutral overlay must
    be performed in the SAME scaled units, or population-scaled diversity is
    not preserved and inference stops being Q-invariant. Concretely, with
    rescaling factor Q: recapitation Ne and event times are divided by Q, the
    recapitation recombination rate is multiplied by Q, and the neutral overlay
    per-generation rate is multiplied by Q (branches are Q-fold shorter, so the
    per-unit-time rate must be Q-fold higher to give the same expected count).
    Q = 1 is an exact no-op.
    """
    Q = float(SLIM_Q)
    ts = tskit.load(trees_path)

    # Recapitation only needs to coalesce the ROOTS of the SLiM trees in the
    # deep past, BEFORE the forward simulation started. The entire MSMC epoch
    # history is already realized inside the SLiM run (applied tick-by-tick via
    # setSubpopulationSize), so it must NOT be re-applied here — doing so both
    # double-counts the recent demography and produces msprime events in the
    # wrong time order (the cause of the "events must be ordered" error with
    # multi-epoch histories). A single constant ancestral Ne is correct and is
    # the standard pyslim recapitation idiom. Ne is divided by Q so the
    # ancestral coalescent is in the same scaled time units as the SLiM trees,
    # and the recombination map rates are multiplied by Q (preserving the map's
    # heterogeneity rather than collapsing to a mean).
    recomb_for_recap = window.recomb_map
    if Q != 1.0:
        recomb_for_recap = msprime.RateMap(
            position=window.recomb_map.position,
            rate=window.recomb_map.rate * Q,
        )
    ts = pyslim.recapitate(
        ts,
        recombination_rate=recomb_for_recap,
        ancestral_Ne=demography.ancestral_ne / Q,
        random_seed=seed,
    )
    overlay_rate = NEUTRAL_MU * Q if Q != 1.0 else NEUTRAL_MU

    alive = pyslim.individuals_alive_at(ts, 0)
    if len(alive) < N_INDIVIDUALS:
        raise RuntimeError(
            f"Only {len(alive)} alive individuals < {N_INDIVIDUALS}"
        )
    chosen = rng.choice(alive, size=N_INDIVIDUALS, replace=False)
    nodes = []
    for ind_id in chosen:
        nodes.extend(ts.individual(ind_id).nodes)
    ts = ts.simplify(samples=nodes, keep_input_roots=True)

    # Normalize allele encoding before overlay
    ts = _normalize_slim_alleles(ts)

    # Build a rate map that's zero at SLiM TE positions, so the overlay
    # can't collide with existing TEs. base_rate is Q-scaled (see above).
    rate_map = _build_no_collision_rate_map(
        ts, base_rate=overlay_rate, sequence_length=ts.sequence_length,
    )
    ts = msprime.sim_mutations(
        ts, rate=rate_map,
        model=msprime.BinaryMutationModel(),
        keep=True,
        discrete_genome=True,
        random_seed=seed,
    )
    return ts


def _is_slim_te(mut):
    """Tight test: True iff mutation has SLiM-origin metadata."""
    meta = mut.metadata
    if not isinstance(meta, dict):
        return False
    ml = meta.get("mutation_list")
    if not ml:
        return False
    return "slim_time" in ml[0]


_COVERAGE_MASK_CACHE = "unset"


def _get_coverage_mask():
    """Lazily build (and cache) the CoverageMask from config; None if disabled."""
    global _COVERAGE_MASK_CACHE
    if _COVERAGE_MASK_CACHE == "unset":
        from .config import COVERAGE_MASK_BED
        if COVERAGE_MASK_BED:
            from .demography import CoverageMask
            _COVERAGE_MASK_CACHE = CoverageMask.from_bed(COVERAGE_MASK_BED)
        else:
            _COVERAGE_MASK_CACHE = None
    return _COVERAGE_MASK_CACHE


def window_covariate(ts, window: GenomicWindow):
    """Covariate vector: [gene_density, recomb_scaled, te_density, snp_density],
    plus a 5th masked-fraction element when MASK_AS_COVARIATE is on."""
    from .config import MASK_AS_COVARIATE
    n_te  = 0
    n_snp = 0
    for var in ts.variants():
        muts = list(var.site.mutations)
        if any(_is_slim_te(m) for m in muts):
            n_te += 1
        else:
            freq = float(var.genotypes.mean())
            if min(freq, 1.0 - freq) >= MAF_MIN:
                n_snp += 1
    w_len_mb = (window.end - window.start) / 1e6
    te_density  = n_te  / w_len_mb if w_len_mb > 0 else 0.0
    snp_density = n_snp / w_len_mb if w_len_mb > 0 else 0.0
    base = [
        window.gene_density,
        window.mean_recomb_scaled,
        te_density,
        snp_density,
    ]
    if MASK_AS_COVARIATE:
        mask = _get_coverage_mask()
        mf = (mask.masked_fraction(window.chrom, window.start, window.end)
              if mask is not None else 0.0)
        base.append(mf)
    return np.array(base, dtype=np.float32)


class DFEBurstSimulator:
    """
    Run N_TRAIN_WINDOWS SLiM simulations under one shared theta (7 params).

    SLiM natively generates m1 mutations at a per-bp rate that is
    lambda * multiplier during the burst interval, lambda otherwise.
    Per-mutation s is drawn from the DFE mixture in a late() callback.

    Override `_run_slim` for tests without SLiM installed.
    """
    def __init__(
        self,
        demography:  MSMCDemography,
        sampler:     WindowSampler,
        n_windows:   int = N_TRAIN_WINDOWS,
        slim_script: Optional[str] = None,
    ):
        self.demography = demography
        self.sampler    = sampler
        self.n_windows  = n_windows
        self.slim_script = slim_script or SLIM_SCRIPT

        self._dem_dir = make_tmpdir("te_dfe_dem_")
        self._dem_path = os.path.join(self._dem_dir, "demog.tsv")
        demography.write_epoch_schedule(self._dem_path)

    def __del__(self):
        try:
            shutil.rmtree(self._dem_dir, ignore_errors=True)
        except Exception:
            pass

    def __call__(self, seed: int, debug_dump_dir: Optional[str] = None,
                  theta_provided=None, n_windows_override=None,
                  window_seeds=None, windows_in=None):
        """
        Run one simulation replicate.

        If `theta_provided` is given (np.array of shape (N_PARAMS,)), use
        it directly as the simulation's parameter vector. Otherwise,
        sample fresh theta from the prior.

        Sharding support (for sharded 01_simulate / benchmark sims):
          * n_windows_override: simulate this many windows instead of
            self.n_windows (one shard simulates a chunk of a replicate).
          * window_seeds: explicit per-window SLiM seeds (length must match the
            window count). Makes each window's seed a deterministic function of
            its global index, so chunk boundaries never change results. When
            given, each window's location is sampled independently rather than
            via the joint covariate-space stratification.

        Returns (windows_out, theta) where windows_out is a list of
        (ts, covariate, recomb_map) per window and theta is the 6-dim
        parameter vector.
        """
        rng = np.random.default_rng(seed)

        # Either use supplied theta or sample fresh
        if theta_provided is not None:
            theta = np.asarray(theta_provided, dtype=np.float32)
            if theta.shape != (6,):
                raise ValueError(
                    f"theta_provided must have shape (6,), got {theta.shape}"
                )
        else:
            theta = DFEBurstPrior.sample(rng)

        # Unpack to the values SLiM needs (pure gamma DFE, no pi_neutral)
        lam         = float(np.exp(float(theta[0])))
        gamma_shape = float(np.exp(float(theta[1])))
        mean_s      = float(np.exp(float(theta[2])))
        t_start_ago = int(round(10.0 ** float(theta[3])))
        duration    = int(round(float(np.exp(float(theta[4])))))
        mult        = float(np.exp(float(theta[5])))

        # Convert burst from (gen ago, duration, multiplier) to SLiM ticks
        burst_tick_start, burst_tick_end = burst_interval_to_ticks(
            t_start_ago, duration, SIM_LENGTH,
        )

        # Q-rescaling of the TIME axis (parse-time loop bounds must be scaled
        # Python-side; rates / Ne / recomb are scaled inside the SLiM script).
        # Q = 1 is exact. We scale SIM_END and the burst ticks consistently and
        # clamp so the burst stays a non-empty interval within [1, sim_end].
        Q = float(SLIM_Q)
        if Q != 1.0:
            sim_end_scaled = max(2, int(round(SIM_LENGTH / Q)))
            bs = int(round(burst_tick_start / Q))
            be = int(round(burst_tick_end / Q))
            bs = max(1, min(bs, sim_end_scaled))
            be = max(bs + 1, min(be, sim_end_scaled + 1))
            burst_tick_start, burst_tick_end = bs, be
        else:
            sim_end_scaled = SIM_LENGTH

        debug_root = None
        if debug_dump_dir is not None:
            from .config import PARAM_NAMES, PARAM_LABELS
            from . import sim_introspect
            debug_root = Path(debug_dump_dir) / f"sim_{seed:06d}"
            debug_root.mkdir(parents=True, exist_ok=True)
            sim_introspect.write_theta(
                debug_root / "theta.txt",
                theta, PARAM_NAMES, PARAM_LABELS,
            )

        n_win = (n_windows_override if n_windows_override is not None
                 else self.n_windows)
        if windows_in is not None:
            # Pre-built windows (e.g. reconstructed from a stratified manifest).
            # Their locations are fixed; window_seeds still drive SLiM RNG.
            if len(windows_in) != n_win:
                raise ValueError(
                    f"windows_in length {len(windows_in)} != n_win {n_win}")
        elif window_seeds is not None:
            if len(window_seeds) != n_win:
                raise ValueError(
                    f"window_seeds length {len(window_seeds)} != n_win {n_win}")
            # Sharded path: each window's location sampled independently.
            windows_in = [self.sampler.sample(rng) for _ in range(n_win)]
        else:
            windows_in = self.sampler.sample_n(
                n_win, rng, cover_covariate_space=True,
            )
        windows_out = []
        # Side channel (does not change the returned tuple shape): the (chrom,
        # start) of each window, so callers can pass absolute coords to the
        # processor for coverage masking without a 4-tuple breaking change.
        self.last_window_coords = []
        for w_idx, gw in enumerate(windows_in):
            # Optional: drop windows that are mostly uncallable under the
            # coverage mask (callable fraction below MIN_CALLABLE_FRACTION).
            # For pre-built / manifest windows we cannot resample here, so we
            # warn; for internally sampled windows the caller should prefer a
            # mask-aware sampler. Off by default.
            from .config import DROP_FULLY_MASKED_WINDOWS, MIN_CALLABLE_FRACTION
            if DROP_FULLY_MASKED_WINDOWS:
                _mask = _get_coverage_mask()
                if _mask is not None:
                    mf = _mask.masked_fraction(gw.chrom, gw.start, gw.end)
                    if (1.0 - mf) < MIN_CALLABLE_FRACTION:
                        import warnings
                        warnings.warn(
                            f"window {gw.chrom}:{gw.start} callable fraction "
                            f"{1.0-mf:.2f} < {MIN_CALLABLE_FRACTION}; consider "
                            f"excluding it (kept, but flagged).")
            if window_seeds is not None:
                w_seed = int(window_seeds[w_idx])
            else:
                w_seed = int(rng.integers(1, 1_000_000_000))
            if debug_root is not None:
                workdir = debug_root / f"_workdir_window_{w_idx:02d}"
                workdir.mkdir(parents=True, exist_ok=True)
                workdir = str(workdir)
                cleanup = False
            else:
                workdir = make_tmpdir("te_dfe_w_")
                cleanup = True
            out_trees  = os.path.join(workdir, "sim.trees")
            try:
                # Background selection: build this window's coding intervals
                # (real GTF or synthetic) and write them for SLiM. Off -> None.
                coding_path = None
                if BG_SELECTION:
                    seq_len_w = gw.end - gw.start
                    coding_iv = build_coding_intervals(gw, seq_len_w, rng)
                    coding_path = os.path.join(workdir, "coding.tsv")
                    write_coding_file(coding_iv, coding_path)
                self._run_slim(
                    demog_path=self._dem_path,
                    recomb_path=gw.recomb_rate_file,
                    out_trees=out_trees,
                    seq_len=gw.end - gw.start,
                    seed=w_seed,
                    lam=lam,
                    gamma_shape=gamma_shape,
                    mean_s=mean_s,
                    burst_tick_start=burst_tick_start,
                    burst_tick_end=burst_tick_end,
                    multiplier=mult,
                    sim_end=sim_end_scaled,
                    coding_path=coding_path,
                )
                ts = postprocess_ts(
                    out_trees, gw, self.demography, w_seed, rng,
                )
                cov = window_covariate(ts, gw)

                if debug_root is not None:
                    sim_introspect.write_window_artifacts(
                        workdir=debug_root,
                        window_idx=w_idx,
                        ts=ts,
                        schedule=[],  # no schedule in rate-driven model
                        covariate=cov,
                        sim_length=SIM_LENGTH,
                        maf_min=MAF_MIN,
                    )
                    ts.dump(
                        str(debug_root / f"sim_full_window_{w_idx:02d}.trees"),
                    )
                    sim_introspect.write_substituted_slim_script(
                        out_path=debug_root / f"slim_substituted_window_{w_idx:02d}.txt",
                        template_path=self.slim_script,
                        seq_len=gw.end - gw.start,
                        sim_end=SIM_LENGTH,
                        lam=lam,
                        gamma_shape=gamma_shape,
                        mean_s=mean_s,
                        burst_tick_start=burst_tick_start,
                        burst_tick_end=burst_tick_end,
                        multiplier=mult,
                        demog_path=self._dem_path,
                        recomb_path=gw.recomb_rate_file,
                        output_path=out_trees,
                        seed=w_seed,
                    )
            finally:
                if cleanup:
                    shutil.rmtree(workdir, ignore_errors=True)
            # 4-tuple return (uniform with the context model): the 4th
            # element is the window's gene_intervals (used by the context
            # channels and by background selection; ignored by the plain
            # simple processor).
            windows_out.append((ts, cov, gw.recomb_map, gw.gene_intervals))
            self.last_window_coords.append((gw.chrom, gw.start))

        return windows_out, theta

    def _run_slim(self, demog_path, recomb_path, out_trees,
                  seq_len, seed,
                  lam, gamma_shape, mean_s,
                  burst_tick_start, burst_tick_end, multiplier,
                  sim_end=SIM_LENGTH, coding_path=None):
        cmd = [
            "slim",
            "-d", f"SEQ_LEN={int(seq_len)}",
            "-d", f"SIM_END={int(sim_end)}",
            "-d", f"Q={float(SLIM_Q):.8f}",
            "-d", f"LAMBDA={lam:.10e}",
            "-d", f"GAMMA_SHAPE={gamma_shape:.8f}",
            "-d", f"MEAN_S={mean_s:.8f}",
            "-d", f"BURST_TICK_START={int(burst_tick_start)}",
            "-d", f"BURST_TICK_END={int(burst_tick_end)}",
            "-d", f"MULTIPLIER={multiplier:.6f}",
            "-d", f"DEMOG_PATH='{demog_path}'",
            "-d", f"RECOMB_PATH='{recomb_path}'",
            "-d", f"OUTPUT_PATH='{out_trees}'",
        ]
        # Background selection: pass USE_BGS + the coding-interval file + the
        # deleterious DFE / rate params. When off, USE_BGS=0 and the SLiM script
        # runs the original uniform-rate, TE-only path (no m2, no coding map).
        if coding_path is not None:
            cmd += [
                "-d", "USE_BGS=1",
                "-d", f"CODING_PATH='{coding_path}'",
                "-d", f"DEL_SHAPE={float(BG_DEL_SHAPE):.8f}",
                "-d", f"DEL_MEAN_S={float(BG_DEL_MEAN_S):.10e}",
                "-d", f"DEL_MU={float(BG_DEL_MU):.10e}",
                "-d", f"DEL_DOMINANCE={float(BG_DEL_DOMINANCE):.6f}",
            ]
        else:
            cmd += ["-d", "USE_BGS=0"]
        cmd += ["-s", str(seed), self.slim_script]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Surface SLiM's actual error so we can debug it
            raise RuntimeError(
                f"SLiM failed (exit {result.returncode}).\n"
                f"--- stdout ---\n{result.stdout}\n"
                f"--- stderr ---\n{result.stderr}\n"
            )
