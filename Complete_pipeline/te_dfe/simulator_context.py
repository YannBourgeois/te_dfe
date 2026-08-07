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
    BG_SELECTION,
    MAX_INSERTIONS_PER_WINDOW,
    N_TRAIN_WINDOWS, MAF_MIN,
)
from .demography import make_tmpdir, MSMCDemography, GenomicWindow, WindowSampler

from .prior import DFEBurstPrior, compute_s_local, compute_lambda_local


SLIM_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "slim", "te_dfe_burst_context.slim",
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


def postprocess_ts(trees_path, window: GenomicWindow,
                    demography: MSMCDemography, seed: int, rng):
    """Recapitate, simplify, overlay neutral SNPs (collision-free)."""
    ts = tskit.load(trees_path)

    demog = msprime.Demography()
    demog.add_population(name="pop_0", initial_size=1)
    demog.add_population(name="p1", initial_size=demography.ancestral_ne)
    for ev in demography.to_msprime_events():
        demog.add_population_parameters_change(
            time=ev.time, initial_size=ev.initial_size, population="p1",
        )
    ts = pyslim.recapitate(
        ts,
        recombination_rate=window.recomb_map,
        demography=demog,
        random_seed=seed,
    )

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
    # can't collide with existing TEs.
    rate_map = _build_no_collision_rate_map(
        ts, base_rate=NEUTRAL_MU, sequence_length=ts.sequence_length,
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


def window_covariate(ts, window: GenomicWindow):
    """4-element covariate: [gene_density, recomb_scaled, te_density, snp_density]."""
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
    return np.array([
        window.gene_density,
        window.mean_recomb_scaled,
        te_density,
        snp_density,
    ], dtype=np.float32)


def _coding_intervals_window_relative(gene_intervals, seq_len, window_start, rng):
    """Window-relative coding intervals for BGS. Real annotation when present,
    else a synthetic exon landscape (so a BGS run is never silently TE-only).
    Mirrors simulator_simple.build_coding_intervals but works from the raw
    gene_intervals + window_start the context simulator already threads."""
    from .config import BG_USE_GTF
    from .simulator_simple import flatten_intervals
    if BG_USE_GTF and gene_intervals is not None and len(gene_intervals) > 0:
        out = []
        for (g0, g1) in gene_intervals:
            a = max(0, int(g0) - int(window_start))
            b = min(int(seq_len), int(g1) - int(window_start))
            if b > a:
                out.append((a, b))
        if out:
            return flatten_intervals(out)   # CDS overlap -> disjoint set
    from .simulator_simple import generate_synthetic_exons
    from .config import (
        BG_MEAN_EXON_LEN, BG_SD_EXON_LEN, BG_MEAN_INTEREXON, BG_SD_INTEREXON,
    )
    return flatten_intervals(generate_synthetic_exons(
        int(seq_len), rng,
        mean_exon=BG_MEAN_EXON_LEN, sd_exon=BG_SD_EXON_LEN,
        mean_gap=BG_MEAN_INTEREXON, sd_gap=BG_SD_INTEREXON,
    ))


def _merge_bin_coding_segments(seq_len, coding_intervals, bin_starts,
                               lambda_local, bin_width):
    """Intersect the uniform bin grid (carrying per-bin TE rate lambda_local)
    with the coding intervals. Returns a contiguous tiling of [0, seq_len):
        seg_ends   inclusive end of each segment (last == seq_len-1)
        seg_coding 1 if the segment is inside a coding interval, else 0
        seg_rate   lambda_local of the bin the segment starts in (used by
                   SLiM only for non-coding segments)
    Boundaries are the union of bin starts and coding edges, so every segment
    is wholly coding or wholly non-coding."""
    L = int(seq_len); bw = int(bin_width); n_lam = len(lambda_local)
    bounds = {0, L}
    for s in bin_starts:
        s = int(s)
        if 0 < s < L:
            bounds.add(s)
    cod = []
    for (a, b) in coding_intervals:
        a = max(0, int(a)); b = min(L, int(b))
        if b > a:
            cod.append((a, b))
            if 0 < a < L: bounds.add(a)
            if 0 < b < L: bounds.add(b)
    bounds = sorted(bounds)
    seg_ends, seg_coding, seg_rate = [], [], []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b <= a:
            continue
        mid = (a + b) // 2
        is_c = any(ca <= mid < cb for (ca, cb) in cod)
        bidx = min(a // bw, n_lam - 1)
        seg_ends.append(b - 1)
        seg_coding.append(1 if is_c else 0)
        seg_rate.append(float(lambda_local[bidx]))
    return seg_ends, seg_coding, seg_rate


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
                 theta_provided=None, n_windows_override: Optional[int] = None,
                 window_seeds: Optional[list] = None):
        """
        Run one simulation replicate.

        Returns (windows_out, theta) where windows_out is a list of
        (ts, covariate, recomb_map, gene_intervals) per window and
        theta is the parameter vector.

        Sharding support (for the benchmark simulator):
          * theta_provided: use this theta instead of drawing from the prior
            (lets every shard of a scenario share one theta).
          * n_windows_override: simulate this many windows instead of
            self.n_windows (one shard simulates a chunk).
          * window_seeds: explicit per-window SLiM seeds (length must match the
            window count). Makes each window's seed a deterministic function of
            its global index, so chunk boundaries never change results. When
            given, each window is sampled independently (no joint
            covariate-space stratification — appropriate for many-window
            benchmark sampling).
        """
        rng = np.random.default_rng(seed)

        # Sample theta directly — no internal/external distinction in Bundle 2.
        # The 9 parameters in theta drive the simulator and are what the
        # network infers.
        ne_harmonic = self.demography.harmonic_mean_ne()
        if theta_provided is not None:
            theta = np.asarray(theta_provided, dtype=np.float64)
        else:
            theta = DFEBurstPrior.sample(rng, ne_harmonic=ne_harmonic)

        # Unpack into named scalars for the simulator
        unpacked = DFEBurstPrior.unpack(theta)
        lambda_global = unpacked["lambda_global"]
        s_global      = unpacked["s_global"]
        alpha_d       = unpacked["alpha_d"]
        alpha_r       = unpacked["alpha_r"]
        gamma_d       = unpacked["gamma_d"]
        gamma_r       = unpacked["gamma_r"]
        t_start_ago   = unpacked["t_start"]
        duration      = unpacked["duration"]
        mult          = unpacked["mult"]

        # Convert burst from (gen ago, duration, multiplier) to SLiM ticks
        burst_tick_start, burst_tick_end = burst_interval_to_ticks(
            t_start_ago, duration, SIM_LENGTH,
        )

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

        n_win = n_windows_override if n_windows_override is not None else self.n_windows
        if window_seeds is not None:
            # Sharded path: each window sampled independently (no joint
            # stratification), with explicit chunk-invariant seeds.
            if len(window_seeds) != n_win:
                raise ValueError(
                    f"window_seeds length {len(window_seeds)} != n_win {n_win}")
            windows_in = [self.sampler.sample(rng) for _ in range(n_win)]
        else:
            windows_in = self.sampler.sample_n(
                n_win, rng, cover_covariate_space=True,
            )
        windows_out = []
        for w_idx, gw in enumerate(windows_in):
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

            # Pre-compute per-bin features and lambda_local for this window.
            # Write to a text file SLiM reads at startup.
            feature_path = os.path.join(workdir, "features.tsv")
            bin_starts, _fd, _fr, lambda_local = self._write_feature_map(
                feature_path,
                seq_len=gw.end - gw.start,
                window_start=gw.start,
                gene_intervals=gw.gene_intervals,
                recomb_map=gw.recomb_map,
                lambda_global=lambda_global,
                gamma_d=gamma_d,
                gamma_r=gamma_r,
            )

            # Background selection: pre-merge the bin grid with the coding
            # intervals so SLiM gets one per-segment table. When BG_SELECTION
            # is off, seg_path stays None and the SLiM script runs the original
            # TE-only (USE_BGS=0) path.
            seg_path = None
            if BG_SELECTION:
                seg_path = os.path.join(workdir, "segments.tsv")
                self._write_segments_map(
                    seg_path,
                    seq_len=gw.end - gw.start,
                    window_start=gw.start,
                    gene_intervals=gw.gene_intervals,
                    bin_starts=bin_starts,
                    lambda_local=lambda_local,
                    rng=np.random.default_rng(w_seed),
                )

            try:
                self._run_slim(
                    demog_path=self._dem_path,
                    recomb_path=gw.recomb_rate_file,
                    out_trees=out_trees,
                    seq_len=gw.end - gw.start,
                    seed=w_seed,
                    lambda_global=lambda_global,
                    s_global=s_global,
                    alpha_d=alpha_d,
                    alpha_r=alpha_r,
                    gamma_d=gamma_d,
                    gamma_r=gamma_r,
                    feature_path=feature_path,
                    burst_tick_start=burst_tick_start,
                    burst_tick_end=burst_tick_end,
                    multiplier=mult,
                    seg_path=seg_path,
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
                        schedule=[],
                        covariate=cov,
                        sim_length=SIM_LENGTH,
                        maf_min=MAF_MIN,
                    )
                    ts.dump(
                        str(debug_root / f"sim_full_window_{w_idx:02d}.trees"),
                    )
            finally:
                if cleanup:
                    shutil.rmtree(workdir, ignore_errors=True)
            windows_out.append(
                (ts, cov, gw.recomb_map, gw.gene_intervals)
            )

        return windows_out, theta

    # -----------------------------------------------------------------
    # Bundle 2 helpers: pre-compute per-bin features for SLiM
    # -----------------------------------------------------------------
    def _write_feature_map(self, path, seq_len, window_start,
                            gene_intervals, recomb_map,
                            lambda_global, gamma_d, gamma_r,
                            bin_width_bp=1000):
        """
        Pre-compute per-bin feature_d, feature_r, and lambda_local. Write
        a TSV file SLiM can read:
            bin_start_bp  feature_d  feature_r  lambda_local
        """
        from .config import (
            DIST_TO_GENE_NORMALIZER,
            RECOMB_TYPICAL, RECOMB_LOG_SCALE, RECOMB_MIN_FOR_LOG,
        )
        n_bins = int(np.ceil(seq_len / bin_width_bp))
        bin_starts = np.arange(n_bins) * bin_width_bp
        bin_mids   = bin_starts + bin_width_bp // 2

        # feature_d: log10(dist to nearest gene in absolute genomic
        # coords) / log10(NORM). gw.gene_intervals are absolute (chrom
        # coords); convert by adding window_start to bin_mids.
        abs_pos = bin_mids + window_start
        if gene_intervals is not None and len(gene_intervals) > 0:
            gi = np.asarray(gene_intervals, dtype=np.float64)
            starts = gi[:, 0]
            ends   = gi[:, 1]
            pos_2d   = abs_pos[:, None]
            starts_2 = starts[None, :]
            ends_2   = ends[None, :]
            inside = (pos_2d >= starts_2) & (pos_2d <= ends_2)
            dist_to_start = np.abs(pos_2d - starts_2)
            dist_to_end   = np.abs(pos_2d - ends_2)
            dist_each = np.where(inside, 0.0,
                                  np.minimum(dist_to_start, dist_to_end))
            dist_per_bin = dist_each.min(axis=1)
        else:
            dist_per_bin = np.full(n_bins, DIST_TO_GENE_NORMALIZER, dtype=np.float64)

        log_norm_dist = np.log10(DIST_TO_GENE_NORMALIZER)
        feature_d = np.log10(np.maximum(dist_per_bin, 1.0)) / log_norm_dist
        feature_d = feature_d.astype(np.float64)

        # feature_r: log10(local recomb rate) / log10(NORM), at bin
        # midpoint position (relative to window start).
        if recomb_map is not None:
            try:
                rates = np.asarray(
                    recomb_map.get_rate(bin_mids.astype(np.float64)),
                    dtype=np.float64,
                )
            except Exception:
                rates = np.full(n_bins, 1e-8, dtype=np.float64)
        else:
            rates = np.full(n_bins, 1e-8, dtype=np.float64)
        rates_clipped = np.maximum(rates, RECOMB_MIN_FOR_LOG)
        feature_r = np.clip(
            np.log10(rates_clipped / RECOMB_TYPICAL) / RECOMB_LOG_SCALE,
            -2.0, 2.0,
        )

        # lambda_local per bin
        log_lam = (np.log(lambda_global)
                   + gamma_d * feature_d
                   + gamma_r * feature_r)
        lambda_local = np.exp(log_lam)
        # Safety cap: lambda * sim_length should not exceed any sane bound
        lambda_local = np.minimum(lambda_local, 1e-5)

        with open(path, "w") as f:
            for i in range(n_bins):
                f.write(f"{int(bin_starts[i])}\t"
                        f"{feature_d[i]:.6f}\t"
                        f"{feature_r[i]:.6f}\t"
                        f"{lambda_local[i]:.10e}\n")
        return bin_starts, feature_d, feature_r, lambda_local

    # -----------------------------------------------------------------
    # Bundle 2 + BGS: pre-merge the bin grid with coding intervals so SLiM
    # gets one per-segment table (the TE rate varies per bin, BGS needs
    # coding segments; their intersection is what the SLiM rate map needs).
    # -----------------------------------------------------------------
    def _write_segments_map(self, path, seq_len, window_start,
                            gene_intervals, bin_starts, lambda_local,
                            rng, bin_width_bp=1000):
        """Write the merged (bin x coding) segment table SLiM reads under BGS:
            seg_end_inclusive  is_coding(0/1)  te_rate
        Coding intervals come from the window's gene annotation (made
        window-relative); if none, a synthetic exon landscape is drawn so a
        BGS run is never silently TE-only.
        """
        coding = _coding_intervals_window_relative(
            gene_intervals, seq_len, window_start, rng)
        seg_ends, seg_cod, seg_rate = _merge_bin_coding_segments(
            seq_len, coding, bin_starts, lambda_local, bin_width_bp)
        with open(path, "w") as f:
            for e, c, r in zip(seg_ends, seg_cod, seg_rate):
                f.write(f"{int(e)}\t{int(c)}\t{float(r):.10e}\n")

    def _run_slim(self, demog_path, recomb_path, out_trees,
                  seq_len, seed,
                  lambda_global, s_global,
                  alpha_d, alpha_r, gamma_d, gamma_r,
                  feature_path,
                  burst_tick_start, burst_tick_end, multiplier,
                  seg_path=None):
        cmd = [
            "slim",
            "-d", f"SEQ_LEN={int(seq_len)}",
            "-d", f"SIM_END={SIM_LENGTH}",
            "-d", f"LAMBDA_GLOBAL={lambda_global:.10e}",
            "-d", f"S_GLOBAL={s_global:.10e}",
            "-d", f"ALPHA_D={alpha_d:.8f}",
            "-d", f"ALPHA_R={alpha_r:.8f}",
            "-d", f"GAMMA_D={gamma_d:.8f}",
            "-d", f"GAMMA_R={gamma_r:.8f}",
            "-d", f"BIN_WIDTH_BP=1000",
            "-d", f"FEATURE_PATH='{feature_path}'",
            "-d", f"BURST_TICK_START={int(burst_tick_start)}",
            "-d", f"BURST_TICK_END={int(burst_tick_end)}",
            "-d", f"MULTIPLIER={multiplier:.6f}",
            "-d", f"DEMOG_PATH='{demog_path}'",
            "-d", f"RECOMB_PATH='{recomb_path}'",
            "-d", f"OUTPUT_PATH='{out_trees}'",
        ]
        # Background selection: pass USE_BGS=1 + the merged per-segment table +
        # the deleterious DFE / rate params. When off, USE_BGS=0 and the SLiM
        # script runs the original per-bin, TE-only path (no m2, no segments).
        if seg_path is not None:
            from .config import (
                BG_DEL_SHAPE, BG_DEL_MEAN_S, BG_DEL_MU, BG_DEL_DOMINANCE,
            )
            cmd += [
                "-d", "USE_BGS=1",
                "-d", f"SEG_PATH='{seg_path}'",
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
            raise RuntimeError(
                f"SLiM failed (exit {result.returncode}).\n"
                f"--- stdout ---\n{result.stdout}\n"
                f"--- stderr ---\n{result.stderr}\n"
            )
