"""
demography.py — Genomic input data utilities.

Three responsibilities:
  1. MSMCDemography     : parse MSMC output → SLiM epoch schedule + msprime events
  2. GenomeAnnotation   : parse GTF/GFF → gene intervals per chromosome
  3. WindowSampler      : sample random 5Mb intervals from the real genome
                          using the real recombination map and gene annotation

The WindowSampler returns GenomicWindow objects containing everything
SLiM needs (per-window recombination rate file, gene density, etc.)
and everything the processor needs (window-relative gene intervals).
"""

import os
import shutil
import tempfile
import weakref
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import msprime
import numpy as np


def make_tmpdir(prefix):
    """mkdtemp under $TE_DFE_TMPDIR if set (created if missing), else the
    default temp location. Lets cluster runs put per-window scratch on a large
    filesystem instead of the small, shared node /tmp (which fills under high
    MAX_CONCURRENT and triggers 'No space left on device')."""
    base = os.environ.get("TE_DFE_TMPDIR") or None
    if base:
        os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix=prefix, dir=base)


# ===========================================================================
# 1. MSMC demography
# ===========================================================================

@dataclass
class DemographicEpoch:
    """One epoch of constant Ne in forward (SLiM) time."""
    slim_gen_start: int
    ne:             int


class MSMCDemography:
    """
    Parses MSMC output and converts to SLiM-compatible epoch schedule.

    MSMC time is in units of (generations × μ); λ = 1 / (2 · Ne · μ).
    We convert:
        Ne    = 1 / (2 · μ · λ)
        t_gen = t_msmc / μ
    """

    def __init__(self, epochs: List[DemographicEpoch], ancestral_ne: int,
                 sim_length: int):
        self.epochs       = epochs
        self.ancestral_ne = ancestral_ne
        self.sim_length   = sim_length

    @classmethod
    def from_file(cls, path: str, mu: float, sim_length: int = 20_000,
                  min_ne: int = 100, max_ne: int = 1_000_000) -> "MSMCDemography":
        """
        Parse an MSMC output file.

        Format: tab-separated, header line ignored. Columns:
            time_left  time_right  lambda_00
        Rows are ordered from most recent (row 0) to most ancient.
        """
        times_msmc, lambdas = [], []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(("time", "#")):
                    continue
                parts = line.split()
                # MSMC2 output columns:
                #   [0] time_index, [1] left_time_boundary,
                #   [2] right_time_boundary, [3] lambda
                # We use right_time_boundary as the epoch start time
                # (in mu-scaled units) and lambda as the inverse coalescence
                # rate. Convert: Ne = 1 / (2 * mu * lambda).
                if len(parts) < 4:
                    continue
                try:
                    times_msmc.append(float(parts[2]))   # right_time_boundary
                    lambdas.append(float(parts[3]))      # lambda
                except ValueError:
                    continue

        if not times_msmc:
            raise ValueError(f"No MSMC data parsed from {path}")

        times_msmc = np.array(times_msmc)
        lambdas    = np.array(lambdas)

        ne_values  = np.clip(1.0 / (2.0 * mu * lambdas), min_ne, max_ne).astype(int)
        t_gen_ago  = times_msmc / mu

        # MSMC rows are sorted by time_right ascending: oldest is last
        # Convert each row to a SLiM forward generation
        epochs       = []
        ancestral_ne = int(ne_values[-1])

        for i in range(len(t_gen_ago)):
            slim_gen = int(round(sim_length - t_gen_ago[i]))
            if slim_gen < 1:
                # This MSMC epoch predates the simulation window
                ancestral_ne = int(ne_values[i])
                continue
            epochs.append(DemographicEpoch(
                slim_gen_start=max(1, slim_gen),
                ne=int(ne_values[i]),
            ))

        # Sort ascending by SLiM generation; deduplicate consecutive same-Ne
        epochs.sort(key=lambda e: e.slim_gen_start)
        deduped = []
        for ep in epochs:
            if deduped and deduped[-1].ne == ep.ne:
                continue
            deduped.append(ep)

        # Ensure the first epoch is generation 1
        if not deduped or deduped[0].slim_gen_start > 1:
            deduped.insert(0, DemographicEpoch(slim_gen_start=1, ne=ancestral_ne))

        return cls(epochs=deduped, ancestral_ne=ancestral_ne, sim_length=sim_length)

    @classmethod
    def constant(cls, ne: int, sim_length: int = 20_000) -> "MSMCDemography":
        """Convenience: constant Ne throughout."""
        return cls(
            epochs=[DemographicEpoch(slim_gen_start=1, ne=ne)],
            ancestral_ne=ne,
            sim_length=sim_length,
        )

    def write_epoch_schedule(self, path: str):
        """Write TSV: slim_generation <TAB> ne, one line per change."""
        with open(path, "w") as f:
            for ep in self.epochs:
                f.write(f"{ep.slim_gen_start}\t{ep.ne}\n")

    def to_msprime_events(self):
        """Full time-ago demography for msprime: (present_size, events).
        
         Returns the present-day population size and a list of
         msprime.PopulationParametersChange events (time in generations-ago) that,
         applied on top of that present size, reproduce the whole history.
        
        Convention (the thing that is easy to get wrong): epochs are stored in
        FORWARD time as (slim_gen_start, ne), sorted ascending, so epochs[-1] is the
        MOST RECENT. Going BACKWARD from the present, the size starts at the most
        recent epoch's Ne and, at each boundary tau = sim_length - epochs[i].
        slim_gen_start, switches to the OLDER neighbour epochs[i-1].ne. The i=1
        boundary carries the ancestral Ne, so the ancestral size is INCLUDED here
        (callers must NOT add a separate ancestral floor, and must set the present
        size to the returned present_size, NOT to ancestral_ne).
        """
        eps = sorted(self.epochs, key=lambda e: e.slim_gen_start)
        present_size = int(eps[-1].ne)
        events = []
        for i in range(1, len(eps)):
            t_ago = self.sim_length - eps[i].slim_gen_start
            if t_ago <= 0:
               continue
            events.append(
                  msprime.PopulationParametersChange(
                  time=t_ago,
                  initial_size=eps[i - 1].ne,   # OLDER neighbour
                  population=0,
               )
           )
        return present_size, events

    def __repr__(self):
        return (f"MSMCDemography(n_epochs={len(self.epochs)}, "
                f"ancestral_Ne={self.ancestral_ne}, sim_length={self.sim_length})")

    def integrated_2Ne(self) -> float:
        """
        Integrate 2*Ne(t) over the simulation, in haploid-generations.
        Used to convert the per-haplotype, per-generation, per-bp insertion
        rate (lambda) into a population-level expected number of insertions
        over the simulation duration:

            E[n_te] = lambda * window_length * integrated_2Ne()

        Treats the demography as piecewise constant (one Ne per epoch,
        with the last epoch extended to sim_length).
        """
        total = 0.0
        boundaries = [ep.slim_gen_start for ep in self.epochs] + [self.sim_length]
        for i, ep in enumerate(self.epochs):
            duration = boundaries[i+1] - boundaries[i]
            if duration <= 0:
                continue
            total += 2.0 * ep.ne * duration
        return total

    def harmonic_mean_ne(self) -> float:
        """
        Time-weighted harmonic mean of Ne over the simulation.

            Ne_h = T / sum_i (T_i / Ne_i)

        where T_i is the duration of epoch i and Ne_i is the constant
        effective population size during that epoch.

        Used for the |2 Ne s| binning thresholds in DFE inference. The
        harmonic mean is the right average for DFE work because it
        captures the integrated drift rate over the relevant time
        (variants spend most of their existence in the smallest-Ne
        epochs they pass through).
        """
        sum_t_over_ne = 0.0
        total_t = 0.0
        boundaries = [ep.slim_gen_start for ep in self.epochs] + [self.sim_length]
        for i, ep in enumerate(self.epochs):
            duration = boundaries[i+1] - boundaries[i]
            if duration <= 0 or ep.ne <= 0:
                continue
            sum_t_over_ne += duration / ep.ne
            total_t += duration
        if sum_t_over_ne <= 0:
            return float(self.ancestral_ne)
        return float(total_t / sum_t_over_ne)


# ===========================================================================
# 2. Gene annotation
# ===========================================================================

@dataclass
class GenomeAnnotation:
    """Gene intervals per chromosome from GTF/GFF. 0-based half-open."""
    gene_intervals: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    chrom_lengths:  Dict[str, int]                  = field(default_factory=dict)

    @staticmethod
    def _parse(path: str, feature_type: str, is_gff: bool):
        intervals: Dict[str, List[Tuple[int, int]]] = {}
        with open(path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 8:
                    continue
                chrom, _, ftype, start_s, end_s = parts[:5]
                if ftype != feature_type:
                    continue
                try:
                    start = int(start_s) - 1   # 1-based → 0-based
                    end   = int(end_s)
                except ValueError:
                    continue
                if end <= start:
                    continue
                intervals.setdefault(chrom, []).append((start, end))
        # Sort intervals per chromosome for efficient lookup
        for chrom in intervals:
            intervals[chrom].sort()
        return intervals

    @classmethod
    def from_gtf(cls, path: str, feature_type: str = "CDS") -> "GenomeAnnotation":
        return cls(gene_intervals=cls._parse(path, feature_type, is_gff=False))

    @classmethod
    def from_gff(cls, path: str, feature_type: str = "CDS") -> "GenomeAnnotation":
        return cls(gene_intervals=cls._parse(path, feature_type, is_gff=True))

    def gene_density_in_window(self, chrom: str, start: int, end: int) -> float:
        """
        Fraction of [start, end) overlapped by genes.

        Handles overlapping genes correctly by computing the union of
        all gene-overlap segments.
        """
        genes = self.gene_intervals.get(chrom, [])
        if not genes:
            return 0.0

        w_len = end - start
        # Build sorted list of (start, end) clipped to window
        clipped = []
        for gs, ge in genes:
            if ge <= start or gs >= end:
                continue
            clipped.append((max(gs, start), min(ge, end)))
        if not clipped:
            return 0.0

        # Merge overlapping intervals
        clipped.sort()
        merged = [clipped[0]]
        for s, e in clipped[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        overlap = sum(e - s for s, e in merged)
        return overlap / w_len

    def intervals_in_window(self, chrom: str, start: int, end: int
                            ) -> List[Tuple[int, int]]:
        """Gene intervals clipped and shifted to window-relative coordinates."""
        genes = self.gene_intervals.get(chrom, [])
        result = []
        for gs, ge in genes:
            if ge <= start or gs >= end:
                continue
            lo = max(gs, start) - start
            hi = min(ge, end)   - start
            if hi > lo:
                result.append((lo, hi))
        return result


# ===========================================================================
# 3. Recombination map
# ===========================================================================

def load_hapmap_recomb(path: str, chrom: str) -> msprime.RateMap:
    """
    Load HapMap-format map for one chromosome.
    Columns: Chromosome  Position(bp)  Rate(cM/Mb)  Map(cM)
    Rate cM/Mb → Morgan/bp = rate × 1e-8.
    Returns msprime.RateMap covering [0, last_position].
    """
    positions = [0]
    rates = []

    with open(path) as f:
        # Skip header
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first:
                first = False
                # Skip if header-like (contains non-numeric)
                tokens = line.split()
                try:
                    int(tokens[1])
                except (ValueError, IndexError):
                    continue
                # Otherwise treat first line as data
                parts = tokens
            else:
                parts = line.split()

            if len(parts) < 3 or parts[0] != chrom:
                continue
            try:
                pos = int(parts[1])
                rate = float(parts[2]) * 1e-8     # cM/Mb → Morgan/bp
            except ValueError:
                continue
            if pos <= positions[-1]:
                continue   # skip out-of-order
            positions.append(pos)
            rates.append(rate)

    if not rates:
        raise ValueError(f"No recomb data for chrom={chrom} in {path}")

    return msprime.RateMap(
        position=np.array(positions, dtype=np.float64),
        rate=np.array(rates, dtype=np.float64),
    )


def write_window_recomb_file(full_map: msprime.RateMap,
                             w_start: int, w_end: int, out_path: str
                             ) -> msprime.RateMap:
    """
    Extract the sub-map for [w_start, w_end), write a SLiM rate file
    (window-relative coordinates), and return the window-relative RateMap.

    SLiM rate file format:  end_position <TAB> rate (one segment per line)
    """
    pos   = np.asarray(full_map.position)
    rates = np.asarray(full_map.rate)

    # Find segment containing w_start
    # `pos` are breakpoints; rates[i] applies to [pos[i], pos[i+1])
    idx = np.searchsorted(pos, w_start, side="right") - 1
    idx = max(0, min(idx, len(rates) - 1))

    win_positions = [0]
    win_rates     = []

    cur_rate = rates[idx]
    cur_idx  = idx + 1

    while cur_idx < len(pos) and pos[cur_idx] < w_end:
        win_positions.append(int(pos[cur_idx] - w_start))
        win_rates.append(cur_rate)
        cur_rate = rates[cur_idx]
        cur_idx += 1

    win_positions.append(w_end - w_start)
    win_rates.append(cur_rate)

    # Convert to numpy
    win_positions = np.array(win_positions, dtype=np.float64)
    win_rates     = np.array(win_rates,     dtype=np.float64)

    # Drop any zero-length segments
    keep = np.diff(win_positions) > 0
    if not keep.all():
        new_pos = [win_positions[0]]
        new_rates = []
        for i, k in enumerate(keep):
            if k:
                new_pos.append(win_positions[i + 1])
                new_rates.append(win_rates[i])
        win_positions = np.array(new_pos, dtype=np.float64)
        win_rates     = np.array(new_rates, dtype=np.float64)

    # Write SLiM file: end_position <TAB> rate
    with open(out_path, "w") as f:
        for i, r in enumerate(win_rates):
            end = int(win_positions[i + 1])
            f.write(f"{end}\t{r:.10e}\n")

    return msprime.RateMap(position=win_positions, rate=win_rates)


# ===========================================================================
# 4. Genomic window sampler
# ===========================================================================

@dataclass
class GenomicWindow:
    """A sampled 5 Mb genomic window with all derived data."""
    chrom:              str
    start:              int
    end:                int
    gene_intervals:     List[Tuple[int, int]]
    gene_density:       float
    recomb_map:         msprime.RateMap
    mean_recomb_scaled: float
    recomb_rate_file:   str


class WindowSampler:
    """
    Sample valid 5 Mb genomic windows from a real annotated genome.

    Per-chromosome HapMap recombination maps are loaded once.
    Per-window rate files are written on demand into `workdir`.

    Caller is responsible for cleaning up workdir if needed.
    """

    def __init__(
        self,
        recomb_map_path: str,
        annotation:      GenomeAnnotation,
        window_size:     int = 5_000_000,
        chromosomes:     Optional[List[str]] = None,
        recomb_scale:    float = 1e-8,
        min_gene_density:float = 0.0,
        workdir:         Optional[str] = None,
    ):
        self.annotation       = annotation
        self.window_size      = window_size
        self.recomb_scale     = recomb_scale
        self.min_gene_density = min_gene_density
        if workdir:
            self.workdir = workdir
        else:
            # Auto-created scratch: relocate off /tmp and clean it up at GC/exit
            # (recomb TSVs here are regenerable; the persisted outputs are the
            # trees under SIM_DIR), so this never leaks orphan te_npe_* dirs.
            self.workdir = make_tmpdir("te_npe_")
            weakref.finalize(self, shutil.rmtree, self.workdir, True)

        # Determine chromosomes
        annot_chroms = list(annotation.gene_intervals.keys())
        if chromosomes is None:
            chromosomes = annot_chroms
        else:
            chromosomes = [c for c in chromosomes if c in annot_chroms]

        # Load per-chromosome maps
        self._chrom_maps: Dict[str, msprime.RateMap] = {}
        for chrom in chromosomes:
            try:
                self._chrom_maps[chrom] = load_hapmap_recomb(recomb_map_path, chrom)
            except ValueError:
                pass

        self.chromosomes = list(self._chrom_maps.keys())
        if not self.chromosomes:
            raise ValueError(
                "No chromosomes have both annotation and recomb data."
            )

        # Pre-compute weights for chromosome sampling (proportional to length)
        self._chrom_weights = np.array([
            float(self._chrom_maps[c].position[-1])
            for c in self.chromosomes
        ])
        self._chrom_weights /= self._chrom_weights.sum()

    def window_from_coords(self, chrom: str, w_start: int) -> "GenomicWindow":
        """
        Build a full GenomicWindow from coordinates alone. All derived fields
        (gene intervals/density, recomb map + file, mean scaled recomb) are
        recomputed deterministically from (chrom, start) + the annotation and
        recombination map. This is what lets a stratified window MANIFEST store
        only coordinates: the heavy objects are reconstructed here at sim time.
        """
        cmap   = self._chrom_maps[chrom]
        w_end  = w_start + self.window_size
        gd     = self.annotation.gene_density_in_window(chrom, w_start, w_end)
        try:
            sub = cmap.slice(left=w_start, right=w_end)
            mean_r = float(sub.mean_rate)
        except Exception:
            mean_r = float(cmap.get_rate(np.linspace(
                w_start, w_end - 1, 200, dtype=int)).mean())
        rate_file = os.path.join(
            self.workdir, f"recomb_{chrom}_{w_start}_{w_end}.tsv")
        w_map = write_window_recomb_file(cmap, w_start, w_end, rate_file)
        return GenomicWindow(
            chrom              = chrom,
            start              = w_start,
            end                = w_end,
            gene_intervals     = self.annotation.intervals_in_window(
                                   chrom, w_start, w_end),
            gene_density       = gd,
            recomb_map         = w_map,
            mean_recomb_scaled = mean_r / self.recomb_scale,
            recomb_rate_file   = rate_file,
        )

    def sample(self, rng: np.random.Generator,
               max_attempts: int = 100) -> GenomicWindow:
        """Sample one valid window."""
        for _ in range(max_attempts):
            chrom = rng.choice(self.chromosomes, p=self._chrom_weights)
            cmap  = self._chrom_maps[chrom]
            chrom_len = int(cmap.position[-1])
            if chrom_len <= self.window_size:
                continue

            w_start = int(rng.integers(0, chrom_len - self.window_size))
            w_end   = w_start + self.window_size

            gd = self.annotation.gene_density_in_window(chrom, w_start, w_end)
            if gd < self.min_gene_density:
                continue

            mean_r = cmap.mean_rate
            # mean_rate is a property; for sub-window mean use slice
            try:
                # msprime >= 1.0 has slice(); compute mean over window
                sub = cmap.slice(left=w_start, right=w_end)
                mean_r = float(sub.mean_rate)
            except Exception:
                # Fallback: numerical integration
                mean_r = float(cmap.get_rate(np.linspace(
                    w_start, w_end - 1, 200, dtype=int)).mean())

            if mean_r <= 0 or not np.isfinite(mean_r):
                continue

            return self.window_from_coords(chrom, w_start)

        raise RuntimeError(f"Could not sample valid window in {max_attempts} attempts.")

    def sample_n(self, n: int, rng: np.random.Generator,
                 cover_covariate_space: bool = True) -> List[GenomicWindow]:
        """
        Sample n valid windows.

        If cover_covariate_space=True, oversample then stratify across
        a 5×5 grid of (gene_density, recomb_rate) quintiles for slope
        identifiability.
        """
        if not cover_covariate_space:
            return [self.sample(rng) for _ in range(n)]

        oversample = max(n * 3, n + 5)
        pool = []
        for _ in range(oversample):
            try:
                pool.append(self.sample(rng))
            except RuntimeError:
                pass
        if not pool:
            raise RuntimeError("Could not sample any windows.")
        if len(pool) <= n:
            return pool

        gd  = np.array([w.gene_density for w in pool])
        rc  = np.array([w.mean_recomb_scaled for w in pool])
        gd_q = np.digitize(gd, np.percentile(gd, [20, 40, 60, 80]))
        rc_q = np.digitize(rc, np.percentile(rc, [20, 40, 60, 80]))
        bin_id = gd_q * 5 + rc_q

        selected = []
        remaining = list(range(len(pool)))
        while len(selected) < n and remaining:
            for b in np.unique(bin_id):
                cand = [i for i in remaining if bin_id[i] == b]
                if cand:
                    pick = int(cand[rng.integers(len(cand))])
                    selected.append(pick)
                    remaining.remove(pick)
                if len(selected) == n:
                    break

        return [pool[i] for i in selected[:n]]


class CoverageMask:
    """
    A BED-defined set of poorly-covered (uncallable) regions to remove from
    simulated data, so training mimics the real callable genome. BED is 0-based
    half-open: 'chrom  start  end' per line (extra columns ignored).

    Optional; constructed only when config.COVERAGE_MASK_BED is set. Provides:
      * is_masked(chrom, pos) for a single position,
      * masked_fraction(chrom, start, end) for a window (for the optional
        masked-fraction covariate and the drop-window decision),
      * filter_positions(chrom, positions) -> boolean keep-mask.
    """
    def __init__(self, intervals_by_chrom):
        # intervals_by_chrom: {chrom: sorted list of (start, end)}
        self.by_chrom = {}
        for ch, ivs in intervals_by_chrom.items():
            ivs = sorted(ivs)
            self.by_chrom[ch] = (
                np.array([a for a, _ in ivs], dtype=np.int64),
                np.array([b for _, b in ivs], dtype=np.int64),
            )

    @classmethod
    def from_bed(cls, path):
        import collections
        ivs = collections.defaultdict(list)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(("#", "track", "browser")):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                ch, a, b = parts[0], int(parts[1]), int(parts[2])
                if b > a:
                    ivs[ch].append((a, b))
        return cls(ivs)

    def filter_positions(self, chrom, positions):
        """Boolean array: True = KEEP (not masked), False = drop (masked)."""
        positions = np.asarray(positions, dtype=np.int64)
        if chrom not in self.by_chrom or len(positions) == 0:
            return np.ones(len(positions), dtype=bool)
        starts, ends = self.by_chrom[chrom]
        # For each position, find the interval whose start is <= pos and check
        # whether pos < that interval's end (intervals are sorted, disjoint or
        # not — we use searchsorted on starts).
        idx = np.searchsorted(starts, positions, side="right") - 1
        keep = np.ones(len(positions), dtype=bool)
        valid = idx >= 0
        masked = np.zeros(len(positions), dtype=bool)
        masked[valid] = positions[valid] < ends[idx[valid]]
        keep[masked] = False
        return keep

    def masked_fraction(self, chrom, start, end):
        """Fraction of [start, end) covered by masked intervals (clipped)."""
        if chrom not in self.by_chrom or end <= start:
            return 0.0
        starts, ends = self.by_chrom[chrom]
        lo = np.maximum(starts, start)
        hi = np.minimum(ends, end)
        overlap = np.clip(hi - lo, 0, None).sum()
        return float(overlap) / float(end - start)

