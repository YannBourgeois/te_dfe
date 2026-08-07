"""
processor.py — Convert tree sequence to 4-channel one-hot tensor (v4).

Three-region tensor layout (see config.py for full description):

  Region 1 — TE anchors          : N_TE_SITES rows
  Region 2 — TE-flanking SNPs    : N_TE_SITES * 2K rows
  Region 3 — Random anchor blocks: N_RANDOM_BLOCKS * (1 + 2K) rows

Each region has a fixed slot count and is padded if underfilled.
The CNN sees a uniform anchor+flanks structure across regions, with
TE-ness signaled by channel 2.
"""
import warnings

import numpy as np
import tskit

from .config import (
    SNP_SAMPLING_STRATEGY,
    FLANK_WINDOW_BP, MAX_FLANK_DISTANCE_BP,
    MIN_TE_FLANK_DIST_BP, MIN_RANDOM_ANCHOR_DIST_BP,
    N_TE_SITES, N_TE_FLANK_SITES, N_RANDOM_BLOCKS, N_RANDOM_BLOCK_SITES,
    N_FLANK_PER_SIDE, BLOCK_SIZE, MAF_MIN, WINDOW_SIZE,
    CM_NORMALIZER, LOCAL_DENSITY_BP, DENSITY_LOG_NORMALIZER,
    EDGE_EXCLUDE_BP,
    FOLD_SFS,
    RECOMB_TYPICAL, RECOMB_LOG_SCALE, RECOMB_MIN_FOR_LOG,
)


def _resolve_missingness(n_individuals):
    """
    Resolve config.MISSINGNESS into a per-individual rate vector, or None if
    disabled. Accepts None, a scalar float (broadcast), a length-n list/array,
    or a path to a one-rate-per-line text file. Cached per process.
    """
    from .config import MISSINGNESS
    if MISSINGNESS is None:
        return None
    import numpy as _np
    val = MISSINGNESS
    if isinstance(val, (int, float)):
        rates = _np.full(n_individuals, float(val), dtype=float)
    elif isinstance(val, str):
        rates = _np.loadtxt(val, dtype=float).ravel()
    else:
        rates = _np.asarray(val, dtype=float).ravel()
    if len(rates) != n_individuals:
        # Broadcast a single value; otherwise require an exact match so a
        # wrong-length missingness vector is caught loudly, not silently tiled.
        if rates.size == 1:
            rates = _np.full(n_individuals, float(rates[0]), dtype=float)
        else:
            raise ValueError(
                f"MISSINGNESS has {rates.size} rates but there are "
                f"{n_individuals} individuals. Provide one per individual, a "
                f"single scalar, or None.")
    if _np.any((rates < 0) | (rates >= 1)):
        raise ValueError("MISSINGNESS rates must be in [0, 1).")
    return rates


def _is_te_mutation(mut):
    """
    True iff this mutation is a TE insertion (SLiM mutation type m1, i.e.
    mutation_type == 1), as opposed to a neutral overlay SNP (msprime) or a
    background-selection deleterious mutation (SLiM type m2).

    SLiM mutations carry metadata with a non-empty mutation_list whose entries
    have slim_time, selection_coeff, and mutation_type. Overlay mutations from
    msprime have empty / default-decoded metadata. We require BOTH that the
    mutation is SLiM-origin (slim_time present) AND that its type is 1 (m1) —
    so that when background selection adds m2 deleterious mutations they are not
    mis-counted as TEs. For sims without m2 this is identical to the old test
    (all SLiM mutations are m1), so it is backward-compatible.
    """
    meta = mut.metadata
    if not isinstance(meta, dict):
        return False
    ml = meta.get("mutation_list")
    if not ml:
        return False
    entry = ml[0]
    if "slim_time" not in entry:
        return False
    # mutation_type may be absent in very old metadata schemas; if so, fall
    # back to the SLiM-origin test (treat as TE), preserving old behavior.
    mt = entry.get("mutation_type")
    return (mt is None) or (mt == 1)


# Default: empty (no genotypes) row state (0 = ancestral)
ANCESTRAL_STATE = 0
SNP_STATE       = 1
TE_STATE        = 2
PADDING_STATE   = 3      # NEW: padding gets its own categorical state
PADDING_FLAG    = -1     # sentinel value for unfilled rows during build


class TEDFEBurstProcessor:
    """
    Tree sequence -> (n_haps, N_SITES) int8 categorical tensor.

    Encoding (one byte per cell):
      0 = ancestral
      1 = SNP-derived
      2 = TE-present
      3 = padding (no real data at this cell)

    The network's forward pass expands this to a 4-channel one-hot tensor
    on GPU before feeding the CNN, so the categorical semantics are
    preserved at compute time.

    Sites are arranged in three regions:
      - Region 1: TE anchor rows
      - Region 2: TE flanking SNP rows (random subsample within FLANK_WINDOW_BP)
      - Region 3: Random anchor blocks (anchor row + flanks)

    When the tree sequence has more than n_te TEs, n_te are subsampled
    UNIFORMLY AT RANDOM with a deterministic seed (derived from the tree
    sequence position and the user-supplied seed) so the same input
    always produces the same output.
    """

    def __init__(
        self,
        n_te:                  int = N_TE_SITES,
        n_te_flank:            int = N_TE_FLANK_SITES,
        n_random_blocks:       int = N_RANDOM_BLOCKS,
        n_flank_per_side:      int = N_FLANK_PER_SIDE,
        maf_min:               float = MAF_MIN,
        flank_window_bp:       int = FLANK_WINDOW_BP,
        min_te_flank_dist_bp:  int = MIN_TE_FLANK_DIST_BP,
        min_random_anchor_dist_bp: int = MIN_RANDOM_ANCHOR_DIST_BP,
        window_size:           int = WINDOW_SIZE,
        sampling_strategy:     str = SNP_SAMPLING_STRATEGY,
    ):
        self.n_te                  = n_te
        self.n_te_flank            = n_te_flank
        self.n_random_blocks       = n_random_blocks
        self.n_random_block_sites  = n_random_blocks * (1 + 2 * n_flank_per_side)
        self.n_flank_per_side      = n_flank_per_side
        self.block_size            = 1 + 2 * n_flank_per_side
        self.n_sites               = n_te + n_te_flank + self.n_random_block_sites
        self.maf_min               = maf_min
        self.flank_window_bp       = flank_window_bp
        self.max_flank_dist_bp     = flank_window_bp   # legacy alias
        self.min_te_flank_dist_bp  = min_te_flank_dist_bp
        self.min_random_anchor_dist_bp = min_random_anchor_dist_bp
        self.window_size           = window_size
        self.sampling_strategy     = sampling_strategy
        # Optional coverage mask, built once from config (None if disabled).
        # window_chrom / window_start are set per call (or as attributes) so the
        # window-relative variant positions can be mapped to absolute coords.
        self.coverage_mask = None
        try:
            from .config import COVERAGE_MASK_BED
            if COVERAGE_MASK_BED:
                from .demography import CoverageMask
                self.coverage_mask = CoverageMask.from_bed(COVERAGE_MASK_BED)
        except Exception as e:
            import warnings
            warnings.warn(f"coverage mask not loaded: {e}")
        self.window_chrom = None
        self.window_start = 0

    def __call__(self, ts: tskit.TreeSequence,
                  seed: int = None,
                  recomb_map = None,
                  window_chrom: str = None,
                  window_start: int = None,
                  pad: bool = True):
        """
        Process a tree sequence into a (geno, pos) tuple.

        Returns
        -------
        geno : (n_haps, N_SITES) int8, values in {0=anc, 1=SNP, 2=TE, 3=pad}
        pos  : (2, N_SITES) float32, per-site position features
                 channel 0: physical position relative to anchor (normalized)
                 channel 1: genetic position relative to anchor (cM, normalized)

        The anchor for each row depends on the region:
          - Region 1 (TE anchors): anchor = the TE itself, so delta = 0
          - Region 2 (TE flanks):  anchor = the TE this flank serves
          - Region 3 (random):     anchor = the first row of the block

        If recomb_map is None, channel 1 of pos is all zeros (still valid input).
        """
        # Sample selection. By default use all of the tree's samples, but
        # enforce that the count matches the configured N_HAPS (catch
        # config/tree mismatches early). When SUBSAMPLE_INDIVIDUALS is on, pick
        # N_INDIVIDUALS individuals keeping each one's haplotype pair together.
        keep_positions = self._select_sample_positions(ts)
        if keep_positions is None:
            n_haps = ts.num_samples
        else:
            n_haps = len(keep_positions)

        if seed is None:
            seed = (int(ts.num_sites) * 1_000_003
                    + int(ts.num_mutations) * 1_009) % (2**31 - 1)
        rng = np.random.default_rng(seed)

        # Pass 1: collect TE rows and SNP rows from the tree sequence
        te_rows, te_pos = [], []
        all_snp_rows, all_snp_pos = [], []

        # Optional coverage mask: drop variants in poorly-covered regions so the
        # simulated data matches the real callable genome. Positions in the tree
        # sequence are window-relative; the mask is in absolute genome coords, so
        # we offset by the window start.
        cov_mask = self.coverage_mask
        win_chrom = window_chrom if window_chrom is not None else self.window_chrom
        win_start = (window_start if window_start is not None
                     else self.window_start)

        for var in ts.variants():
            muts = list(var.site.mutations)
            any_te = any(_is_te_mutation(m) for m in muts)
            genos  = var.genotypes.copy().astype(np.int8)
            if keep_positions is not None:
                genos = genos[keep_positions]   # subsample to selected haplotypes
            derived_count = int(genos.sum())

            # Coverage mask: drop this variant if its (absolute) position falls
            # in a masked region. Applies to BOTH SNPs and TEs.
            if cov_mask is not None and win_chrom is not None:
                abs_pos = win_start + var.site.position
                if not cov_mask.filter_positions(win_chrom, [abs_pos])[0]:
                    continue

            # Filter to polymorphic-in-sample (singletons kept; exclude
            # fixed-in-sample and absent-in-sample). With SLiM's
            # convertToSubstitution = F, fixed-in-sample mutations CAN
            # appear in ts.variants() — we must filter them out here so
            # the simulator data matches what a typical VCF contains.
            if not (0 < derived_count < n_haps):
                continue

            if any_te:
                row = np.where(genos == 1, TE_STATE, ANCESTRAL_STATE).astype(np.int8)
                te_rows.append(row)
                te_pos.append(float(var.site.position))
            else:
                freq = float(genos.mean())
                if min(freq, 1.0 - freq) >= self.maf_min:
                    if FOLD_SFS and freq > 0.5:
                        # Folded SFS: encode minor allele as 1. The derived
                        # allele is the majority here, so flip 0<->1 so that
                        # 1 = minor. (TEs are never folded — handled above.)
                        genos = (1 - genos).astype(np.int8)
                    all_snp_rows.append(genos)        # polarized: 0=anc,1=der
                    all_snp_pos.append(float(var.site.position))    #  folded: 0=major,1=minor

        # Pass 2: build the windowed three-region layout + channels. Shared with
        # the VCF path (vcf_to_tensors), which produces the same variant table
        # (te_rows, te_pos, all_snp_rows, all_snp_pos) from real data and calls
        # this same builder — guaranteeing identical representation.
        from .config import REPRESENTATION
        if REPRESENTATION == "sumstat":
            return self.build_sumstats_from_variant_table(
                te_rows, te_pos, all_snp_rows, all_snp_pos,
                n_haps=n_haps, recomb_map=recomb_map,
                window_chrom=win_chrom, window_start=win_start,
                pad=pad,
            )
        return self.build_from_variant_table(
            te_rows, te_pos, all_snp_rows, all_snp_pos,
            n_haps=n_haps, seed=seed, recomb_map=recomb_map,
            window_chrom=win_chrom, window_start=win_start,
        )

    def _select_sample_positions(self, ts):
        """
        Decide which of the tree's sample nodes to use, as positions into
        ts.samples() (so var.genotypes can be indexed directly).

        Returns None to use ALL samples (the common case). When
        SUBSAMPLE_INDIVIDUALS is on, returns the node positions for
        N_INDIVIDUALS individuals, with each individual's two haplotype nodes
        KEPT TOGETHER and in their original within-individual order — so the
        unphased pairing stays correct and within-individual structure (e.g.
        selfing) is preserved. Selection is the first N_INDIVIDUALS individuals
        by default, or a seeded random draw when SUBSAMPLE_SEED is set.

        With subsampling OFF, enforces ts.num_samples == N_HAPS and raises on
        mismatch (catches config/tree disagreements early).
        """
        from .config import (SUBSAMPLE_INDIVIDUALS, SUBSAMPLE_SEED,
                             N_INDIVIDUALS, N_HAPS)
        samples = np.asarray(ts.samples())
        n_samp = len(samples)

        if not SUBSAMPLE_INDIVIDUALS:
            if n_samp != N_HAPS:
                raise ValueError(
                    f"tree has {n_samp} sample haplotypes but config N_HAPS="
                    f"{N_HAPS}. Set SUBSAMPLE_INDIVIDUALS=True to subsample to "
                    f"N_INDIVIDUALS={N_INDIVIDUALS}, or fix the config / tree.")
            return None

        # Group sample nodes by their individual, preserving order.
        ind_to_positions = {}
        n_no_ind = 0
        for i, nid in enumerate(samples):
            ind = ts.node(int(nid)).individual
            if ind == -1:
                n_no_ind += 1
            ind_to_positions.setdefault(ind, []).append(i)

        # If individual metadata is absent, fall back to consecutive pairs as
        # pseudo-individuals (with a warning) so we degrade safely rather than
        # mispair. This loses true individual identity (no selfing tracking).
        if n_no_ind == n_samp:
            import warnings
            warnings.warn(
                "tree samples carry no individual metadata; pairing consecutive "
                "haplotypes as pseudo-individuals for subsampling. Within-"
                "individual structure (e.g. selfing) will NOT be preserved.")
            if n_samp % 2 != 0:
                raise ValueError("odd number of haplotypes; cannot pair.")
            ind_to_positions = {
                k: [2 * k, 2 * k + 1] for k in range(n_samp // 2)}

        individuals = list(ind_to_positions.keys())
        n_ind_avail = len(individuals)
        if n_ind_avail < N_INDIVIDUALS:
            raise ValueError(
                f"tree has only {n_ind_avail} individuals but "
                f"N_INDIVIDUALS={N_INDIVIDUALS} requested (cannot upsample).")

        # Select individuals: first-N (deterministic) or seeded-random.
        if SUBSAMPLE_SEED is not None:
            rng = np.random.default_rng(int(SUBSAMPLE_SEED))
            sel = sorted(int(s) for s in
                         rng.choice(n_ind_avail, size=N_INDIVIDUALS, replace=False))
            chosen = [individuals[s] for s in sel]
        else:
            chosen = individuals[:N_INDIVIDUALS]

        positions = []
        for ind in chosen:
            positions.extend(ind_to_positions[ind])   # pair order preserved
        return np.asarray(positions, dtype=int)

    def build_from_variant_table(self, te_rows, te_pos,
                                  all_snp_rows, all_snp_pos,
                                  n_haps, seed=None, recomb_map=None,
                                  window_chrom=None, window_start=None):
        """
        Build the (geno, pos) tensors from an already-extracted variant table.
        Shared by the tree-sequence path (__call__) and the VCF path. Inputs:
          te_rows  : list of (n_haps,) int arrays, TE-state rows (TE_STATE/anc)
          te_pos   : list of float window-relative TE positions
          all_snp_rows : list of (n_haps,) int arrays, SNP rows (already folded
                         if FOLD_SFS) — 0/1 coded
          all_snp_pos  : list of float window-relative SNP positions
        """
        if seed is None:
            seed = (len(te_pos) * 1_000_003 + len(all_snp_pos) * 1_009) \
                   % (2**31 - 1)
        rng = np.random.default_rng(seed)
        win_chrom = (window_chrom if window_chrom is not None
                     else self.window_chrom)
        win_start = (window_start if window_start is not None
                     else self.window_start)

        # Pre-sort SNPs by position for binary-search lookups
        if all_snp_rows:
            snp_pos_arr  = np.array(all_snp_pos, dtype=np.float32)
            order        = np.argsort(snp_pos_arr, kind="stable")
            snp_pos_arr  = snp_pos_arr[order]
            snp_rows_arr = np.array([all_snp_rows[i] for i in order], dtype=np.int8)
        else:
            snp_pos_arr  = np.zeros((0,), dtype=np.float32)
            snp_rows_arr = np.zeros((0, n_haps), dtype=np.int8)

        # Exclude SNPs within MIN_TE_FLANK_DIST_BP of ANY TE. These are
        # likely TSD / mapping artefacts in real data and would be filtered
        # by upstream QC; we filter them here too for train/test consistency.
        if len(te_pos) > 0 and len(snp_pos_arr) > 0:
            te_pos_sorted = np.sort(np.array(te_pos, dtype=np.float32))
            # For each SNP, find the closest TE position
            idx_left  = np.searchsorted(te_pos_sorted, snp_pos_arr, side="left")
            idx_left  = np.clip(idx_left, 0, len(te_pos_sorted) - 1)
            dist_left = np.abs(snp_pos_arr - te_pos_sorted[idx_left])
            idx_right = np.clip(idx_left - 1, 0, len(te_pos_sorted) - 1)
            dist_right = np.abs(snp_pos_arr - te_pos_sorted[idx_right])
            min_dist_to_te = np.minimum(dist_left, dist_right)
            keep_mask = min_dist_to_te >= self.min_te_flank_dist_bp
            snp_pos_arr  = snp_pos_arr[keep_mask]
            snp_rows_arr = snp_rows_arr[keep_mask]

        # Subsample TEs if a window has more than the layout represents. With
        # the derived layout the anchor and flank capacities are equal
        # (N_TE_FLANK_SITES = N_TE_SITES * 2 * N_FLANK_PER_SIDE), so this caps at
        # N_TE_SITES; the min() is a defensive guard against any hand-edited
        # config where the two could disagree.
        slots_per_te = 2 * self.n_flank_per_side
        eff_n_te = min(self.n_te, self.n_te_flank // slots_per_te)
        if len(te_pos) > eff_n_te:
            sel = rng.choice(len(te_pos), size=eff_n_te, replace=False)
            sel.sort()  # preserve genomic order for downstream use
            te_rows = [te_rows[i] for i in sel]
            te_pos  = [te_pos[i]  for i in sel]

        # Region 1: TE anchors
        te_anchor_mat, te_anchor_pos = self._fill_te_anchors(
            te_rows, te_pos, n_haps,
        )

        # Region 2: TE flanking SNPs (random subsample within FLANK_WINDOW_BP)
        te_flank_mat, te_flank_pos = self._fill_te_flanks(
            te_pos, snp_rows_arr, snp_pos_arr, n_haps, rng,
        )

        # Region 3: Random anchor blocks
        random_mat, random_pos = self._fill_random_blocks(
            te_pos, snp_rows_arr, snp_pos_arr, n_haps, rng,
        )

        # Concatenate the three regions
        combined_genos = np.vstack([te_anchor_mat, te_flank_mat, random_mat])
        combined_pos   = np.concatenate([te_anchor_pos, te_flank_pos, random_pos])
        is_padding     = (combined_pos < 0)

        # ----- Per-row anchor positions for the position channels -----
        # Region 1: each TE is its own anchor (delta = 0)
        # Region 2: anchor = the TE this flank serves
        # Region 3: anchor = the first row of the block
        n_rows = combined_pos.shape[0]
        anchor_pos = np.full(n_rows, -1.0, dtype=np.float32)

        # Region 1
        for i in range(self.n_te):
            anchor_pos[i] = te_anchor_pos[i]

        # Region 2
        slots_per_te = 2 * self.n_flank_per_side
        for i in range(self.n_te_flank):
            i_te = i // slots_per_te
            anchor_pos[self.n_te + i] = te_anchor_pos[i_te]

        # Region 3
        offset3 = self.n_te + self.n_te_flank
        for block in range(self.n_random_blocks):
            base = offset3 + block * self.block_size
            anchor_p = random_pos[block * self.block_size]
            for k in range(self.block_size):
                anchor_pos[base + k] = anchor_p

        # ----- Physical position channel (normalized) -----
        delta_phys = np.zeros(n_rows, dtype=np.float32)
        valid = (~is_padding) & (anchor_pos >= 0)
        delta_phys[valid] = (
            (combined_pos[valid] - anchor_pos[valid])
            / float(self.flank_window_bp)
        )

        # ----- Genetic position channel (cM, normalized) -----
        delta_gen = np.zeros(n_rows, dtype=np.float32)
        if recomb_map is not None:
            try:
                pos_eval = np.clip(combined_pos[valid], 0.0, None)
                anchor_eval = np.clip(anchor_pos[valid], 0.0, None)
                cm_pos    = np.asarray(
                    recomb_map.get_cumulative_mass(pos_eval), dtype=np.float64,
                ) * 100.0
                cm_anchor = np.asarray(
                    recomb_map.get_cumulative_mass(anchor_eval), dtype=np.float64,
                ) * 100.0
                delta_gen[valid] = (
                    (cm_pos - cm_anchor) / CM_NORMALIZER
                ).astype(np.float32)
            except Exception:
                pass  # silently fall back to zeros if recomb_map lookup fails

        # ----- Local density channels (TE + SNP) -----
        # For each non-padding row, count polymorphic markers within
        # +/- LOCAL_DENSITY_BP of its position. Apply log1p transform and
        # normalize by log1p(DENSITY_LOG_NORMALIZER) to compress the long
        # tail and keep the channel value in ~[0, 1.2].
        #
        # Note: raw counts (not rates per kb). At inference time, the
        # user's VCF must be downsampled to N_HAPS so the counts are
        # comparable to training.
        te_density_arr  = np.zeros(n_rows, dtype=np.float32)
        snp_density_arr = np.zeros(n_rows, dtype=np.float32)

        if valid.any():
            # We pass the FULL sample-polymorphic TE list (te_pos before
            # the 60-subsampling) for accurate counts in dense windows.
            te_pos_for_density = np.sort(np.asarray(te_pos, dtype=np.float32)) \
                                  if len(te_pos) else np.empty(0, dtype=np.float32)
            snp_pos_sorted = snp_pos_arr  # already sorted

            valid_indices = np.where(valid)[0]
            valid_positions = combined_pos[valid_indices]

            log_norm = float(np.log1p(DENSITY_LOG_NORMALIZER))

            if len(te_pos_for_density) > 0:
                lo_idx = np.searchsorted(
                    te_pos_for_density,
                    valid_positions - LOCAL_DENSITY_BP, side="left",
                )
                hi_idx = np.searchsorted(
                    te_pos_for_density,
                    valid_positions + LOCAL_DENSITY_BP, side="right",
                )
                counts = (hi_idx - lo_idx).astype(np.float32)
                # ADDED: length-normalize to remove window-edge underestimation.
                # Rescale the (possibly truncated) count to a full 2*BP window,
                # unbiased if local density is ~uniform. Interior TEs: factor 1.
                avail = (np.minimum(valid_positions + LOCAL_DENSITY_BP, self.window_size)
                         - np.maximum(valid_positions - LOCAL_DENSITY_BP, 0.0))
                counts = counts * (2.0 * LOCAL_DENSITY_BP / np.maximum(avail, 1.0))
                te_density_arr[valid_indices] = (
                    np.log1p(counts) / log_norm
                ).astype(np.float32)

            if len(snp_pos_sorted) > 0:
                lo_idx = np.searchsorted(
                    snp_pos_sorted,
                    valid_positions - LOCAL_DENSITY_BP, side="left",
                )
                hi_idx = np.searchsorted(
                    snp_pos_sorted,
                    valid_positions + LOCAL_DENSITY_BP, side="right",
                )
                counts = (hi_idx - lo_idx).astype(np.float32)
                # ADDED: length-normalize (see TE-density above).
                avail = (np.minimum(valid_positions + LOCAL_DENSITY_BP, self.window_size)
                         - np.maximum(valid_positions - LOCAL_DENSITY_BP, 0.0))
                counts = counts * (2.0 * LOCAL_DENSITY_BP / np.maximum(avail, 1.0))
                snp_density_arr[valid_indices] = (
                    np.log1p(counts) / log_norm
                ).astype(np.float32)

        # Continuous channels. Base 4 are model-agnostic (phys, gen, TE-dens,
        # SNP-dens). The simple model optionally appends ch4 = log local recomb.
        # The context model appends ch4 = dist-to-gene and ch5 = log local
        # recomb (feature_r). The channel ORDER is model-specific, so it is
        # assembled here from the active config.
        from .config import MODEL_NAME
        channels = [delta_phys, delta_gen, te_density_arr, snp_density_arr]

        vi = np.where(valid)[0] if valid.any() else np.array([], dtype=int)
        vpos = combined_pos[vi] if len(vi) else np.array([], dtype=np.float64)

        # Context-only: distance-to-nearest-gene channel (ch4), log-normalized.
        if MODEL_NAME == "context":
            from .config import DIST_TO_GENE_NORMALIZER
            dist_to_gene_arr = np.zeros(n_rows, dtype=np.float32)
            if len(vi):
                gene_intervals = getattr(self, "gene_intervals", None)
                if gene_intervals:
                    gi = np.asarray(gene_intervals, dtype=np.float64)
                    pos2d = vpos[:, None]
                    inside = (pos2d >= gi[None, :, 0]) & (pos2d <= gi[None, :, 1])
                    d = np.where(inside, 0.0,
                                 np.minimum(np.abs(pos2d - gi[None, :, 0]),
                                            np.abs(pos2d - gi[None, :, 1])))
                    dpp = d.min(axis=1)
                else:
                    dpp = np.full(len(vi), DIST_TO_GENE_NORMALIZER)
                log_norm_dist = np.log10(DIST_TO_GENE_NORMALIZER)
                dist_to_gene_arr[vi] = (
                    np.log10(np.maximum(dpp, 1.0)) / log_norm_dist
                ).astype(np.float32)
            channels.append(dist_to_gene_arr)

        # Local recombination-rate channel. Present for the context model
        # (feature_r, ch5) and for the simple model when USE_RECOMB_CHANNEL
        # (ch4). Same log normalization in both for parity.
        want_recomb = (MODEL_NAME == "context") or getattr(
            __import__("te_dfe.config", fromlist=["x"]),
            "USE_RECOMB_CHANNEL", False)
        if want_recomb:
            log_recomb_arr = np.zeros(n_rows, dtype=np.float32)
            if len(vi):
                if recomb_map is not None:
                    try:
                        rates = np.asarray(
                            recomb_map.get_rate(np.clip(vpos, 0, None)),
                            dtype=np.float64)
                    except Exception:
                        rates = np.full_like(vpos, RECOMB_TYPICAL,
                                             dtype=np.float64)
                else:
                    rates = np.full_like(vpos, RECOMB_TYPICAL, dtype=np.float64)
                rates_clipped = np.maximum(rates, RECOMB_MIN_FOR_LOG)
                log_recomb_arr[vi] = np.clip(
                    (np.log10(rates_clipped / RECOMB_TYPICAL)
                     / float(RECOMB_LOG_SCALE)).astype(np.float32),
                    -2.0, 2.0)
            channels.append(log_recomb_arr)

        # Stack continuous channels. Simple: phys,gen,te,snp[,recomb]. Context:
        # phys,gen,te,snp,dist_gene,recomb.
        pos_features = np.stack(channels, axis=0).astype(np.float32)

        # Final encoding for geno: (n_sites, n_haps) int8 in {0,1,2,3}
        out = combined_genos.copy().astype(np.int8)
        out[is_padding] = PADDING_STATE
        out = np.clip(out, 0, 3)

        # Dead-pixel masking: mimic per-individual genotype missingness. A
        # missing diploid genotype means BOTH of an individual's haplotypes are
        # unknown, so we mask haplotype pairs together at the per-site level.
        # TE-anchor cells are masked too (per spec). Missing -> PADDING_STATE.
        rates = _resolve_missingness(n_haps // 2)
        if rates is not None:
            self._apply_missingness(out, rates, rng)

        # Transpose to (n_haps, n_sites)
        geno_out = out.T.astype(np.int8)

        # Unphased mode: collapse haplotype pairs to per-individual diploid
        # dosages (6-state encoding). Done here so every call site that uses the
        # processor gets the same representation; the logic lives in the
        # standalone te_dfe/unphased.py module. Position features are per-site
        # and unchanged.
        from .config import PHASED
        if not PHASED:
            from .unphased import collapse_to_unphased
            geno_out = collapse_to_unphased(geno_out)
        return geno_out, pos_features

    def build_sumstats_from_variant_table(self, te_rows, te_pos,
                                          all_snp_rows, all_snp_pos,
                                          n_haps, recomb_map=None,
                                          window_chrom=None, window_start=None,
                                          pad=True):
        """
        Summary-statistic representation. For each TE, build its per-TE feature
        vector (te_dfe/sumstats.py) from the carrier/non-carrier flanking SNPs.

        pad=True  (default): pad/truncate the variable-size SET of TEs to
                  N_TE_TRAIN_MAX and return (feats[N_TE_TRAIN_MAX, F], mask).
                  Use for a single locus that already holds the full TE set.
        pad=False: return the RAW (n_te, F) features (no padding, no mask), so a
                  caller can POOL TEs across several windows into one set and pad
                  once at the end (the per-replicate TE set is built from the K
                  windows; each TE keeps its own window's flanking SNPs).

        Shared by the tree-seq and VCF paths (same variant table) so train and
        inference see identical statistics.
        """
        from .config import (PHASED, N_TE_TRAIN_MAX, SUMSTAT_FLANK_MAX,
                             FLANK_WINDOW_BP, RECOMB_TYPICAL, RECOMB_LOG_SCALE,
                             RECOMB_MIN_FOR_LOG, DIST_TO_GENE_NORMALIZER,
                             TE_STATE)
        from .sumstats import compute_te_sumstats, N_SUMSTATS_PER_TE

        # SNP matrix (n_snp, n_haps) sorted by position for flank lookups.
        if all_snp_rows:
            snp_pos = np.asarray(all_snp_pos, dtype=np.float64)
            order = np.argsort(snp_pos, kind="stable")
            snp_pos = snp_pos[order]
            snp_mat = np.asarray([all_snp_rows[i] for i in order], dtype=np.int8)
        else:
            snp_pos = np.zeros((0,), dtype=np.float64)
            snp_mat = np.zeros((0, n_haps), dtype=np.int8)

        gene_intervals = getattr(self, "gene_intervals", None)
        gi = (np.asarray(gene_intervals, dtype=np.float64)
              if gene_intervals else None)

        def _dist_to_gene(p):
            if gi is None or len(gi) == 0:
                return 0.0
            inside = (p >= gi[:, 0]) & (p <= gi[:, 1])
            if inside.any():
                return 0.0
            d = np.minimum(np.abs(p - gi[:, 0]), np.abs(p - gi[:, 1])).min()
            return float(np.log10(max(d, 1.0)) / np.log10(DIST_TO_GENE_NORMALIZER))

        def _local_recomb(p):
            if recomb_map is None:
                return 0.0
            try:
                r = float(recomb_map.get_rate(max(p, 0.0)))
            except Exception:
                return 0.0
            r = max(r, RECOMB_MIN_FOR_LOG)
            return float(np.clip(np.log10(r / RECOMB_TYPICAL)
                                 / float(RECOMB_LOG_SCALE), -2.0, 2.0))

        # Local SNP / TE density per TE: count within +/- LOCAL_DENSITY_BP,
        # log1p-normalized exactly as the matrix density channels so both
        # representations carry the same density signal.
        _all_snp_sorted = np.sort(snp_pos) if snp_pos.size else snp_pos
        _all_te_sorted  = (np.sort(np.asarray(te_pos, dtype=np.float64))
                           if len(te_pos) else np.zeros(0))
        _dens_norm = float(np.log1p(DENSITY_LOG_NORMALIZER))
        # ADDED: window bound + edge-rescale factor. Extrapolates a truncated
        # count to a full 2*BP window (>=1 near edges, exactly 1 in interior),
        # removing the systematic density underestimate for TEs within
        # LOCAL_DENSITY_BP of a window edge. Positions are window-relative.
        _win = float(self.window_size)

        def _edge_rescale(p):
            avail = min(p + LOCAL_DENSITY_BP, _win) - max(p - LOCAL_DENSITY_BP, 0.0)
            return (2.0 * LOCAL_DENSITY_BP) / max(avail, 1.0)

        def _local_snp_density(p):
            if _all_snp_sorted.size == 0:
                return 0.0
            lo = np.searchsorted(_all_snp_sorted, p - LOCAL_DENSITY_BP, "left")
            hi = np.searchsorted(_all_snp_sorted, p + LOCAL_DENSITY_BP, "right")
            return float(np.log1p((hi - lo) * _edge_rescale(p)) / _dens_norm)  # ADDED rescale

        def _local_te_density(p):
            if _all_te_sorted.size == 0:
                return 0.0
            lo = np.searchsorted(_all_te_sorted, p - LOCAL_DENSITY_BP, "left")
            hi = np.searchsorted(_all_te_sorted, p + LOCAL_DENSITY_BP, "right")
            cnt = max(hi - lo - 1, 0)                                # exclude focal TE
            return float(np.log1p(cnt * _edge_rescale(p)) / _dens_norm)  # ADDED rescale

        feats = np.zeros((N_TE_TRAIN_MAX, N_SUMSTATS_PER_TE), dtype=np.float32)
        # When pad=False we compute features for EVERY TE (the caller pools
        # across windows and pads once); when pad=True we cap at N_TE_TRAIN_MAX.
        n_te = len(te_pos) if not pad else min(len(te_pos), N_TE_TRAIN_MAX)
        rows = []
        for k in range(n_te):
            te_row = np.asarray(te_rows[k])
            p = float(te_pos[k])
            # ADDED: optionally exclude TEs within EDGE_EXCLUDE_BP of a window
            # edge from the FOCAL (emitted) set — their length-normalized density
            # is the noisiest. Excluded TEs remain in the counting pool above, so
            # interior TEs' neighbour counts are unaffected. 0 = keep all.
            if EDGE_EXCLUDE_BP and (p < EDGE_EXCLUDE_BP
                                    or p > self.window_size - EDGE_EXCLUDE_BP):
                continue
            # Carrier status over OBSERVED TE calls. te_row coding:
            # TE_STATE = carrier, 0 = non-carrier, -1 = missing TE call.
            # Missing-TE individuals are excluded from BOTH groups and from
            # te_freq (observed-count frequency), matching the missing-robust
            # statistics — on complete data (no -1) this is unchanged.
            te_observed = (te_row >= 0)
            carrier_hap = (te_row == TE_STATE)
            noncarrier_hap = (te_row == 0)
            n_te_obs = int(te_observed.sum())
            te_freq = (float(carrier_hap.sum()) / n_te_obs) if n_te_obs else 0.0

            # Flanking SNPs within FLANK_WINDOW_BP, split L/R, ordered outward.
            if snp_pos.size:
                lo, hi = p - FLANK_WINDOW_BP, p + FLANK_WINDOW_BP
                in_win = (snp_pos >= lo) & (snp_pos <= hi)
                idx = np.where(in_win)[0]
                left = idx[snp_pos[idx] < p]
                right = idx[snp_pos[idx] >= p]
                # nearest-first (outward from core), capped per side
                left = left[np.argsort(-(snp_pos[left]))][:SUMSTAT_FLANK_MAX]
                right = right[np.argsort(snp_pos[right])][:SUMSTAT_FLANK_MAX]
            else:
                left = np.array([], dtype=int); right = np.array([], dtype=int)

            flank_cols = np.concatenate([left, right]) if (left.size or right.size) \
                else np.array([], dtype=int)
            # (n_hap, n_flank) flank matrix, then split by carrier status.
            if flank_cols.size:
                fm = snp_mat[flank_cols].T            # (n_haps, n_flank)
            else:
                fm = np.zeros((n_haps, 0), dtype=np.int8)

            if PHASED:
                # Missing-TE haplotypes (-1) fall into neither group.
                carrier = fm[carrier_hap]
                noncarrier = fm[noncarrier_hap]
                fl_idx = np.arange(left.size)
                fr_idx = np.arange(left.size, left.size + right.size)
            else:
                # individual-level: a diploid is a carrier if either haplotype
                # carries the TE; observed only if both TE calls are non-missing.
                # Flank genotypes collapse to per-individual derived-allele
                # presence, preserving missing (-1). On complete data this is
                # identical to the previous dosage collapse.
                ni = n_haps // 2
                te_pair = te_row.reshape(ni, 2)
                te_ind_obs = (te_pair >= 0).all(axis=1)
                is_carr = (te_pair == TE_STATE).any(axis=1)
                carr_ind = te_ind_obs & is_carr
                noncarr_ind = te_ind_obs & (~is_carr)
                if fm.shape[1]:
                    fp = fm.reshape(ni, 2, -1)
                    ind_obs = (fp >= 0).all(axis=1)
                    present = (fp == 1).any(axis=1)
                    fm_ind = np.where(ind_obs, present.astype(np.int8),
                                      -1).astype(np.int8)
                else:
                    fm_ind = np.zeros((ni, 0), dtype=np.int8)
                carrier = fm_ind[carr_ind]
                noncarrier = fm_ind[noncarr_ind]
                fl_idx = np.arange(left.size)
                fr_idx = np.arange(left.size, left.size + right.size)

            rows.append(compute_te_sumstats(
                carrier, noncarrier, fl_idx, fr_idx,
                te_freq=te_freq, phased=PHASED,
                dist_to_gene=_dist_to_gene(p), local_recomb=_local_recomb(p),
                local_snp_density=_local_snp_density(p),
                local_te_density=_local_te_density(p),
                n_flank_max=2 * SUMSTAT_FLANK_MAX))

        if not pad:
            # Raw per-TE features for cross-window pooling (no mask).
            if rows:
                return np.asarray(rows, dtype=np.float32)
            return np.zeros((0, N_SUMSTATS_PER_TE), dtype=np.float32)

        feats = np.zeros((N_TE_TRAIN_MAX, N_SUMSTATS_PER_TE), dtype=np.float32)
        mask = np.zeros((N_TE_TRAIN_MAX,), dtype=np.float32)
        for k, r in enumerate(rows):
            feats[k] = r
            mask[k] = 1.0
        return feats, mask

    @staticmethod
    def _apply_missingness(out, rates, rng):
        """
        out   : (n_sites, n_haps) int8, already encoded {0,1,2,3}
        rates : (n_individuals,) per-individual missingness probabilities
        Masks haplotype pairs (2i, 2i+1) at each site with prob rates[i], setting
        masked cells to PADDING_STATE. Assumes individual i owns consecutive
        haplotypes 2i, 2i+1 (standard tskit/SLiM node ordering).
        """
        n_sites, n_haps = out.shape
        n_ind = n_haps // 2
        for i in range(n_ind):
            r = float(rates[i])
            if r <= 0:
                continue
            miss = rng.random(n_sites) < r          # per-site for this individual
            if miss.any():
                out[miss, 2 * i]     = PADDING_STATE
                out[miss, 2 * i + 1] = PADDING_STATE

    # ---------------------------------------------------------------------
    # Region 1: TE anchors (capped, deterministic-by-position)
    # ---------------------------------------------------------------------

    def _fill_te_anchors(self, te_rows, te_pos, n_haps):
        n = len(te_rows)
        if n == 0:
            return self._padding_block(self.n_te, n_haps)
        mat = np.array(te_rows, dtype=np.int8)
        pos = np.array(te_pos,  dtype=np.float32)
        if n > self.n_te:
            mat = mat[: self.n_te]
            pos = pos[: self.n_te]
        elif n < self.n_te:
            pad_n = self.n_te - n
            mat = np.vstack([
                mat, np.full((pad_n, n_haps), PADDING_FLAG, dtype=np.int8),
            ])
            pos = np.concatenate([
                pos, np.full(pad_n, -1.0, dtype=np.float32),
            ])
        return mat, pos

    # ---------------------------------------------------------------------
    # Region 2: TE flanking SNPs
    #     For each retained TE, randomly subsample K SNPs on each side
    #     within FLANK_WINDOW_BP. SNPs within MIN_TE_FLANK_DIST_BP of any
    #     TE have already been excluded upstream.
    #     The 2K slots per TE are filled left-then-right, sorted by
    #     position within each side. Empty slots are padding.
    # ---------------------------------------------------------------------

    def _fill_te_flanks(self, te_pos_list, snp_rows, snp_pos, n_haps, rng):
        n_total = self.n_te_flank
        out_mat = np.full((n_total, n_haps), PADDING_FLAG, dtype=np.int8)
        out_pos = np.full(n_total, -1.0, dtype=np.float32)

        if len(snp_pos) == 0:
            return out_mat, out_pos

        slots_per_te = 2 * self.n_flank_per_side
        n_real_te = min(len([p for p in te_pos_list if p >= 0]), self.n_te)
        real_te_pos = [p for p in te_pos_list if p >= 0][: n_real_te]

        for i_te, tpos in enumerate(real_te_pos):
            base = i_te * slots_per_te

            # All SNPs in [tpos - W, tpos)
            left_mask  = (snp_pos >= tpos - self.flank_window_bp) & (snp_pos < tpos)
            # All SNPs in (tpos, tpos + W]
            right_mask = (snp_pos > tpos) & (snp_pos <= tpos + self.flank_window_bp)
            left_idx  = np.where(left_mask)[0]
            right_idx = np.where(right_mask)[0]

            # Random subsample (without replacement) of K SNPs per side
            if len(left_idx) > self.n_flank_per_side:
                pick = rng.choice(len(left_idx), self.n_flank_per_side, replace=False)
                left_idx = left_idx[np.sort(pick)]
            if len(right_idx) > self.n_flank_per_side:
                pick = rng.choice(len(right_idx), self.n_flank_per_side, replace=False)
                right_idx = right_idx[np.sort(pick)]

            # Sort within each side by position for downstream consistency
            left_idx  = sorted(left_idx, key=lambda j: snp_pos[j])
            right_idx = sorted(right_idx, key=lambda j: snp_pos[j])

            # Pack into slots
            # Left side: positions [0, K), with closest-to-TE in the last K-1 slot
            # (left_idx is sorted ascending by position, so leftmost->rightmost)
            for k, idx in enumerate(left_idx):
                slot = base + (self.n_flank_per_side - len(left_idx) + k)
                out_mat[slot] = snp_rows[idx]
                out_pos[slot] = snp_pos[idx]
            # Right side: positions [K, 2K)
            for k, idx in enumerate(right_idx):
                slot = base + self.n_flank_per_side + k
                out_mat[slot] = snp_rows[idx]
                out_pos[slot] = snp_pos[idx]

        return out_mat, out_pos

    # ---------------------------------------------------------------------
    # Region 3: Random anchor blocks (anchor + 2K flanks per block)
    #     Anchors are drawn uniformly from positions >= min_random_anchor_dist_bp
    #     from every TE. Anchor row is encoded as ancestral (channel 0); the
    #     network distinguishes from TEs because channel 2 is empty there.
    # ---------------------------------------------------------------------

    def _fill_random_blocks(self, te_pos_list, snp_rows, snp_pos, n_haps, rng):
        n_total = self.n_random_block_sites
        out_mat = np.full((n_total, n_haps), PADDING_FLAG, dtype=np.int8)
        out_pos = np.full(n_total, -1.0, dtype=np.float32)

        # Build eligibility region: positions >= min_random_anchor_dist_bp from every TE
        real_te_pos = np.array(
            [p for p in te_pos_list if p >= 0], dtype=np.float32,
        )

        # Sample random anchors
        anchor_positions = self._sample_random_anchors(real_te_pos, rng)
        if len(anchor_positions) == 0:
            return out_mat, out_pos

        n_anchors = min(len(anchor_positions), self.n_random_blocks)
        anchor_positions = anchor_positions[:n_anchors]

        for i_anchor, apos in enumerate(anchor_positions):
            base = i_anchor * self.block_size

            # Anchor row: ancestral at every haplotype (no mutation)
            out_mat[base] = np.full(n_haps, ANCESTRAL_STATE, dtype=np.int8)
            out_pos[base] = float(apos)

            if len(snp_pos) == 0:
                continue

            ins = int(np.searchsorted(snp_pos, apos, side="left"))

            # Left flanks
            left_idx = []
            j = ins - 1
            count = 0
            while j >= 0 and count < self.n_flank_per_side:
                if abs(snp_pos[j] - apos) > self.max_flank_dist_bp:
                    break
                left_idx.append(j)
                j -= 1
                count += 1
            left_idx = list(reversed(left_idx))

            # Right flanks
            right_idx = []
            j = ins
            count = 0
            while j < len(snp_pos) and count < self.n_flank_per_side:
                if abs(snp_pos[j] - apos) > self.max_flank_dist_bp:
                    break
                right_idx.append(j)
                j += 1
                count += 1

            # Place in block slots: [anchor, L, L, L, R, R, R]
            for k, idx in enumerate(left_idx):
                slot = base + 1 + (self.n_flank_per_side - len(left_idx) + k)
                out_mat[slot] = snp_rows[idx]
                out_pos[slot] = snp_pos[idx]
            for k, idx in enumerate(right_idx):
                slot = base + 1 + self.n_flank_per_side + k
                out_mat[slot] = snp_rows[idx]
                out_pos[slot] = snp_pos[idx]

        return out_mat, out_pos

    # ---------------------------------------------------------------------
    # Anchor position sampling
    # ---------------------------------------------------------------------

    def _sample_random_anchors(self, real_te_pos, rng):
        """
        Sample N_RANDOM_BLOCKS positions in [0, window_size) where the
        nearest TE is at least min_random_anchor_dist_bp away.

        Strategy: rejection sampling. If the window is so TE-dense that
        rejection takes too long (>10x oversample), relax the constraint
        and sample uniformly across the whole window.
        """
        n_target = self.n_random_blocks
        if n_target == 0:
            return np.array([], dtype=np.float32)

        # Empty window → uniform sampling, no exclusion
        if len(real_te_pos) == 0:
            return np.sort(rng.uniform(0, self.window_size, size=n_target)).astype(np.float32)

        # Try rejection sampling
        accepted = []
        oversample_factor = 5
        max_iters = 4

        for _ in range(max_iters):
            need = n_target - len(accepted)
            if need <= 0:
                break
            candidates = rng.uniform(
                0, self.window_size, size=need * oversample_factor,
            )
            min_d = np.min(
                np.abs(candidates[:, None] - real_te_pos[None, :]), axis=1,
            )
            valid = candidates[min_d >= self.min_random_anchor_dist_bp]
            accepted.extend(valid[: need].tolist())

        if len(accepted) < n_target:
            # Fall back to uniform sampling without exclusion. Warn so the
            # user knows their constraints are infeasible at this TE density;
            # silently sampling near TEs would bias the "neutral reference"
            # used for the random-anchor region.
            extra = n_target - len(accepted)
            warnings.warn(
                f"Random-anchor rejection sampling fell back to uniform for "
                f"{extra}/{n_target} anchors. TE density too high for "
                f"min_random_anchor_dist_bp={self.min_random_anchor_dist_bp} "
                f"in window of size {self.window_size}. Consider reducing the "
                f"min-distance, reducing n_random_blocks, or accepting that "
                f"some anchors will be near TEs.",
                stacklevel=3,
            )
            extras = rng.uniform(0, self.window_size, size=extra)
            accepted.extend(extras.tolist())

        accepted = np.array(accepted[:n_target], dtype=np.float32)
        return np.sort(accepted)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _padding_block(n_rows, n_haps):
        return (
            np.full((n_rows, n_haps), PADDING_FLAG, dtype=np.int8),
            np.full(n_rows, -1.0, dtype=np.float32),
        )
