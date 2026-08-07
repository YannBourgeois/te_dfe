#!/usr/bin/env python3
"""
vcf_to_tensors.py — convert real VCF data into the (geno, pos) per-window
tensors the network consumes, so a trained model can do inference on real data.

It produces the SAME representation as the simulator+processor path: per window,
the variant table (TE rows + SNP rows with positions) is fed to the processor's
build_from_variant_table, so the three-region layout, position/density/recomb
channels, folding, masking, and (un)phasing are all applied by the same code as
in training. This guarantees real data lands on the training manifold.

Inputs (TEs and SNPs may be in one joint VCF or two separate files):
  * SNP VCF: standard biallelic SNP calls. Genotypes read per haplotype (phased)
    or per individual (collapsed downstream by the processor when PHASED=False).
  * TE calls: either rows in the same VCF flagged by an INFO field (e.g.
    INFO/SVTYPE=TE or a user-named flag), or a separate VCF / BED-like table of
    TE presence genotypes. TEs are coded as presence (1) / absence (0) per
    haplotype; they are always "derived" and never folded.

Honors:
  * FOLD_SFS  : SNPs folded (minor allele = 1) to match training. TEs untouched.
  * PHASED    : if False, the processor collapses haplotype pairs to diploid
                dosage (handled in build_from_variant_table's downstream
                __call__ path is for ts; here we collapse explicitly after the
                table is built if needed — see note below).
  * coverage mask / window coords : passed through so masked positions are
    dropped exactly as in training.

NOTE on phasing: build_from_variant_table returns a phased (n_haps, n_sites)
matrix; collapsing to unphased is applied here (mirroring the processor) when
PHASED is False, using the same te_dfe.unphased.collapse_to_unphased.

This module requires cyvcf2 (fast VCF parsing). Install: pip install cyvcf2.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def _read_vcf_records(vcf_path, region=None):
    """
    Yield (chrom, pos0, gts_haploid) for biallelic records. gts_haploid is a
    (n_haps,) int array of 0/1 allele indicators (2 per individual, in sample
    order). pos0 is 0-based. Requires cyvcf2.
    """
    try:
        from cyvcf2 import VCF
    except Exception as e:
        raise ImportError(
            "vcf_to_tensors requires cyvcf2 (pip install cyvcf2).") from e
    vcf = VCF(vcf_path)
    it = vcf(region) if region else vcf
    for rec in it:
        if rec.ALT is None or len(rec.ALT) != 1:
            continue  # biallelic only
        # genotype array: (n_samples, 3) -> [a0, a1, phased]
        g = np.asarray(rec.genotypes, dtype=np.int64)
        if g.size == 0:
            continue
        a0 = g[:, 0].copy()
        a1 = g[:, 1].copy()
        # Missing alleles are -1 in cyvcf2; mark them as missing (we pass them
        # through as -1 so the caller can decide; processor padding handles it).
        haps = np.empty(2 * len(a0), dtype=np.int64)
        haps[0::2] = a0
        haps[1::2] = a1
        yield rec.CHROM, int(rec.POS) - 1, haps


def _te_flag(rec_info_getter, te_info_field, te_info_value):
    """True if a record is a TE call by its INFO field."""
    try:
        v = rec_info_getter(te_info_field)
    except Exception:
        return False
    if v is None:
        return False
    if te_info_value is None:
        return True
    return str(v) == str(te_info_value)


def build_window_table(snp_vcf, te_vcf, chrom, win_start, win_size,
                       maf_min, te_info_field=None, te_info_value="TE",
                       joint=False, preserve_missing=True):
    """
    Build the variant table for one window:
      returns (te_rows, te_pos, snp_rows, snp_pos)
    with window-RELATIVE positions, matching the processor's expectations.

    Genotype coding per haplotype: 0 = ref/ancestral, 1 = alt/derived, and
    -1 = MISSING. When preserve_missing=True (default, for the SUMMARY-STATISTIC
    path), missing calls are kept as -1 so the observed-count statistics
    (He, Tajima's D, ZnS) handle them natively. When False (the MATRIX path,
    which needs concrete categorical states), missing is mapped to 0 (ancestral)
    as before. Allele FREQUENCY, MAF filtering, and SFS folding are ALWAYS
    computed over the observed (non-missing) calls.

    snp_vcf : path to SNP VCF.
    te_vcf  : path to TE VCF (or None if joint, using te_info_field on snp_vcf).
    joint   : if True, TEs are flagged inside snp_vcf via te_info_field.
    """
    from te_dfe.config import FOLD_SFS, TE_STATE as _TE_STATE
    from cyvcf2 import VCF

    win_end = win_start + win_size
    region = f"{chrom}:{win_start+1}-{win_end}"

    te_rows, te_pos = [], []
    snp_rows, snp_pos = [], []

    def _haps_from_rec(rec):
        g = np.asarray(rec.genotypes, dtype=np.int64)
        if g.size == 0:
            return None
        haps = np.empty(2 * g.shape[0], dtype=np.int64)
        haps[0::2] = g[:, 0]
        haps[1::2] = g[:, 1]
        # cyvcf2 marks missing alleles as < 0; normalize any negative to -1.
        haps[haps < 0] = -1
        return haps

    # SNPs (and, if joint, TEs flagged in the same file).
    vcf = VCF(snp_vcf)
    for rec in vcf(region):
        if rec.ALT is None or len(rec.ALT) != 1:
            continue
        haps = _haps_from_rec(rec)
        if haps is None:
            continue
        pos_rel = (int(rec.POS) - 1) - win_start
        if not (0 <= pos_rel < win_size):
            continue
        is_te = joint and _te_flag(rec.INFO.get, te_info_field, te_info_value)
        if is_te:
            # TE presence row in the TE_STATE coding used by the processor:
            # present(alt)->TE_STATE(2), absent->0, missing->-1 (if preserved).
            row = np.where(haps == 1, _TE_STATE, 0).astype(np.int8)
            if preserve_missing:
                row = np.where(haps < 0, -1, row).astype(np.int8)
            te_rows.append(row)
            te_pos.append(float(pos_rel))
        else:
            obs = haps >= 0
            n_obs = int(obs.sum())
            if n_obs == 0:
                continue
            dc = int((haps == 1).sum())                 # observed derived count
            if not (0 < dc < n_obs):                    # monomorphic (observed)
                continue
            freq = dc / n_obs                           # observed-count freq
            if min(freq, 1.0 - freq) < maf_min:
                continue
            # row: -1 missing preserved (sumstat) or mapped to 0 (matrix).
            row = haps.astype(np.int8) if preserve_missing \
                else np.where(haps == 1, 1, 0).astype(np.int8)
            if FOLD_SFS and freq > 0.5:
                # flip only the OBSERVED calls; leave -1 missing untouched.
                row = np.where(row < 0, -1, 1 - row).astype(np.int8)
            snp_rows.append(row)
            snp_pos.append(float(pos_rel))

    # Separate TE VCF.
    if te_vcf is not None and not joint:
        tv = VCF(te_vcf)
        for rec in tv(region):
            haps = _haps_from_rec(rec)
            if haps is None:
                continue
            pos_rel = (int(rec.POS) - 1) - win_start
            if not (0 <= pos_rel < win_size):
                continue
            row = np.where(haps == 1, _TE_STATE, 0).astype(np.int8)
            if preserve_missing:
                row = np.where(haps < 0, -1, row).astype(np.int8)
            te_rows.append(row)
            te_pos.append(float(pos_rel))

    return te_rows, te_pos, snp_rows, snp_pos


def vcf_window_to_tensors(processor, snp_vcf, te_vcf, chrom, win_start,
                          win_size, recomb_map=None, seed=0,
                          te_info_field=None, te_info_value="TE", joint=False):
    """
    Convert one window of VCF data to (geno, pos) using the processor's shared
    builder, so the representation matches training exactly. Applies the
    coverage mask (via the processor's window coords) and unphased collapse
    (when config.PHASED is False), mirroring the processor.
    """
    from te_dfe.config import PHASED, MAF_MIN
    te_rows, te_pos, snp_rows, snp_pos = build_window_table(
        snp_vcf, te_vcf, chrom, win_start, win_size, MAF_MIN,
        te_info_field=te_info_field, te_info_value=te_info_value, joint=joint,
        preserve_missing=False)  # matrix path needs concrete states

    # Tell the processor this window's absolute coords (for coverage masking).
    processor.window_chrom = chrom
    processor.window_start = win_start

    geno, pos = processor.build_from_variant_table(
        te_rows, te_pos, snp_rows, snp_pos,
        n_haps=(len(te_rows[0]) if te_rows else
                (len(snp_rows[0]) if snp_rows else 0)),
        seed=seed, recomb_map=recomb_map,
        window_chrom=chrom, window_start=win_start)

    # NOTE: build_from_variant_table already applies the unphased collapse (when
    # config.PHASED is False) and the synthetic missingness augmentation (when
    # config.MISSINGNESS is set). Do NOT re-collapse here. For real-data
    # inference, set config.MISSINGNESS = None — real missing genotypes are
    # already in the VCF; the synthetic augmentation is a TRAINING tool only.
    return geno, pos


def vcf_window_to_sumstats(processor, snp_vcf, te_vcf, chrom, win_start,
                           win_size, recomb_map=None,
                           te_info_field=None, te_info_value="TE", joint=False):
    """
    Convert one window of VCF data to the per-TE SUMMARY-STATISTIC representation
    (feats, mask), using the processor's shared build_sumstats_from_variant_table
    — the SAME code the simulator path uses — so real-data statistics are
    computed identically to training. Honors FOLD_SFS / PHASED / masking exactly
    as the simulator path does (folding is applied in build_window_table; the
    carrier split + phasing-aware stats are inside the shared builder).

    Use this when config.REPRESENTATION == "sumstat". For real-data inference set
    config.MISSINGNESS = None (real missingness is already in the VCF). Returns
    (feats, mask): (N_TE_TRAIN_MAX, N_SUMSTATS_PER_TE) and (N_TE_TRAIN_MAX,).
    """
    from te_dfe.config import MAF_MIN
    te_rows, te_pos, snp_rows, snp_pos = build_window_table(
        snp_vcf, te_vcf, chrom, win_start, win_size, MAF_MIN,
        te_info_field=te_info_field, te_info_value=te_info_value, joint=joint,
        preserve_missing=True)  # sumstat path: -1 reaches observed-count stats

    processor.window_chrom = chrom
    processor.window_start = win_start
    n_haps = (len(te_rows[0]) if te_rows else
              (len(snp_rows[0]) if snp_rows else 0))
    feats, mask = processor.build_sumstats_from_variant_table(
        te_rows, te_pos, snp_rows, snp_pos, n_haps=n_haps,
        recomb_map=recomb_map, window_chrom=chrom, window_start=win_start)
    return feats, mask


def vcf_to_sumstat_observation(processor, snp_vcf, te_vcf, windows,
                               recomb_map=None, te_info_field=None,
                               te_info_value="TE", joint=False, subsample_seed=0,
                               annotation=None):
    """
    Genome-wide SUMMARY-STATISTIC observation from real VCF data, pooling per-TE
    features across MANY windows exactly as 05a + 05a_collate do: each window
    contributes its RAW (unpadded) per-TE features (pad=False); all are
    concatenated into one TE set, then padded/subsampled ONCE to N_TE_TRAIN_MAX.

    windows : iterable of (chrom, win_start, win_size). Total sequence length is
              summed from these (for the optional genome-length covariate).
    annotation : optional GenomeAnnotation; if given, per-window gene intervals
              are set on the processor so dist_to_gene is real (else it is 0).

    Returns (feats, te_mask, seq_len_bp).
    """
    from te_dfe.config import MAF_MIN, N_TE_TRAIN_MAX, N_SUMSTATS_PER_TE
    pooled, seq_len_bp = [], 0.0
    for (chrom, win_start, win_size) in windows:
        seq_len_bp += float(win_size)
        te_rows, te_pos, snp_rows, snp_pos = build_window_table(
            snp_vcf, te_vcf, chrom, win_start, win_size, MAF_MIN,
            te_info_field=te_info_field, te_info_value=te_info_value,
            joint=joint, preserve_missing=True)
        n_haps = (len(te_rows[0]) if te_rows else
                  (len(snp_rows[0]) if snp_rows else 0))
        if n_haps == 0 or not te_rows:
            continue
        processor.window_chrom = chrom
        processor.window_start = win_start
        # Real dist_to_gene needs the window's gene intervals (window-relative).
        if annotation is not None:
            processor.gene_intervals = annotation.intervals_in_window(
                chrom, win_start, win_start + win_size)
        else:
            processor.gene_intervals = None
        raw = processor.build_sumstats_from_variant_table(
            te_rows, te_pos, snp_rows, snp_pos, n_haps=n_haps,
            recomb_map=recomb_map, window_chrom=chrom, window_start=win_start,
            pad=False)
        raw = np.asarray(raw, dtype=np.float32)
        if raw.shape[0]:
            pooled.append(raw)

    allte = (np.concatenate(pooled, axis=0) if pooled
             else np.zeros((0, N_SUMSTATS_PER_TE), dtype=np.float32))
    feats = np.zeros((N_TE_TRAIN_MAX, N_SUMSTATS_PER_TE), dtype=np.float32)
    te_mask = np.zeros((N_TE_TRAIN_MAX,), dtype=np.float32)
    if allte.shape[0] > N_TE_TRAIN_MAX:
        rng = np.random.default_rng(subsample_seed)
        sel = np.sort(rng.choice(allte.shape[0], N_TE_TRAIN_MAX, replace=False))
        feats[:] = allte[sel]; te_mask[:] = 1.0
    elif allte.shape[0]:
        n = allte.shape[0]
        feats[:n] = allte; te_mask[:n] = 1.0
    return feats, te_mask, seq_len_bp


def sumstat_flat_observation(feats, te_mask, seq_len_bp=None):
    """
    Flat observation the trained sumstat posterior expects:
    [feats.flatten() | te_mask.flatten() | (seqlen_cov)], matching
    SumstatSBIWrapper / 05b / 03_train. The covariate is appended only when
    config.SUMSTAT_USE_SEQLEN_COV is on (then seq_len_bp is required).
    """
    feats = np.asarray(feats, dtype=np.float32)
    te_mask = np.asarray(te_mask, dtype=np.float32)
    parts = [feats.ravel(), te_mask.ravel()]
    from te_dfe.config import SUMSTAT_USE_SEQLEN_COV
    if SUMSTAT_USE_SEQLEN_COV:
        if seq_len_bp is None:
            raise ValueError("SUMSTAT_USE_SEQLEN_COV on but seq_len_bp is None.")
        from te_dfe.config import seqlen_covariate
        parts.append(np.array([seqlen_covariate(float(seq_len_bp))],
                              dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)
