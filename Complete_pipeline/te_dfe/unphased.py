#!/usr/bin/env python3
"""
unphased.py — collapse a phased haplotype matrix to an unphased (diploid)
genotype matrix, for the unphased model variant.

Real genotype data is usually unphased: per individual and site we know the
allele DOSAGE (0/1/2) but not which haplotype carries which allele. Collapsing
to one row per individual (rather than shuffling phase) discards exactly the
information that is genuinely unknown — haplotype identity — while retaining
the dosage, which carries most of the signal. (Phase-shuffling, to model
switch errors, is a separate opt-in and is generally overkill; see
shuffle_phase below.)

Encoding. Phased input geno is (n_haps, n_sites) with states
  0 = ancestral, 1 = SNP, 2 = TE, 3 = pad,
where individual i owns consecutive haplotypes 2i, 2i+1. The unphased output is
(n_individuals, n_sites) with a 6-state encoding that preserves both the site
type (SNP vs TE) and the dosage:
  0 = pad
  1 = ancestral (dosage 0)
  2 = SNP heterozygote (dosage 1 at a SNP site)
  3 = SNP homozygote   (dosage 2 at a SNP site)
  4 = TE heterozygote  (dosage 1 at a TE site)
  5 = TE homozygote    (dosage 2 at a TE site)
which one-hots to N_UNPHASED_STATES = 6 channels. A missing/padded haplotype
makes the individual's call at that site pad (you cannot call a diploid genotype
if either copy is missing).

This module is self-contained and unit-testable in isolation; the processor
calls collapse_to_unphased at the end of its build when config.PHASED is False.
"""
import numpy as np

# Phased state codes (must match processor.py).
P_ANC, P_SNP, P_TE, P_PAD = 0, 1, 2, 3

# Unphased state codes.
U_PAD, U_ANC = 0, 1
U_SNP_HET, U_SNP_HOM = 2, 3
U_TE_HET,  U_TE_HOM  = 4, 5
N_UNPHASED_STATES = 6


def collapse_to_unphased(geno_phased):
    """
    geno_phased : (n_haps, n_sites) int, phased states {0,1,2,3}, with
                  individual i = rows 2i, 2i+1.
    Returns (n_individuals, n_sites) int8 in the 6-state unphased encoding.
    """
    g = np.asarray(geno_phased)
    n_haps, n_sites = g.shape
    if n_haps % 2 != 0:
        raise ValueError(f"n_haps={n_haps} is odd; cannot pair into diploids.")
    n_ind = n_haps // 2
    h0 = g[0::2, :]            # (n_ind, n_sites) first haplotype of each indiv
    h1 = g[1::2, :]            # second haplotype

    out = np.full((n_ind, n_sites), U_ANC, dtype=np.int8)

    # Padding: if either copy is padded, the diploid call is missing.
    pad = (h0 == P_PAD) | (h1 == P_PAD)

    # TE dosage (count of TE-present copies) and SNP dosage.
    te_dose  = (h0 == P_TE).astype(np.int8)  + (h1 == P_TE).astype(np.int8)
    snp_dose = (h0 == P_SNP).astype(np.int8) + (h1 == P_SNP).astype(np.int8)

    # TE sites take precedence where a TE copy is present (a column is one site
    # type, so SNP and TE never co-occur in the same column, but this ordering
    # is robust regardless).
    out[snp_dose == 1] = U_SNP_HET
    out[snp_dose == 2] = U_SNP_HOM
    out[te_dose == 1]  = U_TE_HET
    out[te_dose == 2]  = U_TE_HOM
    # dosage 0 stays U_ANC (already filled).
    out[pad] = U_PAD
    return out


def shuffle_phase(geno_phased, switch_rate, rng):
    """
    OPTIONAL switch-error model (opt-in; generally overkill). Instead of
    collapsing, swap the two haplotypes of each individual downstream of random
    switch points along the sequence, so phase is locally scrambled at the given
    per-site switch probability. Keeps the (n_haps, n_sites) shape and the
    4-state phased encoding, so it needs no network change — unlike collapsing.

    geno_phased : (n_haps, n_sites) phased states.
    switch_rate : per-site probability of a phase switch.
    """
    g = np.array(geno_phased, copy=True)
    n_haps, n_sites = g.shape
    n_ind = n_haps // 2
    for i in range(n_ind):
        switches = rng.random(n_sites) < switch_rate
        # cumulative parity: each switch toggles which haplotype is "first"
        flip = np.cumsum(switches) % 2 == 1
        a, b = g[2 * i, flip].copy(), g[2 * i + 1, flip].copy()
        g[2 * i, flip], g[2 * i + 1, flip] = b, a
    return g
