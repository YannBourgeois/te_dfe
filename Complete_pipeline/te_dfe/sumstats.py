#!/usr/bin/env python3
"""
sumstats.py — per-TE summary statistics, shared by the tree-sequence processor
(simulations) and the VCF processor (real data). Computing both from this one
module is the same discipline as build_from_variant_table for the matrix
representation: train and inference see identical statistics.

For each TE we build a fixed-length feature vector. Flanking SNPs (left + right
pooled) are split by TE-carrier status; statistics are computed over ALL pooled
flanking SNPs in each group (no sub-blocks, no left/right split). All statistics
are polarization-free: the TE presence defines the derived/core allele, so no
ancestral-state inference is needed.

Feature vector (FEATURE_NAMES below), constant length regardless of phasing:
  te_freq
  he_carrier_mean, he_carrier_sd, he_noncarrier_mean, he_noncarrier_sd
  tajD_carrier, tajD_noncarrier
  hapext_carrier, hapext_noncarrier   (nSL tract length if phased; ZnS if not)
  dist_to_gene, local_recomb          (context covariates; 0 if unavailable)
  n_flank_snps_norm, valid_tajD, valid_hapext

Phasing: nSL (haplotype identity-tract length) needs phased haplotypes and a
well-defined carrier-haplotype set, so it is used only when phased. When
unphased, the carrier split is at the INDIVIDUAL level (dosage >= 1) and the
haplotype-extension slot falls back to ZnS (genotype LD), which does not require
phase. The slot is the same either way, so the vector length is fixed.
"""
import numpy as np

FEATURE_NAMES = [
    "te_freq",
    "he_carrier_mean", "he_carrier_sd", "he_noncarrier_mean", "he_noncarrier_sd",
    "tajD_carrier", "tajD_noncarrier",
    "hapext_carrier", "hapext_noncarrier",
    "dist_to_gene", "local_recomb", "local_snp_density", "local_te_density",
    "n_flank_snps_norm", "valid_tajD", "valid_hapext",
]
N_SUMSTATS_PER_TE = len(FEATURE_NAMES)


def subsample_valid_indices(mask_row, n_target, rng):
    """Indices of up to `n_target` randomly chosen VALID TEs (mask > 0.5) from
    one replicate's mask row. If `n_target` is None or fewer than n_target TEs
    are valid, returns all valid indices. Shared by the TE-count sweep so the
    NPE observation (05b) and the ABC summaries (07) are each built from a random
    subset of the pooled TE-set of the requested size."""
    valid_idx = np.flatnonzero(np.asarray(mask_row) > 0.5)
    if n_target is None or valid_idx.size <= int(n_target):
        return valid_idx
    return np.sort(rng.choice(valid_idx, size=int(n_target), replace=False))

# Minimum segregating sites / sample size for a statistic to be considered valid.
_MIN_SITES_TAJD = 3
_MIN_HAPS       = 4
_MIN_SITES_HAP  = 2


def expected_het_per_site(geno):
    """Per-SNP expected heterozygosity He_j = 2 p_j (1-p_j). geno: (n, S) with
    0/1 calls and -1 for MISSING. p_j is computed over the OBSERVED (non-missing)
    calls at each site, so He is unbiased under missing data (a site with fewer
    calls just yields a noisier p). Returns (S,) array; empty input -> empty."""
    geno = np.asarray(geno, dtype=float)
    if geno.size == 0 or geno.shape[0] == 0:
        return np.zeros((0,), dtype=float)
    obs = geno >= 0
    n_obs = obs.sum(axis=0)
    derived = np.where(obs, geno, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(n_obs > 0, derived / n_obs, 0.0)
    return 2.0 * p * (1.0 - p)


def tajimas_d(geno):
    """
    Tajima's D over a set of biallelic sites. geno: (n_haps, S) with 0/1 calls
    and -1 for MISSING. Missing data is handled in the pixy style: differences
    and comparisons are counted PER SITE over only the called genotypes, so each
    site contributes according to its own observed sample size n_j rather than
    assuming a common n. Folded/pairwise formulation -> no polarization needed.

      pi      = sum_j  diffs_j / comps_j,   diffs_j = c_j(n_j - c_j),
                                            comps_j = n_j(n_j-1)/2
      theta_W = sum_{seg j} 1 / a1(n_j)     (per-site Watterson correction)

    The variance normalization (for which an exact variable-n form is
    intractable) uses the harmonic-mean effective n across segregating sites;
    D is used here as a FEATURE, not a formal test, so this is appropriate. On
    complete data (all n_j equal) this reduces exactly to the standard estimator.
    Returns (D, valid); valid=False when n or S is too small.
    """
    geno = np.asarray(geno, dtype=float)
    if geno.ndim != 2:
        return 0.0, False
    obs = geno >= 0
    n_j = obs.sum(axis=0).astype(float)               # per-site observed count
    derived = np.where(obs, geno, 0.0).sum(axis=0)     # per-site derived count
    seg = (n_j >= 2) & (derived > 0) & (derived < n_j)
    S = int(seg.sum())
    if S < _MIN_SITES_TAJD:
        return 0.0, False
    nj = n_j[seg]
    cj = derived[seg]
    if np.median(nj) < _MIN_HAPS:                      # effective n too small
        return 0.0, False

    # pixy-style mean pairwise differences: per-site diffs / comps, summed.
    diffs = cj * (nj - cj)
    comps = nj * (nj - 1.0) / 2.0
    pi = float(np.sum(diffs / comps))

    # Watterson with a per-site harmonic number a1(n_j).
    a1j = np.array([np.sum(1.0 / np.arange(1, int(n))) for n in nj])
    theta_w = float(np.sum(1.0 / a1j))

    # Variance via the harmonic-mean effective n (exact variable-n var is
    # intractable; D is a feature, so an effective-n normalization suffices).
    n_eff = S / float(np.sum(1.0 / nj))
    n = max(int(round(n_eff)), 2)
    a1 = np.sum(1.0 / np.arange(1, n))
    a2 = np.sum(1.0 / np.arange(1, n) ** 2)
    b1 = (n + 1) / (3.0 * (n - 1))
    b2 = 2.0 * (n * n + n + 3) / (9.0 * n * (n - 1))
    c1 = b1 - 1.0 / a1
    c2 = b2 - (n + 2) / (a1 * n) + a2 / (a1 * a1)
    e1 = c1 / a1
    e2 = c2 / (a1 * a1 + a2)
    var = e1 * S + e2 * S * (S - 1)
    if var <= 0:
        return 0.0, False
    return float((pi - theta_w) / np.sqrt(var)), True


def zns(geno):
    """
    Kelly's ZnS: mean r^2 over all pairs of segregating sites. geno: (n, S) with
    0/1 calls and -1 for MISSING. r^2 is the squared allele-count correlation, so
    this needs NO phasing (genotypic LD). Under missing data each pair's r^2 is
    computed over the individuals observed at BOTH sites (pairwise-complete), so
    ZnS is missing-robust. When no genotype is missing this takes the fast matrix
    path and matches the standard estimator exactly. Returns (ZnS, valid).
    """
    geno = np.asarray(geno, dtype=float)
    if geno.ndim != 2 or geno.shape[0] < _MIN_HAPS:
        return 0.0, False
    obs = geno >= 0
    n_j = obs.sum(axis=0)
    derived = np.where(obs, geno, 0.0).sum(axis=0)
    seg = (n_j >= _MIN_HAPS) & (derived > 0) & (derived < n_j)
    G = geno[:, seg]
    S = G.shape[1]
    if S < _MIN_SITES_HAP:
        return 0.0, False

    if not np.any(G < 0):
        # Fast complete-data path (identical to the standard matrix estimator).
        n = G.shape[0]
        p = G.mean(axis=0)
        pq = p * (1.0 - p)
        Egg = (G.T @ G) / n
        D = Egg - np.outer(p, p)
        denom = np.outer(pq, pq)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = np.where(denom > 0, D * D / denom, 0.0)
        iu = np.triu_indices(S, k=1)
        vals = r2[iu]
        return (float(np.mean(vals)), True) if vals.size else (0.0, False)

    # Missing-data path: pairwise-complete r^2 (per-pair valid rows).
    r2vals = []
    for a in range(S):
        ga_obs = G[:, a] >= 0
        for b in range(a + 1, S):
            both = ga_obs & (G[:, b] >= 0)
            if int(both.sum()) < _MIN_HAPS:
                continue
            ga = G[both, a]; gb = G[both, b]
            pa = ga.mean(); pb = gb.mean()
            va = pa * (1.0 - pa); vb = pb * (1.0 - pb)
            if va <= 0 or vb <= 0:
                continue
            cov = (ga * gb).mean() - pa * pb
            r2vals.append(cov * cov / (va * vb))
    if not r2vals:
        return 0.0, False
    return float(np.mean(r2vals)), True


def _tract_lengths(haps, core_left, core_right):
    """
    Mean pairwise identity-tract length (in SNPs) around a core, for one
    haplotype group. haps split into left flank (ordered core->outward) and
    right flank (core->outward). Vectorized cumulative-identity.
    haps: (n, S). core_left = boolean index of left-flank cols ordered OUTWARD
    from the core; core_right similarly. Returns mean tract length over pairs.
    """
    n = haps.shape[0]
    if n < 2:
        return 0.0
    total = 0.0
    for cols in (core_left, core_right):
        if cols.size == 0:
            continue
        H = haps[:, cols]                       # (n, L), col 0 nearest core
        # pairwise identity per site: (n, n, L)
        ident = (H[:, None, :] == H[None, :, :])
        run = np.cumprod(ident, axis=2)         # contiguous identity from core
        tract = run.sum(axis=2)                 # (n, n) tract length per pair
        iu = np.triu_indices(n, k=1)
        total += tract[iu].mean() if iu[0].size else 0.0
    return float(total)


def compute_te_sumstats(carrier, noncarrier, flank_left_idx, flank_right_idx,
                        te_freq, phased, dist_to_gene=0.0, local_recomb=0.0,
                        local_snp_density=0.0, local_te_density=0.0,
                        n_flank_max=None):
    """
    Build the per-TE feature vector.

    carrier / noncarrier : (n_haps_or_indiv, n_flank_snps) 0/1 matrices of the
        flanking SNPs, restricted to TE carriers / non-carriers. When phased,
        rows are haplotypes; when unphased, rows are individuals (carrier =
        dosage >= 1) and the matrix holds per-individual minor-allele presence.
    flank_left_idx / flank_right_idx : column indices of the left / right flank
        ordered OUTWARD from the TE core (for the nSL tract walk). For He / D /
        ZnS the columns are pooled, so their order does not matter.
    te_freq : derived (carrier) allele frequency of the TE.
    phased  : selects nSL (True) vs ZnS (False) for the hap-extension slot.
    """
    f = np.zeros(N_SUMSTATS_PER_TE, dtype=np.float32)
    f[0] = te_freq

    he_c = expected_het_per_site(carrier)
    he_n = expected_het_per_site(noncarrier)
    f[1] = he_c.mean() if he_c.size else 0.0
    f[2] = he_c.std()  if he_c.size else 0.0
    f[3] = he_n.mean() if he_n.size else 0.0
    f[4] = he_n.std()  if he_n.size else 0.0

    dC, vC = tajimas_d(carrier)
    dN, vN = tajimas_d(noncarrier)
    f[5], f[6] = dC, dN
    valid_tajD = 1.0 if (vC and vN) else 0.0

    if phased:
        hc = _tract_lengths(carrier, flank_left_idx, flank_right_idx)
        hn = _tract_lengths(noncarrier, flank_left_idx, flank_right_idx)
        vh = (carrier.shape[0] >= 2 and noncarrier.shape[0] >= 2)
    else:
        hc, okc = zns(carrier)
        hn, okn = zns(noncarrier)
        vh = okc and okn
    f[7], f[8] = hc, hn
    valid_hapext = 1.0 if vh else 0.0

    f[9]  = dist_to_gene
    f[10] = local_recomb
    f[11] = local_snp_density
    f[12] = local_te_density
    n_flank = (carrier.shape[1] if carrier.ndim == 2 else 0)
    f[13] = (n_flank / n_flank_max) if n_flank_max else float(n_flank)
    f[14] = valid_tajD
    f[15] = valid_hapext
    return f
