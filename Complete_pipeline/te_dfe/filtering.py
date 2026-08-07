#!/usr/bin/env python3
"""
filtering.py — zero out channels / site-regions / genotype-states, applied
identically at TRAINING (to build a trained-without model) and at BENCHMARK
(the 05b ablation). Sharing one implementation guarantees train and test see
the same masked input, which is what makes a "trained without feature X"
comparison fair.

Operates on numpy arrays in the per-sample layout used by the dataset:
  geno : (..., n_sites) int, last axis = sites; states {0=anc,1=SNP,2=TE,3=pad}
         (typically (n_windows, n_haps, n_sites))
  pos  : (..., n_cont, n_sites) float, continuous channels on the -2 axis
         (typically (n_windows, n_cont, n_sites))

Region layout on the site axis (from config): TE anchors occupy the first
N_TE_SITES columns, flanking SNPs the next N_TE_FLANK_SITES, random background
blocks the remainder. TE anchors are never zeroable (they carry the core TE
signal); only "flank" and "random" can be zeroed.
"""
import numpy as np

# Match processor.py state codes.
_ANC, _SNP, _TE, _PAD = 0, 1, 2, 3


def region_slices(n_sites, n_te_sites, n_flank_sites):
    """Return {region: (start, end)} site-index ranges for the layout."""
    a0, a1 = 0, n_te_sites
    f0, f1 = a1, a1 + n_flank_sites
    r0, r1 = f1, n_sites
    return {"anchor": (a0, a1), "flank": (f0, f1), "random": (r0, r1)}


def apply_feature_filter(geno, pos,
                         zero_channels=(), zero_regions=(), zero_states=(),
                         n_te_sites=None, n_flank_sites=None):
    """
    Return (geno, pos) with the requested features zeroed. Inputs are not
    mutated (copies are made only if something is zeroed).

    zero_channels : iterable of continuous-channel indices (into pos axis -2).
    zero_regions  : subset of {"flank","random"} site regions to blank (geno ->
                    ancestral, pos -> 0 in those columns).
    zero_states   : subset of {"snp","te"} -> set those geno cells to ancestral.
    n_te_sites, n_flank_sites : layout (required if zero_regions given).
    """
    if not (zero_channels or zero_regions or zero_states):
        return geno, pos

    geno = np.array(geno, copy=True)
    pos  = np.array(pos,  copy=True) if pos is not None else None

    # Channels: zero the selected continuous channels.
    if zero_channels and pos is not None:
        n_ch = pos.shape[-2]
        for ch in zero_channels:
            ch = int(ch)
            if not (0 <= ch < n_ch):
                raise ValueError(
                    f"zero_channels {ch} out of range: pos has {n_ch} "
                    f"channels (valid 0..{n_ch-1}).")
            pos[..., ch, :] = 0.0

    # States: set the selected genotype states to ancestral.
    state_map = {"snp": _SNP, "te": _TE}
    for st in zero_states:
        code = state_map.get(st)
        if code is None:
            raise ValueError(f"zero_states '{st}' not in {{'snp','te'}}.")
        geno[geno == code] = _ANC

    # Regions: blank out whole site ranges (geno -> ancestral, pos -> 0).
    if zero_regions:
        if n_te_sites is None or n_flank_sites is None:
            raise ValueError("zero_regions requires n_te_sites and "
                             "n_flank_sites for the layout.")
        n_sites = geno.shape[-1]
        slices = region_slices(n_sites, n_te_sites, n_flank_sites)
        for reg in zero_regions:
            if reg == "anchor":
                raise ValueError("the TE-anchor region cannot be zeroed.")
            if reg not in slices:
                raise ValueError(
                    f"zero_regions '{reg}' not in {{'flank','random'}}.")
            s, e = slices[reg]
            geno[..., s:e] = _ANC
            if pos is not None:
                pos[..., :, s:e] = 0.0

    return geno, pos
