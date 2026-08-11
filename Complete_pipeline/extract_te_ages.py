#!/usr/bin/env python
"""
extract_te_ages.py — pull the TRUE age of every TE out of a SLiM tree sequence.

Why age matters
---------------
A transposition burst and purifying selection are confounded in the marginal
SFS: both leave an excess of rare insertions. They are NOT confounded once the
insertions are partitioned by AGE. A burst puts extra insertions into one age
window and leaves the others alone; selection reshapes the frequency spectrum
WITHIN every window. Those are orthogonal, which is what makes the
age-partitioned fit (fit_dfe_moments_aged.py) able to separate them.

This script provides the ground-truth version of that partition, so the method
can be validated before trusting an age ESTIMATOR on real data.

    # ages from a SLiM tree sequence
    python extract_te_ages.py --trees sim.trees --out ages.tsv

    # with local recombination, needed by the CONTEXT fit
    python extract_te_ages.py --trees win_*.trees --out ages.tsv --recomb auto

    # and a phased VCF (TEs + SNPs) to hand to GEVA / Relate / tsinfer+tsdate
    python extract_te_ages.py --trees sim.trees --out ages.tsv --vcf sites.vcf

GETTING THE TREES
    The pipeline writes SLiM output to a scratch workdir and does not keep it.
    Set the debug dump dir when simulating (simulator_*.py takes debug_dump_dir,
    which writes sim_full_window_NN.trees per window), or re-run a handful of
    replicates with it enabled. You only need enough shards to validate.

COLUMNS
    position, age_gen, derived_count, n_haps, selection_coeff, is_te,
    window, local_recomb

    `window` is the linkage block: TEs in one window are linked, TEs in
    different windows are not. It is what --boot-block-col should be set to in
    the fit, so the bootstrap resamples whole windows and the CI is not
    optimistic. `local_recomb` is the per-TE recombination rate, which the
    context model needs to estimate alpha_r and gamma_r.

CONVENTIONS
    * TEs are mutation type m1 in te_dfe_burst.slim; the background deleterious
      load is m2 and is excluded here.
    * tskit mutation times are in generations BEFORE the end of the simulation,
      which is the same "generations ago" convention the burst parameters use
      (log10_t_start is the older edge of the burst window, in generations ago).
    * Only SEGREGATING insertions are reported: fixed and lost sites carry no
      frequency information and are dropped, matching the fitting code.
"""
import argparse
import sys
from pathlib import Path

import numpy as np


TE_MUTATION_TYPE = 1          # m1 in the SLiM template
BG_MUTATION_TYPE = 2          # m2, the background deleterious load

# Key names the per-window recombination npz might use. The simulator writes
# win_NN_recomb.npz alongside win_NN.trees; different generations of the
# pipeline have used different key spellings, so probe rather than assume.
_POS_KEYS  = ("positions", "position", "pos", "edges", "breakpoints", "ends")
_RATE_KEYS = ("rates", "rate", "recomb", "recomb_rate", "recombination_rate")


def load_recomb_map(path):
    """Return (edges, rates) for a per-window recombination map, or None.

    `edges` has one more entry than `rates`: rate[i] applies on
    [edges[i], edges[i+1]). If the file stores equal-length arrays we treat them
    as (position, rate) samples and step-interpolate from the left instead.
    """
    try:
        d = np.load(str(path), allow_pickle=True)
    except Exception as e:
        print(f"  !! could not read {path}: {type(e).__name__}: {e}",
              file=sys.stderr)
        return None
    pos = next((np.asarray(d[k], float).ravel() for k in _POS_KEYS
                if k in d.files), None)
    rate = next((np.asarray(d[k], float).ravel() for k in _RATE_KEYS
                 if k in d.files), None)
    if rate is None:
        print(f"  !! {Path(path).name} has no recombination-rate array "
              f"(keys: {list(d.files)})", file=sys.stderr)
        return None
    if pos is None:
        # rates only: assume uniform bins over the window, resolved by caller
        return None, rate
    return pos, rate


def recomb_at(positions, rmap, seq_len):
    """Per-TE recombination rate by step lookup into the window's map."""
    if rmap is None:
        return np.full(len(positions), np.nan)
    edges, rates = rmap
    positions = np.asarray(positions, float)
    if edges is None:                      # uniform bins across the window
        n = len(rates)
        idx = np.clip((positions / max(seq_len, 1e-9) * n).astype(int), 0, n - 1)
        return rates[idx]
    if len(edges) == len(rates):           # (position, rate) samples
        idx = np.clip(np.searchsorted(edges, positions, side="right") - 1,
                      0, len(rates) - 1)
        return rates[idx]
    # standard form: len(edges) == len(rates) + 1
    idx = np.clip(np.searchsorted(edges, positions, side="right") - 1,
                  0, len(rates) - 1)
    return rates[idx]


def recomb_intervals(rmap, seq_len):
    """(start, end, rate) arrays covering one window, from a loaded map.

    Handles the three layouts load_recomb_map can return: edges (n+1) with rates
    (n), equal-length (position, rate) samples, and rates-only over uniform bins.
    Writing these out lets the fit compute the SEQUENCE LENGTH behind each
    recombination class exactly, instead of proxying it from site counts -- and
    makes masking trivial later, since a masked region is just a trimmed or
    dropped interval.
    """
    if rmap is None:
        return None
    edges, rates = rmap
    rates = np.asarray(rates, float).ravel()
    if edges is None:                                   # uniform bins
        n = len(rates)
        b = np.linspace(0.0, seq_len, n + 1)
        return b[:-1], b[1:], rates
    edges = np.asarray(edges, float).ravel()
    if len(edges) == len(rates) + 1:                    # standard form
        return edges[:-1], edges[1:], rates
    if len(edges) == len(rates):                        # samples: extend to the end
        end = np.concatenate([edges[1:], [seq_len]])
        return edges, end, rates
    n = min(len(edges) - 1, len(rates))
    return edges[:n], edges[1:n + 1], rates[:n]


def find_recomb_file(trees_path):
    """Sibling recombination npz for one .trees file, or None.

    Tries <stem>_recomb.npz first (win_00.trees -> win_00_recomb.npz), then any
    single *_recomb.npz in the same directory whose stem shares the window index.
    """
    p = Path(trees_path)
    cand = p.with_name(p.stem + "_recomb.npz")
    if cand.exists():
        return cand
    # fall back on matching a trailing window index, e.g. sim_full_window_03
    m = None
    for tok in p.stem.replace("-", "_").split("_")[::-1]:
        if tok.isdigit():
            m = tok
            break
    if m is not None:
        for c in sorted(p.parent.glob("*_recomb.npz")):
            toks = c.stem.replace("-", "_").split("_")
            if any(t.isdigit() and int(t) == int(m) for t in toks):
                return c
    return None


def slim_mutation_types(mut):
    """Mutation type ids carried by one tskit mutation (SLiM metadata).

    A SLiM mutation record can stack several mutations at one site, so the
    metadata holds a list. Returns [] for non-SLiM mutations, e.g. the neutral
    SNPs overlaid with msprime after recapitation.
    """
    md = mut.metadata
    if not isinstance(md, dict):
        return []
    return [m.get("mutation_type") for m in md.get("mutation_list", [])]


def slim_selection_coeffs(mut):
    md = mut.metadata
    if not isinstance(md, dict):
        return []
    return [m.get("selection_coeff") for m in md.get("mutation_list", [])]


def extract(trees_path, include_bg=False, max_sites=None, window_id=0,
            recomb_path=None):
    """Return (rows, sequence_length) for one window.

    `window_id` is carried through so downstream code can tell which of a
    simulation's windows each TE came from. Positions stay LOCAL to the window:
    the windows are non-contiguous genomic regions, so splicing them into one
    coordinate system would imply linkage between windows that does not exist,
    and would corrupt any recombination-based age estimator run on the VCF.

    `recomb_path` is the window's recombination npz; each TE gets the rate at
    its own position, which is what the context model regresses selection and
    insertion against. NaN when no map is available.
    """
    import tskit
    ts = tskit.load(str(trees_path))
    n_haps = int(ts.num_samples)
    seq_len = float(ts.sequence_length)
    rmap = load_recomb_map(recomb_path) if recomb_path else None

    rows = []
    for var in ts.variants():
        site = var.site
        if not site.mutations:
            continue
        # A site may carry several stacked mutations; take the oldest, which is
        # the one that introduced the derived state now segregating.
        muts = sorted(site.mutations, key=lambda m: -m.time)
        m0 = muts[0]
        types = slim_mutation_types(m0)
        if not types:
            continue                                # neutral overlay SNP
        is_te = TE_MUTATION_TYPE in types
        is_bg = BG_MUTATION_TYPE in types
        if not (is_te or (include_bg and is_bg)):
            continue
        g = var.genotypes
        dc = int((g > 0).sum())
        if not (0 < dc < n_haps):                   # segregating only
            continue
        sc = slim_selection_coeffs(m0)
        rows.append((float(site.position), float(m0.time), dc, n_haps,
                     float(sc[0]) if sc and sc[0] is not None else np.nan,
                     int(is_te), int(window_id), np.nan))
        if max_sites and len(rows) >= max_sites:
            break

    arr = np.array(rows, dtype=[("position", "f8"), ("age_gen", "f8"),
                                ("derived_count", "i4"), ("n_haps", "i4"),
                                ("selection_coeff", "f8"), ("is_te", "i4"),
                                ("window", "i4"), ("local_recomb", "f8")])
    if len(arr) and rmap is not None:
        arr["local_recomb"] = recomb_at(arr["position"], rmap, seq_len)
    return arr, seq_len


def extract_neutral_snps(trees_path, window_id=0, max_snps=None, rng=None,
                         recomb_path=None):
    """Ages and frequencies of the NEUTRAL overlay SNPs in one window.

    These are the msprime-overlaid mutations added after recapitation: they
    carry no SLiM metadata, which is exactly how they are told apart from TEs
    (m1) and the background load (m2).

    WHY THEY ARE USEFUL. They pass through the SAME age estimator as the TEs and
    experience the same demography, so their age-partitioned SFS is an EMPIRICAL
    neutral baseline. Calibrating the fit against it absorbs demographic
    misspecification AND the age estimator's own distortion -- neither of which
    any theoretical expectation can capture. This is the same logic as
    calibrating a DFE on synonymous sites, with the bonus that the age-estimator
    error cancels because it acts on both classes alike.

    CAVEAT for real data: "neutral" must really mean neutral. Under background
    selection linked SNPs are not, and the calibration would then import BGS
    into the baseline. Mask regions near genes / under strong BGS before using
    this on real data.

    Neutral SNPs vastly outnumber TEs, so a random subset (--max-snps) is taken:
    the baseline only needs enough sites per age window to be stable.
    """
    import tskit
    ts = tskit.load(str(trees_path))
    n_haps = int(ts.num_samples)
    seq_len = float(ts.sequence_length)
    rmap = load_recomb_map(recomb_path) if recomb_path else None

    rows = []
    for var in ts.variants():
        site = var.site
        if not site.mutations:
            continue
        muts = sorted(site.mutations, key=lambda m: -m.time)
        m0 = muts[0]
        if slim_mutation_types(m0):
            continue                                # SLiM mutation: not neutral
        g = var.genotypes
        dc = int((g > 0).sum())
        if not (0 < dc < n_haps):
            continue
        rows.append((float(site.position), float(m0.time), dc, n_haps,
                     int(window_id), np.nan))

    arr = np.array(rows, dtype=[("position", "f8"), ("age_gen", "f8"),
                                ("derived_count", "i4"), ("n_haps", "i4"),
                                ("window", "i4"), ("local_recomb", "f8")])
    if max_snps and len(arr) > max_snps:
        rng = rng or np.random.default_rng(0)
        arr = arr[np.sort(rng.choice(len(arr), int(max_snps), replace=False))]
    if len(arr) and rmap is not None:
        arr["local_recomb"] = recomb_at(arr["position"], rmap, seq_len)
    return arr, seq_len


def write_vcf(trees_path, vcf_path, contig_id=None):
    """Phased VCF of every site in ONE window, for an external age estimator.

    GEVA, Relate and tsinfer+tsdate all want phased haplotypes, and SLiM tree
    sequences are phased by construction, so this is lossless.

    ONE FILE PER WINDOW, deliberately. The windows are non-contiguous genomic
    regions; merging them into a single coordinate system would imply linkage
    across window boundaries that does not exist, and every one of these
    estimators dates alleles from recombination distance. Each window gets its
    own contig id so a merged file would still be interpreted correctly.

    The VCF does not distinguish TEs from SNPs; join on (window, POS) against
    the age table to recover which is which.
    """
    import tskit
    ts = tskit.load(str(trees_path))
    with open(vcf_path, "w") as fh:
        ts.write_vcf(fh, ploidy=2,
                     contig_id=str(contig_id) if contig_id is not None else "1")
    return ts.num_sites


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trees", nargs="+", required=True,
                    help="one or more SLiM .trees files (windows of one sim)")
    ap.add_argument("--out", required=True, help="output TSV of per-TE ages")
    ap.add_argument("--recomb", nargs="*", default=None,
                    help="per-window recombination npz files, in the SAME order "
                         "as --trees; or the single word 'auto' to look for a "
                         "sibling <stem>_recomb.npz next to each .trees file. "
                         "Needed by the context model (alpha_r / gamma_r); "
                         "without it local_recomb is written as nan.")
    ap.add_argument("--vcf", default=None,
                    help="also write a phased VCF (first --trees file only) for "
                         "GEVA / Relate / tsinfer+tsdate")
    ap.add_argument("--include-background", action="store_true",
                    help="also report m2 background deleterious mutations, for "
                         "checking how much BGS load is present")
    ap.add_argument("--max-sites", type=int, default=None)
    ap.add_argument("--snps-out", default=None, metavar="FILE",
                    help="also write the NEUTRAL overlay SNPs (ages + "
                         "frequencies) to this TSV. Used by the fit's "
                         "--neutral-ages to build an EMPIRICAL neutral baseline "
                         "instead of relying on the demographic expectation.")
    ap.add_argument("--recomb-map-out", default=None, metavar="FILE",
                    help="write the recombination map as explicit intervals "
                         "(window, start, end, rate). The context fit uses this "
                         "to get the SEQUENCE LENGTH behind each recombination "
                         "bin exactly, which is what log_lambda is scaled "
                         "against; without it the sequence share has to be "
                         "proxied and lambda comes out biased.")
    ap.add_argument("--max-snps-per-window", type=int, default=20000,
                    help="random subset of neutral SNPs kept per window "
                         "(they vastly outnumber TEs; the baseline only needs "
                         "enough per age window to be stable).")
    args = ap.parse_args()

    # resolve the recombination map for each window
    rec_paths = [None] * len(args.trees)
    if args.recomb:
        if len(args.recomb) == 1 and str(args.recomb[0]).lower() == "auto":
            rec_paths = [find_recomb_file(tp) for tp in args.trees]
            n_found = sum(p is not None for p in rec_paths)
            print(f"[recomb] auto-detected {n_found}/{len(args.trees)} map(s)")
        elif len(args.recomb) != len(args.trees):
            sys.exit(f"--recomb takes 'auto' or exactly {len(args.trees)} files "
                     f"to match --trees, got {len(args.recomb)}")
        else:
            rec_paths = [Path(p) for p in args.recomb]

    all_rows, total_bp = [], 0.0
    for k, tp in enumerate(args.trees):
        r, seq_len = extract(tp, include_bg=args.include_background,
                             max_sites=args.max_sites, window_id=k,
                             recomb_path=rec_paths[k])
        total_bp += seq_len                 # the TRUE span, not the last TE
        if len(r):
            all_rows.append(r)              # positions stay LOCAL to the window
        rec_note = ""
        if len(r):
            n_rec = int(np.isfinite(r["local_recomb"]).sum())
            if n_rec:
                rec_note = (f", recomb {np.nanmedian(r['local_recomb']):.2e} "
                            f"(median)")
        print(f"  [{k:>3}] {Path(tp).name}: {int((r['is_te'] == 1).sum())} "
              f"segregating TEs over {seq_len/1e6:.2f} Mb"
              + (f", {int((r['is_te'] == 0).sum())} background"
                 if args.include_background else "") + rec_note)
    if args.recomb_map_out:
        iv_rows = []
        for k, tp in enumerate(args.trees):
            if rec_paths[k] is None:
                continue
            import tskit as _tsk
            sl = float(_tsk.load(str(tp)).sequence_length)
            iv = recomb_intervals(load_recomb_map(rec_paths[k]), sl)
            if iv is None:
                continue
            st, en, ra = iv
            for a, b_, r_ in zip(st, en, ra):
                iv_rows.append((k, a, b_, r_))
        if iv_rows:
            with open(args.recomb_map_out, "w") as fh:
                fh.write("window\tstart\tend\trate\n")
                for k, a, b_, r_ in iv_rows:
                    fh.write(f"{k}\t{a:.1f}\t{b_:.1f}\t{r_:.6g}\n")
            tot = sum(b_ - a for _, a, b_, _ in iv_rows)
            print(f"wrote {args.recomb_map_out}: {len(iv_rows)} intervals "
                  f"covering {tot/1e6:.1f} Mb")
        else:
            print("  !! no recombination maps available; --recomb-map-out "
                  "skipped", file=sys.stderr)

    if args.snps_out:
        rng_s = np.random.default_rng(0)
        snp_rows = []
        for k, tp in enumerate(args.trees):
            sr, _ = extract_neutral_snps(tp, window_id=k,
                                         max_snps=args.max_snps_per_window,
                                         rng=rng_s, recomb_path=rec_paths[k])
            if len(sr):
                snp_rows.append(sr)
            print(f"  [{k:>3}] {Path(tp).name}: {len(sr)} neutral SNPs kept")
        if snp_rows:
            snps = np.concatenate(snp_rows)
            hdr_s = (f"# seq_length_bp={total_bp:.0f} n_windows={len(args.trees)}\n"
                     "position\tage_gen\tderived_count\tn_haps\twindow"
                     "\tlocal_recomb")
            np.savetxt(args.snps_out,
                       np.column_stack([snps[c] for c in snps.dtype.names]),
                       delimiter="\t", header=hdr_s, comments="",
                       fmt=["%.1f", "%.4f", "%d", "%d", "%d", "%.6g"])
            print(f"wrote {args.snps_out}: {len(snps)} neutral SNPs "
                  f"(empirical neutral baseline for --neutral-ages)")
        else:
            print("  !! no neutral overlay SNPs found; was the tree sequence "
                  "recapitated and mutated with msprime?", file=sys.stderr)

    if not all_rows:
        sys.exit("no segregating TEs found; is this a SLiM tree sequence?")
    rows = np.concatenate(all_rows)

    # The total sequence length is written into the header so the fit does not
    # have to be told it separately (getting it wrong only shifts log_lambda,
    # but it shifts it silently).
    hdr = (f"# seq_length_bp={total_bp:.0f} n_windows={len(args.trees)}\n"
           "position\tage_gen\tderived_count\tn_haps\tselection_coeff"
           "\tis_te\twindow\tlocal_recomb")
    np.savetxt(args.out, np.column_stack([rows[c] for c in rows.dtype.names]),
               delimiter="\t", header=hdr, comments="",
               fmt=["%.1f", "%.4f", "%d", "%d", "%.6g", "%d", "%d", "%.6g"])
    te = rows[rows["is_te"] == 1]
    print(f"\nwrote {args.out}: {len(te)} TEs over "
          f"{total_bp/1e6:.1f} Mb in {len(args.trees)} window(s)")
    n_rec = int(np.isfinite(rows["local_recomb"]).sum())
    if n_rec == 0:
        print("  local_recomb is nan everywhere: pass --recomb auto (or explicit "
              "npz files) if you intend to fit the CONTEXT model", file=sys.stderr)
    else:
        print(f"  local_recomb present for {n_rec}/{len(rows)} sites")
    if len(te):
        q = np.percentile(te["age_gen"], [5, 25, 50, 75, 95])
        print(f"  age (gen ago) quantiles 5/25/50/75/95: "
              + "  ".join(f"{v:.0f}" for v in q))
        print(f"  frequency: median {np.median(te['derived_count'])/te['n_haps'][0]:.3f}")
        # the burst should show up as an age mode; a flat age distribution means
        # either no burst or one too old to have left survivors
        h, edges = np.histogram(te["age_gen"], bins=10)
        peak = int(np.argmax(h))
        print(f"  age histogram peak in [{edges[peak]:.0f}, {edges[peak+1]:.0f}] gen "
              f"({h[peak]}/{len(te)} TEs)")

    if args.vcf:
        base = Path(args.vcf)
        stem = base.with_suffix("")
        total = 0
        for k, tp in enumerate(args.trees):
            out = (base if len(args.trees) == 1
                   else Path(f"{stem}.window{k:02d}{base.suffix or '.vcf'}"))
            total += write_vcf(tp, out, contig_id=k + 1)
        print(f"wrote {len(args.trees)} VCF file(s), {total} sites total "
              f"(phased, diploid, one contig per window)")
        print("  next: GEVA needs a .bin/.marker conversion; Relate wants haps/sample;")
        print("  tsinfer+tsdate can consume the VCF directly and is the easiest to")
        print("  validate against, since it returns per-site age estimates.")


if __name__ == "__main__":
    main()
