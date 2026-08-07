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

    # and a phased VCF (TEs + SNPs) to hand to GEVA / Relate / tsinfer+tsdate
    python extract_te_ages.py --trees sim.trees --out ages.tsv --vcf sites.vcf

GETTING THE TREES
    The pipeline writes SLiM output to a scratch workdir and does not keep it.
    Set the debug dump dir when simulating (simulator_*.py takes debug_dump_dir,
    which writes sim_full_window_NN.trees per window), or re-run a handful of
    replicates with it enabled. You only need enough shards to validate.

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


def extract(trees_path, include_bg=False, max_sites=None, window_id=0):
    """Return (rows, sequence_length) for one window.

    `window_id` is carried through so downstream code can tell which of a
    simulation's windows each TE came from. Positions stay LOCAL to the window:
    the windows are non-contiguous genomic regions, so splicing them into one
    coordinate system would imply linkage between windows that does not exist,
    and would corrupt any recombination-based age estimator run on the VCF.
    """
    import tskit
    ts = tskit.load(str(trees_path))
    n_haps = int(ts.num_samples)

    # site -> (derived count) for the sample set
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
                     int(is_te), int(window_id)))
        if max_sites and len(rows) >= max_sites:
            break

    return (np.array(rows, dtype=[("position", "f8"), ("age_gen", "f8"),
                                  ("derived_count", "i4"), ("n_haps", "i4"),
                                  ("selection_coeff", "f8"), ("is_te", "i4"),
                                  ("window", "i4")]),
            float(ts.sequence_length))


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
    ap.add_argument("--vcf", default=None,
                    help="also write a phased VCF (first --trees file only) for "
                         "GEVA / Relate / tsinfer+tsdate")
    ap.add_argument("--include-background", action="store_true",
                    help="also report m2 background deleterious mutations, for "
                         "checking how much BGS load is present")
    ap.add_argument("--max-sites", type=int, default=None)
    args = ap.parse_args()

    all_rows, total_bp = [], 0.0
    for k, tp in enumerate(args.trees):
        r, seq_len = extract(tp, include_bg=args.include_background,
                             max_sites=args.max_sites, window_id=k)
        total_bp += seq_len                 # the TRUE span, not the last TE
        if len(r):
            all_rows.append(r)              # positions stay LOCAL to the window
        print(f"  [{k:>3}] {Path(tp).name}: {int((r['is_te'] == 1).sum())} "
              f"segregating TEs over {seq_len/1e6:.2f} Mb"
              + (f", {int((r['is_te'] == 0).sum())} background"
                 if args.include_background else ""))
    if not all_rows:
        sys.exit("no segregating TEs found; is this a SLiM tree sequence?")
    rows = np.concatenate(all_rows)

    # The total sequence length is written into the header so the fit does not
    # have to be told it separately (getting it wrong only shifts log_lambda,
    # but it shifts it silently).
    hdr = (f"# seq_length_bp={total_bp:.0f} n_windows={len(args.trees)}\n"
           "position\tage_gen\tderived_count\tn_haps\tselection_coeff"
           "\tis_te\twindow")
    np.savetxt(args.out, np.column_stack([rows[c] for c in rows.dtype.names]),
               delimiter="\t", header=hdr, comments="",
               fmt=["%.1f", "%.4f", "%d", "%d", "%.6g", "%d", "%d"])
    te = rows[rows["is_te"] == 1]
    print(f"\nwrote {args.out}: {len(te)} TEs over "
          f"{total_bp/1e6:.1f} Mb in {len(args.trees)} window(s)")
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
