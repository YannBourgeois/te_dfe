#!/usr/bin/env python3
"""
inspect_haplotype_age.py — For a single SLiM-produced tree sequence
(post-recap-and-overlay, what 01_simulate.py keeps in debug_dump),
compute per-mutation:
  - allele age (generations before present)
  - selection coefficient (from SLiM metadata)
  - local-tree extent in bp (proxy for haplotype length)
  - present-day allele frequency

Output: a CSV with one row per mutation, plus an ASCII summary table
binned by age. Useful for sanity-checking the relationship between
allele age, selection, and haplotype length in your specific simulation
setup — e.g. to choose a sensible FLANK_WINDOW_BP.

Usage:
    python inspect_haplotype_age.py path/to/sim.trees
    python inspect_haplotype_age.py path/to/sim.trees --csv age_diagnostic.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import tskit


def _is_slim_te(meta):
    """SLiM mutation has mutation_list with slim_time field."""
    if not meta or not isinstance(meta, dict):
        return False
    ml = meta.get("mutation_list")
    if not ml:
        return False
    return "slim_time" in ml[0]


def collect_per_mutation(ts):
    """
    Returns a list of dicts, one per variant. For variants with multiple
    mutations at the same site, we keep the first SLiM mutation's metadata
    and report the variant's overall age/frequency.
    """
    rows = []
    n_haps = ts.num_samples
    sim_end = ts.metadata.get("SLiM", {}).get("tick", None)

    for var in ts.variants():
        muts = list(var.site.mutations)
        if not muts:
            continue
        # Use the earliest mutation at this site
        earliest = min(muts, key=lambda m: m.time)

        # Mutation time in tskit convention: gens before "present" (sample time).
        # SLiM time runs forward; the .time attribute is gen_before_present.
        age_gen = float(earliest.time)

        # Frequency
        derived_count = int(var.genotypes.sum())
        freq = derived_count / n_haps

        # SLiM metadata — get selection coefficient if available
        meta = earliest.metadata
        is_te = _is_slim_te(meta)
        s_coeff = None
        if meta and isinstance(meta, dict):
            ml = meta.get("mutation_list")
            if ml and isinstance(ml, list) and len(ml) > 0:
                s_coeff = ml[0].get("selection_coeff", None)

        # Local tree extent: walk the tree at this variant's position
        tree = ts.at(var.site.position)
        ht_left, ht_right = tree.interval
        ht_length = ht_right - ht_left

        rows.append({
            "position": float(var.site.position),
            "age_gen": age_gen,
            "freq": freq,
            "is_te": int(is_te),
            "selection_coeff": s_coeff if s_coeff is not None else 0.0,
            "haplotype_extent_bp": float(ht_length),
            "tree_left": float(ht_left),
            "tree_right": float(ht_right),
        })
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("trees_path", help="Path to a .trees file")
    p.add_argument("--csv", default=None,
                   help="If provided, save per-mutation table here.")
    p.add_argument("--max-mutations", type=int, default=None,
                   help="Limit analysis to first N mutations (speed up).")
    return p.parse_args()


def main():
    args = parse_args()
    ts = tskit.load(args.trees_path)
    print(f"Loaded {args.trees_path}")
    print(f"  n_sites:     {ts.num_sites}")
    print(f"  n_mutations: {ts.num_mutations}")
    print(f"  n_samples:   {ts.num_samples}")
    print(f"  seq_length:  {ts.sequence_length:.0f} bp")
    print()

    rows = collect_per_mutation(ts)
    if args.max_mutations:
        rows = rows[:args.max_mutations]
    if not rows:
        print("No mutations to analyze.")
        return 1

    arr = np.array([
        (r["age_gen"], r["freq"], r["is_te"],
         r["selection_coeff"], r["haplotype_extent_bp"])
        for r in rows
    ])
    age      = arr[:, 0]
    freq     = arr[:, 1]
    is_te    = arr[:, 2].astype(bool)
    s_coeff  = arr[:, 3]
    ht_bp    = arr[:, 4]

    # =====================================
    # Headline summaries
    # =====================================
    print(f"Total mutations: {len(arr)} ({is_te.sum()} TE, {(~is_te).sum()} SNP)")
    print()

    def print_block(label, mask):
        n = int(mask.sum())
        if n == 0:
            print(f"  {label}: 0 mutations")
            return
        print(f"  {label}: {n} mutations")
        print(f"    age (gen):        median={np.median(age[mask]):.0f}  "
              f"5%={np.percentile(age[mask],5):.0f}  "
              f"95%={np.percentile(age[mask],95):.0f}")
        print(f"    freq:             median={np.median(freq[mask]):.3f}  "
              f"5%={np.percentile(freq[mask],5):.3f}  "
              f"95%={np.percentile(freq[mask],95):.3f}")
        if mask.sum() > 0 and np.any(s_coeff[mask] != 0):
            print(f"    selection_coeff:  median={np.median(s_coeff[mask]):.4f}  "
                  f"5%={np.percentile(s_coeff[mask],5):.4f}  "
                  f"95%={np.percentile(s_coeff[mask],95):.4f}")
        print(f"    hap_extent (bp):  median={np.median(ht_bp[mask]):.0f}  "
              f"5%={np.percentile(ht_bp[mask],5):.0f}  "
              f"95%={np.percentile(ht_bp[mask],95):.0f}")
        print()

    print("=" * 70)
    print("All mutations:")
    print_block("ALL", np.ones(len(arr), dtype=bool))
    print_block("TEs (SLiM)", is_te)
    print_block("SNPs (msprime overlay)", ~is_te)

    # =====================================
    # Binned by age
    # =====================================
    print("=" * 70)
    print("Haplotype extent binned by age:")
    print()
    if age.max() > 0:
        # Geometric bins from min to max age
        log_min = np.log10(max(age[age > 0].min(), 1))
        log_max = np.log10(age.max())
        bin_edges = np.logspace(log_min, log_max, 6)
        print(f"  {'age range (gen)':<30}{'n':>6}{'median hap extent (bp)':>30}")
        print("  " + "-" * 65)
        for i in range(len(bin_edges) - 1):
            lo, hi = bin_edges[i], bin_edges[i+1]
            mask = (age >= lo) & (age < hi)
            if mask.sum() > 0:
                med = np.median(ht_bp[mask])
                print(f"  [{lo:.0f}, {hi:.0f})".ljust(32)
                      + f"{int(mask.sum()):>6}"
                      + f"{med:>30.0f}")
    print()

    # =====================================
    # Theoretical expected haplotype length
    # =====================================
    print("=" * 70)
    print("Theoretical comparison:")
    print("  For age t and recomb rate r, expected haplotype length ~ 1/(2*r*t)")
    print("  (rough order-of-magnitude bound — local-tree extent is somewhat tighter)")
    # Assume the user has a roughly known recomb rate; if the ts metadata
    # tells us, use that, otherwise default
    r_guess = 1e-8  # placeholder; user can interpret
    print(f"  Assuming r = {r_guess:.1e}:")
    for t in [30, 100, 1_000, 10_000, 100_000]:
        L = 1.0 / (2.0 * r_guess * t)
        print(f"    age {t:>7d} gen -> expected ~{L:.0f} bp")
    print()

    # =====================================
    # CSV output
    # =====================================
    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as f:
            writer = _csv.writer(f)
            writer.writerow([
                "position", "age_gen", "freq", "is_te",
                "selection_coeff", "haplotype_extent_bp",
                "tree_left", "tree_right",
            ])
            for r in rows:
                writer.writerow([
                    r["position"], r["age_gen"], r["freq"], r["is_te"],
                    r["selection_coeff"], r["haplotype_extent_bp"],
                    r["tree_left"], r["tree_right"],
                ])
        print(f"Wrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    sys.exit(main())
