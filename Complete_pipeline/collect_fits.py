#!/usr/bin/env python
"""
collect_fits.py — rebuild the sweep summary by parsing every *.fit.txt report.

Replaces the per-task summary writer, which appends one row per array task and
so can lose rows whenever a task dies between the fit and the append (or two
tasks race on the file). This runs ONCE at the end, over whatever reports exist,
and is idempotent: rerun it any time and you get the current, complete table.

    python collect_fits.py dfe_aged_fits/ -o dfe_aged_fits/summary.tsv

Every report that fails to parse is listed at the end with the reason, so a
missing sim is always visible rather than silently absent.

WHAT IT READS
    the fit's own stdout, which already carries everything: the data line, the
    parameter table (estimate AND truth), -logL, the convergence line, the fixed
    parameters, the DFE composition, and any "pinned at bound" warnings (those go
    to stderr, but the sweep runs `2>&1 | tee`, so they land in the report too).

DERIVED COLUMNS
    scale = mean_s / shape and rate = shape / mean_s, for fit and truth. The
    age-partitioned SFS constrains the SCALE, not shape and mean_s separately,
    so scale_fit vs scale_true is the DFE panel that should actually track.
    Selection-class proportions are added on the same footing (--gamma-edges).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# Union of the SIMPLE and CONTEXT parameter sets: one collector serves both.
# Sorted longest-first so the regex alternation cannot match "gamma_r" inside
# "log_gamma_shape".
PARAMS_SIMPLE = ["log_lambda", "log_gamma_shape", "log_mean_s",
                 "log10_t_start", "log_duration", "log_multiplier"]
PARAMS_CONTEXT = ["log_lambda", "log_s_global", "alpha_r", "gamma_r",
                  "log10_t_start", "log_duration", "log_multiplier"]
PARAMS = sorted(set(PARAMS_SIMPLE) | set(PARAMS_CONTEXT), key=len, reverse=True)

# "  log_lambda            -20.4994     1.251e-09    -20.4056"
# "  log10_t_start           2.6405       437 gen      2.7437"   <- natural is 2 tokens
_PARAM_LINE = re.compile(r"^\s+(" + "|".join(PARAMS) + r")\s+(-?\d+\.\d+|nan)\s*(.*)$")


def _f(tok):
    try:
        v = float(tok)
    except (TypeError, ValueError):
        return None
    return None if v != v else v          # drop NaN


def parse_report(path):
    """Return (row_dict, error_or_None) for one .fit.txt."""
    try:
        txt = Path(path).read_text(errors="replace")
    except OSError as e:
        return None, f"unreadable: {e}"

    row = {"sim": Path(path).name.replace(".fit.txt", "")}

    # data line: "data       : 1483 TEs, 100.0 Mb, 7 age windows"
    m = re.search(r"^data\s*:\s*(\d+)\s+TEs,\s*([\d.]+)\s*Mb,\s*(\d+)\s+age windows",
                  txt, re.M)
    if m:
        row["n_te"], row["seq_mb"], row["n_age_windows"] = m.group(1), m.group(2), m.group(3)
    else:
        # context model: "data : 889 TEs, 100.0 Mb, 6 age x 4 recomb cells"
        m = re.search(r"^data\s*:\s*(\d+)\s+TEs,\s*([\d.]+)\s*Mb,\s*(\d+)\s+age\s*x\s*(\d+)\s+recomb",
                      txt, re.M)
        if m:
            row["n_te"], row["seq_mb"] = m.group(1), m.group(2)
            row["n_age_windows"], row["n_recomb_bins"] = m.group(3), m.group(4)
    row["model"] = "context" if "recomb bins (phi_r)" in txt or "n_recomb_bins" in row \
                   else "simple"

    # per-window counts: "   [     0,   25.0) gen ago :     262 TEs"
    win = re.findall(r"^\s*\[\s*[\d.]+,\s*\S+\)\s*gen ago\s*:\s*(\d+)\s+TEs", txt, re.M)
    if win:
        row["te_per_window"] = ";".join(win)
        row["min_te_in_window"] = min(int(w) for w in win)

    m = re.search(r"^demography\s*:\s*N_ref=([\d.]+)", txt, re.M)
    if m:
        row["n_ref"] = m.group(1)

    # optimiser diagnostics
    m = re.search(r"^\s*(\d+)/(\d+)\s+starts within 1 log-unit", txt, re.M)
    if m:
        row["n_near_best"], row["n_runs"] = m.group(1), m.group(2)
        row["converged"] = f"{m.group(1)}/{m.group(2)}"
    m = re.search(r"^\s+-logL\s+(-?[\d.]+)\s*$", txt, re.M)
    if m:
        row["neg_logL"] = m.group(1)

    m = re.search(r"^\s+fixed:\s*(.+)$", txt, re.M)
    row["fixed"] = m.group(1).strip() if m else ""
    m = re.search(r"^\s+DFE search coordinate:\s*(.+)$", txt, re.M)
    row["dfe_coord"] = m.group(1).strip() if m else ""

    # any parameter that ended on a search bound
    pinned = re.findall(r"^\s*!!\s+(\S+)\s+pinned at (LOWER|UPPER) bound", txt, re.M)
    row["pinned"] = ";".join(f"{p}:{d}" for p, d in pinned)

    # --- parameter table: estimate and (optional) truth -------------------
    # Restrict to the table itself. The standard-errors block repeats every
    # parameter name ("log10_t_start  2.41 +/- 0.04  [..]"), so parsing the whole
    # report lets those lines overwrite the real rows -- truth became "[".
    tbl = txt
    h = re.search(r"^\s*parameter\s+estimate\b.*$", txt, re.M)
    if h:
        tail = txt[h.end():]
        e = re.search(r"^\s+-logL\s", tail, re.M)
        tbl = tail[:e.start()] if e else tail
    # The SIMPLE report has columns  parameter | estimate | natural | truth,
    # the CONTEXT report has  parameter | estimate | truth  (no natural). Read
    # the header rather than assuming positions, or the context truth column is
    # taken from the wrong token.
    has_natural = bool(h and "natural" in h.group(0))
    got = 0
    for line in tbl.splitlines():
        m = _PARAM_LINE.match(line)
        if not m:
            continue
        name, est, rest = m.group(1), m.group(2), m.group(3).split()
        if has_natural:
            # natural column is 2 tokens only for log10_t_start ("437 gen")
            n_nat = 2 if name == "log10_t_start" else 1
        else:
            n_nat = 0
        truth = rest[n_nat] if len(rest) > n_nat else None
        row[f"fit_{name}"] = est
        row[f"true_{name}"] = truth if truth is not None else ""
        got += 1
    if got == 0:
        return None, "no parameter table (fit did not finish)"
    if got < len(PARAMS):
        row["parse_warning"] = f"only {got}/{len(PARAMS)} parameters found"

    # --- uncertainties, if the fit was run with --uncerts[-bootstrap] ------
    # "  standard errors (Godambe; fit coordinates, 95% CI):"
    # "    log_gamma_shape      -0.3825 +/- 0.0912   [  -0.5613,  -0.2037]"
    m = re.search(r"standard errors \(([\w-]+)", txt)
    if m:
        row["se_method"] = m.group(1)
        for mm in re.finditer(
                r"^\s{2,}(\w+)\s+(-?\d+\.\d+)\s*\+/-\s*(\d+\.\d+)\s*"
                r"\[\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\s*\]", txt, re.M):
            nm = mm.group(1)
            row[f"se_{nm}"] = mm.group(3)
            row[f"ci_lo_{nm}"] = mm.group(4)
            row[f"ci_hi_{nm}"] = mm.group(5)
    m = re.search(r"^\s+(\d+) bootstrap replicate", txt, re.M)
    if m:
        row["n_bootstrap"] = m.group(1)
    row["hessian_not_pd"] = int("Hessian is not positive definite" in txt)

    # --- context: implied selection / insertion per recombination bin ------
    # "    bin 0 (phi_r=-1.20): s_local=3.1e-03  2Ne*s=3.1  insertion x0.55"
    for mm in re.finditer(r"^\s+bin (\d+) \(phi_r=([-+\d.]+)\):\s*s_local=(\S+)\s+"
                          r"2Ne\*s=(\S+)\s+insertion x(\S+)", txt, re.M):
        b = mm.group(1)
        row[f"phi_r_bin{b}"] = mm.group(2)
        row[f"s_local_bin{b}"] = mm.group(3)
        row[f"gamma_bin{b}"] = mm.group(4)
        row[f"insertion_bin{b}"] = mm.group(5)

    # --- DFE composition, as printed --------------------------------------
    for label, key in [("effectively neutral", "p_neutral"),
                       ("weakly deleterious", "p_weak"),
                       ("moderately deleterious", "p_moderate"),
                       ("strongly deleterious", "p_strong"),
                       ("below drift barrier", "p_below_drift")]:
        m = re.search(rf"^\s+{label}\s+([\d.]+)%", txt, re.M)
        if m:
            row[f"report_{key}"] = float(m.group(1)) / 100.0
    return row, None


def add_derived(row, gamma_edges, ne, gamma_scale):
    """scale/rate plus selection-class proportions, for fit and truth."""
    try:
        from scipy.special import gammainc
    except ImportError:
        gammainc = None
    import math
    for which in ("fit", "true"):
        sh = _f(row.get(f"{which}_log_gamma_shape"))
        ms = _f(row.get(f"{which}_log_mean_s"))
        if sh is None or ms is None:
            continue
        shape, mean_s = math.exp(sh), math.exp(ms)
        scale = mean_s / shape
        row[f"scale_{which}"] = f"{scale:.6e}"
        row[f"rate_{which}"] = f"{1.0/scale:.6e}"
        if gammainc is None:
            continue
        # class edges given on gamma = gamma_scale * Ne * s  ->  s edges
        s_ed = [0.0] + [g / (gamma_scale * ne) for g in gamma_edges]
        cdf = [gammainc(shape, e / scale) for e in s_ed] + [1.0]
        names = (["p_neutral", "p_weak", "p_moderate", "p_strong"]
                 if len(gamma_edges) == 3
                 else [f"p_class{i}" for i in range(len(gamma_edges) + 1)])
        for nm, lo, hi in zip(names, cdf[:-1], cdf[1:]):
            row[f"{nm}_{which}"] = f"{max(hi - lo, 0.0):.6f}"
    return row


def _props_on_s(log_shape, log_mean_s, s_edges):
    """Gamma DFE mass in each s class. s_edges like [0, 1e-3, 1e-2, inf]."""
    from scipy.special import gammainc
    import math
    shape = math.exp(log_shape); scale = math.exp(log_mean_s) / shape
    cdf = [0.0 if e <= 0 else (1.0 if math.isinf(e) else gammainc(shape, e / scale))
           for e in s_edges]
    return [cdf[i + 1] - cdf[i] for i in range(len(cdf) - 1)]


def add_boot_proportions(row, report_path, s_edges, class_names):
    """Percentile CIs for the class proportions, from the per-replicate
    (shape, mean_s) draws the fit wrote to <sim>.boot.tsv. This is the correct
    interval: proportions are non-monotone in mean_s and correlated with shape,
    so an endpoint propagation misbehaves; percentiles of props() over the joint
    draws do not. Also writes point proportions on the SAME edges, so the CI and
    the point match. No-op if the draws file is absent."""
    import numpy as np
    bpath = Path(str(report_path).replace(".fit.txt", ".boot.tsv"))
    if not bpath.exists():
        return row
    try:
        d = np.genfromtxt(bpath, names=True, delimiter="\t")
        ls = np.atleast_1d(d["log_gamma_shape"]); lm = np.atleast_1d(d["log_mean_s"])
    except Exception:
        return row
    if ls.size < 2:
        return row
    P = np.array([_props_on_s(a, b, s_edges) for a, b in zip(ls, lm)])  # (N, K)
    lo = np.percentile(P, 2.5, axis=0); hi = np.percentile(P, 97.5, axis=0)
    for k, nm in enumerate(class_names):
        row[f"ci_lo_{nm}"] = f"{lo[k]:.6f}"
        row[f"ci_hi_{nm}"] = f"{hi[k]:.6f}"
    # point proportions on the same edges, from the fit/true point estimates
    for which in ("fit", "true"):
        sh = _f(row.get(f"{which}_log_gamma_shape"))
        ms = _f(row.get(f"{which}_log_mean_s"))
        if sh is not None and ms is not None:
            pr = _props_on_s(sh, ms, s_edges)
            for k, nm in enumerate(class_names):
                row[f"{nm}_{which}"] = f"{pr[k]:.6f}"
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("indir", nargs="?", default="dfe_aged_fits",
                    help="directory holding the *.fit.txt reports")
    ap.add_argument("-o", "--out", default=None,
                    help="output table; .csv writes comma-separated, otherwise TSV. "
                         "Default: <indir>/summary_parsed.tsv")
    ap.add_argument("--glob", default="*.fit.txt")
    ap.add_argument("--ne", type=float, default=500.0,
                    help="Ne used for the derived selection-class edges")
    ap.add_argument("--gamma-scale", type=float, default=4.0)
    ap.add_argument("--gamma-edges", default="1,10,100",
                    help="class edges on gamma = gamma_scale*Ne*s")
    ap.add_argument("--s-class-edges", default="1e-3,1e-2",
                    help="s edges for the p_nn/p_weak/p_strong proportions and "
                         "their bootstrap CIs. MUST match the edges the R plot "
                         "uses (default 1e-3, 1e-2).")
    args = ap.parse_args()

    s_edges = [0.0] + [float(x) for x in args.s_class_edges.split(",")] + [float("inf")]
    s_classes = ["p_nn", "p_weak", "p_strong"] if len(s_edges) == 4 else \
                [f"p_sclass{i}" for i in range(len(s_edges) - 1)]

    indir = Path(args.indir)
    if not indir.is_dir():
        sys.exit(f"not a directory: {indir}")
    reports = sorted(indir.glob(args.glob))
    if not reports:
        sys.exit(f"no files matching {args.glob} in {indir}")

    gamma_edges = [float(x) for x in args.gamma_edges.split(",")]
    rows, failed = [], []
    for p in reports:
        row, err = parse_report(p)
        if err:
            failed.append((p.name, err))
            continue
        row = add_derived(row, gamma_edges, args.ne, args.gamma_scale)
        row = add_boot_proportions(row, p, s_edges, s_classes)
        rows.append(row)

    if not rows:
        sys.exit(f"parsed 0 of {len(reports)} reports; nothing written")

    # union of keys, in a stable order
    lead = ["sim", "n_te", "seq_mb", "n_age_windows", "min_te_in_window", "n_ref",
            "neg_logL", "converged", "n_near_best", "n_runs", "dfe_coord",
            "fixed", "pinned", "parse_warning"]
    cols = [c for c in lead if any(c in r for r in rows)]
    cols += [f"fit_{p}" for p in PARAMS] + [f"true_{p}" for p in PARAMS]
    rest = sorted({k for r in rows for k in r} - set(cols))
    cols += rest

    out = Path(args.out) if args.out else indir / "summary_parsed.tsv"
    delim = "," if out.suffix.lower() == ".csv" else "\t"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter=delim,
                           restval="NA", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("NA" if r.get(k, "") == "" else r.get(k, "NA"))
                        for k in cols})

    print(f"parsed {len(rows)}/{len(reports)} reports -> {out}")
    if failed:
        print(f"\n{len(failed)} report(s) could not be parsed:")
        for nm, err in failed:
            print(f"  {nm}: {err}")
    n_pin = sum(1 for r in rows if r.get("pinned"))
    if n_pin:
        print(f"\n{n_pin}/{len(rows)} fit(s) ended on a search bound "
              f"(see the 'pinned' column)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
