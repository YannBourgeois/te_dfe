#!/usr/bin/env python3
"""
02_collate.py — Aggregate per-job .npz files into memory-mapped datasets.

The new (pass-3) sim files contain three keys:
  geno  : (n_windows, n_haps, n_sites) int8 categorical {0,1,2,3}
  cov   : (n_windows, n_cov)            float32
  theta : (n_params,)                   float32

We write three artifacts to disk:
  <out>.geno.npy    : (n_sims, n_windows, n_haps, n_sites) int8 memmap
  <out>.cov.npy     : (n_sims, n_windows, n_cov)            float32 memmap
  <out>.theta.pt    : torch tensor of thetas + metadata

This format is 4x smaller on disk than the previous one-hot float32 layout
because the genotype channel dimension lives at compute time, not storage.

Legacy single-key 'obs' files are auto-detected and routed through a
compatibility shim that writes the old <out>.obs.npy memmap instead.

Usage:
    python 02_collate.py --in-dir simulations/ --out results/dataset
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=str, default="simulations/")
    p.add_argument("--pattern", type=str, default="sim_*.npz",
                   help="Glob for shard files. Default sim_*.npz (unchanged). "
                        "Use e.g. '*_msprime.npz' or 'rep_*.npz' to collate shards "
                        "written under a different naming convention without renaming "
                        "them.")
    p.add_argument("--out",    type=str, default="results/dataset",
                   help="Output basename (no extension).")
    p.add_argument("--max-fail-frac", type=float, default=0.10,
                   help="Abort if more than this fraction of jobs failed.")
    p.add_argument("--fast-scan", dest="fast_scan", action="store_true",
                   default=True,
                   help="Pass 1 reads shapes from the FIRST file only instead of "
                        "opening all shards (default ON). For compressed npz, "
                        "reading d['feats'].shape DECOMPRESSES the whole array, so "
                        "the old full scan decompressed the entire dataset TWICE "
                        "(once to scan, once to write). Shards from one sim run "
                        "share Kmax/feature-count/theta-width, so first-file shapes "
                        "are correct; per-file shape validation still happens in "
                        "the write pass (mismatches are skipped there).")
    p.add_argument("--full-scan", dest="fast_scan", action="store_false",
                   help="Old behaviour: open every shard in pass 1 to pre-validate "
                        "shapes. Use only if shards may have heterogeneous shapes.")
    return p.parse_args()


def _detect_format(file_path):
    """Returns 'sumstat' (feats+te_mask+theta), 'newer' (geno+pos+cov+theta),
    'new' (geno+cov+theta), or 'legacy' (obs+theta)."""
    with np.load(file_path) as d:
        keys = set(d.files)
    if "feats" in keys and "te_mask" in keys:
        return "sumstat"
    if "geno" in keys and "pos" in keys and "cov" in keys:
        return "newer"
    if "geno" in keys and "cov" in keys:
        return "new"
    elif "obs" in keys:
        return "legacy"
    raise ValueError(f"Unrecognized .npz format in {file_path}: keys={keys}")


def main():
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.pattern))
    if not files:
        print(f"No files matching '{args.pattern}' in {in_dir}", file=sys.stderr)
        return 1
    print(f"Found {len(files)} simulation files in {in_dir}")

    # Detect format from the first file
    fmt = _detect_format(files[0])
    print(f"Detected format: {fmt}")

    # ---- Pass 1: scan shapes, count loadable files ----
    if args.fast_scan:
        # Read shapes from the FIRST file only. Avoids decompressing every shard
        # just to read .shape (the pass-1 bottleneck for compressed npz). The
        # write pass opens each file anyway and skips any with mismatched shapes,
        # so correctness is preserved; this only removes the redundant scan.
        with np.load(files[0]) as d:
            if fmt == "sumstat":
                feats_shape = d["feats"].shape
                mask_shape  = d["te_mask"].shape
            elif fmt == "newer":
                geno_shape = d["geno"].shape
                pos_shape  = d["pos"].shape
                cov_shape  = d["cov"].shape
            elif fmt == "new":
                geno_shape = d["geno"].shape
                cov_shape  = d["cov"].shape
            else:
                obs_shape = d["obs"].shape
            theta_shape = d["theta"].shape
        valid_files = list(files)
        n_loaded = len(valid_files)
        print(f"  Pass 1 (fast-scan): shapes from {files[0].name}, "
              f"{n_loaded} files queued (validated during write)")
        if fmt == "sumstat":
            return _collate_sumstat(out_base, valid_files, n_loaded,
                                    feats_shape, mask_shape, theta_shape)
        elif fmt == "newer":
            return _collate_newer(out_base, valid_files, n_loaded,
                                  geno_shape, pos_shape, cov_shape, theta_shape)
        elif fmt == "new":
            return _collate_new(out_base, valid_files, n_loaded,
                                geno_shape, cov_shape, theta_shape)
        else:
            return _collate_legacy(out_base, valid_files, n_loaded,
                                   obs_shape, theta_shape)

    # ---- Pass 1 (full scan): open every file to pre-validate shapes ----
    geno_shape = None
    pos_shape  = None
    cov_shape  = None
    obs_shape  = None
    feats_shape = None
    mask_shape  = None
    theta_shape = None
    valid_files = []
    n_failed    = 0

    for fp in files:
        try:
            with np.load(fp) as d:
                if fmt == "sumstat":
                    this_feats_shape = d["feats"].shape
                    this_mask_shape  = d["te_mask"].shape
                elif fmt == "newer":
                    this_geno_shape  = d["geno"].shape
                    this_pos_shape   = d["pos"].shape
                    this_cov_shape   = d["cov"].shape
                elif fmt == "new":
                    this_geno_shape  = d["geno"].shape
                    this_cov_shape   = d["cov"].shape
                else:
                    this_obs_shape   = d["obs"].shape
                this_theta_shape = d["theta"].shape

            if fmt == "sumstat":
                if feats_shape is None:
                    feats_shape, mask_shape = this_feats_shape, this_mask_shape
                    theta_shape = this_theta_shape
                elif (this_feats_shape != feats_shape or
                      this_mask_shape  != mask_shape  or
                      this_theta_shape != theta_shape):
                    print(f"  Skipping {fp.name}: shape mismatch", file=sys.stderr)
                    n_failed += 1
                    continue
            elif fmt == "newer":
                if geno_shape is None:
                    geno_shape, pos_shape, cov_shape = (
                        this_geno_shape, this_pos_shape, this_cov_shape,
                    )
                    theta_shape = this_theta_shape
                elif (this_geno_shape != geno_shape or
                      this_pos_shape  != pos_shape  or
                      this_cov_shape  != cov_shape  or
                      this_theta_shape != theta_shape):
                    print(f"  Skipping {fp.name}: shape mismatch", file=sys.stderr)
                    n_failed += 1
                    continue
            elif fmt == "new":
                if geno_shape is None:
                    geno_shape, cov_shape = this_geno_shape, this_cov_shape
                    theta_shape = this_theta_shape
                elif (this_geno_shape != geno_shape or
                      this_cov_shape  != cov_shape  or
                      this_theta_shape != theta_shape):
                    print(f"  Skipping {fp.name}: shape mismatch", file=sys.stderr)
                    n_failed += 1
                    continue
            else:
                if obs_shape is None:
                    obs_shape   = this_obs_shape
                    theta_shape = this_theta_shape
                elif (this_obs_shape != obs_shape or
                      this_theta_shape != theta_shape):
                    print(f"  Skipping {fp.name}: shape mismatch", file=sys.stderr)
                    n_failed += 1
                    continue
            valid_files.append(fp)
        except Exception as e:
            print(f"  Failed to inspect {fp.name}: {e}", file=sys.stderr)
            n_failed += 1

    n_total  = len(files)
    n_loaded = len(valid_files)
    fail_frac = n_failed / n_total if n_total else 0
    print(f"  Pass 1: {n_loaded}/{n_total} usable ({n_failed} skipped, "
          f"{fail_frac*100:.1f}%)")

    if fail_frac > args.max_fail_frac:
        print(f"FAIL: failure fraction {fail_frac:.2%} > "
              f"max_fail_frac {args.max_fail_frac:.2%}", file=sys.stderr)
        return 2

    if n_loaded == 0:
        print("FAIL: no usable simulation files", file=sys.stderr)
        return 3

    if fmt == "sumstat":
        return _collate_sumstat(out_base, valid_files, n_loaded,
                                feats_shape, mask_shape, theta_shape)
    elif fmt == "newer":
        return _collate_newer(out_base, valid_files, n_loaded,
                                geno_shape, pos_shape, cov_shape, theta_shape)
    elif fmt == "new":
        return _collate_new(out_base, valid_files, n_loaded,
                              geno_shape, cov_shape, theta_shape)
    else:
        return _collate_legacy(out_base, valid_files, n_loaded,
                                  obs_shape, theta_shape)


def _collate_sumstat(out_base, files, n_loaded,
                     feats_shape, mask_shape, theta_shape):
    """Stack per-sim (feats, te_mask, theta) into memmaps for the sumstat
    representation. feats: (N_TE_TRAIN_MAX, N_SUMSTATS_PER_TE) per sim."""
    feats_full = (n_loaded,) + feats_shape
    mask_full  = (n_loaded,) + mask_shape
    theta_full = (n_loaded,) + theta_shape

    feats_bytes = int(np.prod(feats_full)) * np.dtype(np.float32).itemsize
    print(f"  feats   : {feats_full} float32 ({feats_bytes/1e6:.1f} MB)")
    print(f"  te_mask : {mask_full} float32")
    print(f"  theta   : {theta_full} float32")

    feats_path = out_base.with_suffix(".feats.npy")
    mask_path  = out_base.with_suffix(".te_mask.npy")
    theta_path = out_base.with_suffix(".theta.pt")

    feats_mm = np.lib.format.open_memmap(
        feats_path, mode="w+", dtype=np.float32, shape=feats_full)
    mask_mm = np.lib.format.open_memmap(
        mask_path, mode="w+", dtype=np.float32, shape=mask_full)
    thetas = np.empty(theta_full, dtype=np.float32)
    seq_len = np.zeros((n_loaded,), dtype=np.float64)   # per-sim total seq length
    # ADDED: per-sim recomb-decile O/E histograms (summed over windows by the
    # simulator). Shape (n_sims, OE_N_BINS); zeros when the sims didn't emit them
    # (ENRICH_OE_DECILES off) -> theta_meta simply omits them below.
    _oe_bins = None
    oe_O = oe_E = None

    write_i = 0                                      # separate cursor: skips don't gap
    n_skipped = 0
    for fp in files:
        try:
            with np.load(fp) as d:
                _f, _m, _t = d["feats"], d["te_mask"], d["theta"]
                if (_f.shape != feats_shape or _m.shape != mask_shape
                        or _t.shape != theta_shape):
                    print(f"  Skipping {fp.name}: shape mismatch "
                          f"({_f.shape} vs {feats_shape})", file=sys.stderr)
                    n_skipped += 1
                    continue
                feats_mm[write_i] = _f
                mask_mm[write_i]  = _m
                thetas[write_i]   = _t
                if "seq_len_bp" in d.files:
                    seq_len[write_i] = float(d["seq_len_bp"])
                if "oe_O_hist" in d.files:
                    if oe_O is None:
                        _oe_bins = int(np.asarray(d["oe_O_hist"]).shape[0])
                        oe_O = np.zeros((n_loaded, _oe_bins), np.float64)
                        oe_E = np.zeros((n_loaded, _oe_bins), np.float64)
                    oe_O[write_i] = np.asarray(d["oe_O_hist"], np.float64)
                    oe_E[write_i] = np.asarray(d["oe_E_hist"], np.float64)
        except Exception as e:
            print(f"  Skipping {fp.name}: {e}", file=sys.stderr)
            n_skipped += 1
            continue
        write_i += 1
        if write_i % 500 == 0:
            print(f"    loaded {write_i}/{n_loaded}")

    # If any files were skipped, the last (n_loaded - write_i) rows are unwritten
    # (zeros). Record the true count so downstream can slice; the theta/seq/oe
    # arrays are trimmed exactly, and the memmaps are flagged via n_sims in meta.
    if write_i < n_loaded:
        print(f"  {n_skipped} file(s) skipped; dataset has {write_i} valid sims "
              f"(memmap rows {write_i}..{n_loaded-1} are unused padding)")
        thetas   = thetas[:write_i]
        seq_len  = seq_len[:write_i]
        if oe_O is not None:
            oe_O = oe_O[:write_i]; oe_E = oe_E[:write_i]
    n_valid = write_i

    feats_mm.flush(); del feats_mm
    mask_mm.flush();  del mask_mm

    _meta = {
        "thetas":       torch.from_numpy(thetas),
        "seq_len_bp":   torch.from_numpy(seq_len),   # for the seq-length covariate
        "feats_path":   str(feats_path.name),
        "te_mask_path": str(mask_path.name),
        "feats_shape":  (n_valid,) + tuple(feats_full[1:]),
        "te_mask_shape": (n_valid,) + tuple(mask_full[1:]),
        "format":       "sumstat",
        "n_sims":       n_valid,
    }
    if oe_O is not None:                              # ADDED: carry O/E histograms
        _meta["oe_O_hist"] = torch.from_numpy(oe_O)
        _meta["oe_E_hist"] = torch.from_numpy(oe_E)
    torch.save(_meta, theta_path)
    print(f"\n  Wrote sumstat dataset to {out_base}.*")
    print(f"  03_train.py: --dataset {out_base}  (REPRESENTATION=sumstat)")
    return 0


def _collate_newer(out_base, files, n_loaded,
                     geno_shape, pos_shape, cov_shape, theta_shape):
    geno_full  = (n_loaded,) + geno_shape
    pos_full   = (n_loaded,) + pos_shape
    cov_full   = (n_loaded,) + cov_shape
    theta_full = (n_loaded,) + theta_shape

    geno_bytes  = int(np.prod(geno_full))  * np.dtype(np.int8).itemsize
    pos_bytes   = int(np.prod(pos_full))   * np.dtype(np.float32).itemsize
    cov_bytes   = int(np.prod(cov_full))   * np.dtype(np.float32).itemsize
    theta_bytes = int(np.prod(theta_full)) * np.dtype(np.float32).itemsize
    print(f"  geno  : {geno_full} int8    ({geno_bytes/1e9:.2f} GB)")
    print(f"  pos   : {pos_full} float32 ({pos_bytes/1e9:.2f} GB)")
    print(f"  cov   : {cov_full} float32 ({cov_bytes/1e6:.1f} MB)")
    print(f"  theta : {theta_full} float32 ({theta_bytes/1e6:.1f} MB)")

    geno_path  = out_base.with_suffix(".geno.npy")
    pos_path   = out_base.with_suffix(".pos.npy")
    cov_path   = out_base.with_suffix(".cov.npy")
    theta_path = out_base.with_suffix(".theta.pt")

    print(f"  Allocating memmaps...")
    geno_mm = np.lib.format.open_memmap(
        geno_path, mode="w+", dtype=np.int8, shape=geno_full,
    )
    pos_mm = np.lib.format.open_memmap(
        pos_path, mode="w+", dtype=np.float32, shape=pos_full,
    )
    cov_mm = np.lib.format.open_memmap(
        cov_path, mode="w+", dtype=np.float32, shape=cov_full,
    )
    thetas = np.empty(theta_full, dtype=np.float32)

    for i, fp in enumerate(files):
        with np.load(fp) as d:
            geno_mm[i] = d["geno"]
            pos_mm[i]  = d["pos"]
            cov_mm[i]  = d["cov"]
            thetas[i]  = d["theta"]
        if (i + 1) % 500 == 0 or i == n_loaded - 1:
            print(f"    loaded {i+1}/{n_loaded}")

    geno_mm.flush(); del geno_mm
    pos_mm.flush();  del pos_mm
    cov_mm.flush();  del cov_mm

    torch.save({
        "thetas":         torch.from_numpy(thetas),
        "geno_path":      str(geno_path.name),
        "pos_path":       str(pos_path.name),
        "cov_path":       str(cov_path.name),
        "geno_shape":     tuple(geno_full),
        "pos_shape":      tuple(pos_full),
        "cov_shape":      tuple(cov_full),
        "format":         "newer",
        "n_sims":         n_loaded,
    }, theta_path)

    total_gb = (geno_bytes + pos_bytes + cov_bytes + theta_bytes) / 1e9
    print(f"\n  Wrote dataset to {out_base}.*  ({total_gb:.2f} GB total)")
    print(f"  03_train.py: --dataset {out_base}")
    return 0


def _collate_new(out_base, files, n_loaded, geno_shape, cov_shape, theta_shape):
    geno_full = (n_loaded,) + geno_shape
    cov_full  = (n_loaded,) + cov_shape
    theta_full = (n_loaded,) + theta_shape

    geno_bytes  = int(np.prod(geno_full))  * np.dtype(np.int8).itemsize
    cov_bytes   = int(np.prod(cov_full))   * np.dtype(np.float32).itemsize
    theta_bytes = int(np.prod(theta_full)) * np.dtype(np.float32).itemsize
    print(f"  geno  : {geno_full} int8     ({geno_bytes/1e9:.2f} GB on disk)")
    print(f"  cov   : {cov_full} float32  ({cov_bytes/1e6:.1f} MB on disk)")
    print(f"  theta : {theta_full} float32 ({theta_bytes/1e6:.1f} MB on disk)")

    geno_path  = out_base.with_suffix(".geno.npy")
    cov_path   = out_base.with_suffix(".cov.npy")
    theta_path = out_base.with_suffix(".theta.pt")

    print(f"  Allocating memmaps...")
    geno_mm = np.lib.format.open_memmap(
        geno_path, mode="w+", dtype=np.int8, shape=geno_full,
    )
    cov_mm = np.lib.format.open_memmap(
        cov_path, mode="w+", dtype=np.float32, shape=cov_full,
    )
    thetas = np.empty(theta_full, dtype=np.float32)

    for i, fp in enumerate(files):
        with np.load(fp) as d:
            geno_mm[i] = d["geno"]
            cov_mm[i]  = d["cov"]
            thetas[i]  = d["theta"]
        if (i + 1) % 500 == 0 or i == n_loaded - 1:
            print(f"    loaded {i+1}/{n_loaded}")

    geno_mm.flush(); del geno_mm
    cov_mm.flush();  del cov_mm

    torch.save({
        "thetas":         torch.from_numpy(thetas),
        "geno_path":      str(geno_path.name),
        "cov_path":       str(cov_path.name),
        "geno_shape":     tuple(geno_full),
        "cov_shape":      tuple(cov_full),
        "format":         "new",
        "n_sims":         n_loaded,
    }, theta_path)

    total_gb = (geno_bytes + cov_bytes + theta_bytes) / 1e9
    print(f"\n  Wrote dataset to {out_base}.*  ({total_gb:.2f} GB total)")
    print(f"  03_train.py: --dataset {out_base}")
    return 0


def _collate_legacy(out_base, files, n_loaded, obs_shape, theta_shape):
    """Legacy single-tensor format. Same as before."""
    obs_full   = (n_loaded,) + obs_shape
    theta_full = (n_loaded,) + theta_shape

    obs_bytes = int(np.prod(obs_full)) * np.dtype(np.float32).itemsize
    print(f"  obs (legacy) : {obs_full} float32 ({obs_bytes/1e9:.2f} GB)")

    obs_path   = out_base.with_suffix(".obs.npy")
    theta_path = out_base.with_suffix(".theta.pt")

    obs_mm = np.lib.format.open_memmap(
        obs_path, mode="w+", dtype=np.float32, shape=obs_full,
    )
    thetas = np.empty(theta_full, dtype=np.float32)

    for i, fp in enumerate(files):
        with np.load(fp) as d:
            obs_mm[i]  = d["obs"]
            thetas[i]  = d["theta"]
        if (i + 1) % 500 == 0 or i == n_loaded - 1:
            print(f"    loaded {i+1}/{n_loaded}")

    obs_mm.flush(); del obs_mm

    torch.save({
        "thetas":         torch.from_numpy(thetas),
        "obs_memmap_path": str(obs_path.name),
        "obs_shape":      tuple(obs_full),
        "format":         "legacy",
        "n_sims":         n_loaded,
    }, theta_path)

    print(f"  Wrote legacy dataset to {out_base}.*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
