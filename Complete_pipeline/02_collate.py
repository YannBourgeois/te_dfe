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
    p.add_argument("--out",    type=str, default="results/dataset",
                   help="Output basename (no extension).")
    p.add_argument("--max-fail-frac", type=float, default=0.10,
                   help="Abort if more than this fraction of jobs failed.")
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

    files = sorted(in_dir.glob("sim_*.npz"))
    if not files:
        print(f"No sim_*.npz files found in {in_dir}", file=sys.stderr)
        return 1
    print(f"Found {len(files)} simulation files in {in_dir}")

    # Detect format from the first file
    fmt = _detect_format(files[0])
    print(f"Detected format: {fmt}")

    # ---- Pass 1: scan shapes, count loadable files ----
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

    for i, fp in enumerate(files):
        with np.load(fp) as d:
            feats_mm[i] = d["feats"]
            mask_mm[i]  = d["te_mask"]
            thetas[i]   = d["theta"]
            if "seq_len_bp" in d.files:
                seq_len[i] = float(d["seq_len_bp"])
        if (i + 1) % 500 == 0 or i == n_loaded - 1:
            print(f"    loaded {i+1}/{n_loaded}")

    feats_mm.flush(); del feats_mm
    mask_mm.flush();  del mask_mm

    torch.save({
        "thetas":       torch.from_numpy(thetas),
        "seq_len_bp":   torch.from_numpy(seq_len),   # for the seq-length covariate
        "feats_path":   str(feats_path.name),
        "te_mask_path": str(mask_path.name),
        "feats_shape":  tuple(feats_full),
        "te_mask_shape": tuple(mask_full),
        "format":       "sumstat",
        "n_sims":       n_loaded,
    }, theta_path)
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
