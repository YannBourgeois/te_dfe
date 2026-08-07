#!/usr/bin/env python
"""Decode a single msprime (or SLiM sumstat) shard: parameters + feature summary.

    TE_DFE_MODEL=context python inspect_msprime_shard.py sim_shards_msprime_sproxy/rep_000000_msprime.npz
"""
import sys, numpy as np

def main(path):
    d = np.load(path, allow_pickle=True)
    print(f"=== {path} ===")
    print("keys:", list(d.files))
    fmt = str(d["format"]) if "format" in d.files else "?"
    print(f"format={fmt}  master_seed={d['master_seed'] if 'master_seed' in d.files else '?'}"
          f"  replicate_id={d['replicate_id'] if 'replicate_id' in d.files else '?'}")

    # --- parameters ---
    theta = np.asarray(d["theta"], float).reshape(-1)
    try:
        from te_dfe.config import PARAM_NAMES
        names = list(PARAM_NAMES)
    except Exception:
        names = [f"theta[{i}]" for i in range(len(theta))]
    print(f"\n-- theta (width {len(theta)}) --")
    for n, v in zip(names, theta):
        extra = ""
        if n == "log_lambda":     extra = f"   lambda = {np.exp(v):.3e} /bp"
        if n == "log_s_global":   extra = f"   s_global = {np.exp(v):.3e}"
        if n == "log_gamma_shape":extra = f"   shape = {np.exp(v):.3f}"
        if n == "log_mean_s":     extra = f"   mean_s = {np.exp(v):.3e}"
        if n.startswith("log10_t"):extra = f"   t = {10**v:.0f} gen"
        if n == "log_duration":   extra = f"   duration = {np.exp(v):.1f} gen"
        if n == "log_multiplier": extra = f"   multiplier = {np.exp(v):.1f}x"
        print(f"  {n:16s} = {v:+8.4f}{extra}")

    # --- features ---
    feats = np.asarray(d["feats"], float)
    if "te_mask" in d.files:
        m = np.asarray(d["te_mask"]).astype(bool)
        valid = feats[m]
    else:
        valid = feats[(feats != 0).any(axis=1)]
    print(f"\n-- features --  n_te={len(valid)}  shape={feats.shape}")
    try:
        from te_dfe.sumstats import FEATURE_NAMES
        fn = list(FEATURE_NAMES)
    except Exception:
        fn = [f"f{i}" for i in range(feats.shape[1])]
    if len(valid):
        print(f"  {'feature':<22}{'mean':>10}{'sd':>10}{'min':>10}{'max':>10}")
        for i, name in enumerate(fn[:feats.shape[1]]):
            col = valid[:, i]
            print(f"  {name:<22}{col.mean():>10.3f}{col.std():>10.3f}"
                  f"{col.min():>10.3f}{col.max():>10.3f}")
        # TE frequency spectrum quick look (feature 0 is te_freq)
        tf = valid[:, 0]
        print(f"\n  te_freq: median={np.median(tf):.3f}  "
              f"singletons(<0.05)={np.mean(tf<0.05):.0%}  "
              f"common(>0.5)={np.mean(tf>0.5):.0%}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "sim_shards_msprime_sproxy/rep_000000_msprime.npz")
