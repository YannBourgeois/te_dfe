#!/usr/bin/env python
"""
inspect_network_arch.py — recover a trained TESumstatNet's architecture from its
saved weights, including the knobs that are NOT recorded in training_meta.pkl.

training_meta.pkl stores only aggregator_kind, n_attention_heads,
use_embedding_norm and representation. Everything else (per_te_dim, the recomb
trunk features, the pooling sketch, pool_proj_dim) lives in network.py's
constructor defaults, so editing that file silently changes what a checkpoint
means. This script reads it back out of the tensor shapes instead.

Directly recoverable from shapes:
    per_te_dim, n_features   <- encoder.0.weight       (per_te_dim, n_features)
    output_dim, trunk_in     <- trunk.0.weight         (output_dim, trunk_in)
    aggregator_kind          <- 'score' / 'gate' / neither in the keys
    n_attention_heads        <- aggregator.score.2.weight  (n_heads, embed_dim)
    use_embedding_norm       <- presence of enc_norm / context_norm
    pool_proj_dim            <- pool_proj.weight       (pool_proj_dim, per_te_dim)
    n_pool_stats             <- residual // pooled_dim  (see below)

Partly ambiguous: trunk_in is a SUM, so the small additive terms cannot be
separated from each other by shape alone:

    trunk_in = agg_out + n_global_cov
               + len(recomb_tail_quantiles)
               + (3 if recomb_bin_edges else 0)
               + len(corr_feature_pairs)
               + pooled_dim * n_pool_stats

The pooling term is a multiple of pooled_dim (>= 64), while the others are small
integers, so residual // pooled_dim and residual % pooled_dim separate the
pooling path exactly. The remainder is then enumerated over plausible
combinations. pool_mean is also indistinguishable from one more quantile, since
both add 1 to n_pool_stats.

Usage
    python inspect_network_arch.py <run-dir> [<run-dir> ...]
    python inspect_network_arch.py mfnpe/*/ results*/
"""
import sys
from pathlib import Path


def load_state(run_dir):
    import torch
    run = Path(run_dir)
    for name in ("network.pt", "core_network.pt"):
        p = run / name
        if p.exists():
            sd = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            return {k: tuple(v.shape) for k, v in sd.items()
                    if hasattr(v, "shape")}, name
    return None, None


def load_meta(run_dir):
    import pickle
    p = Path(run_dir) / "training_meta.pkl"
    if not p.exists():
        return {}
    try:
        with open(p, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return {}


def enumerate_small(small, max_q=6, max_pairs=6, max_gcov=3):
    """Combinations of (tail_quantiles, bin_edges, corr_pairs, n_global_cov)
    summing to `small`. bin_edges contributes 0 or 3."""
    out = []
    for nq in range(max_q + 1):
        for nb in (0, 3):
            for npair in range(max_pairs + 1):
                for gc in range(max_gcov + 1):
                    if nq + nb + npair + gc == small:
                        out.append((nq, nb // 3, npair, gc))
    return out


def report(run_dir):
    shapes, src = load_state(run_dir)
    print("=" * 74)
    print(f"{run_dir}")
    if shapes is None:
        print("  no network.pt / core_network.pt found")
        return
    meta = load_meta(run_dir)

    # --- direct reads -------------------------------------------------------
    enc = shapes.get("encoder.0.weight")
    trunk0 = shapes.get("trunk.0.weight")
    if enc is None or trunk0 is None:
        print(f"  {src}: not a TESumstatNet (no encoder.0/trunk.0). "
              f"keys: {sorted(shapes)[:6]} ...")
        return
    per_te_dim, n_features = enc
    output_dim, trunk_in = trunk0

    keys = list(shapes)
    if any("aggregator.score" in k for k in keys):
        kind = "attention"
        heads = shapes.get("aggregator.score.2.weight", (None,))[0]
    elif any("aggregator.gate" in k for k in keys):
        kind, heads = "sigmoid_attn", 1
    else:
        kind, heads = "deepsets", None

    emb_norm = any(k.startswith("enc_norm") or k.startswith("context_norm")
                   for k in keys)
    pool_proj = shapes.get("pool_proj.weight")
    pool_proj_dim = pool_proj[0] if pool_proj else None
    pooled_dim = pool_proj_dim or per_te_dim

    # agg_out: embed_dim for both attention variants; deepsets may be 3x
    agg_out = per_te_dim
    note = ""
    if kind == "deepsets":
        note = "  (deepsets agg_out may be 3*embed_dim; check if residual is odd)"

    residual = trunk_in - agg_out
    n_pool_stats = residual // pooled_dim if residual > 0 else 0
    small = residual - n_pool_stats * pooled_dim

    print(f"  source                : {src}")
    print(f"  n_features            : {n_features}")
    print(f"  per_te_dim            : {per_te_dim}")
    print(f"  output_dim            : {output_dim}")
    print(f"  aggregator_kind       : {kind}"
          + (f"  heads={heads}" if heads else "") + note)
    print(f"  use_embedding_norm    : {emb_norm}")
    print(f"  pool_proj_dim         : {pool_proj_dim}"
          f"{'  (no pre-pool projection)' if pool_proj_dim is None else ''}")
    print(f"  trunk_in              : {trunk_in}"
          f"   = {agg_out} (agg) + {n_pool_stats}x{pooled_dim} (pool) + {small}")
    print(f"  POOLING SKETCH        : {n_pool_stats} statistic(s)"
          f" x {pooled_dim} dims = {n_pool_stats * pooled_dim}")
    if n_pool_stats == 0:
        print(f"      -> pool_quantiles=() and pool_mean=False (pooling path OFF)")
    else:
        print(f"      -> len(pool_quantiles) + (1 if pool_mean) = {n_pool_stats}")
        print(f"         (pool_mean is indistinguishable from one more quantile)")

    print(f"  SMALL TRUNK EXTRAS    : {small}"
          f"   = n_global_cov + tail_quantiles + 3*bins + corr_pairs")
    if small == 0:
        print(f"      -> all recomb/corr trunk features OFF, n_global_cov=0")
    else:
        combos = enumerate_small(small)
        print(f"      -> {len(combos)} consistent combination(s); "
              f"(n_tail, bins_on, n_corr, n_gcov):")
        for c in combos[:8]:
            flag = "   <- the old network.py default" if c == (1, 1, 1, 0) else ""
            print(f"         {c}{flag}")
        if len(combos) > 8:
            print(f"         ... {len(combos)-8} more")

    # ---- training_meta.pkl: what IS recorded ---------------------------
    if meta:
        print(f"\n  -- training_meta.pkl --")
        # heads and the variable-K bounds are recorded, so no inference needed
        for k in ("aggregator_kind", "n_attention_heads", "use_embedding_norm",
                  "representation", "model_name", "variable_k", "k_min", "k_max",
                  "n_train_windows", "include_mask", "sumstat_seqlen_cov"):
            if k in meta:
                print(f"    {k:22s}: {meta[k]}")
        if meta.get("infer_names"):
            print(f"    {'infer_names':22s}: {len(meta['infer_names'])} of "
                  f"{len(meta.get('param_names_full', []))} -> "
                  f"{meta['infer_names']}")
        # anything else the file happens to carry (forward-compatible)
        shown = {"aggregator_kind", "n_attention_heads", "use_embedding_norm",
                 "representation", "model_name", "variable_k", "k_min", "k_max",
                 "n_train_windows", "include_mask", "sumstat_seqlen_cov",
                 "infer_names", "param_names_full", "infer_indices", "fix_params"}
        for k in sorted(set(meta) - shown):
            print(f"    {k:22s}: {meta[k]}")
        if meta.get("aggregator_kind") not in (None, kind):
            print(f"    !! META DISAGREES WITH WEIGHTS "
                  f"({meta.get('aggregator_kind')} vs {kind}); trust the weights")
    else:
        print(f"\n  -- training_meta.pkl: absent or unreadable --")

    # ---- learning rate: NOT saved by 03_train.py -----------------------
    if "lr" in meta:
        print(f"\n  learning rate         : {meta['lr']}")
    else:
        print(f"\n  learning rate         : NOT RECORDED. 03_train.py saves neither")
        print(f"      args.lr nor the optimizer state, and never prints it, so it")
        print(f"      cannot be recovered from this directory. Look instead in:")
        print(f"        - logs/*.out for a '[mfnpe] ... lr=' line (MF-NPE runs only,")
        print(f"          echoed by mfnpe_train.sh, not by 03_train.py)")
        print(f"        - the --lr value hardcoded in the slurm script used")
        print(f"      Fix: add \"lr\": args.lr to the training_meta.pkl dump.")

    print(f"\n  rebuild with:")
    print(f"    TESumstatNet(n_features={n_features}, per_te_dim={per_te_dim},")
    print(f"                 output_dim={output_dim}, aggregator_kind='{kind}',")
    print(f"                 n_attention_heads={heads or 1}, "
          f"use_embedding_norm={emb_norm},")
    print(f"                 pool_proj_dim={pool_proj_dim},")
    print(f"                 pool_quantiles=<{n_pool_stats} stat(s)>, "
          f"...)   # + the {small}-dim extras above")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    for d in sys.argv[1:]:
        report(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
