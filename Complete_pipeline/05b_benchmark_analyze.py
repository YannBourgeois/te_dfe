#!/usr/bin/env python3
"""
05b_benchmark_analyze.py — Analyze pre-simulated test sims with a trained
posterior. Produces a benchmark summary, recovery + SBC plots, and CSVs.

Inputs:
  results/posterior.pkl       trained DirectPosterior (from 03_train.py)
  results/training_meta.pkl   aggregator kind + include_mask flag
  test_simulations/test_*.npz pre-simulated test cases (from 05a)

Test .npz formats (auto-detected):
  newer:  keys = geno, pos, cov, theta
  new:    keys = geno, cov, theta
  legacy: keys = obs, theta

Outputs (in --out dir, suffixed _K<N> when --k-target is set):
  benchmark_summary.txt   per-parameter RMSE / bias / coverage / SBC p-value
  recovery_values.csv     one row per (sim, parameter)
  sim_summary.csv         one row per sim with covariate/padding aggregates
  recovery.pdf, sbc.pdf   diagnostic plots

K-variable models (include_mask=True) require a mask column appended to each
observation. This is read from training_meta.pkl and handled automatically.
Use --k-target N to evaluate with only the first N windows (others masked),
which probes how posterior quality scales with the number of windows.

Usage:
    python 05b_benchmark_analyze.py \
        --results-dir results/ \
        --test-dir    test_simulations/ \
        --out         results/benchmark/ \
        --n-post      500 \
        [--k-target N] [--max-tests M] [--no-attention-diag] [--no-reject-prior]
"""
import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
# Model selection before any te_dfe import (config dispatches at import time).
try:
    from model_select import prescan_model, add_model_arg
    prescan_model()
except Exception:
    add_model_arg = None


from te_dfe.config import (
    PARAM_NAMES, N_PARAMS,
    N_TE_SITES, N_TE_FLANK_SITES,
    N_HAPS, N_SITES, N_COVARIATES, N_STATE_CHANNELS, N_TRAIN_WINDOWS,
)

# Continuous-channel count: support both current and older config names.
try:
    from te_dfe.config import N_CONTINUOUS_CHANNELS as N_CONT
except ImportError:
    try:
        from te_dfe.config import N_POSITION_CHANNELS as N_CONT
    except ImportError:
        N_CONT = 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=str, default="results/")
    p.add_argument("--test-dir",    type=str, default="test_simulations/")
    p.add_argument("--out",         type=str, default="results/benchmark/")
    p.add_argument("--n-post",      type=int, default=500,
                   help="Posterior samples per test sim.")
    p.add_argument("--max-tests",   type=int, default=None,
                   help="Cap the number of test sims analyzed.")
    p.add_argument("--k-target",    type=int, default=None,
                   help="Evaluate using only the first K windows. With "
                        "--extrapolate, K may exceed the training window count.")
    p.add_argument("--n-te-target", type=int, default=None,
                   help="(sumstat) Subsample the pooled TE-set to N random valid "
                        "TEs before inference (TE-count sweep). Default: all.")
    p.add_argument("--n-te-seed",   type=int, default=0,
                   help="RNG seed for --n-te-target subsampling.")
    p.add_argument("--window-chunk", type=int, default=20,
                   help="With --extrapolate, encode windows in batches of this "
                        "many to bound GPU/CPU memory (whole-genome K in one "
                        "shot OOMs). Lower if you hit memory limits.")
    p.add_argument("--extrapolate", action="store_true",
                   help="K-EXTRAPOLATION: deploy a model trained at K<=K_train "
                        "on observations with MORE windows (e.g. whole-genome "
                        "200-window scenarios). Bypasses sbi's x-shape check by "
                        "embedding with the K-agnostic core and sampling the "
                        "flow on that embedding. Use for the variable-K + "
                        "sigmoid_attn deployment test.")
    p.add_argument("--zero-regions", type=str, default=None,
                   help="ABLATION: comma-separated genotype regions to zero "
                        "(out of te,flank,random). E.g. 'random' keeps TEs and "
                        "TE-flanking SNPs but removes background SNPs; "
                        "'flank,random' keeps only the TE anchors. Requires "
                        "--extrapolate (operates on the structured path).")
    p.add_argument("--zero-states", type=str, default=None,
                   help="ABLATION: comma-separated cell states to zero "
                        "(out of snp,te), genome-wide. 'snp' keeps TE markers "
                        "but removes SNP polymorphism. Requires --extrapolate.")
    p.add_argument("--zero-channels", type=str, default=None,
                   help="ABLATION: comma-separated continuous-channel indices "
                        "to zero (e.g. '5' for log-recomb, '2,3' for density). "
                        "Requires --extrapolate.")
    p.add_argument("--no-reject-prior", action="store_true",
                   help="Disable rejection of posterior samples outside the "
                        "prior box. Default is REJECTION ON: a well-trained flow "
                        "puts ~all mass in-box, so this enforces prior support "
                        "for clean recovery/SBC. Turn OFF only if a poorly-"
                        "trained flow leaks mass out-of-box and rejection "
                        "acceptance collapses (the ~0%%-acceptance hang).")
    p.add_argument("--max-sampling-time", type=float, default=60.0,
                   help="Per-test wall-clock cap (seconds) on rejection "
                        "sampling, so a pathological cell with near-zero "
                        "acceptance fails fast instead of hanging the whole "
                        "benchmark. Only applies when rejection is on.")
    p.add_argument("--no-attention-diag", action="store_true",
                   help="Skip the optional per-window attention diagnostic "
                        "(meaningless for deepsets; useful for attention models).")
    p.add_argument("--ne-harmonic", type=float, default=None,
                   help="Harmonic-mean Ne for the |2*Ne*s| DFE-bin thresholds. "
                        "Defaults to training_meta['ne_harmonic'] or 500.")
    p.add_argument("--no-bins", action="store_true",
                   help="Skip the derived DFE-bin (p_nn/p_w/p_strong) "
                        "proportions. Only relevant for the pure-gamma DFE model.")
    p.add_argument("--save-samples", action="store_true",
                   help="Dump raw posterior samples per test sim to "
                        "posterior_samples.npz (theta + derived bins + true "
                        "values), enabling joint-posterior / corner plots and "
                        "arbitrary post-hoc CI computation.")
    p.add_argument("--ci-levels", type=str, default="50,80,90",
                   help="Comma-separated central CI levels (percent) to report "
                        "per parameter, e.g. '50,80,90,95'. Reveals non-Gaussian "
                        "shape: if the 50%% CI is much tighter than 90%%/total "
                        "spread, the posterior has long tails.")
    # Register --model so argparse accepts it (prescan_model already
    # consumed its VALUE into TE_DFE_MODEL; this makes the flag legal).
    if add_model_arg is not None:
        add_model_arg(p)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------
def load_training_meta(results_dir: Path) -> dict:
    """Read training_meta.pkl. Returns a dict with sensible legacy defaults."""
    meta_path = results_dir / "training_meta.pkl"
    defaults = {
        "aggregator_kind":    "attention",
        "n_attention_heads":  1,
        "include_mask":       False,
        "use_embedding_norm": False,
    }
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        defaults.update({k: meta.get(k, v) for k, v in defaults.items()})
    return defaults


# ---------------------------------------------------------------------------
# Observation construction
# ---------------------------------------------------------------------------
def _n_data_windows(d):
    """Number of windows present in a test npz (axis 0 of geno/obs). For the
    sumstat representation a scenario is ONE pooled TE-set (not per-window), so
    its 'window count' for the K-machinery is 1."""
    if "feats" in d.files:
        return 1
    if "geno" in d.files:
        return int(np.asarray(d["geno"]).shape[0])
    if "obs" in d.files:
        arr = np.asarray(d["obs"])
        return int(arr.shape[0]) if arr.ndim >= 2 else 1
    return 0


def ablate_observation(geno_int, cont, zero_regions=None, zero_states=None,
                       zero_channels=None):
    """
    Ablate parts of the structured observation in place (clones first) to
    measure each input component's contribution to inference.

    geno_int : (1, W, H, S) integer state codes
    cont     : (1, W, C, S) continuous channels, or None

    zero_regions : iterable of {"te","flank","random"} — zero those SITE
                   COLUMN ranges of the genotype matrix (set to ANCESTRAL=0).
                   Region boundaries come from config (N_TE_SITES,
                   N_TE_FLANK_SITES). "Keep TEs but not random SNPs" =
                   zero_regions={"random"} (optionally also {"flank"}).
    zero_states  : iterable of {"snp","te"} — set cells equal to that state
                   code to ANCESTRAL (0), genome-wide. zero_states={"snp"}
                   keeps TE markers but removes SNP polymorphism.
    zero_channels: iterable of continuous-channel indices to zero (e.g. {5}
                   for log-recomb, {2,3} for the density channels).

    Returns (geno_int, cont) ablated. Pure masking — no re-encoding here;
    the one-hot happens downstream so ANCESTRAL cells become the anc channel.
    """
    from te_dfe.config import N_TE_SITES, N_TE_FLANK_SITES
    SNP_STATE, TE_STATE, ANC = 1, 2, 0

    g = geno_int.clone()
    te_end    = N_TE_SITES
    flank_end = N_TE_SITES + N_TE_FLANK_SITES
    S = g.shape[-1]

    if zero_regions:
        region_cols = {
            "te":     (0, te_end),
            "flank":  (te_end, flank_end),
            "random": (flank_end, S),
        }
        for r in zero_regions:
            if r not in region_cols:
                raise ValueError(f"unknown region '{r}' "
                                 f"(use te/flank/random)")
            lo, hi = region_cols[r]
            g[..., lo:hi] = ANC

    if zero_states:
        state_code = {"snp": SNP_STATE, "te": TE_STATE}
        for s in zero_states:
            if s not in state_code:
                raise ValueError(f"unknown state '{s}' (use snp/te)")
            g[g == state_code[s]] = ANC

    c = cont
    if cont is not None and zero_channels:
        c = cont.clone()
        n_ch = c.shape[2]
        for ch in zero_channels:
            ch = int(ch)
            if not (0 <= ch < n_ch):
                raise ValueError(
                    f"--zero-channels {ch} out of range: this model has "
                    f"{n_ch} continuous channels (valid indices 0..{n_ch-1}). "
                    f"For the simple model: 0=phys pos, 1=genetic pos, "
                    f"2=TE density, 3=SNP density.")
            c[:, :, ch, :] = 0.0
    elif cont is None and zero_channels:
        raise ValueError(
            "--zero-channels was given but this observation has no continuous "
            "channels to zero.")

    return g, c


def structured_observation(d, n_eff):
    """
    Build per-window structured tensors (state_oh-ready geno, cont, cov) for
    the first n_eff windows, for the K-extrapolation bypass path. Returns
    (geno_int (1,W,H,S), cont (1,W,C,S) or None, cov (1,W,n_cov)).
    """
    keys = set(d.files)
    geno = np.asarray(d["geno"])[:n_eff]                       # (W,H,S)
    geno_t = torch.from_numpy(geno.astype(np.int64)).unsqueeze(0)
    if "pos" in keys:
        cont = torch.from_numpy(
            np.asarray(d["pos"])[:n_eff].astype(np.float32)).unsqueeze(0)
    else:
        cont = None
    cov = torch.from_numpy(
        np.asarray(d["cov"])[:n_eff].astype(np.float32)).unsqueeze(0)
    return geno_t, cont, cov


def flat_observation(d, include_mask: bool, k_target: int = None,
                     n_train_windows: int = N_TRAIN_WINDOWS,
                     allow_extrapolate: bool = False,
                     n_te_target: int = None, n_te_rng=None):
    """
    Build the flat tensor the (training-shaped) SBIWrapper expects, slicing
    the test sim to the effective window count.

    n_eff = k_target if given, else (n_data if allow_extrapolate else
            min(n_train_windows, n_data)).

    When include_mask, append an n_eff-length all-ones mask (we slice to
    exactly n_eff real windows, so no padding zeros are needed). Returns
    (flat_array, n_eff), or (None, 0) if the format is unrecognized.
    """
    keys = set(d.files)
    n_data = _n_data_windows(d)

    # Sumstat representation: one pooled TE-set per scenario. The flat layout
    # the SumstatSBIWrapper expects is [feats.flatten() | te_mask.flatten()];
    # there is no window slicing (n_eff is always 1). k_target / extrapolation
    # do not apply (set N at simulation time via the TE pool), so they are
    # ignored here with n_eff = 1.
    if "feats" in keys:
        feats = np.asarray(d["feats"], dtype=np.float32)        # (K, F)
        # SumstatSBIWrapper is ALWAYS built with include_mask=True, so the mask
        # is part of the sumstat observation regardless of the matrix-style
        # `include_mask` flag (which is derived from the aggregator kind and is
        # meaningless here). Always append it. [feats | mask | (cov)].
        parts = [feats.ravel()]
        if "te_mask" in keys:
            te_mask = np.asarray(d["te_mask"], dtype=np.float32)
        else:
            te_mask = np.ones(feats.shape[0], dtype=np.float32)
        # TE-count sweep: keep N random valid TEs, mask out the rest. The
        # count-invariant aggregator then pools exactly the kept set.
        if n_te_target is not None:
            from te_dfe.sumstats import subsample_valid_indices
            rng = n_te_rng if n_te_rng is not None else np.random.default_rng()
            keep = subsample_valid_indices(te_mask, n_te_target, rng)
            new_mask = np.zeros_like(te_mask)
            new_mask[keep] = 1.0
            te_mask = new_mask
        parts.append(te_mask.ravel())
        # Global sequence-length covariate (only when the model was trained with
        # it — gated by config so the flat width matches the wrapper).
        try:
            from te_dfe.config import SUMSTAT_USE_SEQLEN_COV, seqlen_covariate
        except Exception:
            SUMSTAT_USE_SEQLEN_COV = False
        if SUMSTAT_USE_SEQLEN_COV:
            if "seq_len_bp" not in keys:
                raise ValueError(
                    "model expects the seq-length covariate but test file has no "
                    "'seq_len_bp' — regenerate test sims with the current code.")
            parts.append(np.array([seqlen_covariate(float(d["seq_len_bp"]))],
                                  dtype=np.float32))
        return np.concatenate(parts), 1

    if k_target is not None:
        n_eff = min(k_target, n_data)
    elif allow_extrapolate:
        n_eff = n_data
    else:
        n_eff = min(n_train_windows, n_data)

    if {"geno", "pos", "cov"} <= keys:
        parts = [np.asarray(d["geno"])[:n_eff].astype(np.float32).ravel(),
                 np.asarray(d["pos"])[:n_eff].astype(np.float32).ravel(),
                 np.asarray(d["cov"])[:n_eff].astype(np.float32).ravel()]
    elif {"geno", "cov"} <= keys:
        parts = [np.asarray(d["geno"])[:n_eff].astype(np.float32).ravel(),
                 np.asarray(d["cov"])[:n_eff].astype(np.float32).ravel()]
    elif "obs" in keys:
        obs = np.asarray(d["obs"])
        obs = obs[:n_eff] if obs.ndim >= 2 else obs
        parts = [obs.astype(np.float32).ravel()]
    else:
        return None, 0

    if include_mask:
        parts.append(np.ones(n_eff, dtype=np.float32))
    return np.concatenate(parts), n_eff


# ---------------------------------------------------------------------------
# Optional attention diagnostic
# ---------------------------------------------------------------------------
def load_core_network(results_dir: Path, meta: dict):
    """
    Reconstruct the core embedding network (for the attention diagnostic) for
    the checkpoint's representation. Returns the core, or None (diagnostic then
    skipped). Representation is read from meta if present, else inferred from
    the saved state-dict keys (older checkpoints predate the meta field).
    """
    try:
        state = torch.load(results_dir / "network.pt", map_location="cpu",
                           weights_only=False)
        rep = meta.get("representation")
        if rep is None:   # infer from keys: sumstat has 'encoder.*', matrix 'per_window.*'
            rep = ("sumstat" if any(k.startswith("encoder.") for k in state)
                   else "matrix")
        if rep == "sumstat":
            # Build the count-invariant sumstat core so the TE-extrapolation
            # bypass can embed a TE-set of ANY size. The per-TE *attention
            # diagnostic* isn't wired for sumstat (extract_attention reads
            # d["geno"] and returns None for feats-only obs), so the diagnostic
            # is skipped downstream -- but the core itself is needed for
            # --extrapolate, so build and return it rather than None.
            from te_dfe.network import TESumstatNet
            from te_dfe.config import N_SUMSTAT_GLOBAL_COV, N_SUMSTATS_PER_TE
            core = TESumstatNet(
                n_features         = N_SUMSTATS_PER_TE,
                aggregator_kind    = meta["aggregator_kind"],
                n_attention_heads  = meta["n_attention_heads"],
                use_embedding_norm = meta["use_embedding_norm"],
                n_global_cov       = N_SUMSTAT_GLOBAL_COV,
            )
            core.load_state_dict(state)
            core.eval()
            return core
        from te_dfe.network import TECovariateAttentionNet
        core = TECovariateAttentionNet(
            n_individuals     = N_HAPS,
            n_sites           = N_SITES,
            n_covariates      = N_COVARIATES,
            n_state_channels  = N_STATE_CHANNELS,
            n_cont_channels   = N_CONT,
            aggregator_kind   = meta["aggregator_kind"],
            n_attention_heads = meta["n_attention_heads"],
            use_embedding_norm = meta["use_embedding_norm"],
        )
        core.load_state_dict(state)
        core.eval()
        return core
    except Exception as e:
        print(f"  (attention diagnostic disabled: {e})")
        return None


def extract_attention(core, d, n_eff=None):
    """
    Per-window aggregator weights for one sim via the core's forward path
    (legacy, training-shaped K). Returns (W,) numpy array or None on failure.
    For the K-extrapolation path, weights come from sample_via_embedding
    instead (the chunked bypass).
    """
    import torch.nn.functional as F
    try:
        geno = np.asarray(d["geno"])
        if n_eff is not None:
            geno = geno[:n_eff]
        geno_t = torch.from_numpy(geno.astype(np.int64)).unsqueeze(0)
        state_oh = F.one_hot(geno_t, num_classes=N_STATE_CHANNELS) \
                     .permute(0, 1, 4, 2, 3).contiguous().float()
        cont = None
        if "pos" in d.files:
            pos = np.asarray(d["pos"])
            if n_eff is not None:
                pos = pos[:n_eff]
            cont = torch.from_numpy(pos.astype(np.float32)).unsqueeze(0)
        covn = np.asarray(d["cov"])
        if n_eff is not None:
            covn = covn[:n_eff]
        cov = torch.from_numpy(covn.astype(np.float32)).unsqueeze(0)
        W = geno.shape[0]
        mask = torch.ones(1, W)
        with torch.no_grad():
            per_win = core.per_window_embeddings(state_oh, cont)
            _, weights = core.aggregate_from_per_window(per_win, cov, mask)
        return weights.squeeze(0).detach().cpu().numpy()
    except Exception as e:
        print(f"  (attention extraction failed: {e})")
        return None


def sample_via_embedding(posterior, core, geno_int, cont, cov, mask,
                         n_samples, reject_outside_prior=False, prior=None,
                         window_chunk=20):
    """
    K-extrapolation sampling path. Bypasses sbi's x-shape validation and the
    training-shaped SBIWrapper by:
      1. computing the embedding directly with the (K-agnostic) core at the
         observation's actual window count — windows encoded in chunks of
         `window_chunk` to bound peak memory (whole-genome K would OOM in one
         shot), then
      2. sampling the flow conditioned on that precomputed embedding, with the
         flow's internal embedding-net temporarily swapped for identity.

    This is what lets a model trained at K<=20 be deployed/benchmarked at
    K=200 (whole-genome), the entire point of the variable-K + sigmoid_attn
    design. Returns (n_kept, n_params) numpy array.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    # 1. Embed at full K with the K-agnostic core, chunked over windows.
    state_oh = F.one_hot(geno_int, num_classes=N_STATE_CHANNELS) \
                 .permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.no_grad():
        emb, attn_w = core.embedding_chunked_with_weights(
            state_oh, cont, cov, mask=mask, window_chunk=window_chunk)  # (1,D),(1,W)

    # 2. Locate the nflows Flow inside the posterior's density estimator.
    de = posterior.posterior_estimator
    flow = getattr(de, "net", de)
    if not hasattr(flow, "_embedding_net"):
        raise RuntimeError(
            "Could not find flow._embedding_net to bypass; sbi internals may "
            "differ from the expected 0.26.x layout.")

    class _Identity(nn.Module):
        def forward(self, z):
            return z

    orig = flow._embedding_net
    flow._embedding_net = _Identity()
    try:
        with torch.no_grad():
            s = flow.sample(n_samples, context=emb)               # (1, n, P)
        s = s.squeeze(0).detach().cpu().numpy()
    finally:
        flow._embedding_net = orig

    if reject_outside_prior and prior is not None:
        lo = prior.support.base_constraint.lower_bound.cpu().numpy() \
            if hasattr(prior, "support") else None
        # Best-effort: BoxUniform exposes .base_dist.low/.high in sbi
        try:
            lo = prior.base_dist.low.cpu().numpy()
            hi = prior.base_dist.high.cpu().numpy()
            inside = np.all((s >= lo) & (s <= hi), axis=1)
            s = s[inside]
        except Exception:
            pass
    attn_np = attn_w.squeeze(0).detach().cpu().numpy() if attn_w is not None else None
    return s, attn_np


def sample_via_embedding_sumstat(posterior, core, feats, mask, n_samples,
                                 reject_outside_prior=False, prior=None,
                                 global_cov=None):
    """
    Sumstat TE-extrapolation sampling — the sumstat analogue of
    sample_via_embedding. TESumstatNet is count-invariant (per-TE encoder +
    exchangeable pooling), so it embeds a TE-set of ANY size; the only barrier
    to N != N_TE_TRAIN_MAX is sbi's flattened x_o shape check inside
    SumstatSBIWrapper. Bypass it the same way as the matrix path: embed the full
    TE-set directly with the core, then sample the flow conditioned on that
    precomputed embedding with sbi's embedding-net swapped for identity.

    This is what lets a model trained at N<=N_TE_TRAIN_MAX be benchmarked on
    more TEs (the whole point of the variable-N + sigmoid_attn design on the
    sumstat side). Returns (n_kept, n_params) numpy array.
    """
    import torch.nn as nn

    # 1. Embed the full TE-set with the count-invariant core (one pass; a single
    #    observation's K is small even at 1e4, so no chunking needed here).
    with torch.no_grad():
        emb = core.embedding(feats, mask, global_cov)            # (1, output_dim)

    # 2. Locate the nflows Flow inside the posterior's density estimator and
    #    feed the precomputed embedding directly (identity embedding-net).
    de = posterior.posterior_estimator
    flow = getattr(de, "net", de)
    if not hasattr(flow, "_embedding_net"):
        raise RuntimeError(
            "Could not find flow._embedding_net to bypass; sbi internals may "
            "differ from the expected 0.26.x layout.")

    class _Identity(nn.Module):
        def forward(self, z):
            return z

    orig = flow._embedding_net
    flow._embedding_net = _Identity()
    try:
        with torch.no_grad():
            s = flow.sample(n_samples, context=emb)              # (1, n, P)
        s = s.squeeze(0).detach().cpu().numpy()
    finally:
        flow._embedding_net = orig

    if reject_outside_prior and prior is not None:
        try:
            lo = prior.base_dist.low.cpu().numpy()
            hi = prior.base_dist.high.cpu().numpy()
            inside = np.all((s >= lo) & (s <= hi), axis=1)
            s = s[inside]
        except Exception:
            pass
    return s



    """
    Per-window attention/gate weights from one test sim. Returns (W,) numpy
    array, or None if the sim format lacks geno. Feeds the two-branch network
    its three separate inputs (state_oh, cont, covariates).
    """
    import torch.nn.functional as F
    keys = set(d.files)
    if "geno" not in keys:
        return None

    geno_int = torch.from_numpy(np.asarray(d["geno"], dtype=np.int64)).unsqueeze(0)
    state_oh = F.one_hot(geno_int, num_classes=N_STATE_CHANNELS) \
                 .permute(0, 1, 4, 2, 3).contiguous().float()   # (1, W, C, H, S)
    cov = torch.from_numpy(np.asarray(d["cov"], dtype=np.float32)).unsqueeze(0)

    # Continuous channels stay SEPARATE for the two-branch encoder.
    if N_CONT > 0 and "pos" in keys:
        cont = torch.from_numpy(np.asarray(d["pos"], dtype=np.float32)).unsqueeze(0)
    elif N_CONT > 0:
        W, S = geno_int.shape[1], geno_int.shape[3]
        cont = torch.zeros(1, W, N_CONT, S)
    else:
        cont = None

    with torch.no_grad():
        _, weights = core(state_oh, cont, cov)
    return weights.squeeze(0).numpy()


# ---------------------------------------------------------------------------
# Per-sim summary stats
# ---------------------------------------------------------------------------
def compute_sim_stats(d):
    """Per-sim covariate + padding aggregates from one test_*.npz."""
    keys = set(d.files)
    if "geno" not in keys:
        return None
    geno = np.asarray(d["geno"], dtype=np.int8)     # (W, H, S)
    cov  = np.asarray(d["cov"],  dtype=np.float32)  # (W, C)

    te_end    = N_TE_SITES
    flank_end = te_end + N_TE_FLANK_SITES

    n_te_real = n_snp_real = n_pad_te = n_pad_flnk = n_pad_rnd = 0
    for w in range(geno.shape[0]):
        gw = geno[w]
        te_region, flank_region, rand_region = (
            gw[:, :te_end], gw[:, te_end:flank_end], gw[:, flank_end:])
        n_te_real  += int((te_region == 2).any(axis=0).sum())
        n_snp_real += int((flank_region == 1).any(axis=0).sum())
        n_pad_te   += int((te_region == 3).sum())
        n_pad_flnk += int((flank_region == 3).sum())
        n_pad_rnd  += int((rand_region == 3).sum())

    W, n_haps, S = geno.shape
    te_cells    = W * n_haps * te_end
    flank_cells = W * n_haps * N_TE_FLANK_SITES
    rnd_cells   = W * n_haps * (S - flank_end)
    gene_dens, recomb, te_dens, snp_dens = cov[:, 0], cov[:, 1], cov[:, 2], cov[:, 3]

    return {
        "te_dens_mean": float(te_dens.mean()), "te_dens_std": float(te_dens.std()),
        "te_dens_min": float(te_dens.min()),   "te_dens_max": float(te_dens.max()),
        "snp_dens_mean": float(snp_dens.mean()), "snp_dens_std": float(snp_dens.std()),
        "gene_dens_mean": float(gene_dens.mean()), "recomb_mean": float(recomb.mean()),
        "n_te_real_total": int(n_te_real), "n_snp_real_total": int(n_snp_real),
        "n_pad_te_region_total": int(n_pad_te),
        "n_pad_flank_region_total": int(n_pad_flnk),
        "n_pad_random_region_total": int(n_pad_rnd),
        "frac_pad_te_region": float(n_pad_te / max(te_cells, 1)),
        "frac_pad_flank_region": float(n_pad_flnk / max(flank_cells, 1)),
        "frac_pad_random_region": float(n_pad_rnd / max(rnd_cells, 1)),
        "n_windows": int(W), "n_haps": int(n_haps), "n_sites": int(S),
    }


SIM_SUMMARY_KEYS = [
    "te_dens_mean", "te_dens_std", "te_dens_min", "te_dens_max",
    "snp_dens_mean", "snp_dens_std", "gene_dens_mean", "recomb_mean",
    "n_te_real_total", "n_snp_real_total",
    "n_pad_te_region_total", "n_pad_flank_region_total", "n_pad_random_region_total",
    "frac_pad_te_region", "frac_pad_flank_region", "frac_pad_random_region",
    "n_windows", "n_haps", "n_sites",
]


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_summary(path, used, n_post, rmse, bias, coverage, sbc_p, n_failed,
                  names=None):
    names = names if names is not None else PARAM_NAMES
    with open(path, "w") as f:
        f.write(f"Benchmark summary ({len(used)} test sims, "
                f"{n_post} posterior samples each)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'parameter':<25} {'RMSE':>10} {'bias':>10} "
                f"{'cov_90%':>10} {'SBC_p':>10}\n")
        f.write("-" * 70 + "\n")
        for i, name in enumerate(names):
            f.write(f"{name:<25} {rmse[i]:>10.4f} {bias[i]:>+10.4f} "
                    f"{coverage[i]:>10.3f} {sbc_p[i]:>10.4f}\n")
        f.write("\nTarget 90% coverage = 0.90; SBC p-value > 0.05 means "
                "ranks are consistent with uniform (= calibrated).\n")
        if n_failed:
            f.write(f"\n{n_failed} test sims failed analysis.\n")


BINNED_NAMES = ["p_nn", "p_w", "p_strong"]


def write_binned_csv(path, used, true_b, mean_b, lo_b, hi_b, covered_b):
    """Per-sim derived DFE-bin proportions: true vs posterior."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        cols = ["sim_idx", "sim_file"]
        for nm in BINNED_NAMES:
            cols += [f"true_{nm}", f"mean_{nm}", f"ci5_{nm}", f"ci95_{nm}",
                     f"covered_{nm}"]
        w.writerow(cols)
        for i, fp in enumerate(used):
            row = [i, fp.name]
            for j in range(len(BINNED_NAMES)):
                row += [float(true_b[i, j]), float(mean_b[i, j]),
                        float(lo_b[i, j]), float(hi_b[i, j]),
                        int(covered_b[i, j])]
            w.writerow(row)


def write_recovery_csv(path, used, true_t, mean, median, lo, hi, covered,
                       names=None):
    names = names if names is not None else PARAM_NAMES
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sim_idx", "sim_file", "param_name", "true_value",
                    "posterior_mean", "posterior_median", "ci_5pct", "ci_95pct",
                    "covered_90pct"])
        for i, fp in enumerate(used):
            for j, name in enumerate(names):
                w.writerow([i, fp.name, name,
                            float(true_t[i, j]), float(mean[i, j]),
                            float(median[i, j]), float(lo[i, j]), float(hi[i, j]),
                            int(covered[i, j])])


def write_sim_summary_csv(path, used, stats):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sim_idx", "sim_file"] + SIM_SUMMARY_KEYS)
        for i, fp in enumerate(used):
            s = stats[i] or {}
            w.writerow([i, fp.name] + [s.get(k, "") for k in SIM_SUMMARY_KEYS])


def write_attention_csv(path, used, attn, true_t, mean, median, lo, hi, covered,
                        names=None):
    names = names if names is not None else PARAM_NAMES
    W = next(w.shape[0] for w in attn if w is not None)
    param_cols = []
    for name in names:
        param_cols += [f"true_{name}", f"mean_{name}", f"median_{name}",
                       f"ci5_{name}", f"ci95_{name}", f"covered_{name}"]
    attn_cols = [f"attn_w_{k:02d}" for k in range(W)]
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["sim_idx", "sim_file"] + param_cols + attn_cols
                    + ["attn_entropy", "attn_max"])
        for i, fp in enumerate(used):
            row = [i, fp.name]
            for j in range(len(names)):
                row += [float(true_t[i, j]), float(mean[i, j]), float(median[i, j]),
                        float(lo[i, j]), float(hi[i, j]), int(covered[i, j])]
            w = attn[i]
            if w is None:
                row += [""] * W + ["", ""]
            else:
                row += [float(v) for v in w]
                p = np.clip(w, 1e-30, 1.0); p = p / p.sum()
                row += [float(-np.sum(p * np.log(p))), float(w.max())]
            wr.writerow(row)
    return W


def make_plots(out_dir, true_t, mean, lo, hi, sbc_ranks, n_post,
               rmse, coverage, sbc_p, n_used, names=None):
    names = names if names is not None else PARAM_NAMES
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = min(4, len(names))
    nrows = (len(names) + ncols - 1) // ncols

    # Recovery scatter
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for i, name in enumerate(names):
        ax = axes[i]
        ax.errorbar(true_t[:, i], mean[:, i],
                    yerr=[mean[:, i] - lo[:, i], hi[:, i] - mean[:, i]],
                    fmt="o", ms=3, alpha=0.5, capsize=0)
        a = min(true_t[:, i].min(), lo[:, i].min())
        b = max(true_t[:, i].max(), hi[:, i].max())
        ax.plot([a, b], [a, b], "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel("true"); ax.set_ylabel("posterior mean +/- 90% CI")
        ax.set_title(f"{name}\nRMSE={rmse[i]:.3f}, cov={coverage[i]:.2f}", fontsize=10)
    for j in range(len(names), len(axes)):
        axes[j].axis("off")
    fig.tight_layout(); fig.savefig(out_dir / "recovery.pdf", dpi=140); plt.close(fig)
    print(f"Wrote recovery plot -> {out_dir}/recovery.pdf")

    # SBC histograms
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for i, name in enumerate(names):
        ax = axes[i]
        ax.hist(sbc_ranks[:, i] / n_post, bins=20, range=(0, 1),
                edgecolor="black", alpha=0.7)
        ax.axhline(n_used / 20, color="r", lw=1.0, ls="--", label="uniform expected")
        ax.set_xlabel("rank quantile"); ax.set_ylabel("count")
        ax.set_title(f"{name}\nSBC p={sbc_p[i]:.3f}", fontsize=10)
        ax.legend(fontsize=7)
    for j in range(len(names), len(axes)):
        axes[j].axis("off")
    fig.tight_layout(); fig.savefig(out_dir / "sbc.pdf", dpi=140); plt.close(fig)
    print(f"Wrote SBC plot -> {out_dir}/sbc.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    test_dir    = Path(args.test_dir)

    out_dir = Path(args.out)
    if args.k_target is not None:
        out_dir = out_dir.parent / (out_dir.name + f"_K{args.k_target:02d}")
    if args.n_te_target is not None:
        out_dir = out_dir.parent / (out_dir.name + f"_NTE{args.n_te_target}")
    # One rng for the whole run, advanced per scenario -> each replicate gets a
    # distinct, seed-reproducible random TE subsample.
    _obs_te_rng = (np.random.default_rng(args.n_te_seed)
                   if args.n_te_target is not None else None)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Model metadata: aggregator + mask requirement (single source of truth) ---
    meta = load_training_meta(results_dir)
    include_mask = meta["include_mask"]
    print(f"Model: aggregator={meta['aggregator_kind']}, include_mask={include_mask}")

    # --- Posterior ---
    print(f"Loading posterior from {results_dir}/posterior.pkl ...")
    with open(results_dir / "posterior.pkl", "rb") as f:
        posterior = pickle.load(f)

    # --- Optional attention diagnostic (independent of masking) ---
    # The core is ALSO required for the K-extrapolation bypass, so load it
    # whenever attention is on OR we're extrapolating.
    core = None
    if (not args.no_attention_diag) or args.extrapolate:
        core = load_core_network(results_dir, meta)
    if args.extrapolate and core is None:
        print("ERROR: --extrapolate needs the core network (network.pt); "
              "could not load it.", file=sys.stderr)
        return 1

    # --- Ablation flag parsing (structured/extrapolate path only) ---
    def _csv_set(x):
        return {t.strip() for t in x.split(",") if t.strip()} if x else None
    abl_regions  = _csv_set(args.zero_regions)
    abl_states   = _csv_set(args.zero_states)
    abl_channels = ({int(t) for t in args.zero_channels.split(",") if t.strip()}
                    if args.zero_channels else None)
    if (abl_regions or abl_states or abl_channels) and not args.extrapolate:
        print("ERROR: --zero-regions/--zero-states/--zero-channels require "
              "--extrapolate (they operate on the structured embedding path).",
              file=sys.stderr)
        return 1
    if abl_channels:
        from te_dfe.config import N_CONTINUOUS_CHANNELS as _N_CONT
        bad = [c for c in abl_channels if not (0 <= c < _N_CONT)]
        if bad:
            print(f"ERROR: --zero-channels {sorted(bad)} out of range: this "
                  f"model has {_N_CONT} continuous channels "
                  f"(valid 0..{_N_CONT-1}). Simple model: "
                  f"0=phys pos, 1=genetic pos, 2=TE density, 3=SNP density. "
                  f"(Channels 4,5 exist only in the context model.)",
                  file=sys.stderr)
            return 1
    if abl_regions or abl_states or abl_channels:
        print(f"ABLATION: zero_regions={abl_regions} zero_states={abl_states} "
              f"zero_channels={abl_channels}")

    # Prior (for optional out-of-prior rejection on the bypass path).
    prior = None
    if args.extrapolate:
        try:
            from te_dfe.prior import DFEBurstPrior
            prior = DFEBurstPrior.to_sbi_prior()
        except Exception:
            prior = None

    n_train_windows = int(meta.get("n_train_windows", N_TRAIN_WINDOWS))

    # --- K-target validation ---
    if args.k_target is not None:
        if args.extrapolate:
            if args.k_target < 1:
                print("ERROR: --k-target must be >= 1.", file=sys.stderr)
                return 1
            print(f"K-target (extrapolation): first {args.k_target} windows "
                  f"(model trained at K<={n_train_windows})")
        elif not include_mask:
            print("WARNING: --k-target ignored (model has include_mask=False).",
                  file=sys.stderr)
            args.k_target = None
        elif not (1 <= args.k_target <= n_train_windows):
            print(f"ERROR: --k-target must be in [1, {n_train_windows}] "
                  f"without --extrapolate (got {args.k_target}). Pass "
                  f"--extrapolate to evaluate beyond the training window count.",
                  file=sys.stderr)
            return 1
        else:
            print(f"K-target: first {args.k_target} of {n_train_windows} windows")
    elif args.extrapolate:
        if meta.get("representation") == "sumstat":
            print("TE-extrapolation: embedding the full TE-set per sim via the "
                  "count-invariant core (N may exceed N_TE_TRAIN_MAX)")
        else:
            print(f"K-extrapolation: using ALL windows per sim "
                  f"(model trained at K<={n_train_windows})")
    elif include_mask:
        print(f"K-target: first {n_train_windows} windows (training count)")

    reject = not args.no_reject_prior
    print(f"reject_outside_prior = {reject}"
          + (f" (max_sampling_time={args.max_sampling_time:.0f}s/test)"
             if reject else ""))

    # --- Test sims ---
    test_files = sorted(test_dir.glob("test_*.npz"))
    if not test_files:
        print(f"No test_*.npz in {test_dir}", file=sys.stderr)
        return 1
    if args.max_tests:
        test_files = test_files[:args.max_tests]
    print(f"Analyzing {len(test_files)} test simulations...")

    # --- DFE-bin setup (pure-gamma model only) ---
    # Posterior samples are over the INFERRED subset of parameters (in
    # PARAM_NAMES order), so the test file's full theta must be reduced to the
    # same columns before any comparison. (When all params are inferred this is
    # the identity, which is why the matrix runs never hit this.)
    from te_dfe.config import INFER_INDICES
    infer_idx = list(INFER_INDICES)
    n_full = len(PARAM_NAMES)
    is_subset = (infer_idx != list(range(n_full)))
    if is_subset:
        from te_dfe.config import INFER_NAMES
        report_names = list(INFER_NAMES)
        print(f"Inferring {len(infer_idx)} of {n_full} params: {report_names}")
    else:
        report_names = list(PARAM_NAMES)

    do_bins = (not args.no_bins
               and "log_gamma_shape" in PARAM_NAMES
               and "log_mean_s" in PARAM_NAMES)
    if do_bins:
        from te_dfe.prior import compute_binned_props
        idx_shape = PARAM_NAMES.index("log_gamma_shape")
        idx_ms    = PARAM_NAMES.index("log_mean_s")
        # Column positions of these params WITHIN the posterior sample matrix
        # (inferred space). None if the param is fixed (not inferred) -> the
        # gamma-CDF bins cannot be formed from samples, so bins are skipped.
        idx_shape_s = infer_idx.index(idx_shape) if idx_shape in infer_idx else None
        idx_ms_s    = infer_idx.index(idx_ms)    if idx_ms    in infer_idx else None
        if idx_shape_s is None or idx_ms_s is None:
            do_bins = False
            print("DFE bins: OFF (gamma shape / mean_s is FIXED, not inferred — "
                  "cannot form the |2Ne.s| posterior from samples)")
        else:
            ne_h = (args.ne_harmonic if args.ne_harmonic is not None
                    else float(meta.get("ne_harmonic", 500)))
            print(f"DFE bins: ON (ne_harmonic={ne_h:.0f}, "
                  f"thresholds |2Ne.s| < 1, 1-10, >= 10)")
    if not do_bins:
        ne_h = None
        if not args.no_bins and "log_gamma_shape" in PARAM_NAMES \
                and "log_mean_s" in PARAM_NAMES:
            pass  # message already printed above
        elif not args.no_bins:
            print("DFE bins: OFF (model is not the pure-gamma DFE)")

    # --- Multi-level CI setup (tail-shape diagnostic) ---
    ci_levels = sorted({int(x) for x in args.ci_levels.split(",") if x.strip()})
    ci_levels = [c for c in ci_levels if 1 <= c <= 99]
    print(f"CI levels reported: {ci_levels}%")

    true_t, means, medians, lo5, hi95, sbc = [], [], [], [], [], []
    used, sim_stats, attn = [], [], []
    true_bins, mean_bins, lo_bins, hi_bins, sbc_bins = [], [], [], [], []
    # Per-sim CI half-widths at each level: ci_w[level] -> list of (N_PARAMS,)
    ci_w = {c: [] for c in ci_levels}
    saved_samples = [] if args.save_samples else None
    saved_bin_samples = [] if (args.save_samples and do_bins) else None
    n_failed = 0

    for i, fp in enumerate(test_files):
        try:
            d = np.load(fp)
            theta_true_full = d["theta"]
            # Reduce to the inferred columns so it aligns with the posterior
            # samples (which are over INFER_INDICES). Full vector kept for the
            # true-bins lookup, which uses full-space indices.
            theta_true = theta_true_full[infer_idx] if is_subset else theta_true_full

            if args.extrapolate:
                if "feats" in d.files:
                    # --- Sumstat TE-extrapolation bypass --------------------
                    # Embed the full TE-set directly with the count-invariant
                    # core and sample the flow on that embedding -- no sbi
                    # x-shape check, so N may exceed N_TE_TRAIN_MAX.
                    feats_np = np.asarray(d["feats"], dtype=np.float32)   # (K,F)
                    if "te_mask" in d.files:
                        mask_np = np.asarray(d["te_mask"], dtype=np.float32)
                    else:
                        mask_np = np.ones(feats_np.shape[0], dtype=np.float32)
                    # Optional sweep subsample: keep N random valid TEs.
                    if args.n_te_target is not None:
                        from te_dfe.sumstats import subsample_valid_indices
                        keep = subsample_valid_indices(
                            mask_np, args.n_te_target, _obs_te_rng)
                        nm = np.zeros_like(mask_np)
                        nm[keep] = 1.0
                        mask_np = nm
                    n_eff = int(mask_np.sum())
                    feats_t = torch.from_numpy(feats_np).unsqueeze(0)    # (1,K,F)
                    mask_t  = torch.from_numpy(mask_np).unsqueeze(0)     # (1,K)
                    gcov_t = None
                    if getattr(core, "n_global_cov", 0):
                        # seq-length global covariate (only if the model uses it)
                        from te_dfe.config import seqlen_covariate
                        if "seq_len_bp" not in d.files:
                            raise ValueError(
                                "model uses a seq-length covariate but the test "
                                "npz has no 'seq_len_bp'.")
                        gcov = np.asarray(
                            seqlen_covariate(float(d["seq_len_bp"])),
                            dtype=np.float32).reshape(1, -1)
                        gcov_t = torch.from_numpy(gcov)
                    s = sample_via_embedding_sumstat(
                        posterior, core, feats_t, mask_t, args.n_post,
                        reject_outside_prior=reject, prior=prior,
                        global_cov=gcov_t)
                    attn_extrap = None
                    if i == 0:
                        print(f"  [extrapolate sumstat] sim 0 used {n_eff} valid "
                              f"TEs -> {s.shape[0]} samples")
                else:
                    # --- Matrix window K-extrapolation bypass ---------------
                    # K-extrapolation bypass: embed at full K, sample the flow on
                    # the precomputed embedding (no sbi x-shape check, no
                    # training-shaped reshape).
                    _, n_eff = flat_observation(
                        d, include_mask, args.k_target,
                        n_train_windows=n_train_windows, allow_extrapolate=True)
                    geno_int, cont, cov = structured_observation(d, n_eff)
                    if abl_regions or abl_states or abl_channels:
                        geno_int, cont = ablate_observation(
                            geno_int, cont,
                            zero_regions=abl_regions, zero_states=abl_states,
                            zero_channels=abl_channels)
                    mask = (torch.ones(1, n_eff) if include_mask else None)
                    s, attn_extrap = sample_via_embedding(
                        posterior, core, geno_int, cont, cov, mask,
                        args.n_post, reject_outside_prior=reject, prior=prior,
                        window_chunk=args.window_chunk)
                    if i == 0:
                        print(f"  [extrapolate] sim 0 used {n_eff} windows "
                              f"-> {s.shape[0]} samples")
            else:
                x_flat, n_eff = flat_observation(
                    d, include_mask, args.k_target,
                    n_train_windows=n_train_windows, allow_extrapolate=False,
                    n_te_target=args.n_te_target, n_te_rng=_obs_te_rng)
                if x_flat is None:
                    raise ValueError(f"unrecognized npz keys: {set(d.files)}")
                x = torch.from_numpy(x_flat).float().unsqueeze(0)
                sample_kw = dict(reject_outside_prior=reject,
                                 show_progress_bars=False)
                if reject and args.max_sampling_time is not None:
                    sample_kw["max_sampling_time"] = float(args.max_sampling_time)
                with torch.no_grad():
                    s = posterior.sample((args.n_post,), x=x, **sample_kw)
                s = s.detach().cpu().numpy()
                if reject and s.shape[0] < args.n_post:
                    # max_sampling_time hit before n_post in-box draws collected:
                    # low acceptance -> the flow is leaking mass out of the prior
                    # box for this test (poor fit / hard cell). Proceed with what
                    # we got, but flag it so the cell can be investigated.
                    print(f"  WARN {fp.name}: low rejection acceptance — only "
                          f"{s.shape[0]}/{args.n_post} in-box samples in "
                          f"{args.max_sampling_time:.0f}s (flow leaking out of "
                          f"prior; consider --no-reject-prior or more training).",
                          file=sys.stderr)
                attn_extrap = None

            if s.shape[0] == 0:
                raise ValueError("no posterior samples survived rejection")

            true_t.append(theta_true)
            means.append(s.mean(axis=0))
            medians.append(np.median(s, axis=0))
            lo5.append(np.percentile(s, 5, axis=0))
            hi95.append(np.percentile(s, 95, axis=0))
            sbc.append((s < theta_true[None, :]).sum(axis=0))
            used.append(fp)
            sim_stats.append(compute_sim_stats(d))
            # Attention per window: from the bypass weights under --extrapolate,
            # else from the legacy forward path.
            if attn_extrap is not None:
                attn.append(attn_extrap)
            elif core is not None and not args.extrapolate:
                attn.append(extract_attention(core, d))
            else:
                attn.append(None)

            # Multi-level central CI half-widths (tail-shape diagnostic).
            for c in ci_levels:
                lo_p = (100 - c) / 2.0
                hi_p = 100 - lo_p
                w = (np.percentile(s, hi_p, axis=0)
                     - np.percentile(s, lo_p, axis=0)) / 2.0
                ci_w[c].append(w)

            if saved_samples is not None:
                saved_samples.append(s.astype(np.float32))

            if do_bins:
                # Posterior over bins: push each posterior draw through the
                # gamma CDF -> (n_post, 3) array of (p_nn, p_w, p_strong).
                # Index the SAMPLES in inferred space; the TRUE values come from
                # the full theta vector (full-space indices).
                pn, pw, ps = compute_binned_props(
                    s[:, idx_shape_s], s[:, idx_ms_s], ne_h)
                bins_post = np.stack([pn, pw, ps], axis=1)        # (n_post, 3)
                # True bins from the sim's true theta.
                tn, tw, ts = compute_binned_props(
                    np.array([theta_true_full[idx_shape]]),
                    np.array([theta_true_full[idx_ms]]), ne_h)
                true_b = np.array([tn[0], tw[0], ts[0]])
                true_bins.append(true_b)
                mean_bins.append(bins_post.mean(axis=0))
                lo_bins.append(np.percentile(bins_post, 5, axis=0))
                hi_bins.append(np.percentile(bins_post, 95, axis=0))
                sbc_bins.append((bins_post < true_b[None, :]).sum(axis=0))
                if saved_bin_samples is not None:
                    saved_bin_samples.append(bins_post.astype(np.float32))

            if (i + 1) % 25 == 0 or i == len(test_files) - 1:
                print(f"  {i+1}/{len(test_files)} done")
        except Exception as e:
            print(f"  test {fp.name} failed: {e}", file=sys.stderr)
            n_failed += 1

    if not true_t:
        print("FAIL: no successful test analyses", file=sys.stderr)
        return 2

    true_t  = np.stack(true_t)
    means   = np.stack(means)
    medians = np.stack(medians)
    lo5     = np.stack(lo5)
    hi95    = np.stack(hi95)
    sbc     = np.stack(sbc)

    # --- Metrics ---
    rmse = np.sqrt(np.mean((means - true_t) ** 2, axis=0))
    bias = np.mean(means - true_t, axis=0)
    covered = (true_t >= lo5) & (true_t <= hi95)
    coverage = np.mean(covered, axis=0)

    from scipy.stats import kstest, uniform
    sbc_p = []
    for p in range(sbc.shape[1]):          # inferred-param count (== len(report_names))
        _, pv = kstest(sbc[:, p] / args.n_post, uniform(loc=0, scale=1).cdf)
        sbc_p.append(pv)

    # --- Tail-shape diagnostic: mean CI half-width per level + tail ratio ---
    # If a posterior is Gaussian, CI half-widths scale with the normal
    # quantile (z_50/z_90 ~ 0.405). A *smaller* ratio means the 50% CI is
    # tighter than Gaussian relative to the 90% — i.e. a sharp core with long
    # tails. tail_ratio compares the widest to narrowest requested level.
    mean_ci_w = {c: np.stack(ci_w[c]).mean(axis=0) for c in ci_levels}
    from scipy.stats import norm as _norm
    gauss_ref = {c: _norm.ppf(0.5 + c / 200.0) for c in ci_levels}  # z at that level

    # --- Outputs ---
    summary_path = out_dir / "benchmark_summary.txt"
    write_summary(summary_path, used, args.n_post, rmse, bias, coverage,
                  sbc_p, n_failed, names=report_names)
    print(f"Wrote summary -> {summary_path}\n")
    with open(summary_path) as f:
        print(f.read())

    write_recovery_csv(out_dir / "recovery_values.csv", used, true_t, means,
                       medians, lo5, hi95, covered, names=report_names)
    print(f"Wrote recovery values -> {out_dir}/recovery_values.csv")

    # --- Tail-shape: multi-level CI widths appended to summary ---
    if len(ci_levels) >= 2:
        lo_lvl, hi_lvl = ci_levels[0], ci_levels[-1]
        with open(summary_path, "a") as f:
            f.write("\n\nPosterior CI half-widths by level "
                    "(mean over sims) — tail-shape diagnostic\n")
            f.write("=" * 78 + "\n")
            hdr = f"{'parameter':<20}" + "".join(f"{c:>9}%" for c in ci_levels)
            f.write(hdr + f"{'tail_ratio':>14}{'gauss_ref':>11}\n")
            f.write("-" * 78 + "\n")
            for i, name in enumerate(report_names):
                row = f"{name:<20}"
                for c in ci_levels:
                    row += f"{mean_ci_w[c][i]:>10.4f}"
                # observed (narrow/wide) vs Gaussian (narrow/wide)
                obs_ratio = (mean_ci_w[lo_lvl][i]
                             / max(mean_ci_w[hi_lvl][i], 1e-12))
                g_ratio = gauss_ref[lo_lvl] / gauss_ref[hi_lvl]
                row += f"{obs_ratio:>14.3f}{g_ratio:>11.3f}"
                f.write(row + "\n")
            f.write("\ntail_ratio < gauss_ref  => sharper core + longer tails "
                    "than Gaussian\n")
            f.write("(i.e. the inner CI is tighter than a Gaussian with the "
                    "same outer CI would be).\n")
        # Console echo
        print("\nCI half-widths (mean over sims), tail-shape:")
        print(f"  {'param':<18}" + "".join(f"{c:>8}%" for c in ci_levels)
              + f"{'ratio':>9}{'gauss':>8}")
        for i, name in enumerate(report_names):
            line = f"  {name:<18}"
            for c in ci_levels:
                line += f"{mean_ci_w[c][i]:>9.4f}"
            obs_ratio = mean_ci_w[lo_lvl][i] / max(mean_ci_w[hi_lvl][i], 1e-12)
            line += f"{obs_ratio:>9.3f}{gauss_ref[lo_lvl]/gauss_ref[hi_lvl]:>8.3f}"
            print(line)

    # --- Raw posterior samples dump ---
    if saved_samples is not None:
        samp_arr = np.stack(saved_samples)            # (n_sims, n_post, n_infer)
        payload = {
            "samples":     samp_arr,
            "true_theta":  true_t,
            "param_names": np.array(report_names),     # inferred names, matches axis 2
            "sim_files":   np.array([fp.name for fp in used]),
        }
        if saved_bin_samples is not None:
            payload["bin_samples"] = np.stack(saved_bin_samples)  # (n_sims,n_post,3)
            payload["bin_names"]   = np.array(BINNED_NAMES)
            payload["true_bins"]   = np.stack(true_bins)
        np.savez_compressed(out_dir / "posterior_samples.npz", **payload)
        print(f"Wrote raw posterior samples -> {out_dir}/posterior_samples.npz "
              f"(shape {samp_arr.shape}: n_sims x n_post x n_params)")

    write_sim_summary_csv(out_dir / "sim_summary.csv", used, sim_stats)
    print(f"Wrote per-sim summary -> {out_dir}/sim_summary.csv")

    # --- Derived DFE-bin proportions ---
    if do_bins and true_bins:
        true_b = np.stack(true_bins)
        mean_b = np.stack(mean_bins)
        lo_b   = np.stack(lo_bins)
        hi_b   = np.stack(hi_bins)
        sbc_b  = np.stack(sbc_bins)
        rmse_b = np.sqrt(np.mean((mean_b - true_b) ** 2, axis=0))
        bias_b = np.mean(mean_b - true_b, axis=0)
        cov_b  = np.mean((true_b >= lo_b) & (true_b <= hi_b), axis=0)
        sbcp_b = []
        for j in range(len(BINNED_NAMES)):
            _, pv = kstest(sbc_b[:, j] / args.n_post,
                           uniform(loc=0, scale=1).cdf)
            sbcp_b.append(pv)

        write_binned_csv(out_dir / "binned_props.csv", used,
                         true_b, mean_b, lo_b, hi_b,
                         (true_b >= lo_b) & (true_b <= hi_b))
        print(f"Wrote DFE-bin proportions -> {out_dir}/binned_props.csv")

        with open(summary_path, "a") as f:
            f.write("\n\nDerived DFE-bin proportions "
                    f"(p_nn / p_w / p_strong, ne_harmonic={ne_h:.0f})\n")
            f.write("=" * 70 + "\n")
            f.write(f"{'bin':<12} {'RMSE':>10} {'bias':>10} "
                    f"{'cov_90%':>10} {'SBC_p':>10}\n")
            f.write("-" * 70 + "\n")
            for j, nm in enumerate(BINNED_NAMES):
                f.write(f"{nm:<12} {rmse_b[j]:>10.4f} {bias_b[j]:>+10.4f} "
                        f"{cov_b[j]:>10.3f} {sbcp_b[j]:>10.4f}\n")
        print("\nDFE-bin proportions:")
        print(f"  {'bin':<10}{'RMSE':>9}{'bias':>9}{'cov90':>8}{'SBC_p':>9}")
        for j, nm in enumerate(BINNED_NAMES):
            print(f"  {nm:<10}{rmse_b[j]:>9.4f}{bias_b[j]:>+9.4f}"
                  f"{cov_b[j]:>8.3f}{sbcp_b[j]:>9.4f}")

    if any(w is not None for w in attn):
        W = write_attention_csv(out_dir / "attention_per_sim.csv", used, attn,
                                 true_t, means, medians, lo5, hi95, covered, names=report_names)
        print(f"Wrote attention diagnostics -> {out_dir}/attention_per_sim.csv "
              f"(uniform entropy = log({W}) = {np.log(W):.3f})")

    try:
        make_plots(out_dir, true_t, means, lo5, hi95, sbc, args.n_post,
                   rmse, coverage, sbc_p, len(used), names=report_names)
    except Exception as e:
        print(f"  (Plots skipped: {e})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
