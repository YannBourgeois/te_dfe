#!/usr/bin/env python
"""inspect_dataset.py — report the REPRESENTATION (sumstats / te_slice / matrix CNN)
and the demographic MODEL (simple / context) a dataset was generated under.

Usage
-----
    python inspect_dataset.py results/dataset            # collated dataset base
    python inspect_dataset.py results/dataset.theta.pt   # the metadata file itself
    python inspect_dataset.py results/sim_trees/xxx.npz  # a single pre-collate shard

How it decides
--------------
The collated metadata (<base>.theta.pt, written by 02_collate / 05a_collate) is the
definitive source and needs no guessing:

  * REPRESENTATION  <- the `format` field, recorded at collate time:
        sumstat                  -> sumstats
        te_slice                 -> te_slice (sliced CNN)
        newer / new / legacy     -> matrix CNN (whole-window)

  * MODEL           <- the width of the theta matrix (params per sim), which is
        fixed by the config that generated it:
        6            -> simple   (log_lambda, log_gamma_shape, log_mean_s,
                                  log10_t_start, log_multiplier, + 1)
        5 / 7 / 9    -> context  (constant-DFE / rec-or-dist / full context set)
        The model name is NOT stored in the metadata, so this width is the
        primary discriminator; it works for every representation.

  * Corroboration   <- shapes, independent of the theta width:
        CNN (te_slice/matrix): continuous-channel count = 5 (simple) or 6 (context)
                               [base 4 = 2 position + 2 density; simple +1 log-recomb,
                                context +2 dist-to-gene + log-recomb]
        sumstats: feats-per-TE width (model-dependent; printed for reference)

A pre-collate .npz shard has no `format` field, so it is classified structurally
from its array keys (feats -> sumstat; geno+te_mask -> te_slice; geno -> matrix).
"""
import sys
import os

_REP = {
    "sumstat":  "sumstats",
    "te_slice": "te_slice (sliced CNN)",
    "newer":    "matrix CNN (whole-window)",
    "new":      "matrix CNN (whole-window)",
    "legacy":   "matrix CNN (whole-window, legacy)",
}

# theta width -> (model label, illustrative param names)
# Param names per theta width. Taken verbatim from the configs:
#   config_simple.py:33   PARAM_NAMES (6)
#   config_context.py:48  PARAM_NAMES = ["log_lambda","log_s_global"]
#                          + _ACTIVE_CONTEXT + ["log10_t_start","log_duration",
#                            "log_multiplier"]
# _ACTIVE_CONTEXT is the CONTEXT_PARAMS subset of
# ["alpha_d","alpha_r","gamma_d","gamma_r"], IN THAT ORDER.
_CTX_HEAD = ["log_lambda", "log_s_global"]
_CTX_TAIL = ["log10_t_start", "log_duration", "log_multiplier"]


def _ctx(*context):
    return _CTX_HEAD + list(context) + _CTX_TAIL


_MODEL_BY_WIDTH = {
    6: ("simple",
        ["log_lambda", "log_gamma_shape", "log_mean_s",
         "log10_t_start", "log_duration", "log_multiplier"]),
    5: ("context (constant-DFE, no context params)", _ctx()),
    7: ("context (recombination-only or distance-only context)", None),  # ambiguous
    9: ("context (full: alpha_d, alpha_r, gamma_d, gamma_r)",
        _ctx("alpha_d", "alpha_r", "gamma_d", "gamma_r")),
}

# Width 7 is genuinely ambiguous: CONTEXT_PARAMS=["alpha_r","gamma_r"] and
# ["alpha_d","gamma_d"] BOTH give 7 params. Width alone cannot tell them apart,
# so we resolve it from the live config when possible and otherwise show both.
_AMBIGUOUS = {
    7: [("recombination-only  CONTEXT_PARAMS=['alpha_r','gamma_r']",
         _ctx("alpha_r", "gamma_r")),
        ("distance-only       CONTEXT_PARAMS=['alpha_d','gamma_d']",
         _ctx("alpha_d", "gamma_d"))],
}


def _live_param_names(width):
    """Authoritative names from the CURRENT config, if its width matches.
    Requires TE_DFE_MODEL / model_select to resolve; returns None if it cannot
    be imported, or if the dataset was built with a different config."""
    try:
        from te_dfe.config import PARAM_NAMES as _P
        if len(_P) == width:
            return list(_P)
    except Exception:
        pass
    return None


def _theta_width(th):
    """params-per-sim from a theta array: 2-D collated (N, P) -> P; 1-D shard (P,) -> P."""
    if th is None or not hasattr(th, "shape"):
        return None
    sh = tuple(th.shape)
    if len(sh) >= 2:
        return int(sh[1])
    if len(sh) == 1:
        return int(sh[0])
    return None


def classify(meta):
    """meta: dict (collated .theta.pt) or a shard-derived dict. Returns findings."""
    fmt = meta.get("format", "legacy")
    rep = _REP.get(fmt, f"unknown (format='{fmt}')")

    n_params = _theta_width(meta.get("thetas"))
    model, names = _MODEL_BY_WIDTH.get(n_params, ("unknown", None))

    notes = []
    # Prefer the live config's PARAM_NAMES (it knows CONTEXT_PARAMS, so it
    # resolves the width-7 ambiguity); fall back to listing the candidates.
    if names is None and n_params is not None:
        live = _live_param_names(n_params)
        if live is not None:
            names = live
            notes.append("params resolved from the CURRENT config "
                         "(verify the dataset was built with the same "
                         "CONTEXT_PARAMS)")
        elif n_params in _AMBIGUOUS:
            notes.append(f"theta width {n_params} is ambiguous; candidates:")
            for lbl, cand in _AMBIGUOUS[n_params]:
                notes.append(f"    {lbl}  ->  {cand}")
    if fmt in ("te_slice", "newer", "new"):
        ps = meta.get("pos_shape")
        if ps is not None and len(ps) >= 2:
            C = int(ps[-2])               # full shape (n_sims, n_windows, C, S)
            chan = {5: "simple", 6: "context"}.get(C, f"unexpected (C={C})")
            notes.append(f"continuous channels = {C}  ->  {chan}")
    if fmt == "sumstat":
        fs = meta.get("feats_shape")
        if fs is not None:
            notes.append(f"sumstats per TE = {int(fs[-1])}  (model-dependent)")

    shapes = {k: tuple(meta[k]) for k in
              ("geno_shape", "pos_shape", "cov_shape", "feats_shape", "te_mask_shape")
              if meta.get(k) is not None}
    return dict(format=fmt, representation=rep, n_params=n_params,
                model=model, param_names=names, n_sims=meta.get("n_sims"),
                notes=notes, shapes=shapes)


def load_meta(path):
    """Load collated metadata (.theta.pt) or build a meta-like dict from an .npz shard."""
    if path.endswith(".npz"):
        import numpy as np
        with np.load(path) as d:
            meta = {}
            # The shard may record its own format (05a writes format="sumstat").
            # Trust it first; fall back to structural detection below.
            if "format" in d:
                meta["format"] = str(d["format"])
            # Sumstat shards store padded `feats` (01b_reprocess) OR unpadded
            # `feats_raw` (05a benchmark chunks, which pass pad=False and let
            # 02_collate pad). Recognise BOTH, else the shard silently falls
            # through to the "legacy" default and is misreported as matrix CNN.
            if "feats" in d or "feats_raw" in d:
                meta.setdefault("format", "sumstat")
                _k = "feats" if "feats" in d else "feats_raw"
                meta["feats_shape"] = (None,) + d[_k].shape
            elif "geno" in d:
                meta.setdefault("format", "te_slice" if "te_mask" in d else "newer")
                meta["geno_shape"] = (None,) + d["geno"].shape
                if "pos" in d:
                    meta["pos_shape"] = (None,) + d["pos"].shape
                if "cov" in d:
                    meta["cov_shape"] = (None,) + d["cov"].shape
            if "theta" in d:
                meta["thetas"] = np.asarray(d["theta"])   # 1-D shard theta
            return meta

    # collated metadata: accept the base path or the .theta.pt directly
    if not path.endswith(".theta.pt"):
        cand = path + ".theta.pt"
        if os.path.exists(cand):
            path = cand
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:                      # older torch without weights_only kwarg
        return torch.load(path, map_location="cpu")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    target = sys.argv[1]
    if not (os.path.exists(target) or os.path.exists(target + ".theta.pt")):
        print(f"  not found: {target}")
        sys.exit(1)

    r = classify(load_meta(target))
    print(f"  dataset        : {target}")
    print(f"  representation : {r['representation']}   (format='{r['format']}')")
    print(f"  model          : {r['model']}   (theta width = {r['n_params']})")
    if r["param_names"]:
        print(f"  params         : {r['param_names']}")
    if r["n_sims"] is not None:
        print(f"  n_sims         : {r['n_sims']}")
    for n in r["notes"]:
        print(f"  corroboration  : {n}")
    if r["shapes"]:
        print("  shapes:")
        for k, v in r["shapes"].items():
            print(f"    {k:14s}: {v}")


if __name__ == "__main__":
    main()
