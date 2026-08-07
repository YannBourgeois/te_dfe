"""
config.py — model dispatcher.

Selects the simple or context configuration based on the TE_DFE_MODEL
environment variable, then re-exports everything from the chosen overlay so
that `from te_dfe.config import N_PARAMS, PARAM_NAMES, ...` works unchanged
across the whole pipeline.

Model selection (CLI overrides config default):
  * Each pipeline script takes --model {simple,context} and sets
    os.environ["TE_DFE_MODEL"] BEFORE importing te_dfe (see the scripts'
    select_model() helper). This is why the env var, not a Python global,
    drives the choice: it must be fixed before any te_dfe import triggers this
    dispatch.
  * If unset, defaults to DEFAULT_MODEL below.

To read the active model at runtime: te_dfe.config.MODEL_NAME.
"""
import os as _os

DEFAULT_MODEL = "simple"
_MODEL = _os.environ.get("TE_DFE_MODEL", DEFAULT_MODEL).strip().lower()

if _MODEL not in ("simple", "context"):
    raise ValueError(
        f"TE_DFE_MODEL={_MODEL!r} is not valid; use 'simple' or 'context'.")

if _MODEL == "simple":
    from .config_simple import *          # noqa: F401,F403
    from . import config_simple as _active
else:
    from .config_context import *         # noqa: F401,F403
    from . import config_context as _active

# Re-export underscore-prefixed names too (e.g. the context model's
# _ACTIVE_CONTEXT / _CONTEXT_LABELS), which `import *` skips. The simulator and
# prior reference these for the context model.
for _name in dir(_active):
    if _name.startswith("_") and not _name.startswith("__"):
        globals()[_name] = getattr(_active, _name)

MODEL_NAME = _MODEL

# Resolve the inferred-parameter subset (marginalize-out). INFER_INDICES are
# columns into the full theta vector to KEEP, in canonical PARAM_NAMES order;
# INFER_NAMES are the matching names. When INFER_PARAMS is None, all are kept.
# Scripts slice theta with INFER_INDICES at train/infer time and record
# INFER_NAMES in training_meta so the column mapping is stable.
def _resolve_infer_subset():
    full = list(PARAM_NAMES)
    sel  = globals().get("INFER_PARAMS", None)
    fix  = dict(globals().get("FIX_PARAMS", {}) or {})

    # Validate fixed names + no overlap with explicitly-inferred names.
    bad_fix = [s for s in fix if s not in full]
    if bad_fix:
        raise ValueError(
            f"FIX_PARAMS {bad_fix} not in this model's PARAM_NAMES {full}.")
    if sel:
        bad = [s for s in sel if s not in full]
        if bad:
            raise ValueError(
                f"INFER_PARAMS {bad} not in this model's PARAM_NAMES {full}.")
        clash = [s for s in sel if s in fix]
        if clash:
            raise ValueError(
                f"cannot both fix and infer the same parameter(s): {clash}.")

    # Fixed-parameter index/value vectors (positions into the full theta).
    fix_idx = [full.index(n) for n in full if n in fix]   # canonical order
    fix_val = [float(fix[full[i]]) for i in fix_idx]

    # Inferred set = (explicit INFER_PARAMS or all) MINUS fixed params.
    base_keep = list(sel) if sel else list(full)
    keep = [n for n in full if n in set(base_keep) and n not in fix]
    idx  = [full.index(n) for n in keep]
    return idx, keep, fix_idx, fix_val

INFER_INDICES, INFER_NAMES, FIX_INDICES, FIX_VALUES = _resolve_infer_subset()
N_INFER_PARAMS = len(INFER_INDICES)


def apply_fixed_params(theta):
    """
    Overwrite the fixed-parameter entries of a sampled full theta vector with
    their constants (native/transformed units). No-op when FIX_PARAMS is empty.
    Called by the prior's sample() so fixed parameters are truly constant in the
    simulations.
    """
    if not FIX_INDICES:
        return theta
    import numpy as _np
    theta = _np.asarray(theta, dtype=_np.float32).copy()
    for i, v in zip(FIX_INDICES, FIX_VALUES):
        theta[i] = v
    return theta
