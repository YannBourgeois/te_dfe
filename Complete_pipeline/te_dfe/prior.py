"""
prior.py — model dispatcher for the prior + DFE-bin helpers.

Re-exports the simple or context prior based on the active model
(te_dfe.config.MODEL_NAME, itself set from TE_DFE_MODEL). Both overlays define
DFEBurstPrior (with model-appropriate sample/unpack/to_sbi_prior) plus their
DFE-bin helpers; this dispatch makes `from te_dfe.prior import DFEBurstPrior`
resolve to the right one across the pipeline.
"""
from .config import MODEL_NAME as _MODEL

if _MODEL == "context":
    from .prior_context import *      # noqa: F401,F403
    from . import prior_context as _active
else:
    from .prior_simple import *       # noqa: F401,F403
    from . import prior_simple as _active

for _name in dir(_active):
    if _name.startswith("_") and not _name.startswith("__"):
        globals()[_name] = getattr(_active, _name)


def sbi_prior_subset(indices=None):
    """
    Return the sbi prior restricted to the inferred-parameter subset (the
    marginal over the kept dims). For a BoxUniform / independent prior this is
    just slicing the per-dim bounds — exact, since the prior factorizes. If
    indices is None, uses config.INFER_INDICES. Returns the full prior when all
    dims are kept.
    """
    import torch
    full = DFEBurstPrior.to_sbi_prior()
    if indices is None:
        from .config import INFER_INDICES
        indices = list(INFER_INDICES)
    try:
        n = int(full.low.shape[0])
    except Exception:
        return full                      # non-box prior: return as-is
    if list(indices) == list(range(n)):
        return full
    from sbi.utils import BoxUniform
    idx = torch.as_tensor(indices, dtype=torch.long)
    return BoxUniform(low=full.low[idx], high=full.high[idx])
