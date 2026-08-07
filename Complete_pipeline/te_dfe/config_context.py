"""
config_context.py — context-model overlay (context-dependent DFE + insertion
preference). Selected when TE_DFE_MODEL=context.

Inherits shared infrastructure and realism from config_base, then adds the
context model's parameter set, prior ranges, and the two extra continuous
channels (dist-to-gene, log-recomb) that drive the context regressions.

Generative model (recombination-only default):
  s_local(p)      = exp(log_s_global + alpha_r * feature_r(p))
  lambda_local(p) = lambda_global * exp(gamma_r * feature_r(p))
where feature_r(p) is the normalized log local recombination rate. The
simple-model limit is alpha_* = gamma_* = 0.

CONTEXT_PARAMS selects which context coefficients are FREE (inferred); any
omitted coefficient is pinned to 0 in the simulator and dropped from theta, so
the flow's parameter space stays non-degenerate.
"""
from .config_base import *          # noqa: F401,F403
import numpy as np

MODEL_NAME = "context"

# --- Prior ranges -----------------------------------------------------------
LOG_LAMBDA_RANGE   = (np.log(5e-10), np.log(2e-9))
LOG_S_GLOBAL_RANGE = (np.log(3e-4), np.log(0.1))
ALPHA_D_RANGE      = (-2.0, 2.0)
ALPHA_R_RANGE      = (-2.0, 2.0)
GAMMA_D_RANGE      = (-2.0, 2.0)
GAMMA_R_RANGE      = (-2.0, 2.0)
# (burst ranges inherited from base.)

# --- Parameter bookkeeping (configurable context set) -----------------------
#   Recombination-only (default): ["alpha_r","gamma_r"]            -> 7 params
#   Distance-only:                ["alpha_d","gamma_d"]            -> 7 params
#   Full:                         ["alpha_d","alpha_r","gamma_d","gamma_r"] -> 9
#   Constant-DFE (no context):    []                               -> 5 params
CONTEXT_PARAMS = ["alpha_r", "gamma_r"]

_ALL_CONTEXT_ORDER = ["alpha_d", "alpha_r", "gamma_d", "gamma_r"]
_ACTIVE_CONTEXT = [c for c in _ALL_CONTEXT_ORDER if c in CONTEXT_PARAMS]
_CONTEXT_LABELS = {
    "alpha_d": "alpha_d (selection slope on log dist-to-gene)",
    "alpha_r": "alpha_r (selection slope on log local recomb)",
    "gamma_d": "gamma_d (insertion preference on log dist-to-gene)",
    "gamma_r": "gamma_r (insertion preference on log local recomb)",
}
PARAM_NAMES = (
    ["log_lambda", "log_s_global"]
    + list(_ACTIVE_CONTEXT)
    + ["log10_t_start", "log_duration", "log_multiplier"]
)
PARAM_LABELS = (
    ["log lambda (baseline insertion rate at zero context)",
     "log s_global (baseline |s| at zero context)"]
    + [_CONTEXT_LABELS[c] for c in _ACTIVE_CONTEXT]
    + ["log10 t_start (gen ago)", "log duration (gen)", "log multiplier"]
)
N_PARAMS = len(PARAM_NAMES)

# --- Continuous channels ----------------------------------------------------
# Base 4 + 2 context channels: ch4 = dist-to-gene, ch5 = log-recomb (feature_r).
# These are ALWAYS present in the context model (feature_r drives the context
# regression even when a coefficient is pinned to 0).
N_CONTEXT_CHANNELS    = 2
N_CONTINUOUS_CHANNELS = (N_POSITION_CHANNELS + N_DENSITY_CHANNELS
                         + N_CONTEXT_CHANNELS)
N_CNN_INPUT_CHANNELS = N_STATE_CHANNELS + N_CONTINUOUS_CHANNELS
