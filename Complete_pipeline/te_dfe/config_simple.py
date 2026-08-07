"""
config_simple.py — simple-model overlay (6-parameter gamma-DFE + burst).

Inherits all shared infrastructure and realism options from config_base, then
adds the simple model's parameter set, prior ranges, and continuous-channel
layout. Selected when TE_DFE_MODEL=simple (the default).

Inferred parameter vector (6):
  0 log_lambda       baseline per-bp per-gen insertion rate (log)
  1 log_gamma_shape  shape of the gamma DFE over |s| (log)
  2 log_mean_s       mean |s| of the gamma DFE (log)
  3 log10_t_start    burst start, generations ago (log10)
  4 log_duration     burst duration in generations (log)
  5 log_multiplier   burst insertion-rate multiplier, >=1x (log)

The raw gamma (shape, mean_s) is weakly identified; report the post-hoc
DFE-bin proportions (p_nn / p_w / p_strong via the gamma CDF), which are much
better identified. See prior.py.
"""
from .config_base import *          # noqa: F401,F403
import numpy as np

MODEL_NAME = "simple"

# --- Prior ranges -----------------------------------------------------------
LOG_LAMBDA_RANGE      = (np.log(5e-10), np.log(2e-9))
LOG_GAMMA_SHAPE_RANGE = (np.log(0.1),  np.log(2.0))
LOG_MEAN_S_RANGE      = (np.log(1e-4), np.log(0.1))
# (burst ranges inherited from base: LOG10_T_START_RANGE etc.)

# --- Parameter bookkeeping --------------------------------------------------
N_PARAMS    = 6
PARAM_NAMES = [
    "log_lambda", "log_gamma_shape", "log_mean_s",
    "log10_t_start", "log_duration", "log_multiplier",
]
PARAM_LABELS = [
    "log lambda (baseline insertion rate)",
    "log gamma shape", "log mean |s|",
    "log10 t_start (gen ago)", "log duration (gen)", "log multiplier",
]

# --- Continuous channels ----------------------------------------------------
# Base 4 (phys, gen, TE-dens, SNP-dens) plus an optional local-recomb channel
# (ch4). Enabling it changes the input width and invalidates checkpoints.
USE_RECOMB_CHANNEL  = True
N_RECOMB_CHANNELS   = 1 if USE_RECOMB_CHANNEL else 0
N_CONTINUOUS_CHANNELS = (N_POSITION_CHANNELS + N_DENSITY_CHANNELS
                         + N_RECOMB_CHANNELS)
N_CNN_INPUT_CHANNELS = N_STATE_CHANNELS + N_CONTINUOUS_CHANNELS
