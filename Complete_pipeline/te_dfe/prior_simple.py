"""
prior.py — v4 PURE-GAMMA prior, 6 params.

Network parameter space == simulator parameter space (no internal/external
distinction). The DFE is a pure gamma on |s|:
    |s| ~ Gamma(shape, scale)    with mean(|s|) = shape * scale = mean_s
No point mass at s=0, no mixture. All TEs are deleterious to some degree
(consistent with most modern DFE inference, e.g. polyDFE/Tataru).

Inferred parameters (6):
    [log_lambda, log_gamma_shape, log_mean_s,
     log10_t_start, log_duration, log_multiplier]

Derived quantities (computed at reporting time via the gamma CDF):
    p_nn     = P(|2*Ne*s| < SEL_THRESHOLD_NN)
    p_w      = P(SEL_THRESHOLD_NN <= |2*Ne*s| < SEL_THRESHOLD_STRONG)
    p_strong = P(|2*Ne*s| >= SEL_THRESHOLD_STRONG)

See compute_binned_props() for the helper to push posterior samples
through to derived quantities.
"""
import numpy as np
from scipy.stats import gamma as gamma_dist

from .config import (
    LOG_LAMBDA_RANGE,
    LOG_GAMMA_SHAPE_RANGE, LOG_MEAN_S_RANGE,
    LOG10_T_START_RANGE, LOG_DURATION_RANGE, LOG_MULTIPLIER_RANGE,
    SEL_THRESHOLD_NN, SEL_THRESHOLD_STRONG,
    N_PARAMS,
)


def sample_s_values(n_te, gamma_shape, mean_s, rng):
    """Draw n_te selection coefficients from the pure-gamma DFE.
    Returns array of negative s values (deleterious convention)."""
    if n_te == 0:
        return np.array([], dtype=float)
    scale = mean_s / max(gamma_shape, 1e-6)
    s_vals = -rng.gamma(gamma_shape, scale, size=n_te)
    return np.clip(s_vals, -1.0, 0.0)


def compute_binned_props(log_gamma_shape, log_mean_s, ne_harmonic):
    """
    Compute (p_nn, p_w, p_strong) from gamma DFE parameters.
    All inputs can be scalars or vectors (numpy broadcasts).

    Returns: (p_nn, p_w, p_strong) arrays of the same shape as inputs.
    """
    shape = np.exp(np.asarray(log_gamma_shape))
    mean_s = np.exp(np.asarray(log_mean_s))
    scale = mean_s / np.maximum(shape, 1e-12)

    s_thr_nn     = SEL_THRESHOLD_NN     / (2.0 * ne_harmonic)
    s_thr_strong = SEL_THRESHOLD_STRONG / (2.0 * ne_harmonic)

    cdf_nn     = gamma_dist.cdf(s_thr_nn,     shape, scale=scale)
    cdf_strong = gamma_dist.cdf(s_thr_strong, shape, scale=scale)

    p_nn     = cdf_nn
    p_w      = cdf_strong - cdf_nn
    p_strong = 1.0 - cdf_strong
    return p_nn, p_w, p_strong


class DFEBurstPrior:
    """
    v4 pure-gamma prior. The network's parameters are exactly the
    simulator's parameters.

    Theta (6 params):
      [log_lambda, log_gamma_shape, log_mean_s,
       log10_t_start, log_duration, log_multiplier]
    """

    @staticmethod
    def sample(rng, ne_harmonic=None):
        """Sample theta from the prior. Returns 6-element float32.
        ne_harmonic is unused (kept for API compatibility)."""
        theta = np.array([
            rng.uniform(*LOG_LAMBDA_RANGE),
            rng.uniform(*LOG_GAMMA_SHAPE_RANGE),
            rng.uniform(*LOG_MEAN_S_RANGE),
            rng.uniform(*LOG10_T_START_RANGE),
            rng.uniform(*LOG_DURATION_RANGE),
            rng.uniform(*LOG_MULTIPLIER_RANGE),
        ], dtype=np.float32)
        # Overwrite any fixed parameters with their constants (no-op if none),
        # so fixed params are truly held constant across all simulations.
        from .config import apply_fixed_params
        return apply_fixed_params(theta)

    @staticmethod
    def unpack(theta):
        """Returns (lam, gamma_shape, mean_s, t_start, dur, mult)."""
        if len(theta) != N_PARAMS:
            raise ValueError(
                f"Expected theta of length {N_PARAMS}, got {len(theta)}."
            )
        return (
            float(np.exp(float(theta[0]))),   # lambda
            float(np.exp(float(theta[1]))),   # gamma_shape
            float(np.exp(float(theta[2]))),   # mean_s
            int(round(10.0 ** float(theta[3]))),   # t_start
            int(round(float(np.exp(float(theta[4]))))),   # duration
            float(np.exp(float(theta[5]))),   # multiplier
        )

    @staticmethod
    def to_sbi_prior():
        """Box-uniform prior over theta, used by sbi."""
        import torch
        from sbi.utils import BoxUniform
        low = np.array([
            LOG_LAMBDA_RANGE[0],
            LOG_GAMMA_SHAPE_RANGE[0],
            LOG_MEAN_S_RANGE[0],
            LOG10_T_START_RANGE[0],
            LOG_DURATION_RANGE[0],
            LOG_MULTIPLIER_RANGE[0],
        ], dtype=np.float32)
        high = np.array([
            LOG_LAMBDA_RANGE[1],
            LOG_GAMMA_SHAPE_RANGE[1],
            LOG_MEAN_S_RANGE[1],
            LOG10_T_START_RANGE[1],
            LOG_DURATION_RANGE[1],
            LOG_MULTIPLIER_RANGE[1],
        ], dtype=np.float32)
        return BoxUniform(low=torch.tensor(low), high=torch.tensor(high))
