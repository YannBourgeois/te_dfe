"""
te_dfe/snpe_refine.py — SNPE refinement helpers for the simple model.

The simple model has two parameter spaces:
  - Internal (7 params): drives the simulator. The 3 DFE internals
    (logit_pi, log_shape, log_mean_s) are gamma+mixture.
  - External (7 params): what the network infers. The 3 DFE externals
    (p_nn, p_w, log_mean_s_overall) are binned summaries.

The transform internal -> external is deterministic and not injective:
multiple internals map to the same external. For SNPE refinement, we
sample new internals via *importance sampling*:

  1. Draw many candidate internals from the broad internal prior.
  2. Transform each to external.
  3. Compute importance weight:
       w = p_posterior(external | x_obs) * |J(internal)| / p_prior(internal)
     where J is the Jacobian of the internal->external transform.
  4. Resample internals weighted by w; use these as simulator inputs.

The Jacobian is computed numerically via finite differences over the
3-D DFE block (the other 4 dims are identity, so their Jacobian
contribution is 1). This avoids the need to invert the transform.
"""
import numpy as np
from scipy.special import expit
from scipy.stats import gamma as gamma_dist

from .config import (
    LOG_LAMBDA_RANGE,
    LOGIT_PI_NEUTRAL_RANGE, LOG_GAMMA_SHAPE_RANGE, LOG_MEAN_S_RANGE,
    LOG10_T_START_RANGE, LOG_DURATION_RANGE, LOG_MULTIPLIER_RANGE,
    SEL_THRESHOLD_NN, SEL_THRESHOLD_STRONG,
)
from .prior import compute_derived_dfe


# Internal parameter order, used for arrays:
INTERNAL_PARAM_NAMES = [
    "log_lambda",
    "logit_pi",
    "log_shape",
    "log_mean_s",
    "log10_t_start",
    "log_duration",
    "log_mult",
]

# Bounds of each internal param (corresponding order)
INTERNAL_BOUNDS = [
    LOG_LAMBDA_RANGE,
    LOGIT_PI_NEUTRAL_RANGE,
    LOG_GAMMA_SHAPE_RANGE,
    LOG_MEAN_S_RANGE,
    LOG10_T_START_RANGE,
    LOG_DURATION_RANGE,
    LOG_MULTIPLIER_RANGE,
]

# Index of DFE block within internal space (the only nontrivial transform)
DFE_INTERNAL_IDX = [1, 2, 3]   # logit_pi, log_shape, log_mean_s


# ---------------------------------------------------------------------------
# Sampling and density evaluation in internal space
# ---------------------------------------------------------------------------
def sample_internal_prior(rng, n):
    """Draw n samples from the uniform internal prior (n, 7)."""
    out = np.empty((n, len(INTERNAL_BOUNDS)), dtype=np.float64)
    for i, (lo, hi) in enumerate(INTERNAL_BOUNDS):
        out[:, i] = rng.uniform(lo, hi, size=n)
    return out


def log_prob_internal_prior(theta_int):
    """
    Log-density of theta_int (shape (n, 7)) under the uniform internal prior.

    For a uniform box prior, the log-density is constant over the support
    and -inf outside. We return the constant (or -inf for outside-support
    rows).
    """
    theta_int = np.atleast_2d(theta_int)
    n = theta_int.shape[0]
    log_p = np.zeros(n, dtype=np.float64)
    for i, (lo, hi) in enumerate(INTERNAL_BOUNDS):
        outside = (theta_int[:, i] < lo) | (theta_int[:, i] > hi)
        log_p[outside] = -np.inf
    # The uniform density is 1/volume; for log-prob purposes the constant
    # cancels in importance weights. Just use 0 for inside-support rows.
    return log_p


# ---------------------------------------------------------------------------
# Internal -> external transform (vectorized)
# ---------------------------------------------------------------------------
def internal_to_external_batch(theta_int, ne_harmonic):
    """
    Apply the internal -> external transform to a batch of internals.

    Vectorized via scipy's batched gamma CDF; O(n) speed.

    Input:  theta_int shape (n, 7)
    Output: theta_ext shape (n, 7)
    """
    theta_int = np.atleast_2d(theta_int)
    n = theta_int.shape[0]
    out = np.empty((n, 7), dtype=np.float64)

    out[:, 0] = theta_int[:, 0]  # log_lambda

    logit_pi   = theta_int[:, DFE_INTERNAL_IDX[0]]
    log_shape  = theta_int[:, DFE_INTERNAL_IDX[1]]
    log_mean_s = theta_int[:, DFE_INTERNAL_IDX[2]]
    p_nn, p_w, lms = _dfe_transform_vec(logit_pi, log_shape, log_mean_s, ne_harmonic)
    out[:, 1] = p_nn
    out[:, 2] = p_w
    out[:, 3] = lms

    out[:, 4] = theta_int[:, 4]  # log10_t_start
    out[:, 5] = theta_int[:, 5]  # log_duration
    out[:, 6] = theta_int[:, 6]  # log_mult

    return out


# ---------------------------------------------------------------------------
# Jacobian of the internal -> external transform
# ---------------------------------------------------------------------------
def _dfe_transform_vec(logit_pi, log_shape, log_mean_s, ne_harmonic):
    """
    Vectorized DFE transform: arrays of (logit_pi, log_shape, log_mean_s)
    -> (p_nn, p_w, log_mean_s_overall), each shape (n,).
    """
    logit_pi   = np.asarray(logit_pi,   dtype=np.float64)
    log_shape  = np.asarray(log_shape,  dtype=np.float64)
    log_mean_s = np.asarray(log_mean_s, dtype=np.float64)

    pi_neutral  = expit(logit_pi)
    gamma_shape = np.exp(log_shape)
    mean_s      = np.exp(log_mean_s)

    s_thr_nn     = SEL_THRESHOLD_NN     / (2.0 * ne_harmonic)
    s_thr_strong = SEL_THRESHOLD_STRONG / (2.0 * ne_harmonic)
    scale = mean_s / np.maximum(gamma_shape, 1e-12)

    # Vectorized gamma CDF (scipy supports broadcasting)
    cdf_nn     = gamma_dist.cdf(s_thr_nn,     gamma_shape, scale=scale)
    cdf_strong = gamma_dist.cdf(s_thr_strong, gamma_shape, scale=scale)

    p_nn   = pi_neutral + (1.0 - pi_neutral) * cdf_nn
    p_w    = (1.0 - pi_neutral) * (cdf_strong - cdf_nn)
    mean_s_overall = (1.0 - pi_neutral) * mean_s
    log_mean_s_overall = np.log(np.maximum(mean_s_overall, 1e-12))

    return p_nn, p_w, log_mean_s_overall


def log_abs_det_jacobian(theta_int, ne_harmonic, eps=1e-5):
    """
    Vectorized log|det J| of the internal->external transform.
    Identity dims contribute 1; only the 3D DFE block matters.
    Uses 6 vectorized DFE-transform calls (one per finite-difference
    side) instead of per-row loops, so this scales to large N.
    """
    theta_int = np.atleast_2d(theta_int)
    n = theta_int.shape[0]

    logit_pi   = theta_int[:, DFE_INTERNAL_IDX[0]]
    log_shape  = theta_int[:, DFE_INTERNAL_IDX[1]]
    log_mean_s = theta_int[:, DFE_INTERNAL_IDX[2]]

    # Build the 3x3 Jacobian via central differences.
    # The 3 input dims are (logit_pi, log_shape, log_mean_s) and the 3
    # output dims are (p_nn, p_w, log_mean_s_overall). We'll assemble
    # J[k, output_dim, input_dim] for k in [0, n).
    J = np.zeros((n, 3, 3), dtype=np.float64)

    # d/d_logit_pi
    p_nn_p, p_w_p, lms_p = _dfe_transform_vec(logit_pi + eps, log_shape, log_mean_s, ne_harmonic)
    p_nn_m, p_w_m, lms_m = _dfe_transform_vec(logit_pi - eps, log_shape, log_mean_s, ne_harmonic)
    J[:, 0, 0] = (p_nn_p - p_nn_m) / (2 * eps)
    J[:, 1, 0] = (p_w_p  - p_w_m ) / (2 * eps)
    J[:, 2, 0] = (lms_p  - lms_m ) / (2 * eps)

    # d/d_log_shape
    p_nn_p, p_w_p, lms_p = _dfe_transform_vec(logit_pi, log_shape + eps, log_mean_s, ne_harmonic)
    p_nn_m, p_w_m, lms_m = _dfe_transform_vec(logit_pi, log_shape - eps, log_mean_s, ne_harmonic)
    J[:, 0, 1] = (p_nn_p - p_nn_m) / (2 * eps)
    J[:, 1, 1] = (p_w_p  - p_w_m ) / (2 * eps)
    J[:, 2, 1] = (lms_p  - lms_m ) / (2 * eps)

    # d/d_log_mean_s
    p_nn_p, p_w_p, lms_p = _dfe_transform_vec(logit_pi, log_shape, log_mean_s + eps, ne_harmonic)
    p_nn_m, p_w_m, lms_m = _dfe_transform_vec(logit_pi, log_shape, log_mean_s - eps, ne_harmonic)
    J[:, 0, 2] = (p_nn_p - p_nn_m) / (2 * eps)
    J[:, 1, 2] = (p_w_p  - p_w_m ) / (2 * eps)
    J[:, 2, 2] = (lms_p  - lms_m ) / (2 * eps)

    # Vectorized log-abs-det via slogdet (numpy supports stacked arrays
    # of (..., 3, 3))
    log_abs_det = np.full(n, -np.inf, dtype=np.float64)
    valid = np.all(np.isfinite(J.reshape(n, -1)), axis=1)
    if valid.any():
        signs, ld = np.linalg.slogdet(J[valid])
        log_abs_det_valid = np.where(
            (signs != 0) & np.isfinite(ld), ld, -np.inf,
        )
        log_abs_det[valid] = log_abs_det_valid
    return log_abs_det


def _dfe_transform_only(logit_pi, log_shape, log_mean_s, ne_harmonic):
    """Return [p_nn, p_w, log_mean_s_overall] given 3 internals + Ne."""
    pi_neutral  = float(expit(logit_pi))
    gamma_shape = float(np.exp(log_shape))
    mean_s      = float(np.exp(log_mean_s))
    try:
        p_nn, p_w, _, mean_s_overall = compute_derived_dfe(
            pi_neutral, gamma_shape, mean_s, ne_harmonic,
        )
    except (ValueError, FloatingPointError):
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    return np.array(
        [p_nn, p_w, np.log(max(mean_s_overall, 1e-12))], dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Importance sampling for SNPE refinement
# ---------------------------------------------------------------------------
def sample_internals_for_round(
    n_sims_needed,
    posterior_external,
    x_obs_flat,                  # already-flattened observation tensor
    ne_harmonic,
    rng,
    n_candidates=None,
    oversample_factor=20,
    posterior_log_prob_fn=None,
    verbose=True,
    target_ess_normalized=0.01,
    max_candidates=2_000_000,
):
    """
    Sample n_sims_needed internal theta vectors via importance sampling
    against the current external posterior.

    Adapts the candidate pool size if ESS is below `target_ess_normalized`,
    doubling up to `max_candidates`. This keeps the importance sample
    statistically meaningful even when the posterior is sharper than the
    prior.

    Returns
    -------
    sampled_internals : (n_sims_needed, 7) np.float64 array
    diagnostics       : dict with keys n_candidates, n_finite, ess_raw,
                         ess_normalized (= ess_raw / n_candidates),
                         weights_summary
    """
    if n_candidates is None:
        n_candidates = max(n_sims_needed * oversample_factor, 1000)

    attempt = 0
    while True:
        attempt += 1
        sampled, diag = _one_importance_attempt(
            n_sims_needed=n_sims_needed,
            posterior_external=posterior_external,
            x_obs_flat=x_obs_flat,
            ne_harmonic=ne_harmonic,
            rng=rng,
            n_candidates=n_candidates,
            posterior_log_prob_fn=posterior_log_prob_fn,
            verbose=verbose,
        )
        if diag["ess_normalized"] >= target_ess_normalized:
            diag["adaptive_attempts"] = attempt
            return sampled, diag
        if n_candidates >= max_candidates:
            if verbose:
                print(f"  [snpe sample] WARNING: hit max_candidates "
                      f"({max_candidates}) with ESS still below target "
                      f"({diag['ess_normalized']:.4f} < {target_ess_normalized}). "
                      f"Continuing with the last batch.")
            diag["adaptive_attempts"] = attempt
            return sampled, diag
        if verbose:
            print(f"  [snpe sample] ESS too low "
                  f"({diag['ess_normalized']:.4f} < {target_ess_normalized}); "
                  f"doubling candidate pool to {2*n_candidates}.")
        n_candidates = min(2 * n_candidates, max_candidates)


def _one_importance_attempt(
    n_sims_needed,
    posterior_external,
    x_obs_flat,
    ne_harmonic,
    rng,
    n_candidates,
    posterior_log_prob_fn,
    verbose,
):
    """Single importance-sampling pass with a fixed candidate count."""

    # 1. Sample candidate internals
    candidates = sample_internal_prior(rng, n_candidates)

    # 2. Transform to external
    ext = internal_to_external_batch(candidates, ne_harmonic)

    # 3. Filter out candidates with NaN-external (numerical edge cases)
    finite_ext = np.all(np.isfinite(ext), axis=1)
    # And those with ext outside the external posterior's support
    # (external p_nn, p_w must be in [0,1])
    in_support = finite_ext & (
        (ext[:, 1] >= 0) & (ext[:, 1] <= 1) &
        (ext[:, 2] >= 0) & (ext[:, 2] <= 1) &
        (ext[:, 1] + ext[:, 2] <= 1.0001)
    )
    # Also filter on log_mean_s_overall range to be safe
    in_support &= (ext[:, 3] >= LOG_MEAN_S_RANGE[0] - 5) & \
                  (ext[:, 3] <= LOG_MEAN_S_RANGE[1] + 1)

    # Default posterior log_prob caller, unless user-overridden
    if posterior_log_prob_fn is None:
        import torch
        def _log_prob(ext_batch):
            ext_t = torch.from_numpy(ext_batch.astype(np.float32))
            x_t   = torch.from_numpy(x_obs_flat.astype(np.float32))
            if x_t.dim() == 1:
                x_t = x_t.unsqueeze(0)
            with torch.no_grad():
                lp = posterior_external.log_prob(ext_t, x=x_t)
            return lp.cpu().numpy()
        posterior_log_prob_fn = _log_prob

    # 4. Evaluate log_p_posterior(ext) at valid candidates only
    log_p_post = np.full(n_candidates, -np.inf, dtype=np.float64)
    if in_support.any():
        ext_valid = ext[in_support].astype(np.float32)
        log_p_post_valid = posterior_log_prob_fn(ext_valid)
        log_p_post[in_support] = log_p_post_valid

    # 5. Log-Jacobian at valid candidates
    log_J = np.full(n_candidates, -np.inf, dtype=np.float64)
    if in_support.any():
        log_J_valid = log_abs_det_jacobian(
            candidates[in_support], ne_harmonic,
        )
        log_J[in_support] = log_J_valid

    # 6. Log-prior on internals: constant for uniform box prior; just
    # filter for in-support
    log_p_int = log_prob_internal_prior(candidates)
    # For uniform box, the constant log_volume cancels in normalized
    # weights, so we can set to 0 for in-support rows.

    # 7. Importance weights: w = p_post(ext) * |J| / p_prior_int
    #    log w = log_p_post + log_J - log_p_int (which is 0 for uniform)
    log_w = log_p_post + log_J - log_p_int

    finite_w = np.isfinite(log_w)
    n_finite = int(finite_w.sum())
    if n_finite == 0:
        raise RuntimeError(
            "All candidate internals had non-finite importance weights "
            "(posterior, Jacobian, or prior support failed). Increase "
            "oversample_factor or check the posterior."
        )

    # Stable normalization
    log_w[~finite_w] = -np.inf
    log_w_finite = log_w[finite_w]
    log_w_finite -= log_w_finite.max()
    w_finite = np.exp(log_w_finite)
    w_full = np.zeros(n_candidates, dtype=np.float64)
    w_full[finite_w] = w_finite
    w_sum = w_full.sum()
    if w_sum <= 0:
        raise RuntimeError("Importance weights summed to zero; aborting.")
    w_full /= w_sum

    # ESS diagnostic
    ess_raw = 1.0 / (w_full**2).sum()
    ess_normalized = ess_raw / n_candidates

    if verbose:
        print(f"  [snpe sample] n_candidates={n_candidates}, "
              f"n_finite={n_finite} ({100.0*n_finite/n_candidates:.1f}%), "
              f"ESS={ess_raw:.0f} ({100.0*ess_normalized:.1f}% normalized)")

    if ess_normalized < 0.005:
        print(f"  [snpe sample] WARNING: ESS very low ({ess_normalized:.4f}). "
              f"Posterior may be too concentrated for the candidate cloud. "
              f"Consider increasing oversample_factor or adding rounds.")

    # 8. Resample (with replacement; standard SNPE)
    idx = rng.choice(n_candidates, size=n_sims_needed,
                      replace=True, p=w_full)
    sampled = candidates[idx]

    diag = {
        "n_candidates": int(n_candidates),
        "n_finite":     int(n_finite),
        "ess_raw":      float(ess_raw),
        "ess_normalized": float(ess_normalized),
        "weights_max":  float(w_full.max()),
        "weights_min_nonzero": float(w_full[w_full > 0].min()) if (w_full > 0).any() else 0.0,
    }
    return sampled, diag
