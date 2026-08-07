"""
prior.py — 9-parameter joint prior for the context-dependent DFE +
insertion preference + burst model (Bundle 2).

External (inferred) theta (9 params):
    [log_lambda, log_s_global, alpha_d, alpha_r,
     gamma_d, gamma_r,
     log10_t_start, log_duration, log_multiplier]

Per-locus DFE: all TEs deleterious, with
    s_local(p) = exp(log_s_global + alpha_d * feature_d(p) + alpha_r * feature_r(p))

Per-locus insertion rate:
    lambda_local(p) = lambda_global * exp(gamma_d * feature_d(p)
                                          + gamma_r * feature_r(p))

When alpha_d = alpha_r = gamma_d = gamma_r = 0, the model collapses to
the simple constant-rate, constant-s model.

There is no separate internal/external parameterization — the network
infers exactly the parameters that drive the simulator. This is a
simpler design than the gamma+mixture model in Bundle 1.
"""
import numpy as np

from .config import (
    LOG_LAMBDA_RANGE, LOG_S_GLOBAL_RANGE,
    ALPHA_D_RANGE, ALPHA_R_RANGE,
    GAMMA_D_RANGE, GAMMA_R_RANGE,
    LOG10_T_START_RANGE, LOG_DURATION_RANGE, LOG_MULTIPLIER_RANGE,
    N_PARAMS, PARAM_NAMES, _ACTIVE_CONTEXT,
)

# Per-coefficient prior ranges, looked up by name so the active context set
# (config.CONTEXT_PARAMS) drives sampling/unpacking without index juggling.
_CONTEXT_RANGE = {
    "alpha_d": ALPHA_D_RANGE,
    "alpha_r": ALPHA_R_RANGE,
    "gamma_d": GAMMA_D_RANGE,
    "gamma_r": GAMMA_R_RANGE,
}


def compute_s_local(s_global, alpha_d, alpha_r, feature_d, feature_r):
    """
    Compute the local selection coefficient at a position with given
    context features. Returns positive s_local; caller should negate for
    SLiM (deleterious).

    All inputs are scalars or numpy arrays (broadcastable).
    """
    log_s = (np.log(s_global)
             + alpha_d * np.asarray(feature_d)
             + alpha_r * np.asarray(feature_r))
    return np.exp(log_s)


def compute_lambda_local(lambda_global, gamma_d, gamma_r,
                          feature_d, feature_r):
    """
    Compute the local insertion rate at a position. Same form as s_local
    but for the rate.
    """
    log_lam = (np.log(lambda_global)
               + gamma_d * np.asarray(feature_d)
               + gamma_r * np.asarray(feature_r))
    return np.exp(log_lam)


class DFEBurstPrior:
    """
    9-parameter prior. The inferred parameters are exactly the simulator
    parameters; no separate internal/external distinction.

    Bundle 2 parameter vector:
        [log_lambda, log_s_global, alpha_d, alpha_r,
         gamma_d, gamma_r,
         log10_t_start, log_duration, log_multiplier]
    """

    @staticmethod
    def sample(rng, ne_harmonic=None):
        """
        Sample theta from the joint uniform-on-each-dim prior.

        The parameter vector is:
            [log_lambda, log_s_global, <active context coefs>,
             log10_t_start, log_duration, log_multiplier]
        where <active context coefs> follows config.CONTEXT_PARAMS in the
        canonical order (alpha_d, alpha_r, gamma_d, gamma_r). Omitted
        coefficients are not sampled (they are fixed at 0.0 in unpack).

        The ne_harmonic argument is accepted for API compatibility and unused.

        Returns N_PARAMS-element float32 array.
        """
        vals = [
            rng.uniform(*LOG_LAMBDA_RANGE),       # log_lambda
            rng.uniform(*LOG_S_GLOBAL_RANGE),     # log_s_global
        ]
        for c in _ACTIVE_CONTEXT:
            vals.append(rng.uniform(*_CONTEXT_RANGE[c]))
        vals += [
            rng.uniform(*LOG10_T_START_RANGE),    # log10_t_start
            rng.uniform(*LOG_DURATION_RANGE),     # log_duration
            rng.uniform(*LOG_MULTIPLIER_RANGE),   # log_multiplier
        ]
        theta = np.array(vals, dtype=np.float32)
        from .config import apply_fixed_params
        return apply_fixed_params(theta)

    @staticmethod
    def unpack(theta):
        """
        Unpack theta into interpretable scalar values. Context coefficients
        not in config.CONTEXT_PARAMS are returned as 0.0 (axis disabled), so
        the simulator and SLiM receive all four coefficients regardless of
        which are inferred.

        Returns dict with keys:
          lambda_global, s_global, alpha_d, alpha_r, gamma_d, gamma_r,
          t_start (gen ago, int), duration (gen, int), mult (linear).
        """
        if len(theta) != N_PARAMS:
            raise ValueError(
                f"Expected theta of length {N_PARAMS}, got {len(theta)}."
            )
        # Globals are always the first two; burst is always the last three.
        out = {
            "lambda_global": float(np.exp(theta[0])),
            "s_global":      float(np.exp(theta[1])),
            "alpha_d":       0.0,
            "alpha_r":       0.0,
            "gamma_d":       0.0,
            "gamma_r":       0.0,
        }
        # Active context coefs sit between the globals and the burst block.
        for i, c in enumerate(_ACTIVE_CONTEXT):
            out[c] = float(theta[2 + i])
        burst0 = 2 + len(_ACTIVE_CONTEXT)
        out["t_start"]  = int(round(10.0 ** float(theta[burst0])))
        out["duration"] = int(round(float(np.exp(theta[burst0 + 1]))))
        out["mult"]     = float(np.exp(theta[burst0 + 2]))
        return out

    @staticmethod
    def to_sbi_prior():
        """
        Box-uniform prior for sbi's posterior normalization. Derived from the
        active context set so it always matches PARAM_NAMES / N_PARAMS.
        """
        import torch
        from sbi.utils import BoxUniform
        lows  = [LOG_LAMBDA_RANGE[0], LOG_S_GLOBAL_RANGE[0]]
        highs = [LOG_LAMBDA_RANGE[1], LOG_S_GLOBAL_RANGE[1]]
        for c in _ACTIVE_CONTEXT:
            lo, hi = _CONTEXT_RANGE[c]
            lows.append(lo); highs.append(hi)
        lows  += [LOG10_T_START_RANGE[0], LOG_DURATION_RANGE[0], LOG_MULTIPLIER_RANGE[0]]
        highs += [LOG10_T_START_RANGE[1], LOG_DURATION_RANGE[1], LOG_MULTIPLIER_RANGE[1]]
        low  = np.array(lows,  dtype=np.float32)
        high = np.array(highs, dtype=np.float32)
        return BoxUniform(low=torch.tensor(low), high=torch.tensor(high))


# Standard population-genetics selection-class thresholds in |2 Ne s| units:
#   nearly-neutral |2Ne s| < 1,  weak 1..10,  strong >= 10.
SEL_THRESHOLD_NN     = 1.0
SEL_THRESHOLD_STRONG = 10.0


def compute_context_binned_props(s_global, alpha_r, feature_r, ne_harmonic,
                                 alpha_d=0.0, feature_d=0.0,
                                 gamma_r=None, gamma_d=0.0, weighted=False):
    """
    Realized-DFE binned proportions for the CONTEXT model.

    The context model induces a *distribution of selection coefficients across
    sites* via
        s_local(site) = s_global * exp(alpha_d*feature_d + alpha_r*feature_r),
    where feature_r is the per-site recombination feature (pos channel 5).

    Two distributions are meaningful (see compute_context_dfe_both):

    weighted=False (AVAILABLE DFE): every site counts equally — the selective
        effects available across the genome if TEs inserted uniformly. This is
        the neutral reference.

    weighted=True (REALIZED DFE): each site is weighted by its local insertion
        probability lambda_local(site) ∝ exp(gamma_d*feature_d + gamma_r*
        feature_r), so the bins reflect the selection actually EXPERIENCED by
        insertions given preference. This is the biologically realized DFE to
        report to users. Requires gamma_r (insertion-preference slope).

    The shift between the two isolates the effect of insertion preference on the
    selective burden: if TEs preferentially insert where selection is weaker,
    the realized (weighted) DFE shifts toward neutral relative to the available
    (unweighted) one.

    s_global, alpha_r, gamma_r are scalars (one posterior draw); feature_r is
    the per-site vector. Returns (p_nn, p_w, p_strong) for that draw.
    """
    feature_r = np.asarray(feature_r, dtype=float).ravel()
    s_local = compute_s_local(s_global, alpha_d, alpha_r, feature_d, feature_r)
    scaled = 2.0 * ne_harmonic * np.abs(s_local)        # |2 Ne s| per site

    if weighted:
        if gamma_r is None:
            raise ValueError("weighted=True requires gamma_r (insertion slope)")
        # Insertion weights ∝ lambda_local; lambda_global cancels on
        # normalization, so only the context slopes matter.
        log_w = gamma_d * np.asarray(feature_d) + gamma_r * feature_r
        log_w = log_w - np.max(log_w)                   # stabilize exp
        w = np.exp(log_w)
        w = w / np.sum(w)
        p_nn     = float(np.sum(w[scaled <  SEL_THRESHOLD_NN]))
        p_strong = float(np.sum(w[scaled >= SEL_THRESHOLD_STRONG]))
    else:
        p_nn     = float(np.mean(scaled <  SEL_THRESHOLD_NN))
        p_strong = float(np.mean(scaled >= SEL_THRESHOLD_STRONG))
    p_w = 1.0 - p_nn - p_strong
    return p_nn, p_w, p_strong


def compute_context_dfe_both(s_global, alpha_r, gamma_r, feature_r,
                             ne_harmonic, alpha_d=0.0, gamma_d=0.0,
                             feature_d=0.0):
    """
    Convenience: both the AVAILABLE (unweighted) and REALIZED (insertion-
    weighted) binned DFEs for one posterior draw, plus the realized-minus-
    available shift per bin.

    Returns dict with keys:
        avail    : (p_nn, p_w, p_strong) unweighted
        realized : (p_nn, p_w, p_strong) insertion-weighted by lambda_local
        shift    : realized - avail, per bin (negative p_strong shift => TEs
                   preferentially insert where selection is weaker, i.e.
                   preference REDUCES the strong-deleterious burden).
    """
    avail = compute_context_binned_props(
        s_global, alpha_r, feature_r, ne_harmonic,
        alpha_d=alpha_d, feature_d=feature_d, weighted=False)
    realized = compute_context_binned_props(
        s_global, alpha_r, feature_r, ne_harmonic,
        alpha_d=alpha_d, feature_d=feature_d,
        gamma_r=gamma_r, gamma_d=gamma_d, weighted=True)
    shift = tuple(r - a for r, a in zip(realized, avail))
    return {"avail": avail, "realized": realized, "shift": shift}
