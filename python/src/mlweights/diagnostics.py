"""Diagnostics for level-1 weights and simple summaries of cluster variation."""
import numpy as np
from scipy.optimize import brentq
from .scaling import cluster_inflation, is_constant_within
from ._utils import as_arrays

_Z75 = 0.6744897501960817


def icc_logit(sigma2):
    """Latent-scale intraclass correlation for a random-intercept logit model."""
    return np.asarray(sigma2) / (np.asarray(sigma2) + np.pi ** 2 / 3)


def mor(sigma2):
    """Median odds ratio."""
    return np.exp(np.sqrt(2 * np.asarray(sigma2)) * _Z75)


def check_level1_weights(w1, cluster, tol=0.05):
    """Summarise how far level-1 weights are from being scaled.

    Returns a dict with the mean, minimum and maximum inflation factor, the share
    of clusters whose factor differs from 1 by more than tol, whether the weight
    is constant within clusters, and a short message.
    """
    w1, _, inv = as_arrays(w1, cluster)
    c = np.bincount(inv, weights=w1) / np.bincount(inv)
    const = is_constant_within(w1, cluster)
    off = float(np.mean(np.abs(c - 1) > tol))
    if off == 0:
        msg = "Level-1 weights are scaled."
    elif const:
        msg = ("Level-1 weights are constant within clusters but not equal to 1. "
               "Rescaled, they all become 1: set the level-1 weight to 1 and keep the cluster weight.")
    else:
        msg = "Level-1 weights are not scaled. Rescale them within clusters before fitting."
    return {"mean_c": float(c.mean()), "min_c": float(c.min()), "max_c": float(c.max()),
            "share_unscaled": off, "constant_within": const, "message": msg}


def approx_inflated_variance(sigma2, v, c, w2=None):
    """First-order value of the cluster variance obtained with level-1 weights inflated by c.

    sigma2 : cluster variance from the correctly scaled fit.
    v      : per-cluster sampling variance of the cluster effect on the logit scale,
             about 1 / sum_i p_ij (1 - p_ij).
    c      : inflation factor per cluster (scalar or array).
    w2     : cluster weights (optional).

    Solves sum_j w2_j [sigma2 + v_j (1 - 1/c_j) - s] / (s + v_j / c_j)^2 = 0.
    With equal v and c this is s = sigma2 + (1 - 1/c) v. The normal approximation
    behind it ignores clusters in which all outcomes are equal. For c >= 1 the
    true inflation is usually larger, so read the result as a lower bound.
    """
    v = np.atleast_1d(np.asarray(v, float))
    c = np.broadcast_to(np.asarray(c, float), v.shape)
    w2 = np.ones_like(v) if w2 is None else np.asarray(w2, float) / np.mean(w2)
    if np.any(c <= 0) or np.any(v <= 0) or sigma2 < 0:
        raise ValueError("c and v must be positive and sigma2 non-negative")
    f = lambda s: np.sum(w2 * (sigma2 + v * (1 - 1 / c) - s) / (s + v / c) ** 2)
    lo, hi = 0.0, sigma2 + float(np.max(v)) + 1.0
    if abs(f(lo)) < 1e-14:
        return 0.0
    return float(brentq(f, lo, hi)) if f(lo) > 0 else 0.0
