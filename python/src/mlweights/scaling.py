"""Scaling of level-1 (within-cluster) weights."""
import numpy as np
from ._utils import as_arrays


def scale_level1(w, cluster, method="size"):
    """Rescale level-1 weights within each cluster.

    method="size":      weights sum to the number of sampled units in the cluster.
    method="effective": weights sum to the effective sample size (sum w)^2 / sum w^2.
    Both give 1 for every unit when the weight is constant within the cluster.
    """
    w, _, inv = as_arrays(w, cluster)
    s1 = np.bincount(inv, weights=w)
    if method == "size":
        target = np.bincount(inv).astype(float)
    elif method == "effective":
        target = s1 ** 2 / np.bincount(inv, weights=w ** 2)
    else:
        raise ValueError("method must be 'size' or 'effective'")
    return w * (target / s1)[inv]


def cluster_inflation(w, cluster):
    """Inflation factor c_j = (sum of level-1 weights) / (number of units), one value per unit.

    c_j = 1 means the weights are already scaled. c_j > 1 makes the cluster look
    c_j times larger than it is in the pseudo-likelihood.
    """
    w, _, inv = as_arrays(w, cluster)
    return (np.bincount(inv, weights=w) / np.bincount(inv))[inv]


def is_constant_within(w, cluster, rtol=1e-8):
    """True if the weight does not vary within any cluster."""
    w, _, inv = as_arrays(w, cluster)
    mx = np.full(inv.max() + 1, -np.inf); mn = np.full(inv.max() + 1, np.inf)
    np.maximum.at(mx, inv, w); np.minimum.at(mn, inv, w)
    return bool(np.all(mx - mn <= rtol * np.abs(mx)))
