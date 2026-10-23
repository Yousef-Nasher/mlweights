import numpy as np


def as_arrays(w, cluster):
    w = np.asarray(w, dtype=float)
    cluster = np.asarray(cluster)
    if w.ndim != 1 or cluster.ndim != 1 or len(w) != len(cluster):
        raise ValueError("weights and cluster must be 1-D arrays of the same length")
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("weights must be finite and positive")
    _, inv = np.unique(cluster, return_inverse=True)
    return w, cluster, inv
