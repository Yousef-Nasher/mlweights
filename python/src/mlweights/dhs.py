"""Level-weights for DHS-type two-stage samples, with level-1 rescaling."""
import numpy as np
from .scaling import scale_level1


def dhs_level_weights(v005, stratum, cluster, A_h, M_h, S_h, M_total, m_c, alpha=0.5):
    """Split the DHS final weight into a cluster weight and a within-cluster weight.

    Follows the approximation in the DHS Program framework for level-weights
    (de-normalise, then split the variation factor f with exponent alpha), and
    adds the step that the framework's code omits: the within-cluster weight is
    rescaled to sum to the cluster sample size.

    v005     : DHS final weight (six implied decimals), one value per unit.
    stratum  : stratum code per unit.
    cluster  : cluster code per unit.
    A_h, M_h : dicts {stratum: number of clusters in the frame, average households per cluster}.
    S_h      : households selected per cluster (scalar or dict by stratum).
    M_total  : households in the frame. m_c: completed household interviews.

    Returns a dict with arrays: d (de-normalised weight), w2 (cluster weight),
    w1 (within-cluster weight, unscaled), w1_scaled (to use in the model).
    """
    v005 = np.asarray(v005, float); stratum = np.asarray(stratum); cluster = np.asarray(cluster)
    get = lambda m: np.array([m[s] for s in stratum], float) if isinstance(m, dict) else np.full(len(stratum), float(m))
    A, M, S = get(A_h), get(M_h), get(S_h)
    a = np.zeros(len(stratum))
    for s in np.unique(stratum):
        k = stratum == s
        a[k] = len(np.unique(cluster[k]))
    d = v005 / 1e6 * (M_total / m_c)
    f = d / ((A / a) * (M / S))
    w2 = (A / a) * f ** alpha
    w1 = d / w2
    return {"d": d, "w2": w2, "w1": w1, "w1_scaled": scale_level1(w1, cluster)}
