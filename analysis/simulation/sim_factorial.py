"""Simulation 1 (factorial): a constant level-1 weight c in a random-intercept logit model.

Data: J = 400 clusters of n units, x ~ N(0,1) at level 1, z ~ Bernoulli(0.5) at level 2,
logit P(y = 1) = b0 + 0.5 x + 0.5 z + u_j, u_j ~ N(0, s2). Every unit has the same weight, so
c = 1 is the correct analysis. Each data set is fitted with c = 1, 2, 4 and with
mean-one weights that vary between clusters (like a normalised DHS weight at level 1).
Usage: python sim_factorial.py <reps> <seed0> <out.json>"""
import sys, json, time, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.special import expit, logit
from mlweights import WeightedRILogit as RILogit, approx_inflated_variance

J = 400
NS, S2S, PS = (5, 10, 20), (0.1, 0.4, 1.0), (0.1, 0.5)
CS = {"c1": 1.0, "c2": 2.0, "c4": 4.0}
B1, B2 = 0.5, 0.5

def predict(m, X, cl, c):
    """First-order value from approx_inflated_variance, using the c = 1 fit."""
    return approx_inflated_variance(m.sigma2, m.cluster_info(), c)

def one(seed, n, s2, p):
    rng = np.random.default_rng(seed)
    cl = np.repeat(np.arange(J), n)
    x = rng.normal(size=J * n); z = rng.binomial(1, .5, J)
    b0 = logit(p) - 0.25
    u = rng.normal(0, np.sqrt(s2), J)
    y = rng.binomial(1, expit(b0 + B1 * x + B2 * z[cl] + u[cl]))
    X = np.column_stack([np.ones(J * n), x, z[cl]])
    m1 = RILogit(y, X, cl).fit()
    out = {"prev": float(y.mean())}
    cvar = rng.lognormal(-0.5 * np.log(1 + .6 ** 2), np.sqrt(np.log(1 + .6 ** 2)), J)
    cvar = cvar / cvar.mean()
    for k, c in list(CS.items()) + [("cvar", None)]:
        cc = cvar if c is None else np.full(J, c)
        m = m1 if k == "c1" else RILogit(y, X, cl, w1=cc[cl]).fit(m1.theta.copy())
        out[k] = dict(b0=float(m.beta[0]), b1=float(m.beta[1]), b2=float(m.beta[2]), s2=m.sigma2, conv=bool(m.converged),
                      pred=float(predict(m1, X, cl, cc)) if k != "c1" else m1.sigma2)
    return out

if __name__ == "__main__":
    reps, seed0, fn = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    t = time.time(); res = []
    for ci, (n, s2, p) in enumerate(itertools.product(NS, S2S, PS)):
        for r in range(reps):
            res.append(dict(n=n, s2=s2, p=p, rep=r, **one(seed0 + 100000 * ci + r, n, s2, p)))
        print(n, s2, p, "%.0fs" % (time.time() - t), flush=True)
    json.dump(res, open(fn, "w"), default=float)
