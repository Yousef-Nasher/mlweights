"""
Simulation 2: weighting choices for a random-intercept logistic model
fitted to a DHS-like stratified two-stage sample.

Design (as described for DHS surveys):
  strata = region x urban/rural; clusters drawn with probability proportional
  to census size (systematic PPS); a fixed number of households drawn with
  equal probability from the listing; every eligible woman interviewed.
Truth is known because we generate the data.
"""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.special import expit
from mlweights import WeightedRILogit as RILogit

S_HH = 25                                  # households selected per cluster
BETA = dict(b0=-0.5, x1=0.5, x2=-0.3, urban=0.4, comm=-0.4)
SIGMA2 = 0.8                               # ICC about 0.20
NAMES = ["b0", "x1", "x2", "urban", "comm"]


def population(rng, scen):
    """Cluster-level frame: stratum, size, random effect, community covariate."""
    R = 10
    rows = []
    reg_size = np.exp(rng.normal(np.log(900), 0.8, R))         # rural clusters per region
    if scen == "C":
        # region effect on the outcome, larger in small (oversampled) regions; omitted by the analyst
        z = -(np.log(reg_size) - np.log(reg_size).mean()) / np.log(reg_size).std()
        gam = 0.6 * z
    else:
        gam = np.zeros(R)
    h = 0
    for r in range(R):
        for urb in (0, 1):
            A = int(max(150, reg_size[r] * (0.35 if urb else 1.0)))
            u = rng.normal(0, np.sqrt(SIGMA2), A)
            e = rng.normal(size=A)
            rho = 0.5 if scen == "B" else 0.0                     # size related to the cluster effect
            M = np.exp(np.log(100) + 0.35 * (rho * u / np.sqrt(SIGMA2) + np.sqrt(1 - rho ** 2) * e))
            M = np.maximum(30, np.round(M))
            comm = rng.binomial(1, 0.35 if urb else 0.6, A)
            rows.append(dict(h=h, reg=r, urb=urb, A=A, u=u, M=M, comm=comm, gam=gam[r]))
            h += 1
    return rows


def pps_systematic(rng, M, a):
    perm = rng.permutation(len(M))
    cum = np.cumsum(M[perm])
    I = cum[-1] / a
    pts = rng.uniform(0, I) + I * np.arange(a)
    return perm[np.searchsorted(cum, pts, side="right")]


def draw_sample(rng, pop, n_clusters=400, p_dom=0.5, equal_prob=False):
    Mh = np.array([s["M"].sum() for s in pop])
    alloc = np.maximum(8, np.round(n_clusters * Mh ** 0.4 / (Mh ** 0.4).sum())).astype(int)
    if equal_prob:                       # benchmark design: clusters with equal probability
        Ah = np.array([s["A"] for s in pop])
        alloc = np.maximum(2, np.round(n_clusters * Ah / Ah.sum())).astype(int)
    recs, j = [], 0
    for s, a in zip(pop, alloc):
        if equal_prob:
            sel = rng.choice(s["A"], a, replace=False); pi1 = np.full(a, a / s["A"])
        else:
            sel = pps_systematic(rng, s["M"], a); pi1 = a * s["M"][sel] / s["M"].sum()
        for k, c in enumerate(sel):
            M = s["M"][c]
            L = max(S_HH, round(M * np.exp(rng.normal(0, 0.15))))   # listing differs from census size
            n = rng.binomial(S_HH, p_dom)                            # women in the analysis group
            if n == 0:
                continue
            x1 = rng.binomial(1, 0.6 if s["urb"] else 0.3, n)
            x2 = rng.normal(size=n)
            eta = (BETA["b0"] + BETA["x1"] * x1 + BETA["x2"] * x2 + BETA["urban"] * s["urb"]
                   + BETA["comm"] * s["comm"][c] + s["gam"] + s["u"][c])
            y = rng.binomial(1, expit(eta))
            recs.append(dict(j=j, h=s["h"], n=n, y=y, x1=x1, x2=x2, urb=s["urb"], comm=s["comm"][c],
                             w2_true=1 / pi1[k], w1_true=L / S_HH,
                             A_h=s["A"], a_h=a, Mbar_h=s["M"].mean()))
            j += 1
    return recs, float(Mh.sum())


def assemble(recs, M_total):
    n = np.array([r["n"] for r in recs])
    cl = np.repeat(np.arange(len(recs)), n)
    y = np.concatenate([r["y"] for r in recs])
    X = np.column_stack([np.ones(len(y)), np.concatenate([r["x1"] for r in recs]),
                         np.concatenate([r["x2"] for r in recs]),
                         np.repeat([r["urb"] for r in recs], n), np.repeat([r["comm"] for r in recs], n)])
    g = lambda k: np.array([r[k] for r in recs], float)
    w2t, w1t = g("w2_true"), g("w1_true")
    w = w2t * w1t                                   # overall weight; the same for everyone in a cluster
    m_c = S_HH * len(recs)                          # completed households
    wn = w * m_c / (S_HH * w).sum()                 # normalised like HV005/V005 (mean 1 over households)
    # --- DHS Program approximation of level-weights, alpha = 0.5 ---
    d = wn * M_total / m_c                          # step 1: de-normalise
    f = d / ((g("A_h") / g("a_h")) * (g("Mbar_h") / S_HH))
    w2a = (g("A_h") / g("a_h")) * f ** 0.5          # step 2
    w1a = d / w2a                                   # step 3 (left unscaled in the published code)
    rep = lambda v: np.repeat(v, n)
    one = np.ones(len(recs))
    W = {  # name: (level-1 weight per cluster, level-2 weight per cluster)
        "1 Unweighted":                              (one, one),
        "2 Final weight at level 1":                 (wn, one),
        "3 Final weight at level 2":                 (one, wn),
        "4 DHS approximation, as published":        (w1a, w2a),
        "5 DHS approximation, level 1 rescaled":    (one, w2a),
        "6 True level-weights, unscaled":            (w1t, w2t),
        "7 True level-weights, level 1 rescaled":    (one, w2t),
    }
    return y, X, cl, {k: (rep(a), b) for k, (a, b) in W.items()}, dict(cv_w=float(wn.std() / wn.mean()),
                                                                     mean_n=float(n.mean()), w1a=float(w1a.mean()))


def one_rep(seed, scen, p_dom):
    rng = np.random.default_rng(seed)
    pop = population(rng, scen)
    recs, M_total = draw_sample(rng, pop, p_dom=p_dom)
    y, X, cl, W, info = assemble(recs, M_total)
    out = {"info": info}
    start = RILogit(y, X, cl).fit().theta
    for k, (w1, w2) in W.items():
        m = RILogit(y, X, cl, w1=w1, w2=w2).fit(start.copy())
        out[k] = dict(zip(NAMES, m.beta.tolist()), s2=m.sigma2, conv=bool(m.converged))
    return out


def benchmark(scen, seed=999, n_clusters=6000, reps=6):
    """Pseudo-true values for scenario C (model omits region): unweighted fit to large
    equal-probability samples of clusters."""
    est = []
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        pop = population(rng, scen)
        recs, M_total = draw_sample(rng, pop, n_clusters=n_clusters, p_dom=1.0, equal_prob=True)
        y, X, cl, _, _ = assemble(recs, M_total)
        m = RILogit(y, X, cl).fit()
        est.append(dict(zip(NAMES, m.beta.tolist()), s2=m.sigma2))
    return {k: float(np.mean([e[k] for e in est])) for k in est[0]}


if __name__ == "__main__":
    scen, p_dom, reps, seed0 = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    t = time.time()
    res = [one_rep(seed0 + i, scen, p_dom) for i in range(reps)]
    truth = dict(BETA, s2=SIGMA2) if scen != "C" else benchmark(scen)
    json.dump(dict(scen=scen, p_dom=p_dom, truth=truth, res=res),
              open(f"out_{scen}_{p_dom}_{seed0}.json", "w"), default=float)
    print(scen, p_dom, reps, "reps  %.0fs" % (time.time() - t))
