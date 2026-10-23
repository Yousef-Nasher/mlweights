"""Simulation 3: standard errors and confidence interval coverage in the DHS-like design.

Same data-generating mechanism as sim_dhs.py. Each model is fitted by weighted
pseudo-maximum likelihood and its covariance estimated with the design-based
sandwich (first-stage clusters within strata, as survey software does).
Usage: python sim_coverage.py <setting A|B|C> <share of women in the analysis> <reps> <seed0>
"""
import sys, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from mlweights import WeightedRILogit
from sim_dhs import population, draw_sample, assemble, benchmark, BETA, SIGMA2, NAMES

KEEP = ["1 Unweighted", "3 Final weight at level 2", "4 DHS approximation, as published",
        "5 DHS approximation, level 1 rescaled", "7 True level-weights, level 1 rescaled"]


def one_rep(seed, scen, p_dom):
    rng = np.random.default_rng(seed)
    pop = population(rng, scen)
    recs, M_total = draw_sample(rng, pop, p_dom=p_dom)
    y, X, cl, W, info = assemble(recs, M_total)
    strata = np.array([r["h"] for r in recs])            # one per cluster, clusters numbered 0..J-1
    start = WeightedRILogit(y, X, cl).fit().theta
    out = {}
    for k in KEEP:
        w1, w2 = W[k]
        m = WeightedRILogit(y, X, cl, w1=w1, w2=w2).fit(start)
        s = m.summary(strata=strata)
        out[k] = dict(beta=m.beta.tolist(), se=s["se_beta"].tolist(), s2=m.sigma2, se_s2=s["se_sigma2"],
                      lo_s2=s["lo_sigma2"], hi_s2=s["hi_sigma2"], conv=bool(m.converged))
    return out


if __name__ == "__main__":
    scen, p_dom, reps, seed0 = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    t = time.time()
    res = [one_rep(seed0 + i, scen, p_dom) for i in range(reps)]
    truth = dict(BETA, s2=SIGMA2) if scen != "C" else benchmark(scen)
    json.dump(dict(scen=scen, p_dom=p_dom, truth=truth, names=NAMES, res=res),
              open(f"out/cov_{scen}_{p_dom}_{seed0}.json", "w"), default=float)
    print(scen, p_dom, reps, "reps  %.0fs" % (time.time() - t))
