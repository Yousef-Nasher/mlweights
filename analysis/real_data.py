"""Dose-response on real data: the same model with the level-1 weight multiplied by c,
and two common ways of passing the final weight to the model."""
import sys
import numpy as np
import pandas as pd
from mlweights import WeightedRILogit, approx_inflated_variance
from dhs_data import load, design

SURVEYS = sys.argv[1:] or ["ZW2015", "KE2022", "UG2024"]
rows, info = [], []
for s in SURVEYS:
    d = load(s)
    X, names = design(d)
    y, cl = d.y.values, d.v001.values
    g = d.groupby("v001")
    w2 = g.wn.first().sort_index().values             # final weight, one per cluster
    one = np.ones(len(d))
    base = WeightedRILogit(y, X, cl, w2=w2).fit()
    v = base.cluster_info()
    methods = {"Unweighted": (one, None), "Final weight at level 2, level 1 = 1": None,
               "Final weight at level 1 (v005/1e6)": (d.wn.values, None)}
    for c in (2, 4, 8):
        methods[f"Level-1 weight = {c}, final weight at level 2"] = (c * one, w2)
    methods["Raw v005 at level 1"] = (d.v005.values.astype(float), None)
    for k, spec in methods.items():
        m = base if spec is None else WeightedRILogit(y, X, cl, w1=spec[0], w2=spec[1]).fit(base.theta)
        sm = m.summary(strata=d.v022.values)
        r = dict(survey=s, method=k, sigma2=m.sigma2, SE_sigma2=sm["se_sigma2"], CI_low_sigma2=sm["lo_sigma2"],
                 CI_high_sigma2=sm["hi_sigma2"], ICC=m.icc, MOR=m.mor, converged=m.converged,
                 **{f"OR {n}": float(np.exp(b)) for n, b in zip(names[1:], m.beta[1:])})
        if k.startswith("Level-1 weight ="):
            r["approx_sigma2"] = approx_inflated_variance(base.sigma2, v, float(k.split("= ")[1].split(",")[0]), w2)
        rows.append(r)
    same = g.y.agg(lambda t: t.nunique() == 1)
    info.append(dict(survey=s, women=len(d), clusters=d.v001.nunique(), mean_per_cluster=len(d) / d.v001.nunique(),
                     weighted_prevalence=float(np.average(d.y, weights=d.wn)), share_clusters_all_same=float(same.mean()),
                     weight_constant_within_cluster=bool((g.v005.nunique() == 1).all()), mean_v=float(v.mean())))
pd.DataFrame(rows).to_csv("results/real_data.csv", index=False)
pd.DataFrame(info).to_csv("results/real_data_samples.csv", index=False)
print(pd.DataFrame(info).round(3).to_string(index=False))
print(pd.DataFrame(rows)[["survey", "method", "sigma2", "approx_sigma2", "ICC", "MOR"]].round(3).to_string(index=False))
