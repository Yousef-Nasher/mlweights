"""Re-run the DHS Program worked example (Zimbabwe 2015, modern contraception):
level-weights as in the published code, and the same weights with level 1 rescaled."""
import numpy as np
import pandas as pd
from mlweights import WeightedRILogit, dhs_level_weights, check_level1_weights
from dhs_data import load, design, ZW_A_H, ZW_M_H, ZW_S_H, ZW_M_TOTAL, ZW_M_C

d = load("ZW2015")
X, names = design(d)
y, cl = d.y.values, d.v001.values
one = np.ones(len(d))
d_hh = d.wn.values * ZW_M_TOTAL / ZW_M_C
models = {}
for alpha in (0.0, 0.5, 1.0):
    w = dhs_level_weights(d.v005.values, d.v022.values, cl, ZW_A_H, ZW_M_H, ZW_S_H, ZW_M_TOTAL, ZW_M_C, alpha)
    if alpha == 0.5:
        print(check_level1_weights(w["w1"], cl))
    models[f"alpha={alpha:g}, as published"] = (w["w1"], w["w2"])
    models[f"alpha={alpha:g}, level 1 rescaled"] = (w["w1_scaled"], w["w2"])
models["Final weight at level 2, level 1 = 1"] = (one, d_hh)
models["Unweighted"] = (one, one)

start = WeightedRILogit(y, X, cl).fit().theta
rows = []
for k, (w1, w2) in models.items():
    m = WeightedRILogit(y, X, cl, w1=w1, w2=w2).fit(start)
    s = m.summary(strata=d.v022.values)            # design-based SEs, strata as in the published code
    rows.append(dict(model=k, **dict(zip(names, np.round(m.beta, 3))),
                     **{f"SE {n}": round(float(v), 3) for n, v in zip(names, s["se_beta"])},
                     sigma2=m.sigma2, SE_sigma2=s["se_sigma2"], CI_low_sigma2=s["lo_sigma2"], CI_high_sigma2=s["hi_sigma2"],
                     ICC=m.icc, MOR=m.mor, converged=m.converged))
out = pd.DataFrame(rows)
print(f"women {len(d)}, clusters {d.v001.nunique()}")
print(out.round(3).to_string(index=False))
out.to_csv("results/zimbabwe_mr27.csv", index=False)
