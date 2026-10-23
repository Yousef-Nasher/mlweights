"""Summarise simulation output: bias, Monte Carlo SE, empirical SE and RMSE.
Usage: python summarise.py <output folder>"""
import sys, json, glob, os
import numpy as np
import pandas as pd

ICC = lambda s: s / (s + np.pi ** 2 / 3)
MOR = lambda s: np.exp(np.sqrt(2 * s) * 0.6744897501960817)
out = sys.argv[1] if len(sys.argv) > 1 else "out"

# Simulation 1 (factorial)
res = []
for f in sorted(glob.glob(os.path.join(out, "fact_*.json"))):
    res += json.load(open(f))
rows = []
for (n, s2, p), g in (pd.DataFrame(res).groupby(["n", "s2", "p"]) if res else []):
    for k, lab in [("c1", "c = 1 (scaled)"), ("c2", "c = 2"), ("c4", "c = 4"), ("cvar", "mean-one weights at level 1")]:
        e = pd.DataFrame(list(g[k]))
        e = e[e.conv]
        R = len(e)
        rows.append(dict(n=n, sigma2=s2, prevalence=p, method=lab, reps=R, conv_rate=R / len(g),
                         mean_sigma2=e.s2.mean(), median_sigma2=e.s2.median(),
                         relbias_sigma2=100 * (e.s2.mean() - s2) / s2, mcse_relbias=100 * e.s2.std() / np.sqrt(R) / s2,
                         empse_sigma2=e.s2.std(), rmse_sigma2=np.sqrt(((e.s2 - s2) ** 2).mean()),
                         approx_sigma2=e.pred.mean(), mean_ICC=ICC(e.s2).mean(), true_ICC=ICC(s2),
                         mean_MOR=MOR(e.s2).mean(), true_MOR=MOR(s2),
                         relbias_b1=100 * (e.b1.mean() - .5) / .5, relbias_b2=100 * (e.b2.mean() - .5) / .5,
                         observed_prev=g.prev.mean()))
t1 = pd.DataFrame(rows)
if len(t1):
    t1.to_csv(os.path.join(out, "sim1_summary.csv"), index=False)

# Simulation 2 (DHS-like)
rows = []
for f in sorted(glob.glob(os.path.join(out, "out_*.json"))):
    J = json.load(open(f)); truth = J["truth"]
    methods = [k for k in J["res"][0] if k != "info" and not k.startswith("8")]
    mean_n = np.mean([r["info"]["mean_n"] for r in J["res"]])
    for k in methods:
        e = pd.DataFrame([r[k] for r in J["res"]]); e = e[e.conv]; R = len(e)
        r = dict(setting=J["scen"], women_per_cluster=mean_n, method=k, reps=R, conv_rate=R / len(J["res"]),
                 true_sigma2=truth["s2"], mean_sigma2=e.s2.mean(),
                 relbias_sigma2=100 * (e.s2.mean() - truth["s2"]) / truth["s2"],
                 mcse_relbias=100 * e.s2.std() / np.sqrt(R) / truth["s2"],
                 rmse_sigma2=np.sqrt(((e.s2 - truth["s2"]) ** 2).mean()),
                 mean_ICC=ICC(e.s2).mean(), true_ICC=ICC(truth["s2"]), mean_MOR=MOR(e.s2).mean(), true_MOR=MOR(truth["s2"]))
        for b in ["b0", "x1", "x2", "urban", "comm"]:
            r[f"bias_{b}"] = e[b].mean() - truth[b]
            r[f"relbias_{b}"] = 100 * (e[b].mean() - truth[b]) / truth[b]   # > 0: further from zero
        rows.append(r)
t2 = pd.DataFrame(rows)
if len(t2):
    t2.to_csv(os.path.join(out, "sim2_summary.csv"), index=False)
pd.set_option("display.width", 250)
if len(t1):
    print(t1[["n", "sigma2", "prevalence", "method", "conv_rate", "mean_sigma2", "relbias_sigma2", "mcse_relbias", "approx_sigma2", "relbias_b1", "relbias_b2"]].round(2).to_string(index=False))
if len(t2):
    print(t2[["setting", "women_per_cluster", "method", "conv_rate", "mean_sigma2", "relbias_sigma2", "mcse_relbias", "mean_ICC", "mean_MOR", "relbias_x1", "relbias_urban", "relbias_comm", "relbias_b0"]].round(2).to_string(index=False))

# Simulation 3 (coverage of 95% confidence intervals, design-based standard errors)
rows = []
for f in sorted(glob.glob(os.path.join(out, "cov_*.json"))):
    J = json.load(open(f)); truth = J["truth"]; names = J["names"]
    mean_n = {"0.25": 6.3, "0.5": 12.5, "1.0": 25.0}[str(J["p_dom"])]
    for k in J["res"][0]:
        e = [r[k] for r in J["res"] if r[k]["conv"]]
        R = len(e)
        B = np.array([x["beta"] for x in e]); S = np.array([x["se"] for x in e])
        r = dict(setting=J["scen"], women_per_cluster=mean_n, method=k, reps=R)
        for i, nm in enumerate(names):
            t = truth[nm]
            cov = np.mean((B[:, i] - 1.96 * S[:, i] <= t) & (t <= B[:, i] + 1.96 * S[:, i]))
            r[f"cover_{nm}"] = 100 * cov
            r[f"se_ratio_{nm}"] = S[:, i].mean() / B[:, i].std(ddof=1)
        s2 = np.array([x["s2"] for x in e]); lo = np.array([x["lo_s2"] for x in e]); hi = np.array([x["hi_s2"] for x in e])
        r["cover_s2"] = 100 * np.mean((lo <= truth["s2"]) & (truth["s2"] <= hi))
        r["se_ratio_s2"] = np.mean([x["se_s2"] for x in e]) / s2.std(ddof=1)
        r["mcse_cover"] = 100 * np.sqrt(0.95 * 0.05 / R)
        rows.append(r)
t3 = pd.DataFrame(rows)
if len(t3):
    t3.to_csv(os.path.join(out, "sim3_coverage.csv"), index=False)
    print(t3[["setting", "women_per_cluster", "method", "reps"] + [c for c in t3.columns if c.startswith("cover_")] + ["se_ratio_s2", "se_ratio_x1"]].round(1).to_string(index=False))
