"""Figures for the paper. Run from analysis/ after real_data.py and simulation/summarise.py."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85", "#262624"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GREY, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "lines.linewidth": 2})
os.makedirs("figures", exist_ok=True)
NAMES = {"ZW2015": "Zimbabwe DHS 2015", "KE2022": "Kenya DHS 2022", "UG2024": "Uganda MIS 2024-25"}

# Figure 1: real data, cluster variance against the level-1 inflation factor
r = pd.read_csv("results/real_data.csv")
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=False)
for ax, s in zip(axes, ["ZW2015", "KE2022", "UG2024"]):
    d = r[r.survey == s]
    ref = d[d.method.str.startswith("Final weight at level 2")].iloc[0]
    inf = d[d.method.str.startswith("Level-1 weight =")]
    c = [1] + [int(m.split("= ")[1].split(",")[0]) for m in inf.method]
    obs = [ref.sigma2] + list(inf.sigma2)
    lo = [ref.CI_low_sigma2] + list(inf.CI_low_sigma2)
    hi = [ref.CI_high_sigma2] + list(inf.CI_high_sigma2)
    apx = [ref.sigma2] + list(inf.approx_sigma2)
    ax.errorbar(c, obs, yerr=[[o - l for o, l in zip(obs, lo)], [h - o for o, h in zip(obs, hi)]], color=BLUE,
                marker="o", ms=5, capsize=2, elinewidth=1, label="Estimate (95% CI)")
    ax.plot(c, apx, color=GREY, ls="--", marker="s", ms=4, label="First-order value")
    if s == "ZW2015":
        z = pd.read_csv("results/zimbabwe_mr27.csv")
        pub = z[z.model == "alpha=0.5, as published"].iloc[0]
        ax.plot([3.76], [pub.sigma2], marker="D", ms=7, color=ORANGE, ls="none", label="DHS code as published")
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_title(NAMES[s], fontsize=9, color=INK)
    ax.set_ylim(bottom=0)
axes[0].set_ylabel("Cluster variance")
axes[1].set_xlabel("Inflation factor c of the level-1 weight")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(f"figures/figure1.{ext}", dpi=600 if ext != "pdf" else None)
plt.close(fig)

# Figure 2: simulation 1, estimated / true cluster variance
if not os.path.exists("simulation/out/sim1_summary.csv"):
    print("figure 1 written; run simulation/summarise.py for figure 2"); raise SystemExit(0)
t = pd.read_csv("simulation/out/sim1_summary.csv")
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True, sharey=True)
styles = [("c = 2", BLUE, "o"), ("c = 4", ORANGE, "s"), ("mean-one weights at level 1", AQUA, "^")]
for i, p in enumerate([0.1, 0.5]):
    for k, s2 in enumerate([0.1, 0.4, 1.0]):
        ax = axes[i, k]
        for lab, col, mk in styles:
            d = t[(t.prevalence == p) & (t.sigma2 == s2) & (t.method == lab)].sort_values("n")
            ax.plot(d.n, d.mean_sigma2 / s2, color=col, marker=mk, ms=5, label=lab)
        ax.axhline(1, color=GREY, lw=1)
        ax.set_yscale("log"); ax.set_xscale("log")
        ax.set_yticks([1, 2, 5, 10, 20, 50]); ax.set_yticklabels(["1", "2", "5", "10", "20", "50"])
        ax.set_xticks([5, 10, 20]); ax.set_xticklabels(["5", "10", "20"]); ax.minorticks_off()
        ax.set_title(f"Prevalence {0.13 if p == 0.1 else 0.5:g}, true variance {s2:g}", fontsize=8, color=INK)
fig.supxlabel("Respondents per cluster", fontsize=9, color=INK, y=0.075)
fig.supylabel("Estimated / true cluster variance", fontsize=9, color=INK)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, ["Level-1 weight 2", "Level-1 weight 4", "Normalised weight at level 1"], loc="lower center", ncol=3,
           frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.09, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(f"figures/figure2.{ext}", dpi=600 if ext != "pdf" else None)
print("figures written")
