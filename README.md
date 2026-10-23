# mlweights

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23601431.svg)](https://doi.org/10.5281/zenodo.23601431)

Survey weights in two-level random-intercept models, with a focus on Demographic and Health Survey (DHS) data.

In a weighted multilevel model the within-cluster (level-1) weight enters the likelihood as given. If it is not rescaled within clusters, each cluster looks larger than it is, and the model reads sampling noise as real differences between clusters. The cluster variance, the intraclass correlation (ICC) and the median odds ratio (MOR) all come out too large. `svy: melogit` in Stata has no option to rescale level-1 weights, so this has to be done by the user.

The DHS final weight does not vary within clusters. Rescaled, the level-1 weight is therefore 1 for every respondent, and a correct weighted analysis puts the weight at level 2 only.

This repository contains:

| Folder | Content |
|---|---|
| `python/` | Python package `mlweights`: weight scaling, checks, DHS level-weights with rescaling, a weighted random-intercept logit estimator |
| `R/mlweights/` | R package `mlweights` with the same weight functions |
| `stata/` | Stata example for rescaling before `svy: melogit` |
| `analysis/` | Code for every result in the paper |
| `analysis/results/` | Model estimates (no microdata) |

## Install

Python (3.9 or later):

```bash
pip install ./python
```

R:

```r
# from a clone of this repository
install.packages("R/mlweights", repos = NULL, type = "source")
# or
remotes::install_github("Yousef-Nasher/mlweights", subdir = "R/mlweights")
```

## Use

Python:

```python
import numpy as np
from mlweights import check_level1_weights, scale_level1, WeightedRILogit

check_level1_weights(w1, cluster)          # is the level-1 weight scaled?
w1s = scale_level1(w1, cluster)            # sums to the cluster sample size
m = WeightedRILogit(y, X, cluster, w1=w1s, w2=w2).fit()
m.sigma2, m.icc, m.mor
m.summary(strata=stratum_per_cluster)   # design-based SEs and 95% CIs
```

R:

```r
library(mlweights)
check_level1_weights(w1, cluster)
d$w1s <- scale_level1(d$w1, d$v001)
# then, for example, WeMix::mix(..., weights = c("w1s", "w2"), cWeights = TRUE)
```

Splitting a DHS final weight into level-weights, with the rescaling step:

```python
from mlweights import dhs_level_weights
w = dhs_level_weights(v005, v022, v001, A_h, M_h, S_h, M_total, m_c, alpha=0.5)
w["w2"], w["w1_scaled"]        # use these two in the model
```

`A_h`, `M_h`, `S_h`, `M_total` and `m_c` come from the survey's sampling appendix.

## Reproduce the paper

The DHS data cannot be shared. Register at the DHS Program, download the women's recode files in Stata format, and put them in one folder:

- `ZWIR72FL.DTA` (Zimbabwe DHS 2015)
- `KEIR8CFL.DTA` (Kenya DHS 2022)
- `UGIR91FL.DTA` (Uganda Malaria Indicator Survey 2024-25)

Then:

```bash
pip install ./python pandas
export DHS_DATA_DIR=/path/to/folder
cd analysis
python zimbabwe_mr27.py              # reanalysis of the DHS worked example
Rscript zimbabwe_mr27_wemix.R        # the same with WeMix
Rscript zimbabwe_level1_lme4.R       # level-1 mechanism with lme4
python real_data.py                  # three surveys
cd simulation
python sim_factorial.py 125 1 out/fact_a.json
python sim_factorial.py 125 50000 out/fact_b.json
for s in "A 0.25 1000" "A 0.5 2000" "A 1.0 5000" "B 0.5 4000" "C 0.5 3000"; do set -- $s; python sim_dhs.py $1 $2 300 $3; done
mv out_*.json out/
for s in "A 0.5 12000" "A 0.25 11000" "B 0.5 14000" "C 0.5 13000"; do set -- $s; python sim_coverage.py $1 $2 500 $3; done
python summarise.py out
cd .. && python figures.py
```

The simulations do not need DHS data.

## Tests

```bash
pip install ./python pytest && pytest python/tests
R CMD build R/mlweights && R CMD check mlweights_0.1.0.tar.gz
```

## Citation

If you use this code, please cite the paper and the software (see `CITATION.cff`):

Nasher YM, Alqumbaey M. Unscaled within-cluster weights inflate cluster variance in multilevel logistic models of Demographic and Health Survey data: a reanalysis and simulation study. Manuscript submitted for publication, 2026.

Nasher YM, Alqumbaey M. mlweights: scaling and checking survey weights for two-level models (version 0.1.0) [software]. Zenodo, 2026. https://doi.org/10.5281/zenodo.23601431

## Licence

MIT. See `LICENSE`.
