* Weighted two-level logistic model with DHS data in Stata.
* svy: melogit applies the level-1 weight as given; it has no option to rescale it.

gen double wt = v005/1000000

* Option A (DHS final weight): weight at level 2, level-1 weight 1.
* The DHS weight does not vary within clusters, so this is what a rescaled
* level-1 weight reduces to.
gen byte one = 1
svyset v001, weight(wt) strata(v022) , singleunit(centered) || _n, weight(one)
svy: melogit y x1 x2 || v001:

* Option B (separate level-weights wt2 = cluster, wt1 = within cluster):
* rescale wt1 so that it sums to the number of respondents in each cluster.
* Run this on the estimation sample only (drop cases with missing values first).
bysort v001: egen double sw1 = total(wt1)
bysort v001: gen n1 = _N
gen double wt1s = wt1*n1/sw1
svyset v001, weight(wt2) strata(v022) , singleunit(centered) || _n, weight(wt1s)
svy: melogit y x1 x2 || v001:
