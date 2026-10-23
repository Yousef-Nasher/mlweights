"""Read and prepare the DHS women's recode files used in the analysis.

The DHS files are not distributed with this repository. Register with the DHS
Program, download the Stata files and set DHS_DATA_DIR to the folder holding:
ZWIR72FL.DTA (Zimbabwe 2015), KEIR8CFL.DTA (Kenya 2022), UGIR91FL.DTA (Uganda 2024-25).
"""
import os
import numpy as np
import pandas as pd

DATA = os.environ.get("DHS_DATA_DIR", "data")
FILES = {"ZW2015": "ZWIR72FL.DTA", "KE2022": "KEIR8CFL.DTA", "UG2024": "UGIR91FL.DTA"}

# Zimbabwe 2015 frame values used in the DHS Program worked example (Appendix B of its report).
ZW_A_H = dict(zip(range(1, 20), [673, 3340, 162, 2451, 463, 2843, 839, 2298, 165, 1343, 218, 1280, 981, 2230, 372, 2907, 4920, 198, 1682]))
ZW_M_H = dict(zip(range(1, 20), [108, 102, 98, 100, 99, 98, 101, 100, 113, 106, 97, 99, 99, 99, 100, 101, 104, 144, 98]))
ZW_S_H, ZW_M_TOTAL, ZW_M_C = 28, 2976119, 10534


def _path(survey):
    for name in (FILES[survey], FILES[survey].lower(), FILES[survey].replace(".DTA", ".dta")):
        p = os.path.join(DATA, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{FILES[survey]} not found in {DATA}")


def load(survey):
    """Analysis sample, outcome y, covariates and the normalised final weight wn."""
    base = ["v001", "v005", "v012", "v022", "v025", "v106"]
    if survey in ("ZW2015", "KE2022"):
        d = pd.read_stata(_path(survey), convert_categoricals=False,
                          columns=base + ["v021", "v213", "v218", "v313", "v501", "v605"] if survey == "ZW2015"
                          else base + ["v213", "v218", "v313", "v501", "v605"])
        # currently married, not pregnant, not declared infecund; outcome: modern method
        d = d[(d.v501 == 1) & ~((d.v213 == 1) | (d.v605 == 7))].copy()
        d["y"] = (d.v313 == 3).astype(float)
        d["grp"] = d.v218.clip(upper=4)            # living children 0,1,2,3,4+
    elif survey == "UG2024":
        d = pd.read_stata(_path(survey), convert_categoricals=False, columns=base + ["v201", "v208", "m14_1"])
        # women with a birth in the last five years; outcome: four or more antenatal visits (last birth)
        d = d[(d.v208 > 0) & d.m14_1.notna()].copy()
        d["y"] = ((d.m14_1 >= 4) & (d.m14_1 < 98)).astype(float)
        d["grp"] = pd.cut(d.v201, [0, 1, 3, 5, 99], labels=False).astype(float)   # 1, 2-3, 4-5, 6+
    else:
        raise ValueError(survey)
    d = d[d.v005 > 0].copy()                      # zero weight: outside the weighted population
    d["age"] = d.v012.astype(float)
    d["secondary"] = np.where(d.v106.isin([2, 3]), 1.0, np.where(d.v106.isin([0, 1]), 0.0, np.nan))
    d["urban"] = (d.v025 == 1).astype(float)
    d = d.dropna(subset=["y", "age", "secondary", "urban", "grp"]).copy()
    d["wn"] = d.v005 / 1e6
    return d.reset_index(drop=True)


def design(d):
    levels = sorted(d.grp.unique())[1:]
    X = np.column_stack([np.ones(len(d)), d.age, d.secondary, d.urban] + [(d.grp == k).astype(float) for k in levels])
    names = ["Intercept", "Age", "Secondary", "Urban"] + [f"Group {int(k)}" for k in levels]
    return X, names
