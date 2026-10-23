import numpy as np
from mlweights import dhs_level_weights, approx_inflated_variance, icc_logit, mor


def _toy():
    cl = np.repeat(np.arange(6), 5)
    st = np.repeat([1, 1, 1, 2, 2, 2], 5)
    v005 = np.repeat([800000, 1200000, 1000000, 900000, 1500000, 600000], 5)
    return v005, st, cl


def test_split_multiplies_back_and_rescales_to_one():
    v005, st, cl = _toy()
    out = dhs_level_weights(v005, st, cl, A_h={1: 300, 2: 500}, M_h={1: 110, 2: 95}, S_h=25,
                            M_total=60000, m_c=150, alpha=0.5)
    assert np.allclose(out["w1"] * out["w2"], out["d"])
    assert np.allclose(out["w1_scaled"], 1.0)
    assert out["w1"].mean() > 1.5


def test_alpha_one_puts_all_variation_in_cluster_weight():
    v005, st, cl = _toy()
    out = dhs_level_weights(v005, st, cl, {1: 300, 2: 500}, {1: 110, 2: 95}, 25, 60000, 150, alpha=1.0)
    assert np.allclose(out["w1"], np.where(st == 1, 110, 95) / 25)


def test_approx_inflated_variance_limits():
    v = np.full(100, 0.5)
    assert np.isclose(approx_inflated_variance(0.3, v, 1.0), 0.3)
    assert np.isclose(approx_inflated_variance(0.3, v, 4.0), 0.3 + 0.5 * 0.75)
    assert np.isclose(approx_inflated_variance(0.3, v, 1e9), 0.8, atol=1e-6)
    assert approx_inflated_variance(0.3, v, 0.5) < 0.3


def test_icc_mor():
    assert np.isclose(icc_logit(np.pi ** 2 / 3), 0.5)
    assert np.isclose(mor(0.0), 1.0)
    assert np.isclose(mor(0.8), 2.3469, atol=1e-3)
