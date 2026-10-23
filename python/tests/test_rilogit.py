import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import approx_fprime
from scipy.special import expit, logsumexp
from mlweights import WeightedRILogit, scale_level1


def _data(J=300, n=12, s2=0.8, seed=1):
    rng = np.random.default_rng(seed)
    cl = np.repeat(np.arange(J), n)
    X = np.column_stack([np.ones(J * n), rng.binomial(1, .4, J * n), rng.normal(size=J * n)])
    u = rng.normal(0, np.sqrt(s2), J)
    y = rng.binomial(1, expit(X @ np.array([-.5, .5, -.3]) + u[cl]))
    return y, X, cl, rng


def test_gradient_matches_numerical():
    y, X, cl, rng = _data()
    w1 = np.repeat(rng.uniform(2, 6, 300), 12); w2 = rng.uniform(.5, 3, 300)
    m = WeightedRILogit(y, X, cl, w1=w1, w2=w2)
    th = np.array([-.4, .4, -.2, -.1])
    _, g = m.negloglik(th)
    gn = approx_fprime(th, lambda t: m.negloglik(t, grad=False), 1e-6)
    assert np.max(np.abs(g - gn)) < 1e-5


def test_adaptive_quadrature_matches_brute_force():
    y, X, cl, rng = _data(J=100, n=8)
    w1 = np.repeat(rng.uniform(1, 3, 100), 8); w2 = rng.uniform(.5, 3, 100)
    m = WeightedRILogit(y, X, cl, w1=w1, w2=w2)
    th = np.array([-.4, .4, -.2, -.1])
    xq, wq = hermgauss(201); z = np.sqrt(2) * xq
    eta = (X @ th[:3])[:, None] + np.exp(th[3]) * z[None, :]
    cll = np.zeros((100, 201)); np.add.at(cll, cl, w1[:, None] * (y[:, None] * eta - np.logaddexp(0, eta)))
    logI = logsumexp(cll + np.log(wq)[None, :], axis=1) - .5 * np.log(np.pi)
    brute = -((w2 / w2.mean()) * logI).sum() / m._scale
    assert abs(m.negloglik(th, grad=False) - brute) < 1e-7


def test_recovers_truth_unweighted():
    y, X, cl, _ = _data(J=1500, n=15, seed=3)
    m = WeightedRILogit(y, X, cl).fit()
    assert m.converged
    assert np.allclose(m.beta, [-.5, .5, -.3], atol=0.08)
    assert abs(m.sigma2 - 0.8) < 0.12


def test_unscaled_constant_weight_inflates_and_scaling_fixes_it():
    y, X, cl, _ = _data(J=600, n=10, s2=0.5, seed=4)
    base = WeightedRILogit(y, X, cl).fit()
    w1 = np.full(len(y), 4.0)
    infl = WeightedRILogit(y, X, cl, w1=w1).fit(base.theta)
    fixed = WeightedRILogit(y, X, cl, w1=scale_level1(w1, cl)).fit(base.theta)
    assert infl.sigma2 > 1.4 * base.sigma2
    assert np.isclose(fixed.sigma2, base.sigma2, rtol=1e-4)


def test_level2_weight_scale_is_irrelevant():
    y, X, cl, rng = _data(J=200, n=10, seed=5)
    w2 = rng.uniform(.5, 3, 200)
    a = WeightedRILogit(y, X, cl, w2=w2).fit()
    b = WeightedRILogit(y, X, cl, w2=1000 * w2).fit()
    assert np.allclose(a.theta, b.theta, atol=1e-5)
    assert len(a.cluster_info()) == 200 and np.all(a.cluster_info() > 0)


def test_scores_vanish_at_optimum_and_sandwich_matches_model_se():
    y, X, cl, _ = _data(J=800, n=10, s2=0.5, seed=7)
    m = WeightedRILogit(y, X, cl).fit()
    assert np.max(np.abs(m._cluster_scores(m.theta).sum(0))) / m._scale < 1e-4
    V = m.vcov()
    # inverse of the observed information (model-based) for comparison
    eps, k = 1e-5, len(m.theta)
    H = np.zeros((k, k))
    for i in range(k):
        e = np.zeros(k); e[i] = eps
        H[:, i] = (m._cluster_scores(m.theta + e).sum(0) - m._cluster_scores(m.theta - e).sum(0)) / (2 * eps)
    se_model = np.sqrt(np.diag(np.linalg.inv(-H)))
    assert np.allclose(np.sqrt(np.diag(V)), se_model, rtol=0.15)


def test_summary_and_strata_shapes():
    y, X, cl, rng = _data(J=200, n=8, seed=8)
    m = WeightedRILogit(y, X, cl, w2=rng.uniform(.5, 2, 200)).fit()
    st_cluster = np.arange(200) % 10
    s1 = m.summary(strata=st_cluster)
    s2 = m.summary(strata=st_cluster[cl])              # one code per unit gives the same answer
    assert np.allclose(s1["se_beta"], s2["se_beta"])
    assert s1["lo_sigma2"] < s1["sigma2"] < s1["hi_sigma2"]
    assert np.all(s1["se_beta"] > 0)
