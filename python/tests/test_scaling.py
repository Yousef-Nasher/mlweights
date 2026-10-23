import numpy as np
import pytest
from mlweights import scale_level1, cluster_inflation, is_constant_within, check_level1_weights


def test_size_scaling_sums_to_cluster_size():
    rng = np.random.default_rng(0)
    cl = np.repeat(np.arange(30), rng.integers(2, 9, 30))
    w = rng.uniform(0.5, 6, len(cl))
    s = scale_level1(w, cl)
    for j in np.unique(cl):
        assert np.isclose(s[cl == j].sum(), (cl == j).sum())
        assert np.allclose(s[cl == j] / w[cl == j], s[cl == j][0] / w[cl == j][0])


def test_effective_scaling_sums_to_effective_size():
    cl = np.array([1, 1, 1, 2, 2])
    w = np.array([1.0, 2.0, 3.0, 4.0, 4.0])
    s = scale_level1(w, cl, method="effective")
    assert np.isclose(s[:3].sum(), 36 / 14)
    assert np.allclose(s[3:], 1.0)


def test_constant_weights_become_one():
    cl = np.repeat([10, 20, 30], [4, 5, 6])
    w = np.repeat([3.8, 4.1, 0.2], [4, 5, 6])
    assert np.allclose(scale_level1(w, cl), 1.0)
    assert np.allclose(scale_level1(w, cl, "effective"), 1.0)
    assert is_constant_within(w, cl)
    assert not is_constant_within(np.r_[w[:-1], 0.3], cl)


def test_inflation_factor():
    cl = np.repeat([1, 2], [3, 2])
    w = np.array([4.0, 4.0, 4.0, 1.0, 3.0])
    assert np.allclose(cluster_inflation(w, cl), [4, 4, 4, 2, 2])


def test_check_messages():
    cl = np.repeat([1, 2], [3, 3])
    assert check_level1_weights(np.ones(6), cl)["share_unscaled"] == 0
    r = check_level1_weights(np.full(6, 3.8), cl)
    assert r["constant_within"] and np.isclose(r["mean_c"], 3.8) and "set the level-1 weight to 1" in r["message"]


@pytest.mark.parametrize("bad", [np.array([1.0, -1.0]), np.array([1.0, np.nan])])
def test_rejects_bad_weights(bad):
    with pytest.raises(ValueError):
        scale_level1(bad, np.array([1, 1]))
