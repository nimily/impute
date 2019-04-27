import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import numpy.testing as npt

import pytest

from impute.sample_set import EntrySampleSet
from impute.fpc import FpcImpute


def test_fpc_alpha_max():
    npr.seed(314159265)

    shape = 300, 300
    n_rows, n_cols = shape

    ss = EntrySampleSet(shape)

    for i in range(n_rows):
        for j in range(n_cols):
            if (i + j) % 2 == 1:
                x = (i, j, 1)
                y = 1 + npr.random()

                ss.add_obs(x, y)

    imputer = FpcImpute(shape)
    alpha = imputer.alpha_max(ss)

    zs = imputer.fit(ss, [alpha, alpha * 0.999])

    actual = zs[0].to_matrix()
    expect = np.zeros(shape)
    npt.assert_array_almost_equal(actual, expect)

    actual = npl.norm(zs[1].to_matrix())
    assert actual > 1e-5


@pytest.mark.parametrize(
    'rank',
    range(1, 5)
)
def test_fpc_debias(rank):
    npr.seed(314159265)

    shape = 50, 40
    n_rows, n_cols = shape
    rank = 3

    ss = EntrySampleSet(shape)

    for i in range(n_rows):
        for j in range(n_cols):
            if (i + j) % 2 == 1:
                x = (i, j, 1)
                y = npr.randn()

                ss.add_obs(x, y)

    u = npr.randn(n_rows, rank)
    v = npr.randn(rank, n_cols)

    svd = FpcImpute.debias(ss, u, v)

    z = svd.to_matrix()
    g = u.T @ ss.rss_grad(z) @ v.T

    actual = np.diagonal(g)
    expect = np.zeros(rank)

    npt.assert_array_almost_equal(actual, expect)


def gen_random_data(n_rows, n_cols, rank, n_obs):
    shape = n_rows, n_cols

    bl = npr.randn(n_rows, rank)
    br = npr.randn(n_cols, rank)
    b = bl @ br.T

    ss = EntrySampleSet(shape)

    indices = npr.choice(n_rows * n_cols, size=n_obs, replace=False)

    xs = [(i // n_cols, i % n_cols, 1) for i in indices]
    ys = [b[i, j] + npr.randn() for i, j, _ in xs]

    ss.add_all_obs(xs, ys)

    return b, ss


def loss(ss: EntrySampleSet, b, alpha):
    ys = np.array(ss.ys)
    yh = ss.value(b)
    return np.sum((ys - yh) ** 2) / 2 + alpha * npl.norm(b, 'nuc')


@pytest.mark.parametrize(
    'seed, alpha_ratio',
    zip(range(5), np.logspace(0, -3, num=5, base=2))
)
def test_fpc_strong_optimality(seed, alpha_ratio):
    npr.seed(seed)

    shape = 200, 250
    n_rows, n_cols = shape
    rank = 5
    n_obs = 25 * 1000

    _, ss = gen_random_data(n_rows, n_cols, rank, n_obs)

    imputer = FpcImpute(shape)
    alpha = imputer.alpha_max(ss) * alpha_ratio

    z = imputer.impute(ss, alpha=alpha, xtol=1e-10)
    m = z.to_matrix()
    g = ss.rss_grad(m)

    u = z.u
    v = z.v
    w = g + alpha * u @ v

    actual = u.T @ w
    expect = np.zeros(actual.shape)
    npt.assert_almost_equal(actual, expect, decimal=3)

    actual = w @ v.T
    expect = np.zeros(actual.shape)
    npt.assert_almost_equal(actual, expect)

    actual = npl.norm(w, 2)
    expect = alpha * (1 + 1e-5)
    assert actual < expect


@pytest.mark.parametrize(
    'seed, alpha_ratio',
    zip(range(10), np.logspace(0, -10, num=10, base=2))
)
def test_fpc_weak_optimality(seed, alpha_ratio):
    npr.seed(seed)

    shape = 200, 250
    n_rows, n_cols = shape
    rank = 5
    n_obs = 25 * 1000

    b, ss = gen_random_data(n_rows, n_cols, rank, n_obs)

    imputer = FpcImpute(shape)
    alpha = imputer.alpha_max(ss) * alpha_ratio

    z = imputer.impute(ss, alpha=alpha, gtol=1e-3)
    m = z.to_matrix()

    loss_b = loss(ss, b, alpha)
    loss_m = loss(ss, m, alpha)

    assert loss_m < loss_b
