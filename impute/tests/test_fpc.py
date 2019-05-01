import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import numpy.testing as npt

import pytest

from impute import Dataset
from impute.fpc import FpcImpute

from .test_base import create_alternating_entry_dataset


def loss(ds: Dataset, b, alpha):
    ys = np.array(ds.ys)
    yh = ds.op(b)
    return np.sum((ys - yh) ** 2) / 2 + alpha * npl.norm(b, 'nuc')


@pytest.fixture(params=[
    (1, (50, 45), 2),
    (2, (100, 80), 5),
    (3, (200, 250), 8),
], name='rae_case')  # randomized alternating-entry dataset
def rae_dataset_fixture(request):
    seed = request.param[0]
    shape = request.param[1]
    rank = request.param[2]

    npr.seed(seed)

    n_row, n_col = shape

    bl = npr.randn(n_row, rank)
    br = npr.randn(n_col, rank)
    b = bl @ br.T

    yield b, create_alternating_entry_dataset(shape, b)


@pytest.fixture(
    params=np.logspace(0, -3, num=5, base=2),
    ids=[f'alpha[{i}]' for i in range(5)],
    name='alpha_ratio'
)
def alpha_ratio_fixture(request):
    yield request.param


@pytest.mark.usefixtures('rae_case', 'alpha_ratio')
class TestFpc:

    @staticmethod
    def test_alpha_max(rae_case):
        b, ds = rae_case
        shape = b.shape

        imputer = FpcImpute(shape)
        alpha = imputer.alpha_max(ds)

        zs = imputer.fit(ds, [alpha, alpha * 0.999])

        actual = zs[0].to_matrix()
        expect = np.zeros_like(actual)
        npt.assert_array_almost_equal(actual, expect)

        actual = npl.norm(zs[1].to_matrix())
        assert actual > 1e-5

    @staticmethod
    def test_strong_optimality(rae_case, alpha_ratio):
        b, ds = rae_case
        shape = b.shape

        imputer = FpcImpute(shape)
        alpha = imputer.alpha_max(ds) * alpha_ratio

        z = imputer.impute(ds, alpha=alpha, xtol=1e-10)
        m = z.to_matrix()
        g = ds.rss_grad(m)

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

    @staticmethod
    def test_weak_optimality(rae_case, alpha_ratio):
        b, ds = rae_case

        shape = b.shape
        imputer = FpcImpute(shape)
        alpha = imputer.alpha_max(ds) * alpha_ratio

        z = imputer.impute(ds, alpha=alpha, gtol=1e-3)
        m = z.to_matrix()

        loss_b = loss(ds, b, alpha) + 1e-5
        loss_m = loss(ds, m, alpha)

        assert loss_m < loss_b

    @staticmethod
    def test_debias(rae_case):
        b, ds = rae_case

        n_row, n_col = b.shape
        rank = 10

        npr.seed(10)

        u = npr.randn(n_row, rank)
        v = npr.randn(n_col, rank)

        svd = FpcImpute.debias(ds, u, v.T)

        z = svd.to_matrix()
        g = u.T @ ds.rss_grad(z) @ v

        actual = np.diagonal(g)
        expect = np.zeros(rank)

        npt.assert_array_almost_equal(actual, expect)
