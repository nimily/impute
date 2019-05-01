import numpy as np
import numpy.random as npr
import numpy.testing as npt

import pytest

from impute.fpc import FpcImpute

from .test_base import create_alternating_entry_dataset


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


@pytest.mark.usefixtures('rae_case')
class TestFpc:

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
