import numpy as np
import numpy.random as npr
import numpy.testing as npt

import pytest

from impute import FpcImpute


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
