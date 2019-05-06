import numpy.testing as npt

import pytest

from impute.decomposition.randomized_svd import randomized_svd


@pytest.mark.usefixtures('low_rank_matrix')
class TestRandomizedSvd:

    @staticmethod
    def test_fixed_precision(low_rank_matrix):
        b, r, _, ed, _ = low_rank_matrix

        tol = 0.5
        _, ad, _ = randomized_svd(b, tol=tol, max_rank=r)
        npt.assert_array_almost_equal(ad, ed[ed >= tol])

        tol = 0.
        _, ad, _ = randomized_svd(b, tol=tol, max_rank=r)
        npt.assert_array_almost_equal(ad, ed)

        tol = 1.
        _, ad, _ = randomized_svd(b, tol=tol, max_rank=r)
        npt.assert_array_almost_equal(ad, 0)

        if r > 3:
            tol = ed[3] + 1e-5
            _, ad, _ = randomized_svd(b, tol=tol, max_rank=r)
            npt.assert_array_almost_equal(ad, ed[:3], decimal=3)
