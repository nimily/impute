import numpy as np
import numpy.testing as npt

import pytest

from impute.decomposition.randomized_svd import randomized_svd


@pytest.mark.usefixtures('low_rank_matrix')
class TestRandomizedSvd:

    @staticmethod
    def test_fixed_rank(low_rank_matrix):
        b, r, _, ed, _ = low_rank_matrix

        au, ad, av = randomized_svd(b, rank=r)

        npt.assert_array_almost_equal(ad, ed)

        actual = au.T @ au
        expect = np.eye(actual.shape[0])

        npt.assert_almost_equal(actual, expect)

        actual = av @ av.T
        expect = np.eye(actual.shape[0])

        npt.assert_almost_equal(actual, expect)

        actual = au @ np.diag(ad) @ av
        expect = b

        npt.assert_array_almost_equal(actual, expect)
