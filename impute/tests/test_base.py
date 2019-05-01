import numpy as np
import numpy.testing as npt

import pytest


@pytest.mark.usefixtures('nae_case')
class TestDataset:

    def test_op(self, nae_case):
        b, ds = nae_case

        actual = ds.op(b)
        expect = ds.ys

        npt.assert_array_almost_equal(actual, expect)

    def test_rss_grad(self, nae_case):
        b, ds = nae_case

        actual = ds.rss_grad(b)
        expect = np.zeros_like(b)

        npt.assert_array_almost_equal(actual, expect)
