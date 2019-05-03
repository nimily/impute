import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import numpy.testing as npt

from impute.decomposition.randomized_svd import randomized_svd as svd


def test():
    npr.seed(1)

    n = 1500
    m = 1000
    r = 10

    eu = npl.qr(npr.randn(n, r))[0]
    ed = np.arange(r, 0, -1)
    ev = npl.qr(npr.randn(m, r))[0]
    b = eu @ np.diag(ed) @ ev.T

    au, ad, av = svd(b, rank=r)

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
