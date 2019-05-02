import numpy as np
import numpy.random as npr
import numpy.testing as npt

from impute.decomposition.randomized_svd import randomized_expander as expander
from impute.decomposition.randomized_svd import randomized_svd as svd
from sklearn.utils.extmath import randomized_svd as sk_svd


def test():
    npr.seed(1)

    n = 500
    m = 400
    r = 20

    bl = npr.randn(n, r)
    br = npr.randn(m, r)
    b = bl @ br.T

    # q = np.zeros((n, 0))
    # s = np.zeros((0,))
    # q, s, v = expander(b, q, s)

    # assert q.shape == (n, 10)
    # assert s.shape == (10, )
    #
    # print(q.T @ q)

    u, actual_d, v = svd(b, rank=r)
    _, expect_d, _ = sk_svd(b, r)

    npt.assert_array_almost_equal(actual_d, expect_d)

    actual = u.T @ u
    expect = np.eye(actual.shape[0])

    npt.assert_almost_equal(actual, expect)

    actual = v @ v.T
    expect = np.eye(actual.shape[0])

    npt.assert_almost_equal(actual, expect)
