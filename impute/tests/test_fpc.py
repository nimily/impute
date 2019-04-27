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
    "rank",
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
