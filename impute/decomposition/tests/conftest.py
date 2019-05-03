import numpy as np
import numpy.linalg as npl
import numpy.random as npr

import pytest


@pytest.fixture(params=[
    (1, 50, 40, 2),
    (2, 90, 100, 5),
    (3, 1500, 1000, 10),
], name='low_rank_matrix')
def low_rank_matrix_fixture(request):
    seed, n, m, r = request.param

    npr.seed(seed)

    u = npl.qr(npr.randn(n, r))[0]
    s = np.sort(npr.uniform(0, 1, r))[::-1]
    v = npl.qr(npr.randn(m, r))[0]
    b = u @ np.diag(s) @ v.T

    return b, r, u, s, v
