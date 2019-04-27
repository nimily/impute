from itertools import product

import numpy.random as npr
import numpy.testing as npt

import pytest

from impute.sample_set import EntrySampleSet
# from impute.soft import SoftImpute
from impute.bregman import BregmanImpute


@pytest.mark.parametrize(
    'seed',
    [1]
)
def test_exact_fpc(seed):
    npr.seed(seed)

    m, n, r = 40, 40, 2
    p = 800

    shape = m, n

    bl = npr.randn(m, r)
    br = npr.randn(n, r)

    b = bl @ br.T

    entries = list(product(range(n), range(m)))
    indices = npr.choice(range(n * m), size=p, replace=False)

    xs = [entries[i] + (1, ) for i in indices]
    ys = [b[i, j] for i, j, _ in xs]

    ss = EntrySampleSet(shape)
    ss.add_all_obs(xs, ys)

    # soft = SoftImpute(shape)
    breg = BregmanImpute(shape)

    bh = breg.impute(ss, alpha=1e-8).to_matrix()

    import numpy.linalg as npl
    rel_err = npl.norm(bh - b, 'fro') / npl.norm(b, 'fro')
    print(rel_err)

    npt.assert_almost_equal(bh, b)
