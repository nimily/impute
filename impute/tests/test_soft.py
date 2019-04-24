import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import numpy.testing as npt

from impute.measurement import EntryMeasurement
from impute.sample_set import EntrySampleSet
from impute.soft import SoftImpute


def test_alpha_max():
    npr.seed(314159265)

    shape = 20, 20
    n_rows, n_cols = shape

    ss = EntrySampleSet(shape)

    for i in range(n_rows):
        for j in range(n_cols):
            if (i + j) % 2 == 1:
                x = EntryMeasurement(shape, i, j, 1)
                y = 1 + npr.random()

                ss.add_obs(x, y)

    imputer = SoftImpute(shape)

    alpha = imputer.alpha_max(ss)
    zs = imputer.fit(ss, [alpha, alpha * 0.999])

    actual = zs[0]
    expect = np.zeros(shape)
    npt.assert_array_almost_equal(actual, expect)

    actual = npl.norm(zs[1])
    assert actual > 1e-5
