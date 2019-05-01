import numpy as np
import numpy.random as npr
import numpy.testing as npt

import pytest

from impute import Dataset
from impute.linear_ops import EntryTraceLinearOp


def create_alternating_entry_dataset(shape, b, sd=1.0) -> Dataset:
    n_rows, n_cols = shape

    op = EntryTraceLinearOp(shape)
    ys = []
    for i in range(n_rows):
        for j in range(n_cols):
            if (i + j) % 2 == 0:
                x = (i, j, 1)
                y = b[i, j] + npr.randn() * sd

                op.append(x)
                ys.append(y)

    return Dataset(op, ys)


def create_randomized_entry_dataset(seed, shape, n_sample) -> Dataset:
    npr.seed(seed)


@pytest.fixture(params=[
    (1, (50, 45), 2),
    (2, (100, 80), 5),
], name='nae_case')  # noiseless alternating-entry dataset
def nae_dataset_fixture(request):

    seed = request.param[0]
    shape = request.param[1]
    rank = request.param[2]

    npr.seed(seed)

    n_row, n_col = shape

    bl = npr.randn(n_row, rank)
    br = npr.randn(n_col, rank)
    b = bl @ br.T

    yield b, create_alternating_entry_dataset(shape, b, sd=0.0)


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
