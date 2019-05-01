from typing import Tuple, List

import numpy as np
import numpy.random as npr
import numpy.testing as npt

import pytest

from impute.linear_ops import DotLinearOp, DenseTraceLinearOp, RowTraceLinearOp, EntryTraceLinearOp
from impute.utils import one_hot


def random_one_hot(shape) -> Tuple[Tuple[int, ...], float, np.ndarray]:
    pos = tuple(npr.randint(s) for s in shape)
    val = npr.rand()

    assert all(isinstance(p, int) for p in pos)
    assert isinstance(val, float)

    return pos, val, one_hot(shape, pos, val)


def create_dot_op(shape, n_sample):
    op = DotLinearOp(shape)

    entries = [random_one_hot(shape) for _ in range(n_sample)]
    op.extend([x for _, _, x in entries])

    return op, entries


@pytest.mark.parametrize(
    'seed',
    range(5)
)
def test_dot_linear_op_norm(seed):
    npr.seed(seed)

    shape = (25,)

    n_sample = 1000

    op, entries = create_dot_op(shape, n_sample)

    m = np.zeros(shape)
    for p, v, _ in entries:
        m[p] += v ** 2

    actual = op.xxt.norm()
    expect = m.max()

    npt.assert_array_almost_equal(actual, expect)


class TraceData:

    def __init__(self, shape):
        self.shape = shape

        self.pos: List[Tuple[int, ...]] = []
        self.val: List[float] = []
        self.xs: List[np.ndarray] = []

        self.b: np.ndarray = np.zeros((0,))
        self.bt: np.ndarray = np.zeros((0,))

        self.expect_evaluate: np.ndarray = np.zeros((0,))
        self.expect_evaluate_t: np.ndarray = np.zeros((0,))

        self.expect_xtx: np.ndarray = np.zeros((0,))
        self.expect_xxt: np.ndarray = np.zeros((0,))

        self.expect_norm: float = 0.0


def create_data(seed, shape, n_sample):
    npr.seed(seed)

    data = TraceData(shape)
    for _ in range(n_sample):
        p, v, x = random_one_hot(shape)

        data.pos.append(p)
        data.val.append(v)
        data.xs.append(x)

    data.b = npr.randn(*shape)
    data.bt = npr.randn(n_sample)

    data.expect_evaluate = np.array([data.b[p] * v for p, v in zip(data.pos, data.val)])
    data.expect_evaluate_t = sum(c * x for c, x in zip(data.bt, data.xs))

    data.expect_xtx = sum(data.b[p] * v * x for p, v, x in zip(data.pos, data.val, data.xs))
    data.expect_xxt = np.array([
        data.expect_evaluate_t[p] * v for p, v in zip(data.pos, data.val)
    ])

    m = np.zeros(shape)
    rows = [r for r, _ in data.pos]
    cols = [c for _, c in data.pos]
    vals = np.power(data.val, 2)
    np.add.at(m, (rows, cols), vals)
    data.expect_norm = m.max() ** 0.5

    return data


def add_to_dense_op(op, data):
    op.extend(data.xs)


def add_to_row_op(op, data):
    op.extend([
        (r, one_hot(data.shape[1], c, v)) for (r, c), v in zip(data.pos, data.val)
    ])


def add_to_entry_op(op, data):
    op.extend([
        (r, c, v) for (r, c), v in zip(data.pos, data.val)
    ])


@pytest.fixture(params=[
    (1, (25, 20), 400),
    (2, (25, 20), 400),
    (3, (25, 20), 400),
    (4, (100, 100), 100),
    (5, (100, 100), 100),
], name='data')
def data_fixture(request):
    yield create_data(*request.param)


@pytest.fixture(params=[
    (DenseTraceLinearOp, add_to_dense_op),
    (RowTraceLinearOp, add_to_row_op),
    (EntryTraceLinearOp, add_to_entry_op),
], ids=[
    'dense',
    'row',
    'entry',
], name='op_info')
def op_info_fixture(request):
    yield request.param


@pytest.mark.usefixtures('op_info', 'data')
class TestTraceLinearOp:

    @staticmethod
    def test_evaluate(op_info, data):
        op = TestTraceLinearOp.get_loaded_op(op_info, data)

        actual = op.evaluate(data.b)
        expect = data.expect_evaluate

        npt.assert_array_almost_equal(actual, expect)

    @staticmethod
    def test_evaluate_t(op_info, data):
        op = TestTraceLinearOp.get_loaded_op(op_info, data)

        actual = op.evaluate_t(data.bt)
        expect = data.expect_evaluate_t

        npt.assert_array_almost_equal(actual, expect)

    @staticmethod
    def test_xtx(op_info, data):
        op = TestTraceLinearOp.get_loaded_op(op_info, data)

        actual = op.xtx(data.b)
        expect = data.expect_xtx

        npt.assert_array_almost_equal(actual, expect)

    @staticmethod
    def test_xxt(op_info, data):
        op = TestTraceLinearOp.get_loaded_op(op_info, data)

        actual = op.xxt(data.bt)
        expect = data.expect_xxt

        npt.assert_array_almost_equal(actual, expect)

    @staticmethod
    def test_norm(op_info, data):
        op = TestTraceLinearOp.get_loaded_op(op_info, data)

        actual = op.norm()
        expect = data.expect_norm

        npt.assert_array_almost_equal(actual, expect)

    @staticmethod
    def get_loaded_op(op_info, data):
        op_cls, op_add = op_info

        op = op_cls(data.shape)
        op_add(op, data)

        return op
