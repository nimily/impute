from typing import Tuple, List

import numpy as np
import numpy.random as npr

import pytest

from impute import Dataset
from impute import FpcImpute, SoftImpute
from impute import DotLinearOp, DenseTraceLinearOp, RowTraceLinearOp, EntryTraceLinearOp
from impute.utils import random_one_hot, one_hot


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


def create_randomized_entry_dot_op(shape, n_sample):
    op = DotLinearOp(shape)

    entries = [random_one_hot(shape) for _ in range(n_sample)]
    op.extend([x for _, _, x in entries])

    return op, entries


# trace operator dataset
class TraceLinearOpTestCase:

    # pylint: disable=too-many-instance-attributes
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


def create_trace_linear_op_test_case(seed, shape, n_sample):
    npr.seed(seed)

    data = TraceLinearOpTestCase(shape)
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


#  fixtures
@pytest.fixture(params=[
    (1, 50, 10),
    (2, 100, 1000),
], name='re_dot_case')  # randomized alternating-entry dot-operator
def re_dataset_fixture(request):
    seed = request.param[0]
    shape = request.param[1]
    n_sample = request.param[2]

    npr.seed(seed)

    return create_randomized_entry_dot_op(shape, n_sample)


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

    return b, create_alternating_entry_dataset(shape, b, sd=0.0)


@pytest.fixture(params=[
    (1, (50, 45), 2),
    (2, (100, 80), 5),
    (3, (200, 250), 8),
], name='rae_case')  # randomized alternating-entry dataset
def rae_dataset_fixture(request):
    seed = request.param[0]
    shape = request.param[1]
    rank = request.param[2]

    npr.seed(seed)

    n_row, n_col = shape

    bl = npr.randn(n_row, rank)
    br = npr.randn(n_col, rank)
    b = bl @ br.T

    return b, create_alternating_entry_dataset(shape, b)


@pytest.fixture(
    params=np.logspace(0, -3, num=5, base=2),
    ids=[f'alpha[{i}]' for i in range(5)],
    name='alpha_ratio'
)
def alpha_ratio_fixture(request):
    return request.param


@pytest.fixture(
    params=[FpcImpute, SoftImpute],
    name='imputer_cls'
)
def imputer_cls_fixture(request):
    return request.param


@pytest.fixture(params=[
    (1, (25, 20), 400),
    (2, (25, 20), 400),
    (3, (25, 20), 400),
    (4, (100, 100), 100),
    (5, (100, 100), 100),
], name='trace_op_case')
def trace_op_case_fixture(request):
    yield create_trace_linear_op_test_case(*request.param)


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
