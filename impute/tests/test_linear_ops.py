import numpy as np
import numpy.random as npr
import numpy.testing as npt

import pytest

from impute.linear_ops import TraceLinearOp
from impute.utils import one_hot


def random_one_hot(shape):
    pos = npr.randint(shape[0]), npr.randint(shape[1])
    val = npr.rand()

    return pos, val, one_hot(shape, pos, val)


def create_trace_op(shape, n_sample):
    op = TraceLinearOp(shape)

    entries = [random_one_hot(shape) for _ in range(n_sample)]
    op.add_all([e[2] for e in entries])

    return op, entries


@pytest.mark.parametrize(
    'seed',
    range(5)
)
def test_evaluate(seed):
    npr.seed(seed)

    shape = 50, 50

    n_sample = 100

    op, entries = create_trace_op(shape, n_sample)

    b = npr.rand(*shape)

    actual = op.evaluate(b)
    expect = np.array([b[pos] * val for pos, val, _ in entries])

    npt.assert_array_almost_equal(actual, expect)

    # repeat all entries
    op.add_all(op.xs)

    actual = op.evaluate(b)
    expect = np.array([b[pos] * val for pos, val, _ in entries] * 2)

    npt.assert_array_almost_equal(actual, expect)


@pytest.mark.parametrize(
    'seed',
    range(5)
)
def test_evaluate_t(seed):
    npr.seed(seed)

    shape = 50, 50

    n_sample = 100

    op, entries = create_trace_op(shape, n_sample)

    b = npr.rand(n_sample)

    actual = op.evaluate_t(b)
    expect = sum(c * e[2] for c, e in zip(b, entries))

    npt.assert_array_almost_equal(actual, expect)

    # repeat all entries
    op.add_all(op.xs)

    b = npr.rand(2 * n_sample)

    actual = op.evaluate_t(b)
    expect = sum(c * e[2] for c, e in zip(b[:n_sample] + b[n_sample:], entries))

    npt.assert_array_almost_equal(actual, expect)


@pytest.mark.parametrize(
    'seed',
    range(5)
)
def test_xtx(seed):
    npr.seed(seed)

    shape = 50, 50

    n_sample = 100

    op, entries = create_trace_op(shape, n_sample)

    b = npr.rand(*shape)

    actual = op.xtx(b)
    expect = sum(b[p] * v * x for p, v, x in entries)

    npt.assert_array_almost_equal(actual, expect)


@pytest.mark.parametrize(
    'seed',
    range(5)
)
def test_xxt(seed):
    npr.seed(seed)

    shape = 50, 50

    n_sample = 100

    op, entries = create_trace_op(shape, n_sample)

    b = npr.rand(n_sample)

    actual = op.xxt(b)
    expect = op.evaluate(op.evaluate_t(b))

    npt.assert_array_almost_equal(actual, expect)


@pytest.mark.parametrize(
    'seed',
    range(5)
)
def test_norm(seed):
    npr.seed(seed)

    shape = 25, 25

    n_sample = 1000

    op, entries = create_trace_op(shape, n_sample)

    b = npr.rand(n_sample)

    m = np.zeros(shape)
    for p, v, _ in entries:
        m[p] += v ** 2

    actual = op.norm()
    expect = m.max() ** 0.5

    npt.assert_array_almost_equal(actual, expect)


@pytest.mark.parametrize(
    'seed',
    range(5)
)
def test_xtx_norm(seed):
    npr.seed(seed)

    shape = 25, 25

    n_sample = 1000

    op, entries = create_trace_op(shape, n_sample)

    b = npr.rand(n_sample)

    m = np.zeros(shape)
    for p, v, _ in entries:
        m[p] += v ** 2

    actual = op.xtx.norm()
    expect = m.max()

    npt.assert_array_almost_equal(actual, expect)


@pytest.mark.parametrize(
    'seed',
    range(5)
)
def test_xxt_norm(seed):
    npr.seed(seed)

    shape = 25, 25

    n_sample = 1000

    op, entries = create_trace_op(shape, n_sample)

    b = npr.rand(n_sample)

    m = np.zeros(shape)
    for p, v, _ in entries:
        m[p] += v ** 2

    actual = op.xxt.norm()
    expect = m.max()

    npt.assert_array_almost_equal(actual, expect)
