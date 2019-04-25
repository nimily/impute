import numpy as np
import numpy.testing as npt

from impute.measurement import EntryMeasurement
from impute.sample_set import SampleSet, RowSampleSet, EntrySampleSet


def test_sample_set_value():
    shape = 5, 5

    rows = [i for i in range(shape[0])] * 2
    cols = [j // 2 for j in range(2 * shape[0])]
    vals = [i ** 0.5 for i in range(shape[0] * 2)]

    xs = [EntryMeasurement(shape, r, c, v) for r, c, v in zip(rows, cols, vals)]
    ys = [i ** 2 for i in range(2 * shape[0])]

    ss = SampleSet(shape)
    ss.add_all_obs(xs, ys)

    m = np.zeros(shape)
    m[2, 3] = 2.5
    actual = ss.value(m)
    for i in range(shape[0]):
        assert actual[i] == (i == 7) * 2.5 * i ** 0.5


def test_sample_set_adj_value():
    shape = 5, 5

    rows = [i for i in range(shape[0])] * 2
    cols = [j // 2 for j in range(2 * shape[0])]
    vals = [i ** 0.5 for i in range(shape[0] * 2)]

    xs = [EntryMeasurement(shape, r, c, v) for r, c, v in zip(rows, cols, vals)]
    ys = [i ** 2 for i in range(2 * shape[0])]

    ss = SampleSet(shape)
    ss.add_all_obs(xs, ys)

    v = [0, 1] * shape[0]
    actual = ss.adj_value(v)
    expect = np.array([
        [0, 0, 5, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 7, 0],
        [0, 3, 0, 0, 0],
        [0, 0, 0, 0, 9],
    ]) ** 0.5
    for i in range(shape[0]):
        for j in range(shape[0]):
            assert actual[i, j] == expect[i, j]


def test_sample_set_rss_grad():
    shape = 5, 5

    rows = [i for i in range(shape[0])] * 2
    cols = [j // 2 for j in range(2 * shape[0])]
    vals = [i ** 0.5 for i in range(shape[0] * 2)]

    xs = [EntryMeasurement(shape, r, c, v) for r, c, v in zip(rows, cols, vals)]
    ys = [i ** 1.5 for i in range(2 * shape[0])]

    ss = SampleSet(shape)
    ss.add_all_obs(xs, ys)

    m = np.zeros(shape)
    m[2, 3] = 3
    m[1, 2] = 2
    actual = ss.rss_grad(m)
    expect = - np.array([
        [0, 0, 25, 0, 0],
        [1, 0, 0, 36, 0],
        [0, 4, 0, 28, 0],
        [0, 9, 0, 0, 64],
        [0, 0, 16, 0, 81],
    ])
    npt.assert_array_almost_equal(actual, expect, 4)


def test_sample_set_op_norm():
    shape = 5, 5

    rows = [i for i in range(shape[0])] * 2
    cols = [j // 2 for j in range(2 * shape[0])]
    vals = [i ** 0.5 for i in range(shape[0] * 2)]

    xs = [EntryMeasurement(shape, r, c, v) for r, c, v in zip(rows, cols, vals)]
    ys = [i ** 1.5 for i in range(2 * shape[0])]

    ss = SampleSet(shape)
    ss.add_all_obs(xs, ys)

    actual = ss.op_norm()
    expect = max(vals)
    npt.assert_almost_equal(actual, expect)


def test_row_sample_set_op_norm1():
    shape = 5, 5

    rows = [i for i in range(shape[0])] * 2
    cols = [j // 2 for j in range(2 * shape[0])]
    vals = [i ** 0.5 for i in range(shape[0] * 2)]

    xs = [EntryMeasurement(shape, r, c, v) for r, c, v in zip(rows, cols, vals)]
    ys = [i ** 1.5 for i in range(2 * shape[0])]

    ss = RowSampleSet(shape)
    ss.add_all_obs(xs, ys)

    actual = ss.op_norm()
    expect = max(vals)
    npt.assert_almost_equal(actual, expect)


def test_row_sample_set_op_norm2():
    shape = 5, 5

    rows = [i for i in range(shape[0])] * 2
    cols = [j // 2 for j in range(2 * shape[0])]
    vals = [i ** 0.5 for i in range(shape[0] * 2)]

    def one_hot(i):
        v = np.zeros(shape[1])
        v[i] = 1

        return v

    xs = [(r, one_hot(c) * v) for r, c, v in zip(rows, cols, vals)]
    ys = [i ** 1.5 for i in range(2 * shape[0])]

    ss = RowSampleSet(shape)
    ss.add_all_obs(xs, ys)

    actual = ss.op_norm()
    expect = max(vals)
    npt.assert_almost_equal(actual, expect)


def test_entry_sample_set_op_norm1():
    shape = 5, 5

    rows = [i for i in range(shape[0])] * 2
    cols = [j // 2 for j in range(2 * shape[0])]
    vals = [i ** 0.5 for i in range(shape[0] * 2)]

    xs = [EntryMeasurement(shape, r, c, v) for r, c, v in zip(rows, cols, vals)]
    ys = [i ** 1.5 for i in range(2 * shape[0])]

    ss = EntrySampleSet(shape)
    ss.add_all_obs(xs, ys)

    actual = ss.op_norm()
    expect = max(vals)
    npt.assert_almost_equal(actual, expect)


def test_entry_sample_set_op_norm2():
    shape = 5, 5

    rows = [i for i in range(shape[0])] * 2
    cols = [j // 2 for j in range(2 * shape[0])]
    vals = [i ** 0.5 for i in range(shape[0] * 2)]

    xs = list(zip(rows, cols, vals))
    ys = [i ** 1.5 for i in range(2 * shape[0])]

    ss = EntrySampleSet(shape)
    ss.add_all_obs(xs, ys)

    actual = ss.op_norm()
    expect = max(vals)
    npt.assert_almost_equal(actual, expect)
