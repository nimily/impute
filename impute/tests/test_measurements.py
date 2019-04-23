import numpy as np
import numpy.testing as npt

from impute.measurement import EntryMeasurement
from impute.measurement import RowMeasurement
from impute.measurement import FullMeasurement


def test_entry_measurements():
    shape = 5, 5
    i, j, v = 1, 4, 3

    m = EntryMeasurement(shape, i, j, v)

    assert m.as_matrix()[i, j] == v

    total = np.zeros(shape)

    m.add_to(total, 0.5)
    assert total[i, j] == v * 0.5

    b = np.arange(shape[0] * shape[1]).reshape(shape)
    assert m.sense(b) == v * b[i, j]


def test_row_measurements():
    n_rows, n_cols = 5, 5
    i, v = 1, np.arange(n_cols)

    m = RowMeasurement(n_rows, i, v)

    as_matrix = m.as_matrix()
    for j in range(n_rows):
        if j == i:
            npt.assert_array_almost_equal(as_matrix[j], v)
        else:
            npt.assert_array_almost_equal(as_matrix[j], np.zeros(n_cols))

    total = np.zeros((n_rows, n_cols))

    m.add_to(total, 0.5)
    for j in range(n_rows):
        if j == i:
            npt.assert_array_almost_equal(total[j], v * 0.5)
        else:
            npt.assert_array_almost_equal(total[j], np.zeros(n_cols))

    b = np.arange(n_rows * n_cols).reshape((n_rows, -1))
    npt.assert_array_almost_equal(m.sense(b), v @ b[i])


def test_full_measurements():
    shape = 5, 5
    data = np.arange(shape[0] * shape[1]).reshape(shape)

    m1 = FullMeasurement(data)

    as_matrix = m1.as_matrix()
    for i in range(shape[0]):
        for j in range(shape[1]):
            assert as_matrix[i, j] == i * shape[1] + j

    total = np.zeros(shape)

    m1.add_to(total, 0.5)
    for i in range(shape[0]):
        for j in range(shape[1]):
            assert total[i, j] == (i * shape[1] + j) * 0.5

    data = np.arange(shape[0] * shape[1], 0, -1).reshape(shape)
    m2 = FullMeasurement(data)

    m2.add_to(total, 0.5)
    for i in range(shape[0]):
        for j in range(shape[1]):
            assert total[i, j] == (shape[0] * shape[1]) * 0.5
