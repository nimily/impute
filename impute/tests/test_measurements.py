import numpy as np
import numpy.linalg as npl

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
    assert m.sense(b) == v * (b[i, j])


def test_row_measurements():
    shape = 5, 5
    i, v = 1, np.arange(shape[1])

    m = RowMeasurement(shape, i, v)

    for j in range(shape[1]):
        assert m.as_matrix()[i, j] == v[j]

    total = np.zeros(shape)

    m.add_to(total, 0.5)
    for j in range(shape[1]):
        assert total[i, j] == v[j] * 0.5

    b = np.arange(shape[0] * shape[1]).reshape(shape)
    assert npl.norm(m.sense(b) - v @ b[i]) <= 1e-5


def test_full_measurements():
    shape = 5, 5
    data = np.arange(shape[0] * shape[1]).reshape(shape)

    m1 = FullMeasurement(data)

    for i in range(shape[0]):
        for j in range(shape[1]):
            assert m1.as_matrix()[i, j] == i * shape[1] + j

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
