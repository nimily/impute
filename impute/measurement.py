import abc

import numpy as np


class Measurement:

    @abc.abstractmethod
    def as_matrix(self):
        pass

    @property
    def shape(self):
        return self.n_rows, self.n_cols

    @property
    @abc.abstractmethod
    def n_rows(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def n_cols(self) -> int:
        pass

    def add_to(self, m, scale):
        m += scale * self.as_matrix()

    def sense(self, m):
        return self.as_matrix().dot(m.T).trace()


class FullMeasurement(Measurement):

    def __init__(self, val):
        self.val = val

    @property
    def n_rows(self):
        return self.val.shape[0]

    @property
    def n_cols(self):
        return self.val.shape[1]

    def as_matrix(self):
        return self.val.copy()


class RowMeasurement(FullMeasurement):

    def __init__(self, n_rows, i, val):
        super().__init__(((n_rows, i), val))

    @property
    def shape(self):
        return self.n_rows, self.n_cols

    @property
    def n_rows(self):
        return self.val[0][0]

    @property
    def n_cols(self):
        return self.val[1].shape[0]

    @property
    def row_index(self):
        return self.val[0][1]

    @property
    def row_value(self):
        return self.val[1]

    def as_matrix(self):
        x = np.zeros(self.shape)

        i = self.row_index
        v = self.row_value
        x[i] = v

        return x

    def add_to(self, m, scale):
        i = self.row_index
        v = self.row_value

        m[i] += scale * v

    def sense(self, m):
        i = self.row_index
        v = self.row_value

        return v @ m[i]
  

class EntryMeasurement(RowMeasurement):

    def __init__(self, shape, i, j, val):
        n_rows, n_cols = shape
        super().__init__(n_rows, i, (n_cols, j, val))

    @property
    def n_cols(self):
        return self.val[1][0]

    @property
    def col_index(self):
        return self.val[1][1]

    @property
    def entry_value(self):
        return self.val[1][2]

    @property
    def row_value(self):
        j = self.col_index
        v = np.zeros(self.n_cols)
        v[j] = self.entry_value

        return v

    def add_to(self, m, scale):
        i = self.row_index
        j = self.col_index
        v = self.entry_value

        m[i, j] += scale * v

    def sense(self, m):
        i = self.row_index
        j = self.col_index
        v = self.entry_value

        return v * m[i, j]
