import abc

import numpy as np


class Measurement:

    @abc.abstractmethod
    def as_matrix(self):
        pass

    def add_to(self, m, scale):
        m += scale * self.as_matrix()

    def sense(self, m):
        return self.as_matrix().dot(m.T).trace()


class EntryMeasurement(Measurement):

    def __init__(self, shape, row, col, val):
        self.shape = shape

        self.row = row
        self.col = col
        self.val = val

    def as_matrix(self):
        x = np.zeros(self.shape)

        x[self.row, self.col] = self.val

        return x

    def add_to(self, m, scale):
        m[self.row, self.col] += scale * self.val

    def sense(self, m):
        return self.val * m[self.row, self.col]


class RowMeasurement(Measurement):

    def __init__(self, shape, row, val):
        self.shape = shape

        self.row = row
        self.val = val

    def as_matrix(self):
        x = np.zeros(self.shape)

        x[self.row] = self.val

        return x

    def add_to(self, m, scale):
        m[self.row] += scale * self.val

    def sense(self, m):
        return self.val.dot(m[self.row])


class FullMeasurement(Measurement):

    def __init__(self, val):
        self.val = val

    def as_matrix(self):
        return self.val.copy()
