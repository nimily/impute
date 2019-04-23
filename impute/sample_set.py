import numpy as np

from .measurement import Measurement


class SampleSet:

    def __init__(self, shape):
        self.shape = shape

        self.xs = []
        self.ys = []

    def add_obs(self, x: Measurement, y: float):
        self.add_all_obs([x], [y])

    def add_all_obs(self, xs, ys):
        self.xs.extend(xs)
        self.ys.extend(ys)

    def value(self, m):
        return np.array([x.sense(m) for x in self.xs])

    def adj_value(self, v):
        return sum(y * x.as_matrix() for y, x in zip(v, self.xs))

    def rss_grad(self, b):
        return self.adj_value(self.value(b) - self.ys)
