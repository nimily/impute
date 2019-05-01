from typing import Tuple

import numpy as np
import numpy.linalg as npl
import numpy.random as npr


class SVD:

    def __init__(self, u, s, v):
        self.u = u
        self.s = s
        self.v = v

    @staticmethod
    def to_svd(w) -> 'SVD':
        u, s, v = npl.svd(w, full_matrices=False)

        return SVD(u, s, v).trim()

    def to_matrix(self):
        return self.u @ np.diag(self.s) @ self.v

    def trim(self):
        r = max(1, len(self.s[self.s > 0]))

        u = self.u[:, :r]
        s = self.s[:r]
        v = self.v[:r, :]

        return SVD(u, s, v)


def svt(w, thresh):
    u, s, v = npl.svd(w, full_matrices=False)

    return SVD(u, thresh(s), v).trim()


def soft_svt(w, level):
    func = np.vectorize(lambda x: max(x - level, 0))

    return svt(w, func)


def hard_svt(w, level):
    func = np.vectorize(lambda x: 0 if x < level else x)

    return svt(w, func)


def one_hot(shape, pos, val: float = 1) -> np.ndarray:
    x = np.zeros(shape)
    x[pos] = val

    return x


def random_one_hot(shape) -> Tuple[Tuple[int, ...], float, np.ndarray]:
    if isinstance(shape, tuple):
        pos = tuple(npr.randint(s) for s in shape)
    else:
        pos = (npr.randint(shape), )

    val = npr.rand()

    assert all(isinstance(p, int) for p in pos)
    assert isinstance(val, float)

    return pos, val, one_hot(shape, pos, val)
