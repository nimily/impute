import numpy as np
import numpy.linalg as npl


class SVD:

    def __init__(self, u: np.ndarray, s: np.ndarray, v: np.ndarray):
        self.u = u
        self.s = s
        self.v = v

    @staticmethod
    def to_svd(w: np.ndarray) -> 'SVD':
        u, s, v = npl.svd(w, full_matrices=False)

        return SVD(u, s, v).trim()

    def to_matrix(self) -> np.ndarray:
        return self.u @ np.diag(self.s) @ self.v

    def trim(self) -> 'SVD':
        r = max(1, len(self.s[self.s > 0]))

        u = self.u[:, :r]
        s = self.s[:r]
        v = self.v[:r, :]

        return SVD(u, s, v)


def soft_thresh(level):
    return np.vectorize(lambda x: 0 if x < level else x - level)


def hard_thresh(level):
    return np.vectorize(lambda x: 0 if x < level else x)

