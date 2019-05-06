from typing import NamedTuple

import numpy as np
import numpy.linalg as npl


class SVD(NamedTuple):
    u: np.ndarray
    s: np.ndarray
    v: np.ndarray

    @property
    def rank(self):
        return sum(self.s > 1e-7)

    @property
    def t(self):
        return SVD(self.v.T, self.s, self.u.T)

    @staticmethod
    def to_svd(w: np.ndarray) -> 'SVD':
        u, s, v = npl.svd(w, full_matrices=False)

        return SVD(u, s, v).trim()

    def to_matrix(self) -> np.ndarray:
        return self.u @ np.diag(self.s) @ self.v

    def trim(self, r=None) -> 'SVD':
        if r:
            r = min(r, sum(self.s > 0))
        else:
            r = sum(self.s > 0)

        r = max(r, 1)

        u = self.u[:, :r]
        s = self.s[:r]
        v = self.v[:r, :]

        return SVD(u, s, v)
