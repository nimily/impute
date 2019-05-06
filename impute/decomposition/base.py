from typing import NamedTuple

import numpy as np
import numpy.linalg as npl


class SVD(NamedTuple):
    u: np.ndarray
    s: np.ndarray
    v: np.ndarray

    @property
    def shape(self):
        return self.u.shape[0], self.v.shape[1]

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

    def trim(self, thresh: float = 0, rank=None) -> 'SVD':
        if rank:
            rank = min(rank, sum(self.s > thresh))
        else:
            rank = sum(self.s > thresh)

        if rank == 0:

            u = self.u[:, :1]
            s = np.zeros(1)
            v = self.v[:1, :]

            return SVD(u, s, v)

        u = self.u[:, :rank]
        s = self.s[:rank]
        v = self.v[:rank, :]

        return SVD(u, s, v)
