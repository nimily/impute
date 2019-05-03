from typing import Tuple, Optional

from math import ceil

import numpy as np
import numpy.random as npr

from numpy.linalg import eigh
from scipy.linalg import qr, polar

from sklearn.utils.extmath import svd_flip

from .base import SVD


def sym_eig(x):
    d, v = eigh(x)

    d = np.flip(d)
    v = np.flip(v, axis=1)

    return v, d


def partial_orthogonalization(
        y: np.ndarray,
        q: np.ndarray,
        overwrite_y: bool = False) -> np.ndarray:
    if not overwrite_y:
        y = y.copy()

    y -= q @ (q.T @ y)

    y = qr(y, mode='economic')[0]

    return y


def randomized_expander(
        z: np.ndarray,
        q: np.ndarray,
        n_col: int = 10,
        n_iter: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # step A
    m = z.shape[1]
    p = min(n_col, m - q.shape[1])

    w = npr.randn(m, p)
    y = z @ w
    del w

    qy = partial_orthogonalization(y, q, overwrite_y=True)
    q = np.append(q, qy, axis=1)

    for _ in range(n_iter):
        q = z @ (z.T @ q)
        q = qr(q, mode='economic')[0]

    # step B
    h, c = qr(z.T @ q, mode='economic')
    w, p = polar(c)
    v, d = sym_eig(p)

    return q @ v, d, (h @ w @ v).T


def randomized_svd(
        z: np.ndarray,
        tol: Optional[float] = None,
        max_rank: Optional[int] = None,
        n_oversamples: int = 10,
        n_iter='auto',
        a: int = 2,
        rho: float = 0.05,
        gamma: float = 0.1) -> SVD:
    m, n = z.shape

    transpose = m < n

    if transpose:
        z = z.T
        m, n = n, m

    if max_rank is None:
        max_rank = n

    if n_iter == 'auto':
        n_iter = 7 if max_rank < .1 * n else 4

    u = np.zeros((m, 0), dtype=np.float64)
    s = np.zeros((0, ), dtype=np.float64)
    v = None

    if tol is None:
        # fixed-rank case
        n_col = max_rank + n_oversamples
        u, s, v = randomized_expander(z, u, n_col, n_iter)

    else:
        # fixed-precision case
        b = ceil(gamma * n)
        r = 0
        l = 0.1 * b

        converged = False
        while not converged:
            p = a if r < l else ceil(rho * n)
            l = min(r + p, b)
            r = sum(s >= tol)

            u, s, v = randomized_expander(z, u, p, n_iter)

            converged = min(s) < tol and u.shape[1] < n

        max_rank = min(max_rank, sum(s > tol))

    if max_rank == 0:
        u = np.ones((m, 1)) / m ** 0.5
        s = np.zeros((1, ))
        v = np.ones((n, 1)) / n ** 0.5

        max_rank = 1

    u, v = svd_flip(u, v)

    if transpose:
        u, s, v = v.T, s, u.T

    return SVD(u, s, v).trim(max_rank)
