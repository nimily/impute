from typing import Optional, Tuple

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


def _fixed_rank_svd(
        z: np.ndarray,
        rank: int = 5,
        n_oversamples: int = 10,
        n_iter='auto') -> SVD:
    m, n = z.shape

    transpose = m < n

    if transpose:
        z = z.T
        m, n = n, m

    if n_iter == 'auto':
        n_iter = 7 if rank < .1 * n else 4

    u = np.zeros((m, 0), dtype=np.float64)

    n_col = rank + n_oversamples
    u, s, v = randomized_expander(z, u, n_col, n_iter)

    u, v = svd_flip(u, v)

    if transpose:
        u, s, v = v.T, s, u.T

    return SVD(u, s, v).trim(rank=rank)


def randomized_svd(
        z: np.ndarray,
        tol: float = 0.0,
        guess: int = 5,
        max_rank: Optional[int] = None) -> SVD:
    if max_rank is None:
        max_rank = min(*z.shape)

    u, s, v = np.zeros(1), np.zeros(1), np.zeros(1)

    stop = False
    rank = guess
    while not stop:
        u, s, v = _fixed_rank_svd(z, rank, n_oversamples=10, n_iter=7)

        if s[-1] < tol or rank == max_rank:
            stop = True
        else:
            rank = min(max_rank, 2 * rank)

    return SVD(u, s, v).trim(thresh=tol, rank=max_rank)
