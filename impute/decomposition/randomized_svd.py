from typing import Tuple, Optional

import numpy as np
import numpy.random as npr

from numpy.linalg import eigh, qr
from scipy.linalg import polar

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

    y = qr(y)[0]

    return y


def randomized_expander(
        a: np.ndarray,
        q: np.ndarray,
        s: np.ndarray,
        n_col: int = 10,
        n_iter: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # step A
    m = a.shape[1]
    p = n_col

    w = npr.randn(m, p)
    y = a @ w
    del w

    qy = partial_orthogonalization(y, q, overwrite_y=True)
    q = np.append(q, qy, axis=1)

    for _ in range(n_iter):
        q = a @ (a.T @ q)
        q = qr(q)[0]

    # step B
    h, c = qr(a.T @ q)
    w, p = polar(c)
    v, d = sym_eig(p)

    s = np.append(s, d)

    return q @ v, s, (h @ w @ v).T


def randomized_svd(
        a: np.ndarray,
        tol: Optional[float] = None,
        rank: Optional[int] = None,
        n_oversamples: Optional[int]=10,
        n_iter='auto',
        transpose='auto') -> SVD:
    n_row, n_col = a.shape
    max_cols = rank + n_oversamples

    if n_iter == 'auto':
        n_iter = 7 if rank < .1 * min(a.shape) else 4

    if transpose == 'auto':
        transpose = n_row < n_col

    if transpose:
        a = a.T

    u = np.zeros((n_row, 0), dtype=np.float64)
    s = np.zeros((0,), dtype=np.float64)
    v = None
    if tol is None:
        u, s, v = randomized_expander(a, u, s, max_cols, n_iter)

    while u.shape[1] < max_cols:
        u, s, v = randomized_expander(a, u, s, max_cols, n_iter)

    u, v = svd_flip(u, v)

    if transpose:
        u, s, v = v.T, s, u.T

    return SVD(u, s, v).trim(rank)
