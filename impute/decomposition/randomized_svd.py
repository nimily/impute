from typing import Optional

import numpy as np

from numpy.linalg import svd
from sklearn.decomposition import TruncatedSVD

from .base import SVD


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
        trunc_svd = TruncatedSVD(n_components=rank, n_iter=20)
        trunc_svd.fit(z)

        v = np.array(trunc_svd.components_)
        u, s, _ = svd(z @ v.T)

        if s[-1] < tol or rank == max_rank:
            stop = True
        else:
            rank = min(max_rank, 2 * rank)

    return SVD(u, s, v).trim(thresh=tol, rank=max_rank)
