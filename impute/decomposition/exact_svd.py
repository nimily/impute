from typing import Optional

import numpy as np
import numpy.linalg as npl

from .base import SVD


def exact_svd(
        w: np.ndarray,
        tol: Optional[float] = None,
        rank: Optional[int] = None) -> SVD:
    n, m = w.shape

    if rank is None:
        rank = min(n, m)

    u, s, v = npl.svd(w, full_matrices=False)

    if tol is not None:
        rank = min(rank, max(sum(s > tol), 1))

    return SVD(u, s, v).trim(rank)
