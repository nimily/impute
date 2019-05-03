from typing import Optional

import numpy as np
import numpy.linalg as npl

from .base import SVD, soft_thresh, hard_thresh


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


def exact_soft_svt(
        w: np.ndarray,
        tol: Optional[float] = None,
        rank: Optional[int] = None) -> SVD:
    thresh = soft_thresh(tol)

    u, s, v = exact_svd(w, tol, rank)

    return SVD(u, thresh(s), v)


def exact_hard_svt(
        w: np.ndarray,
        tol: Optional[float] = None,
        rank: Optional[int] = None) -> SVD:
    thresh = hard_thresh(tol)

    u, s, v = exact_svd(w, tol, rank)

    return SVD(u, thresh(s), v)
