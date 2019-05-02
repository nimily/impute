from typing import Optional, Callable

import numpy as np
import numpy.linalg as npl

from .base import SVD, soft_thresh, hard_thresh


def exact_svd(
        w: np.ndarray,
        thresh: Optional[Callable] = None,
        init: Optional[SVD] = None,
        rank: Optional[int] = None,
        precision: Optional[float] = None) -> SVD:
    u, s, v = npl.svd(w, full_matrices=False)

    if thresh:
        s = thresh(s)

    if rank:
        u = u[:, :rank]
        s = s[:rank]
        v = v[:rank, :]

    return SVD(u, s, v).trim()


def exact_soft_svt(
        w: np.ndarray,
        level: float = 0.0,
        init: Optional[SVD] = None,
        rank: Optional[int] = None,
        precision: Optional[float] = None) -> SVD:
    thresh = soft_thresh(level)

    return exact_svd(w, thresh, init, rank, precision)


def exact_hard_svt(
        w: np.ndarray,
        level: float = 0.0,
        init: Optional[SVD] = None,
        rank: Optional[int] = None,
        precision: Optional[float] = None) -> SVD:
    thresh = hard_thresh(level)

    return exact_svd(w, thresh, init, rank, precision)
