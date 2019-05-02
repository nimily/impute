from typing import Optional, Callable

import numpy as np
import numpy.linalg as npl

from .base import SVD, soft_thresh, hard_thresh


def exact_svd(
        w: np.ndarray,
        initial: Optional[SVD] = None,
        tolerance: Optional[float] = None,
        n_components: Optional[int] = None,
        thresh: Optional[Callable] = None) -> SVD:
    u, s, v = npl.svd(w, full_matrices=False)

    if thresh:
        s = thresh(s)

    if n_components:
        u = u[:, :n_components]
        s = s[:n_components]
        v = v[:n_components, :]

    return SVD(u, s, v).trim()


def exact_soft_svt(
        w: np.ndarray,
        level: float = 0.0,
        initial: Optional[SVD] = None,
        tolerance: Optional[float] = None,
        n_components: Optional[int] = None) -> SVD:
    thresh = soft_thresh(level)

    return exact_svd(w, initial, tolerance, n_components, thresh)


def exact_hard_svt(
        w: np.ndarray,
        level: float = 0.0,
        initial: Optional[SVD] = None,
        tolerance: Optional[float] = None,
        n_components: Optional[int] = None) -> SVD:
    thresh = hard_thresh(level)

    return exact_svd(w, initial, tolerance, n_components, thresh)
