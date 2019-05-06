import numpy as np
import numpy.linalg as npl

from .base import SVD


def exact_svd(
        w: np.ndarray,
        *_, **__) -> SVD:

    u, s, v = npl.svd(w, full_matrices=False)

    return SVD(u, s, v)
