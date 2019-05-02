from typing import Optional, Callable

import numpy as np
import numpy.random as npr

import scipy.linalg as scl
from scipy.linalg import qr

from sklearn.utils.extmath import svd_flip, safe_sparse_dot

from .base import SVD


def partial_orthogonalization(q: np.ndarray,
                              y: np.ndarray,
                              overwrite_y: bool = False) -> np.ndarray:
    if not overwrite_y:
        y = y.copy()

    y -= q.T @ (q @ y)

    qr(y, overwrite_a=True, mode='economic')

    return y


def randomized_range_finder(
        w: np.ndarray,
        tolerance: Optional[float],
        size: Optional[int],
        n_iter,
        power_iteration_normalizer='auto'):
    # Generating normal random vectors with shape: (A.shape[1], size)
    q = npr.normal(size=(w.shape[1], size))
    if w.dtype.kind == 'f':
        # Ensure f32 is preserved as f32
        q = q.astype(w.dtype, copy=False)

    # Deal with "auto" mode
    if power_iteration_normalizer == 'auto':
        if n_iter <= 2:
            power_iteration_normalizer = 'none'
        else:
            power_iteration_normalizer = 'LU'

    # Perform power iterations with Q to further 'imprint' the top
    # singular vectors of A in Q
    for _ in range(n_iter):
        if power_iteration_normalizer == 'none':
            q = safe_sparse_dot(w, q)
            q = safe_sparse_dot(w.T, q)
        elif power_iteration_normalizer == 'LU':
            q, *_ = scl.lu(safe_sparse_dot(w, q), permute_l=True)
            q, *_ = scl.lu(safe_sparse_dot(w.T, q), permute_l=True)
        elif power_iteration_normalizer == 'QR':
            q, *_ = scl.qr(safe_sparse_dot(w, q), mode='economic')
            q, *_ = scl.qr(safe_sparse_dot(w.T, q), mode='economic')

    # Sample the range of A using by linear projection of Q
    # Extract an orthonormal basis
    q, *_ = scl.qr(safe_sparse_dot(w, q), mode='economic')
    return q


def rand_svd(w: np.ndarray,
             tolerance: Optional[float] = None,
             n_components: Optional[int] = None,
             thresh: Optional[Callable] = None,
             n_oversamples=10, n_iter='auto',
             power_iteration_normalizer='auto', transpose='auto',
             flip_sign=True) -> SVD:
    n_random = n_components + n_oversamples
    n_samples, n_features = w.shape

    if n_iter == 'auto':
        # Checks if the number of iterations is explicitly specified
        # Adjust n_iter. 7 was found a good compromise for PCA. See #5299
        n_iter = 7 if n_components < .1 * min(w.shape) else 4

    if transpose == 'auto':
        transpose = n_samples < n_features
    if transpose:
        # this implementation is a bit faster with smaller shape[1]
        w = w.T

    q = randomized_range_finder(w, tolerance, n_random, n_iter,
                                power_iteration_normalizer)

    # project M to the (k + p) dimensional space using the basis vectors
    b = safe_sparse_dot(q.T, w)

    # compute the SVD on the thin matrix: (k + p) wide
    uh, s, v = scl.svd(b, full_matrices=False)

    del b
    u = np.dot(q, uh)

    if flip_sign:
        if not transpose:
            u, v = svd_flip(u, v)
        else:
            # In case of transpose u_based_decision=false
            # to actually flip based on u and not v.
            u, v = svd_flip(u, v, u_based_decision=False)

    if thresh:
        s = thresh(s)

    if transpose:
        u, s, v = v.T, s, u.T

    return SVD(u, s, v).trim(n_components)
