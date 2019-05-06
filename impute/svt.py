from typing import Optional, Union, Callable

import numpy as np

from .decomposition import SVD, exact_svd, randomized_svd


def soft_thresh(level):
    return np.vectorize(lambda x: 0 if x < level else x - level)


def hard_thresh(level):
    return np.vectorize(lambda x: 0 if x < level else x)


def svt(
        w: np.ndarray,
        level: float,
        guess: Optional[int] = None,
        thresh: str = 'soft',
        svd: Union[Callable, str] = 'exact') -> SVD:
    if thresh == 'soft':
        thresh = soft_thresh(level)
    else:
        thresh = hard_thresh(level)

    if isinstance(svd, str):
        assert svd in ('randomized', 'exact')

        if svd == 'randomized':
            svd = randomized_svd
        else:
            svd = exact_svd

    if guess is None:
        guess = 5
    u, s, v = svd(w, level, guess)

    assert callable(thresh)
    return SVD(u, thresh(s), v).trim()


def tuned_svt(thresh: str = 'soft',
              svd: Union[Callable, str] = 'exact'):
    def _svt(w: np.ndarray,
             level: float,
             guess: Optional[int] = None):
        return svt(w, level, guess, thresh, svd)

    return _svt
