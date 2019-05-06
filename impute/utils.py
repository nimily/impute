from typing import Tuple

import numpy as np
import numpy.random as npr


def one_hot(shape, pos, val: float = 1) -> np.ndarray:
    x = np.zeros(shape)
    x[pos] = val

    return x


def random_one_hot(shape) -> Tuple[Tuple[int, ...], float, np.ndarray]:
    if isinstance(shape, tuple):
        pos = tuple(npr.randint(s) for s in shape)
    else:
        pos = (npr.randint(shape), )

    val = npr.rand()

    assert all(isinstance(p, int) for p in pos)
    assert isinstance(val, float)

    return pos, val, one_hot(shape, pos, val)


def trace_inner(a: np.ndarray, b: np.ndarray) -> float:
    return (a * b).sum()
