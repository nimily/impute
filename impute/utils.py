import numpy as np
import numpy.linalg as npl


def svt(w, thresh):
    u, s, v = npl.svd(w, full_matrices=False)

    return u @ np.diag(thresh(s)) @ v


def soft_svt(w, level):
    func = np.vectorize(lambda x: max(x - level, 0))

    return svt(w, func)


def hard_svt(w, level):
    func = np.vectorize(lambda x: 0 if x < level else x)

    return svt(w, func)
