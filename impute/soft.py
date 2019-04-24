from typing import List, Tuple

import numpy as np
import numpy.linalg as npl

from .base import BaseImpute
from .sample_set import EntrySampleSet
from .utils import SVD, soft_svt


class SoftImpute(BaseImpute):

    def __init__(self, shape, svt_op=soft_svt):
        super().__init__(shape)

        self.svt_op = svt_op

    def update_once(self, ss: EntrySampleSet, alpha: float) -> Tuple[float, float]:
        assert self.z_new is not None
        z_old = self.z_new
        m_old = z_old.to_matrix()

        y_new = m_old - ss.rss_grad(m_old)
        z_new = self.svt(y_new, alpha)
        m_new = z_new.to_matrix()

        self.z_old = z_old
        self.z_new = z_new

        return npl.norm(m_new - m_old), npl.norm(m_old)

    def fit(self,
            ss: EntrySampleSet,
            alphas: List[float],
            max_iters: int = 100,
            tol: float = 1e-5,
            warm_start: bool = True) -> List[SVD]:

        if not warm_start:
            self._init_z()

        zs: List[SVD] = []

        for alpha in alphas:
            for _ in range(max_iters):
                delta_norm, old_norm = self.update_once(ss, alpha)

                if delta_norm ** 2 <= tol * old_norm ** 2:
                    break

            assert self.z_new is not None
            zs.append(self.z_new)

        return zs

    def svt(self, w, alpha: float) -> SVD:
        return self.svt_op(w, alpha)

    def alpha_max(self, ss: EntrySampleSet):
        grad = ss.rss_grad(self.zero())

        return npl.norm(grad, 2)

    def zero(self):
        return np.zeros(self.shape)
