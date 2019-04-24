from typing import Tuple, Any, List

import numpy as np
import numpy.linalg as npl

from .base import BaseImpute
from .sample_set import EntrySampleSet, SampleSet
from .utils import SVD, soft_svt

DEFAULT_TOL = 1e-5


class SoftImpute(BaseImpute):

    def __init__(self, shape, svt_op=soft_svt):
        super().__init__(shape)

        self.tol = 0
        self.svt_op = svt_op

    def update_once(self, ss: SampleSet, alpha: float) -> Tuple[float, float]:
        assert isinstance(ss, EntrySampleSet)

        assert self.z_new is not None
        z_old = self.z_new
        m_old = z_old.to_matrix()

        y_new = m_old - ss.rss_grad(m_old)
        z_new = self.svt(y_new, alpha)
        m_new = z_new.to_matrix()

        self.z_old = z_old
        self.z_new = z_new

        return npl.norm(m_new - m_old), npl.norm(m_old)

    def should_stop(self, metrics: Any) -> bool:
        delta_norm, old_norm = metrics

        return delta_norm ** 2 < self.tol * old_norm ** 2

    def _prefit(self,
                ss: SampleSet,
                alphas: List[float],
                max_iters: int = 100,
                warm_start: bool = True,
                **kwargs):

        assert isinstance(ss, EntrySampleSet)

        if 'tol' in kwargs:
            assert isinstance(kwargs['tol'], float)

            self.tol = kwargs['tol']
        else:
            self.tol = DEFAULT_TOL

    def svt(self, w, alpha: float) -> SVD:
        return self.svt_op(w, alpha)

    def alpha_max(self, ss: EntrySampleSet):
        grad = ss.rss_grad(self.zero())

        return npl.norm(grad, 2)

    def zero(self):
        return np.zeros(self.shape)
