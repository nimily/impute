from typing import Tuple, Any, List

import numpy.linalg as npl

from .base import LagrangianImpute, Dataset
from .linear_ops import EntryTraceLinearOp
from .utils import SVD, soft_svt

DEFAULT_TOL = 1e-5


class SoftImpute(LagrangianImpute):

    def __init__(self, shape, svt_op=soft_svt):
        super().__init__(shape)

        self.tol = 0
        self.svt_op = svt_op

    def update_once(self, ds: Dataset, alpha: float) -> Tuple[float, float]:
        self.ensure_entry_op(ds)

        assert self.z_new is not None
        z_old = self.z_new
        m_old = z_old.to_matrix()

        y_new = m_old - ds.rss_grad(m_old)
        z_new = self.svt(y_new, alpha)
        m_new = z_new.to_matrix()

        self.z_old = z_old
        self.z_new = z_new

        return npl.norm(m_new - m_old), npl.norm(m_old)

    def should_stop(self, metrics: Any) -> bool:
        delta_norm, old_norm = metrics

        return delta_norm ** 2 < self.tol * old_norm ** 2

    def _prefit(self,
                ds: Dataset,
                alphas: List[float],
                max_iters: int = 100,
                warm_start: bool = True,
                **kwargs):

        self.ensure_entry_op(ds)

        if 'tol' in kwargs:
            assert isinstance(kwargs['tol'], float)

            self.tol = kwargs['tol']
        else:
            self.tol = DEFAULT_TOL

    def svt(self, w, alpha: float) -> SVD:
        return self.svt_op(w, alpha)

    @staticmethod
    def ensure_entry_op(ds: Dataset):
        assert isinstance(ds.op, EntryTraceLinearOp)
