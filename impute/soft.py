from typing import Tuple, Any, List

import numpy.linalg as npl

from .base import SvtLagrangianImpute, Dataset
from .linear_ops import EntryTraceLinearOp
from .svt import tuned_svt

DEFAULT_TOL = 1e-5
DEFAULT_SVT = tuned_svt()


class SoftImpute(SvtLagrangianImpute):

    def __init__(self, shape, svt_op=None):
        super().__init__(shape, svt_op)

        self.tol = 0

    def get_threshold(self, alpha: float):
        return alpha

    def update_once(self,
                    ds: Dataset,
                    alpha: float,
                    prev_rank: int = 0) -> Tuple[float, float]:
        self.ensure_entry_op(ds)

        assert self.z_new is not None
        z_old = self.z_new
        m_old = z_old.to_matrix()

        y_new = m_old - ds.rss_grad(m_old)
        z_new = self.svt(y_new, alpha, prev_rank)
        m_new = z_new.to_matrix()

        self.z_old = z_old
        self.z_new = z_new

        return npl.norm(m_new - m_old), npl.norm(m_old)

    def should_stop(self, metrics: Any, goal: float) -> bool:
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

    @staticmethod
    def ensure_entry_op(ds: Dataset):
        assert isinstance(ds.op, EntryTraceLinearOp)
