from math import inf, isfinite
from typing import List, Tuple, Any
from collections import namedtuple

import numpy as np
import numpy.linalg as npl

from .base import vector, SvtLagrangianImpute, Dataset
from .decomposition import SVD

DEFAULT_XTOL = 1e-3
DEFAULT_GTOL = 0.0
DEFAULT_DTOL = inf

FpcMetrics = namedtuple('Metric', 'd_norm o_norm opt_cond')


class FpcImpute(SvtLagrangianImpute):

    def __init__(self, shape: Tuple[int, int], svt_op=None):
        super().__init__(shape, svt_op)

        self.tau: float = 0.0

        self.xtol: float = DEFAULT_XTOL
        self.gtol: float = DEFAULT_GTOL
        self.dtol: float = DEFAULT_DTOL

    def get_threshold(self, alpha: float):
        return self.tau * alpha

    def update_once(self,
                    ds: Dataset,
                    alpha: float,
                    prev_rank: int = 0) -> FpcMetrics:
        tau = self.tau

        z_old = self.z_new  # type: ignore
        m_old = z_old.to_matrix()

        g_old = ds.rss_grad(m_old)
        y_new = m_old - tau * g_old
        z_new = self.svt(y_new, alpha, prev_rank)
        m_new = z_new.to_matrix()

        d_norm = npl.norm(m_new - m_old)

        # debiasing
        if isfinite(self.dtol):
            g_norm = npl.norm(g_old, 2)

            if g_norm > self.dtol * d_norm:
                u = z_new.u
                v = z_new.v

                z_new = FpcImpute.debias(ds, u, v)
                m_new = z_new.to_matrix()
                d_norm = npl.norm(m_new - m_old)

        if self.gtol > 0.0:
            opt_cond = npl.norm(z_new.u @ z_new.v + g_old / alpha, 2) - 1
        else:
            opt_cond = None

        self.z_old = z_old
        self.z_new = z_new

        return FpcMetrics(
            d_norm=d_norm,
            o_norm=npl.norm(m_old),
            opt_cond=opt_cond
        )

    @staticmethod
    def debias(ds: Dataset, u: vector, v: vector) -> SVD:

        def transform_xs():
            uxvs = ds.op.to_matrix_list(u.T, v.T)
            uxvs = [x.diagonal() if np.ndim(x) == 2 else x for x in uxvs]

            return np.array(uxvs)

        xs = transform_xs()
        ys = ds.ys

        s, *_ = npl.lstsq(xs, ys, rcond=None)

        return SVD(u, s, v)

    def should_stop(self, metrics: Any) -> bool:
        assert isinstance(metrics, FpcMetrics)

        d_norm = metrics.d_norm
        o_norm = metrics.o_norm

        if d_norm < self.xtol * max(1, o_norm):
            return True

        if self.gtol > 0 and metrics.opt_cond < self.gtol:
            return True

        return False

    def _prefit(self,
                ds: Dataset,
                alphas: List[float],
                max_iters: int = 100,
                warm_start: bool = True,
                **kwargs):

        if 'tau' in kwargs:
            self.tau = kwargs['tau']
        else:
            self.tau = 1 / ds.op.norm() ** 2

        if 'xtol' in kwargs:
            assert isinstance(kwargs['xtol'], float)

            self.xtol = kwargs['xtol']
        else:
            self.xtol = DEFAULT_XTOL

        if 'gtol' in kwargs:
            assert isinstance(kwargs['gtol'], float)
            self.gtol = kwargs['gtol']

        if 'dtol' in kwargs:
            assert isinstance(kwargs['dtol'], float)
            self.dtol = kwargs['dtol']
