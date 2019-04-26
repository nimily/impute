from math import inf, isfinite
from typing import List, Tuple, Any
from collections import namedtuple

import numpy as np
import numpy.linalg as npl

from .base import LagrangianImpute
from .sample_set import SampleSet
from .measurement import Measurement
from .utils import soft_svt, SVD

DEFAULT_XTOL = 1e-3
DEFAULT_GTOL = 0.0
DEFAULT_DTOL = inf

FpcMetrics = namedtuple('Metric', 'd_norm o_norm opt_cond')


class FpcImpute(LagrangianImpute):

    def __init__(self, shape: Tuple[int, int]):
        super().__init__(shape)

        self.tau: float = 0.0

        self.xtol: float = DEFAULT_XTOL
        self.gtol: float = DEFAULT_GTOL
        self.dtol: float = DEFAULT_DTOL

    def update_once(self,
                    ss: SampleSet,
                    alpha: float) -> FpcMetrics:
        tau = self.tau

        assert self.z_new is not None
        z_old = self.z_new
        m_old = z_old.to_matrix()

        g_old = ss.rss_grad(m_old)
        y_new = m_old - tau * g_old
        z_new = soft_svt(y_new, tau * alpha)
        m_new = z_new.to_matrix()

        d_norm = npl.norm(m_new - m_old)

        # debiasing
        if isfinite(self.dtol):
            g_norm = npl.norm(g_old, 2)

            if g_norm > self.dtol * d_norm:
                u = z_new.u
                v = z_new.v

                z_new = FpcImpute.debias(ss, u, v)
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
    def debias(ss: SampleSet, u, v) -> SVD:

        def transform(x: Measurement):
            m = x.as_matrix(u.T, v.T)

            if np.ndim(m) == 2:
                return m.diagonal()

            return [m]

        xs = np.array([transform(x) for x in ss.xs])
        ys = ss.ys

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
                ss: SampleSet,
                alphas: List[float],
                max_iters: int = 100,
                warm_start: bool = True,
                **kwargs):

        if 'tau' in kwargs:
            self.tau = kwargs['tau']
        else:
            self.tau = 1 / ss.op_norm() ** 2

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
