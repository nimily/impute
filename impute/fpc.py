from typing import List, Tuple, Any
from collections import namedtuple

import numpy as np
import numpy.linalg as npl

from .base import LagrangianImpute
from .sample_set import SampleSet
from .measurement import Measurement
from .utils import soft_svt, SVD

DEFAULT_XTOL = 1e-5

FpcMetrics = namedtuple('Metric', 'd_norm o_norm')


class FpcImpute(LagrangianImpute):

    def __init__(self, shape: Tuple[int, int]):
        super().__init__(shape)

        self.tau: float = 0.0

        self.xtol: float = 0.0
        self.gtol: float = 0.0

    def update_once(self,
                    ss: SampleSet,
                    alpha: float) -> Tuple[float, float]:
        tau = self.tau

        assert self.z_new is not None
        z_old = self.z_new
        m_old = z_old.to_matrix()

        y_new = m_old - tau * ss.rss_grad(m_old)
        z_new = soft_svt(y_new, tau * alpha)
        m_new = z_new.to_matrix()

        self.z_old = z_old
        self.z_new = z_new

        return FpcMetrics(
            d_norm=npl.norm(m_new - m_old),
            o_norm=npl.norm(m_old)
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
