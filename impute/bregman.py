from typing import Tuple

import numpy as np

from .base import BaseImpute, LagrangianImpute
from .fpc import FpcImpute
from .sample_set import SampleSet, EntrySampleSet
from .utils import SVD


class BregmanImpute(BaseImpute):

    def __init__(self, shape: Tuple[int, int], lag_impute=None):
        super().__init__(shape)

        if lag_impute is None:
            lag_impute = FpcImpute(shape)

        assert isinstance(lag_impute, LagrangianImpute)
        self.lag_impute = lag_impute

    def impute(self, ss: SampleSet, **kwargs) -> SVD:
        assert 'alpha' in kwargs

        alpha = kwargs['alpha']
        assert isinstance(alpha, float)

        if 'eta' in kwargs:
            eta = kwargs['eta']
            assert isinstance(eta, float)
        else:
            eta = 0.25

        z = self.lag_impute.impute(ss, alpha=alpha, eta=eta)

        xs = ss.xs
        y0 = np.array(ss.ys)
        ys = np.array(ss.ys)
        for _ in range(500):
            m = z.to_matrix()
            ys = y0 + (ys - ss.value(m))
            print(np.linalg.norm(ys))
            ss = EntrySampleSet(ss.shape)
            ss.add_all_obs(xs, ys)
            z = self.lag_impute.fit(ss, [alpha], max_iters=500, xtol=1e-10, gtol=1e-4, tau=1)[0]

        assert z is not None

        return z
