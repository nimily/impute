import abc

from typing import Optional, List, Tuple, Any

import numpy as np
import numpy.linalg as npl

from .sample_set import SampleSet
from .utils import SVD


class BaseImpute:

    def __init__(self, shape: Tuple[int, int]):
        self.shape = shape

        self.starting_point: Optional[SVD] = None

        self.z_old: Optional[SVD] = None
        self.z_new: Optional[SVD] = None

        self._init_starting_point()
        self._init_z()

    def _init_starting_point(self, value=None):

        if value is None:
            sp = self.zero()
        else:
            sp = np.copy(value)

        self.starting_point = SVD.to_svd(sp)

    def _init_z(self):
        self.z_old = None
        self.z_new = self.starting_point

    def zero(self):
        return np.zeros(self.shape)

    @abc.abstractmethod
    def impute(self, ss: SampleSet, **kwargs) -> SVD:
        pass


class LagrangianImpute(BaseImpute):

    @abc.abstractmethod
    def update_once(self,
                    ss: SampleSet,
                    alpha: float) -> Any:
        pass

    @abc.abstractmethod
    def should_stop(self, metrics: Any) -> bool:
        pass

    def _prefit(self,
                ss: SampleSet,
                alphas: List[float],
                max_iters: int = 100,
                warm_start: bool = True,
                **kwargs):
        pass

    def fit(self,
            ss: SampleSet,
            alphas: List[float],
            max_iters: int = 100,
            warm_start: bool = True,
            **kwargs) -> List[SVD]:

        self._prefit(ss, alphas, max_iters, warm_start, **kwargs)

        if not warm_start:
            self._init_z()

        zs: List[SVD] = []

        for alpha in alphas:
            for _ in range(max_iters):
                metrics = self.update_once(ss, alpha)

                if self.should_stop(metrics):
                    break

            assert self.z_new is not None
            zs.append(self.z_new)

        return zs

    def impute(self, ss: SampleSet, **kwargs) -> SVD:
        assert 'alpha' in kwargs
        alpha_min = kwargs['alpha']

        if 'eta' in kwargs:
            eta = kwargs['eta']
            assert isinstance(eta, float)
        else:
            eta = 0.25

        alphas = self.get_alpha_seq(ss, alpha_min, eta)

        return self.fit(ss, alphas, **kwargs)[-1]

    def alpha_max(self, ss: SampleSet) -> float:
        grad = ss.rss_grad(self.zero())

        return npl.norm(grad, 2)

    def get_alpha_seq(self, ss: SampleSet, alpha_min: float, eta: float) -> List[float]:
        alphas = []

        alpha = self.alpha_max(ss)
        while alpha > alpha_min:
            alphas.append(alpha)

            alpha *= eta

        alphas.append(alpha_min)

        return alphas
