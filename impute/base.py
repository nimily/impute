import abc

from typing import Optional, List, Tuple, Any, Union

import numpy as np
import numpy.linalg as npl

from .linear_ops import TraceLinearOp, vector
from .utils import SVD


class Dataset:
    op: TraceLinearOp
    ys: Union[vector, List[float]]
    aty: Optional[vector] = None

    def __init__(self,
                 op: TraceLinearOp,
                 ys: Union[vector, List[float]],
                 ay: Optional[vector] = None):
        self.op: TraceLinearOp = op
        self.ys: vector = ys if isinstance(ys, np.ndarray) else np.array(ys)
        self.ay: Optional[vector] = ay

    def rss_grad(self, b: vector) -> vector:
        self.ay = self.op.t(self.ys)
        return self.op.xtx(b) - self.ay

    @property
    def xs(self):
        return self.op.to_matrix_list()

    def extend(self, xs, ys):
        self.op.extend(xs)

        if isinstance(self.ys, np.ndarray):
            self.ys = np.concatenate([self.ys, ys])


def penalized_loss(ds: Dataset, b, alpha):
    ys = np.array(ds.ys)
    yh = ds.op(b)
    return np.sum((ys - yh) ** 2) / 2 + alpha * npl.norm(b, 'nuc')


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
    def impute(self, ds: Dataset, **kwargs) -> SVD:
        pass


class LagrangianImpute(BaseImpute):

    @abc.abstractmethod
    def update_once(self,
                    ds: Dataset,
                    alpha: float) -> Any:
        pass

    @abc.abstractmethod
    def should_stop(self, metrics: Any) -> bool:
        pass

    def _prefit(self,
                ds: Dataset,
                alphas: List[float],
                max_iters: int = 100,
                warm_start: bool = True,
                **kwargs):
        pass

    def fit(self,
            ds: Dataset,
            alphas: List[float],
            max_iters: int = 100,
            warm_start: bool = True,
            **kwargs) -> List[SVD]:

        self._prefit(ds, alphas, max_iters, warm_start, **kwargs)

        if not warm_start:
            self._init_z()

        zs: List[SVD] = []

        for alpha in alphas:
            for _ in range(max_iters):
                metrics = self.update_once(ds, alpha)

                if self.should_stop(metrics):
                    break

            assert self.z_new is not None
            zs.append(self.z_new)

        return zs

    def impute(self, ds: Dataset, **kwargs) -> SVD:
        assert 'alpha' in kwargs
        alpha_min = kwargs['alpha']

        if 'eta' in kwargs:
            eta = kwargs['eta']
            assert isinstance(eta, float)
        else:
            eta = 0.25

        alphas = self.get_alpha_seq(ds, alpha_min, eta)

        return self.fit(ds, alphas, **kwargs)[-1]

    def alpha_max(self, ds: Dataset) -> float:
        grad = ds.rss_grad(self.zero())

        return npl.norm(grad, 2)

    def get_alpha_seq(self, ds: Dataset, alpha_min: float, eta: float) -> List[float]:
        alphas = []

        alpha = self.alpha_max(ds)
        while alpha > alpha_min:
            alphas.append(alpha)

            alpha *= eta

        alphas.append(alpha_min)

        return alphas
