import abc

from typing import Optional, List, Tuple, Any, Union

import numpy as np
import numpy.linalg as npl

from .svt import tuned_svt
from .ops import TraceLinearOp, vector
from .decomposition import SVD


class Dataset:
    op: TraceLinearOp
    ys: Union[vector, List[float]]
    aty: Optional[vector] = None

    def __init__(self,
                 op: TraceLinearOp,
                 ys: Union[vector, List[float]],
                 xty: Optional[vector] = None,
                 yty: Optional[float] = None):
        self.op: TraceLinearOp = op
        self.ys: vector = ys if isinstance(ys, np.ndarray) else np.array(ys)

        self.xty: Optional[vector] = xty
        self.yty: Optional[float] = yty

        self.fresh = False

    def loss(self,
             b: Union[SVD, vector],
             alphas: Union[float, List[float], np.ndarray]) -> np.ndarray:
        if isinstance(alphas, float):
            alphas = [alphas]

        alphas = np.array(alphas)

        if isinstance(b, SVD):
            matrix = b.to_matrix()
            nuc_norm = np.sum(b.s)
        else:
            matrix = b
            nuc_norm = npl.norm(b, 'nuc')

        rss = self.rss(matrix)
        reg = nuc_norm

        return rss + alphas * reg

    def rss(self, b: vector) -> float:
        self.ensure_freshness()

        assert self.yty is not None
        assert self.xty is not None

        return np.sum(b * (self.op.xtx(b) - 2 * self.xty)) + self.yty

    def rss_grad(self, b: vector) -> vector:
        self.ensure_freshness()
        return self.op.xtx(b) - self.xty

    def ensure_freshness(self):
        if not self.fresh:
            self.refresh()

    def refresh(self):
        self.fresh = True

        self.xty = self.op.t(self.ys)
        self.yty = sum(np.power(self.ys, 2))

    @property
    def xs(self):
        return self.op.to_matrix_list()

    def extend(self, xs, ys):
        self.op.extend(xs)

        assert isinstance(self.ys, np.ndarray)
        self.ys = np.concatenate([self.ys, ys])

        self.fresh = False


class BaseImpute:

    def __init__(self, shape: Tuple[int, int]):
        self.shape = shape

        self.starting_point: Optional[SVD] = None

        self.z_old: Optional[SVD] = None
        self.z_new: Optional[SVD] = None

        self._init_starting_point()
        self._init_z()

    def _init_starting_point(self, value: np.ndarray = None):

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
                    alpha: float,
                    prev_rank: int = 0) -> Any:
        pass

    @abc.abstractmethod
    def should_stop(self, metrics: Any, goal: float) -> bool:
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
            goals: Optional[Union[List[float], np.ndarray]] = None,
            **kwargs) -> List[SVD]:

        self._prefit(ds, alphas, max_iters, warm_start, **kwargs)

        if not warm_start:
            self._init_z()

        zs: List[SVD] = []

        if goals is None:
            goals = np.zeros_like(alphas)

        assert self.z_new is not None
        for alpha, goal in zip(alphas, goals):
            for _ in range(max_iters):
                prev_rank = self.z_new.rank
                metrics = self.update_once(ds, alpha, prev_rank)

                if self.should_stop(metrics, goal):
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


class SvtLagrangianImpute(LagrangianImpute):
    DEFAULT_SVT = tuned_svt()

    def __init__(self, shape: Tuple[int, int], svt_op=None):
        super().__init__(shape)

        if svt_op is None:
            svt_op = SvtLagrangianImpute.DEFAULT_SVT

        self.svt_op = svt_op

    def svt(self,
            w,
            alpha: float,
            prev_rank: int = 0) -> SVD:
        thresh = self.get_threshold(alpha)
        guess = prev_rank + 4
        return self.svt_op(w, thresh, guess=guess)

    @abc.abstractmethod
    def get_threshold(self, alpha: float):
        pass
