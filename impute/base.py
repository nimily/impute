import abc

from typing import Optional, List, Tuple, Any, Union

import numpy as np
import numpy.linalg as npl

from .svt import tuned_svt
from .linear_ops import TraceLinearOp, vector
from .decomposition import SVD
from .utils import trace_inner as inner
from .vector import Vector, Matrix, FullMatrix, RowMatrix, EntryMatrix

vector_repr = Union[vector, SVD]


class Dataset:

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

    @staticmethod
    def nuc_norm(b: vector_repr) -> float:
        if isinstance(b, np.ndarray):
            return npl.norm(b, 'nuc')
        else:
            return b.s.sum()

    def loss(self, b, alpha: float):
        return self.rss(b) + alpha * Dataset.nuc_norm(b)

    def rss(self, b: vector) -> float:
        self.ensure_refreshness()

        assert isinstance(self.xty, np.ndarray)
        assert isinstance(self.yty, float)

        return (inner(b, self.op.xtx(b) - 2 * self.xty) + self.yty) / 2

    def rss_grad(self, b: vector) -> vector:
        self.ensure_refreshness()
        return self.op.xtx(b) - self.xty

    @property
    def xs(self):
        return self.op.to_matrix_list()

    def ensure_refreshness(self):
        if not self.fresh:
            self.refresh()

    def refresh(self):
        self.fresh = True

        self.xty = self.op.t(self.ys)
        self.yty = (self.ys ** 2).sum()

    def extend(self, xs, ys):
        self.op.extend(xs)

        assert isinstance(self.ys, np.ndarray)
        self.ys = np.concatenate([self.ys, ys])

        self.fresh = False


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
            goal: float = 0,
            **kwargs) -> List[SVD]:

        self._prefit(ds, alphas, max_iters, warm_start, **kwargs)

        if not warm_start:
            self._init_z()

        zs: List[SVD] = []

        assert self.z_new is not None
        for alpha in alphas:
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
        return self.svt_op(w, thresh)

    @abc.abstractmethod
    def get_threshold(self, alpha: float):
        pass
