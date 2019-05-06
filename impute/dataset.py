from typing import Optional, List, Union

import numpy as np

from .vector import Vector, Matrix, FullMatrix, RowMatrix, EntryMatrix
from .linear_ops import LinearOp, TraceLinearOp
from .utils import trace_inner as inner


class Dataset:

    def __init__(self,
                 op: TraceLinearOp,
                 ys: Union[np.ndarray, List[float]],
                 xty: Optional[Matrix] = None,
                 yty: Optional[float] = None):

        self.op: TraceLinearOp = op
        self.ys: Vector = ys if isinstance(ys, np.ndarray) else np.array(ys)

        self._xty: Optional[Matrix] = xty
        self._yty: Optional[float] = yty

        self.fresh = False

    def loss(self, b: Matrix, alpha: float):
        return self.rss(b) + alpha * b.norm('nuc')

    def rss(self, b: Matrix) -> float:
        self.ensure_refreshness()

        assert isinstance(self.xty, np.ndarray)
        assert isinstance(self.yty, float)

        return (inner(b, self.op.xtx(b) - 2 * self.xty) + self.yty) / 2

    def rss_grad(self, b: Matrix) -> Matrix:
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
