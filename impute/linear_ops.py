import abc

from typing import Tuple, List, Union, Optional, Type, TypeVar, Generic, Iterator
from functools import reduce

import numpy as np
import numpy.linalg as npl

from .utils import one_hot


vector = Union[np.ndarray]

full_matrix = vector
row_matrix = Tuple[int, vector]
entry_matrix = Tuple[int, int, float]


class LinearOp(abc.ABC):

    LinearOpType = Type['LinearOp']

    def __init__(self, adjoint: Optional[Union['LinearOp', str]] = None, **kwargs):
        super().__init__(**kwargs)

        if adjoint == 'self':
            adjoint = self

        assert adjoint is None or isinstance(adjoint, LinearOp)
        self.adjoint: Optional[LinearOp] = adjoint

    @property
    @abc.abstractmethod
    def i_shape(self):
        pass

    @property
    @abc.abstractmethod
    def o_shape(self):
        pass

    def __call__(self, b):
        return self.evaluate(b)

    @abc.abstractmethod
    def evaluate(self, b: vector) -> vector:
        pass

    @abc.abstractmethod
    def evaluate_t(self, b: vector) -> vector:
        pass

    @property
    def t(self) -> 'LinearOp':
        if self.adjoint is None:
            self.adjoint = self.lazy_t()

        return self.adjoint

    def lazy_t(self) -> 'LinearOp':
        return TransposeLinearOp(self)

    @property
    def xtx(self) -> 'LinearOp':
        return AdjointCompositeLinearOp(self)

    @property
    def xxt(self) -> 'LinearOp':
        return AdjointCompositeLinearOp(self.t)

    @abc.abstractmethod
    def norm(self) -> float:
        pass


class CompositeLinearOp(LinearOp):

    def __init__(self,
                 ops: Tuple[LinearOp],
                 **kwargs):
        for i in range(len(ops) - 1):
            op1 = ops[i]
            op2 = ops[i - 1]

            assert op2.i_shape == op1.o_shape

        super().__init__(**kwargs)

        self.ops: Tuple[LinearOp, ...] = ops

    @property
    def i_shape(self):
        return self.ops[-1].i_shape

    @property
    def o_shape(self):
        return self.ops[0].o_shape

    def evaluate(self, b: vector) -> vector:
        return reduce(lambda y, op: op.evaluate(y), reversed(self.ops), b)

    def evaluate_t(self, b: vector) -> vector:
        return reduce(lambda y, op: op.evaluate_t(y), self.ops, b)

    def norm(self) -> float:
        raise NotImplementedError


class SelfAdjointOp(LinearOp):

    def __init__(self, **kwargs):
        kwargs.update({
            'adjoint': self,
        })

        super().__init__(**kwargs)

    def evaluate_t(self, b: vector) -> vector:
        return self.evaluate(b)

    def xtx(self) -> 'SelfAdjointOp':
        return self.square

    def xxt(self) -> 'SelfAdjointOp':
        return self.square

    @property
    @abc.abstractmethod
    def square(self) -> 'SelfAdjointOp':
        pass


class AdjointCompositeLinearOp(SelfAdjointOp, CompositeLinearOp):

    def __init__(self, op: LinearOp, **kwargs):
        kwargs.update({
            'ops': (op.t, op)
        })

        super().__init__(**kwargs)

    @property
    def square(self) -> 'AdjointCompositeLinearOp':
        return AdjointCompositeLinearOp(self)

    def norm(self) -> float:
        return self.ops[1].norm() ** 2


class TransposeLinearOp(LinearOp):

    def __init__(self, adjoint: LinearOp, **kwargs):
        super().__init__(**kwargs)

        self.adjoint: LinearOp = adjoint

    @property
    def i_shape(self):
        return self.adjoint.o_shape

    @property
    def o_shape(self):
        return self.adjoint.i_shape

    def evaluate(self, b: vector) -> vector:
        return self.adjoint.evaluate_t(b)

    def evaluate_t(self, b: vector):
        return self.adjoint.evaluate(b)

    @property
    def xtx(self) -> LinearOp:
        return self.adjoint.xxt

    @property
    def xxt(self) -> LinearOp:
        return self.adjoint.xtx

    def norm(self) -> float:
        return self.adjoint.norm()


class HadamardLinearOp(SelfAdjointOp):

    def __init__(self, coefs: vector, **kwargs):
        super().__init__(**kwargs)

        self.coefs = coefs

    @property
    def i_shape(self):
        return self.coefs.shape

    @property
    def o_shape(self):
        return self.coefs.shape

    def evaluate(self, b: vector) -> vector:
        return np.multiply(self.coefs, b)

    def norm(self) -> float:
        pass

    @property
    def square(self) -> 'SelfAdjointOp':
        return HadamardLinearOp(self.coefs ** 2)


X = TypeVar('X')


class IncrementalData(Generic[X]):

    def __init__(self):
        self.xs: List[X] = []
        self.fresh: bool = False

    @property
    @abc.abstractmethod
    def i_shape(self):
        pass

    def append(self, x: X):
        self.extend([x])

    def extend(self, xs: List[X]):
        self.preprocess_data(xs)

        self.xs.extend(xs)

        self.postprocess_data(xs)

    def preprocess_data(self, xs: List[X]):
        pass

    def postprocess_data(self, xs: List[X]):
        if xs:
            self.fresh = False

    def to_matrix_list(self,
                       left: Optional[vector] = None,
                       right: Optional[vector] = None) -> Iterator[vector]:
        return map(lambda x: self.to_matrix(x, left, right), self.xs)

    def to_matrix(self, x: X,
                  left: Optional[vector] = None,
                  right: Optional[vector] = None) -> vector:
        assert isinstance(x, np.ndarray)

        if left is not None:
            x = x @ left

        if right is not None:
            x = x @ right

        return x


class DotLinearOp(LinearOp, IncrementalData[vector]):

    def __init__(self, i_shape: Union[int, Tuple[int]]):
        super().__init__()

        if isinstance(i_shape, int):
            i_shape = (i_shape, )

        self._i_shape = i_shape

        self.matrix = np.zeros((0, i_shape[0]))
        self._norm: float = 0.0

    @property
    def i_shape(self):
        return self._i_shape

    @property
    def o_shape(self):
        return tuple([len(self.xs)])

    def evaluate(self, b: vector) -> vector:
        self.ensure_freshness()

        return self.matrix @ b

    def evaluate_t(self, b: vector) -> vector:
        self.ensure_freshness()

        return self.matrix.T @ b

    def norm(self) -> float:
        self.ensure_freshness()

        return self._norm

    def ensure_freshness(self):
        if not self.fresh:
            self.refresh()

    def refresh(self):
        if self.xs:
            self.matrix = np.vstack(self.xs)
            self._norm = npl.norm(self.matrix, 2)
        else:
            self.matrix = np.empty((0, self.i_shape[0]))
            self._norm = 0.0

        self.fresh = True


class DenseTraceLinearOp(LinearOp, IncrementalData[vector]):

    def __init__(self, i_shape: Tuple[int, int]):
        super().__init__()

        self._i_shape = i_shape

        self._norm = 0

    @property
    def i_shape(self):
        return self._i_shape

    @property
    def o_shape(self):
        return tuple([len(self.xs)])

    def evaluate(self, b: vector) -> vector:
        return np.array([np.trace(x @ b.T) for x in self.xs])

    def evaluate_t(self, b: vector) -> vector:
        return sum(c * x for c, x in zip(b, self.xs))

    def norm(self) -> float:
        if not self.fresh:
            self.refresh_norm()

        return self._norm

    def refresh_norm(self):
        xs = np.array([x.flatten() for x in self.xs])

        self._norm = npl.norm(xs, 2)


class RowTraceLinearOp(LinearOp, IncrementalData[row_matrix]):

    def __init__(self, i_shape: Tuple[int, int]):
        super().__init__()

        self._i_shape = i_shape

        self._norm = 0

        n_row, n_col = i_shape

        self.row_ops: List[DotLinearOp] = [DotLinearOp(n_col) for _ in range(n_row)]
        self.row_to_global: List[List[int]] = [[] for _ in range(n_row)]

    def postprocess_data(self, xs: List[row_matrix]):
        start = len(self.xs) - len(xs)

        for i, x in enumerate(xs):
            r, v = x
            self.row_ops[r].append(v)
            self.row_to_global[r].append(start + i)

        super().postprocess_data(xs)

    @property
    def i_shape(self):
        return self._i_shape

    @property
    def n_row(self):
        return self.i_shape[0]

    @property
    def n_col(self):
        return self.i_shape[1]

    @property
    def o_shape(self):
        return tuple([len(self.xs)])

    def evaluate(self, b: vector) -> vector:
        per_rows = [op(b[r]) for r, op in enumerate(self.row_ops)]

        y = np.zeros(self.o_shape)
        for i, row in enumerate(per_rows):
            indices = self.row_to_global[i]
            y[indices] = row

        return y

    def evaluate_t(self, b: vector) -> vector:
        row_ops = self.row_ops
        row_vecs = [b[indices] for indices in self.row_to_global]

        return np.vstack([op.t(vec) for vec, op in zip(row_vecs, row_ops)])

    def norm(self) -> float:
        if not self.fresh:
            self.refresh_norm()

        return self._norm

    def refresh_norm(self):
        self._norm = max(op.norm() for op in self.row_ops)
        self.fresh = True

    def to_matrix(self, x: row_matrix,
                  left: Optional[vector] = None,
                  right: Optional[vector] = None) -> vector:
        r, v = x
        m = np.outer(one_hot(self.n_row, r), v)

        if left is not None:
            m = left @ m

        if right is not None:
            m = m @ right

        return m


class EntryTraceLinearOp(LinearOp, IncrementalData[entry_matrix]):

    def __init__(self, i_shape: Tuple[int, int]):
        super().__init__()

        self._i_shape = i_shape

        self._norm = 0

        self.rows: List[int] = []
        self.cols: List[int] = []
        self.vals: List[float] = []

        self._xtx = np.zeros(i_shape)

    def postprocess_data(self, xs: List[entry_matrix]):
        for x in xs:
            r, c, v = x
            self.rows.append(r)
            self.cols.append(c)
            self.vals.append(v)

            self._xtx[r, c] += v ** 2

        self.refresh_norm()

        super().postprocess_data(xs)

    @property
    def i_shape(self):
        return self._i_shape

    @property
    def o_shape(self):
        return tuple([len(self.xs)])

    def evaluate(self, b: vector) -> vector:
        return np.multiply(b[self.rows, self.cols], self.vals)

    def evaluate_t(self, b: vector) -> vector:
        t = np.zeros(self.i_shape)

        np.add.at(t, (self.rows, self.cols), np.multiply(self.vals, b))

        return t

    @property
    def xtx(self) -> LinearOp:
        return HadamardLinearOp(self._xtx)

    def norm(self) -> float:
        if not self.fresh:
            self.refresh_norm()

        return self._norm

    def refresh_norm(self):
        self._norm = self._xtx.max() ** 0.5
        self.fresh = True

    def to_matrix(self, x: entry_matrix,
                  left: Optional[vector] = None,
                  right: Optional[vector] = None) -> vector:
        i, j, v = x
        m = one_hot(self.i_shape, (i, j), v)

        if left is not None:
            m = left @ m

        if right is not None:
            m = m @ right

        return m


TraceLinearOp = Union[DenseTraceLinearOp, RowTraceLinearOp, EntryTraceLinearOp]
