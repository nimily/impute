import abc

from typing import Tuple, List, Union, Optional, Type, TypeVar, Generic
from functools import reduce

import numpy as np
import numpy.linalg as npl


vector = Union[np.ndarray]


class LinearOp(abc.ABC):

    LinearOpType = Type['LinearOp']

    def __init__(self, adjoint: Optional[Union['LinearOp', str]] = None):
        super().__init__()

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
                 *ops: LinearOp,
                 adjoint: Optional[Union[LinearOp, str]] = None):
        for i in range(len(ops) - 1):
            op1 = ops[i]
            op2 = ops[i - 1]

            assert op2.i_shape == op1.o_shape

        super().__init__(adjoint)

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


class AdjointCompositeLinearOp(CompositeLinearOp):

    def __init__(self, op: LinearOp):
        super().__init__(op.t, op)

    def norm(self) -> float:
        return self.ops[1].norm() ** 2


class TransposeLinearOp(LinearOp):

    def __init__(self, adjoint: LinearOp):
        super().__init__(None)

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


X = TypeVar('X')


class IncrementalData(Generic[X]):

    def __init__(self):
        self.xs: List[X] = []
        self.fresh: bool = False

    @property
    @abc.abstractmethod
    def i_shape(self):
        pass

    def add(self, x: X):
        self.add_all([x])

    def add_all(self, xs: List[X]):
        self.preprocess_data(xs)

        self.xs.extend(xs)

        self.postprocess_data(xs)

    def preprocess_data(self, xs: List[X]):
        pass
        # for x in xs:
        #     if hasattr(x, 'shape'):
        #         assert x.shape == self.i_shape

    def postprocess_data(self, xs: List[X]):
        if not xs:
            self.fresh = False


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
        self.matrix = np.array(self.xs)
        self._norm = npl.norm(self.matrix, 2)
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


MatrixLinearOp = Union[DenseTraceLinearOp]
