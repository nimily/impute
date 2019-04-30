import abc

from typing import Tuple, List, Union, Optional, Type
from functools import reduce

import numpy as np


vector = Union[np.ndarray]


class LinearOp(abc.ABC):

    LinearOpType = Type['LinearOp']

    def __init__(self, adjoint: Optional[Union['LinearOp', str]] = None):

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
        return CompositeLinearOp(self.t, self, adjoint='self')

    @property
    def xxt(self) -> 'LinearOp':
        return CompositeLinearOp(self, self.t, adjoint='self')


class CompositeLinearOp(LinearOp):

    def __init__(self,
                 *ops: LinearOp,
                 adjoint: Optional[Union['LinearOp', str]] = None):
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


class TraceLinearOp(LinearOp):

    def __init__(self, i_shape: Tuple[int, int]):
        super().__init__()

        self._i_shape = i_shape

        self._init_xs()

    @property
    def i_shape(self):
        return self._i_shape

    @property
    def o_shape(self):
        return tuple([len(self.xs)])

    def _init_xs(self):
        self.xs: List[vector] = []

    def add(self, x: vector):
        self.add_all([x])

    def add_all(self, xs: List[vector]):
        for x in xs:
            assert x.shape == self.i_shape

        self.xs.extend(xs)

    def evaluate(self, b: vector) -> vector:
        return np.array([np.trace(x @ b.T) for x in self.xs])

    def evaluate_t(self, b: vector) -> vector:
        return sum(c * x for c, x in zip(b, self.xs))
