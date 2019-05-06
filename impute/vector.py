import abc

from typing import Optional, Tuple

import numpy as np
import numpy.linalg as npl


class Vector:

    def __init__(self, array: Optional[np.ndarray] = None, *_, **__):
        self.array = array

    @property
    @abc.abstractmethod
    def shape(self) -> Tuple[int, ...]:
        pass

    @property
    def as_array(self) -> np.ndarray:
        if self.array is None:
            self.array = self.to_array()

        return self.array

    @abc.abstractmethod
    def to_array(self) -> np.ndarray:
        pass

    def norm(self, ord) -> float:
        return npl.norm(self.as_array, ord)

    def __add__(self, other):
        return self.as_array + Vector._unwrap(other)

    def __sub__(self, other):
        return self.as_array - Vector._unwrap(other)

    def __mul__(self, other):
        return self.as_array * Vector._unwrap(other)

    def __matmul__(self, other):
        return self.as_array @ Vector._unwrap(other)

    def __neg__(self):
        return - self.as_array

    def __pos__(self):
        return + self.as_array

    def __abs__(self):
        return abs(self.as_array)

    def __radd__(self, other):
        return other + self.as_array

    def __rsub__(self, other):
        return other - self.as_array

    def __rmul__(self, other):
        return other * self.as_array

    def __rmatmul__(self, other):
        return other @ self.as_array

    @staticmethod
    def _unwrap(obj):
        if isinstance(obj, Vector):
            return obj.as_array
        else:
            return obj


class ArrayVector(np.ndarray, Vector):

    def __array_finalize__(self, _):
        self.array = self

    @property
    def as_array(self) -> np.ndarray:
        return self.view(np.ndarray)

    def to_array(self) -> np.ndarray:
        return self.view(np.ndarray).copy()

    def __add__(self, other):
        return self.as_array + Vector._unwrap(other)

    def __sub__(self, other):
        return self.as_array - Vector._unwrap(other)

    def __mul__(self, other):
        return self.as_array * Vector._unwrap(other)

    def __matmul__(self, other):
        return self.as_array @ Vector._unwrap(other)

    def __neg__(self):
        return - self.as_array

    def __pos__(self):
        return + self.as_array

    def __abs__(self):
        return abs(self.as_array)

    def __radd__(self, other):
        return other + self.as_array

    def __rsub__(self, other):
        return other - self.as_array

    def __rmul__(self, other):
        return other * self.as_array

    def __rmatmul__(self, other):
        return other @ self.as_array


class Matrix(Vector):

    def __getitem__(self, item):
        return self.as_array[item]

    @property
    @abc.abstractmethod
    def shape(self) -> Tuple[int, int]:
        pass

    @abc.abstractmethod
    def to_array(self) -> np.ndarray:
        pass


class Svd(Matrix):

    def __init__(self,
                 u: np.ndarray,
                 s: np.ndarray,
                 v: np.ndarray):
        super().__init__()

        self.u: np.ndarray = u
        self.s: np.ndarray = s
        self.v: np.ndarray = v

    @property
    def shape(self) -> Tuple[int, int]:
        return self.u.shape[0], self.v.shape[1]

    @property
    def rank(self):
        return sum(self.s > 1e-10)

    @property
    def t(self):
        return Svd(self.v.T, self.s, self.u.T)

    @property
    def to_array(self) -> np.ndarray:
        return self.u @ np.diag(self.s) @ self.v

    def trim(self, thresh=0, r=None) -> 'Svd':
        if r:
            r = min(r, sum(self.s > thresh))
        else:
            r = sum(self.s > thresh)

        r = max(r, 1)

        u = self.u[:, :r]
        s = self.s[:r]
        v = self.v[:r, :]

        return Svd(u, s, v)

    def norm(self, ord) -> float:
        if ord == 'nuc':
            return sum(self.s)
        if ord == 'fro':
            return sum(self.s ** 2) ** 0.5
        if ord == 2:
            return max(self.s)

        return npl.norm(self.as_array, ord)


class FullMatrix(Matrix):

    def __init__(self, array: np.ndarray):
        super().__init__(array)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.array.shape

    def to_array(self) -> np.ndarray:
        return self.array


class RowMatrix(Matrix):

    def __init__(self, n_row: int, i_row: int, row: np.ndarray):
        super().__init__()

        self.row: np.ndarray = row
        self.n_row: int = n_row
        self.i_row: int = i_row

    @property
    def shape(self) -> Tuple[int, int]:
        return self.n_row, len(self.row)

    def to_array(self) -> np.ndarray:
        array = np.zeros(self.shape)

        array[self.i_row] = self.row

        return array

    def norm(self, ord) -> float:
        if ord in (2, 'nuc', 'fro'):
            return npl.norm(self.row)

        return npl.norm(self.as_array, ord)


class EntryMatrix(Matrix):

    def __init__(self,
                 n_row: int, i_row: int,
                 n_col: int, i_col: int,
                 val: float):
        super().__init__()

        self.n_row: int = n_row
        self.i_row: int = i_row

        self.n_col: int = n_col
        self.i_col: int = i_col

        self.val: float = val

    @property
    def shape(self) -> Tuple[int, int]:
        return self.n_row, self.n_col

    def to_array(self) -> np.ndarray:
        array = np.zeros(self.shape)

        array[self.i_row, self.i_col] = self.val

        return array

    def norm(self, ord) -> float:
        if ord in (2, 'nuc', 'fro'):
            return abs(self.val)

        return npl.norm(self.as_array, ord)
