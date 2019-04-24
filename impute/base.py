from typing import Optional, Tuple

import numpy as np

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
