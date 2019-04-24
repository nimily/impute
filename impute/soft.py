import numpy as np
import numpy.linalg as npl

from typing import List, Tuple

from .sample_set import EntrySampleSet
from .utils import soft_svt


class SoftImpute:

    def __init__(self, shape, svt_op=soft_svt):
        self.shape = shape
        self.svt_op = svt_op

        self.starting_point = None
        self.z_old = None
        self.z_new = None

        self._init_starting_point()
        self._init_z()

    def _init_starting_point(self, value=None):

        if value is None:
            self.starting_point = self.zero()
        else:
            self.starting_point = np.copy(value)

    def _init_z(self):
        self.z_old = None
        self.z_new = self.starting_point

    def update_once(self, ss: EntrySampleSet, alpha: float) -> Tuple[float, float]:
        z_old = self.z_new

        y_new = z_old - ss.rss_grad(z_old)
        z_new = self.svt(y_new, alpha)

        self.z_old = z_old
        self.z_new = z_new

        return npl.norm(z_new - z_old), npl.norm(z_old)

    def svt(self, w, alpha):
        return self.svt_op(w, alpha)

    def fit(self,
            ss: EntrySampleSet,
            alphas: List[float],
            max_iters: int = 100,
            tol: float = 1e-5,
            warm_start: bool = True):

        if not warm_start:
            self._init_z()

        zs = []

        for alpha in alphas:
            for i in range(max_iters):
                delta_norm, old_norm = self.update_once(ss, alpha)

                if delta_norm ** 2 <= tol * old_norm ** 2:
                    break

            zs.append(self.z_new)

        return zs

    def alpha_max(self, ss: EntrySampleSet):
        grad = ss.rss_grad(self.zero())

        return npl.norm(grad, 2)

    def zero(self):
        return np.zeros(self.shape)
