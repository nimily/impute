import numpy as np
import numpy.linalg as npl
import numpy.testing as npt

import pytest

from impute.svt import tuned_svt


def create_imputer(imputer_cls, svt, shape):

    return imputer_cls(shape, svt_op=tuned_svt(svd=svt))


@pytest.mark.usefixtures('imputer_cls', 'rae_case', 'alpha_ratio')
class TestLagrangianImputer:

    @staticmethod
    @pytest.mark.usefixtures('svt')
    def test_alpha_max(imputer_cls, svt, rae_case):
        b, ds = rae_case
        shape = b.shape

        imputer = create_imputer(imputer_cls, svt, shape)
        alpha = imputer.alpha_max(ds)

        zs = imputer.fit(ds, [alpha, alpha * 0.999])

        actual = zs[0].to_matrix()
        expect = np.zeros_like(actual)
        npt.assert_array_almost_equal(actual, expect)

        actual = npl.norm(zs[1].to_matrix())
        assert actual > 1e-5

    @staticmethod
    def test_strong_optimality(imputer_cls, rae_case, alpha_ratio):
        svt = 'exact'

        b, ds = rae_case
        shape = b.shape

        imputer = create_imputer(imputer_cls, svt, shape)
        alpha = imputer.alpha_max(ds) * alpha_ratio

        z = imputer.impute(ds, alpha=alpha, xtol=1e-10)
        m = z.to_matrix()
        g = ds.rss_grad(m)

        u = z.u
        v = z.v
        w = g + alpha * u @ v

        actual = u.T @ w
        expect = np.zeros(actual.shape)
        npt.assert_almost_equal(actual, expect, decimal=3)

        actual = w @ v.T
        expect = np.zeros(actual.shape)
        npt.assert_almost_equal(actual, expect)

        actual = npl.norm(w, 2)
        expect = alpha * (1 + 1e-5)
        assert actual < expect

    @staticmethod
    @pytest.mark.usefixtures('svt')
    def test_weak_optimality(imputer_cls, svt, rae_case, alpha_ratio):
        b, ds = rae_case
        shape = b.shape

        imputer = create_imputer(imputer_cls, svt, shape)
        alpha = imputer.alpha_max(ds) * alpha_ratio

        z = imputer.impute(ds, alpha=alpha, gtol=1e-3)
        m = z.to_matrix()

        loss_b = ds.loss(b, alpha) + 1e-5
        loss_m = ds.loss(m, alpha)

        assert loss_m < loss_b
