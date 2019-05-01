import numpy as np
import numpy.testing as npt

import pytest


@pytest.mark.usefixtures('re_dot_case')
class TestDotLinearOp:

    @staticmethod
    def test_op_norm(re_dot_case):
        op, entries = re_dot_case

        m = np.zeros(op.i_shape)
        for j, v, _ in entries:
            m[j] += v ** 2

        actual = op.norm()
        expect = m.max() ** 0.5

        npt.assert_almost_equal(actual, expect)

    @staticmethod
    def test_xtx_op_norm(re_dot_case):
        op, entries = re_dot_case

        m = np.zeros(op.i_shape)
        for j, v, _ in entries:
            m[j] += v ** 2

        actual = op.xtx.norm()
        expect = m.max()

        npt.assert_almost_equal(actual, expect)

    @staticmethod
    def test_xxt_op_norm(re_dot_case):
        op, entries = re_dot_case

        m = np.zeros(op.i_shape)
        for j, v, _ in entries:
            m[j] += v ** 2

        actual = op.xxt.norm()
        expect = m.max()

        npt.assert_almost_equal(actual, expect)


@pytest.mark.usefixtures('op_info', 'trace_op_case')
class TestTraceLinearOp:

    @staticmethod
    def test_evaluate(op_info, trace_op_case):
        data = trace_op_case
        op = TestTraceLinearOp.get_loaded_op(op_info, data)

        actual = op.evaluate(data.b)
        expect = data.expect_evaluate

        npt.assert_array_almost_equal(actual, expect)

    @staticmethod
    def test_evaluate_t(op_info, trace_op_case):
        data = trace_op_case
        op = TestTraceLinearOp.get_loaded_op(op_info, data)

        actual = op.evaluate_t(data.bt)
        expect = data.expect_evaluate_t

        npt.assert_array_almost_equal(actual, expect)

    @staticmethod
    def test_xtx(op_info, trace_op_case):
        data = trace_op_case
        op = TestTraceLinearOp.get_loaded_op(op_info, data)

        actual = op.xtx(data.b)
        expect = data.expect_xtx

        npt.assert_array_almost_equal(actual, expect)

    @staticmethod
    def test_xxt(op_info, trace_op_case):
        data = trace_op_case
        op = TestTraceLinearOp.get_loaded_op(op_info, data)

        actual = op.xxt(data.bt)
        expect = data.expect_xxt

        npt.assert_array_almost_equal(actual, expect)

    @staticmethod
    def test_norm(op_info, trace_op_case):
        data = trace_op_case
        op = TestTraceLinearOp.get_loaded_op(op_info, data)

        actual = op.norm()
        expect = data.expect_norm

        npt.assert_array_almost_equal(actual, expect)

    @staticmethod
    def get_loaded_op(op_info, trace_op_case):
        data = trace_op_case
        op_cls, op_add = op_info

        op = op_cls(data.shape)
        op_add(op, data)

        return op
