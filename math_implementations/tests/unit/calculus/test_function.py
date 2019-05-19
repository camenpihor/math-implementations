import unittest

import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal

from math_implementations.calculus import Array, Function  # test import


class TestArray(unittest.TestCase):
    fn = Function(function_def=lambda x, y: (x ** 2) + (y ** 2), num_inputs=2, output_dims=None)

    def test_call_lists(self):
        actual = self.fn([1, 2, 3], [1, 2, 3])
        expected = [2, 8, 18]
        assert_array_equal(expected, actual)

    def test_call_numbers(self):
        actual = self.fn(1, 2)
        expected = 5
        assert_array_equal(expected, actual)

    def test_differentiate(self):
        fn_prime = self.fn.differentiate
        assert isinstance(fn_prime, Function)
        assert fn_prime.input_dim == 2
        assert fn_prime.output_dims == (2,)

        x = Array([-10, -5, 0, 5, 10])
        y = x
        expected = [x * 2, y * 2]
        actual = fn_prime(x, y)
        assert_array_almost_equal(expected, actual, decimal=2)

    def test_integrate_r2_to_r(self):
        F = self.fn.integrate
        assert isinstance(F, Function)
        assert F.input_dim == 2
        assert F.output_dims is None

        bounds = [0, 2]
        expected = 10.66667
        actual = F(*bounds)
        assert_almost_equal(actual, expected, decimal=0)  # takes too long to be good :/

    def test_integrate_partial_derivatives(self):
        f_star = self.fn.differentiate.integrate
        bounds = [0, 2]

        expected = self.fn(2, 2)
        actual = f_star(*bounds)
        assert_almost_equal(actual, expected, decimal=2)
