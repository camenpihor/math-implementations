import unittest

import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from math_implementations.calculus import Array, Function  # test import
from math_implementations.calculus.function import DimensionMismatchExpcetion


class TestArray(unittest.TestCase):
    fn = Function(function_def=lambda x, y: (x ** 2) + (y ** 2), num_dims=2)

    def test_call_lists(self):
        actual = self.fn([1, 2, 3], [1, 2, 3])
        expected = [2, 8, 18]
        assert_array_equal(expected, actual)

    def test_call_numbers(self):
        actual = self.fn(1, 2)
        expected = [5]
        assert_array_equal(expected, actual)

    def test_check_dimensionality(self):
        with pytest.raises(DimensionMismatchExpcetion):
            self.fn(1, 2, 3)  # three arguments instead of the expected 2

    def test_differentiate(self):
        fn_prime = self.fn.differentiate
        assert isinstance(fn_prime, Function)

        x = Array([-10, -5, 0, 5, 10])
        y = x
        expected = [x * 2, y * 2]
        actual = fn_prime(x, y)
        assert_array_almost_equal(expected, actual, decimal=2)
