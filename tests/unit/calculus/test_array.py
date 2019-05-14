import operator
import unittest

import numpy as np
import pytest

from math_implementations.calculus import Array  # test import
from math_implementations.calculus.array import InvalidInputException


class TestArray(unittest.TestCase):
    three_one = Array([1, 2, 3])
    two_three = Array([[1, -2, 3], [4, 5, -6]])
    two_two_three = Array([[[-1, 2, 3], [4, -5, 6]], [[7, -8, 9], [-10, 11, 12]]])
    test_cases = [three_one, two_three, two_two_three]

    def test_shape(self):
        for test_case in self.test_cases:
            assert test_case.shape == np.array(test_case).shape

    def test_alias_T(self):
        np.testing.assert_array_equal(self.three_one.T, self.three_one.transpose)

    def test_transpose(self):
        for test_case in self.test_cases:
            np.testing.assert_array_equal(test_case.transpose, np.array(test_case).T)

    def test_abs(self):
        for test_case in self.test_cases:
            np.testing.assert_array_equal(test_case.abs(), np.abs(test_case))

    def test_operators(self):
        math_operators = [
            operator.add,
            operator.sub,
            operator.eq,
            operator.truediv,
            operator.le,
            operator.ge,
            operator.gt,
            operator.lt,
            operator.ne,
            operator.mul,
            operator.pow,
        ]

        for test_case in self.test_cases:
            for math_op in math_operators:
                np.testing.assert_array_equal(
                    math_op(test_case, 5),
                    math_op(np.array(test_case), 5),
                    f"Failed on: {test_case}",
                )

                np.testing.assert_array_equal(
                    math_op(test_case, test_case.abs()),
                    math_op(np.array(test_case), test_case.abs()),
                    f"Failed on: {test_case}",
                )

    def test_indexing(self):
        # Result should not be of type list, in not, these will fail
        assert self.three_one[0] == np.array(self.three_one)[0]
        assert self.two_three[0, 0] == np.array(self.two_three)[0, 0]
        assert self.two_two_three[0, 0, 0] == np.array(self.two_two_three)[0, 0, 0]

        # Result may be of type list
        np.testing.assert_array_equal(
            self.two_three[:, 0], np.array(self.two_three)[:, 0]
        )
        np.testing.assert_array_equal(
            self.two_three[0, :], np.array(self.two_three[0, :])
        )
        np.testing.assert_array_equal(
            self.two_two_three[:-1, 0, 0], np.array(self.two_two_three[:-1, 0, 0])
        )
        np.testing.assert_array_equal(
            self.two_two_three[0, :-1, 0], np.array(self.two_two_three[0, :-1, 0])
        )
        np.testing.assert_array_equal(
            self.two_two_three[0, 0, :-1], np.array(self.two_two_three[0, 0, :-1])
        )
        np.testing.assert_array_equal(
            self.two_two_three[:1, :1, 0], np.array(self.two_two_three[:1, :1, 0])
        )

    def test_check_input(self):
        with pytest.raises(InvalidInputException):
            Array("A")
