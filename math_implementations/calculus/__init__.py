"""Implmentation of some numerical calculus methods without relying on NumPy.

Provides
    1. Array: a subclass of list to make list operations easier since we do not rely on
    Numpy.
    2. Function: a wrapper of `function`s which implements differentiation and
    integration.
"""
from .array import Array
from .function import Function
