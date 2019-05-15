"""Class which implements differentiation and integration."""
from inspect import signature

from .array import Array


class Function:
    """Function wrapper.

    Provides additional methods to aid differentiation and integration.
    """

    e = 1e-2

    def __init__(self, function_def):
        """Wrap functions to aid differeentiation and integration.

        Parameters
        ----------
        function_def : function
            function from R^N -> R.
            The number of function arguments must equal its dimensionality.

        """
        self.fn = function_def
        self.n_dims = len(signature(self.fn).parameters)

    def __call__(self, *args):
        """Call function.

        Parameters
        ----------
        *args : int | float | list
            Inputs to Function

        Returns
        -------
        Array
            Array of dimension `self.n_dims`

        """
        self.check_dimensionality(args)
        f_args = [Array(arg) if isinstance(arg, list) else Array([arg]) for arg in args]
        return Array(self.fn(*f_args))

    def check_dimensionality(self, args):
        """Ensure that the dimensionality of the input is equal to that of `self.f`."""
        if len(args) != self.n_dims:
            raise DimensionMismatchExpcetion(f"Invalid number of arguments! Must be {self.n_dims}")

    def __differentiate(self, data):
        if not isinstance(data, list):
            data = [data]

        data = [Array(datum) for datum in data]
        num_dims = len(data)

        partials = []
        for dim_number in range(num_dims):
            new_data = [
                data[idx] + self.e if idx == dim_number else data[idx] for idx in range(num_dims)
            ]
            partials.append((self(*new_data) - self(*data)) / self.e)

        return partials

    def __integrate_left(self, x):
        # TODO rework for multiple dimensions
        cum_sum = 0
        integral = []
        for element in x:
            at_point = self(element) * self.e
            integral.append(at_point + cum_sum)
            cum_sum += at_point
        return Array(integral)

    def differentiate(self):
        """Differentiate function.

        Returns
        -------
        Function
            f -> f'(x)

        """
        return Function(self.__differentiate)

    def integrate(self):
        """Integrate function.

        Returns
        -------
        Function
            f -> F(x)

        """
        return Function(self.__integrate_left)


class DimensionMismatchExpcetion(Exception):
    """Raised when the number of arguments don't equal the dimension of the function."""
