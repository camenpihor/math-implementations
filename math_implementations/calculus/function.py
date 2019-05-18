"""Class which implements differentiation and integration."""
from .array import Array


class Function:
    """Function wrapper.

    Provides additional methods to aid differentiation and integration.
    """

    e = 1e-4

    def __init__(self, function_def, num_inputs, output_dims=None):
        """Wrap functions to aid differeentiation and integration.

        Parameters
        ----------
        function_def : function
            function from R^N -> R^M.
            The number of function arguments must equal its dimensionality.
        input_dim : int
            N in R^N -> R^M
        output_dims : None | tuple
            the dimensionality of the matrix returned by the function for each input. Defaults to
            None, in the case of a function from R^N to R, since no matrix is returned for each
            input.
        """
        self.fn = function_def
        self.input_dim = num_inputs
        self.output_dims = output_dims

    def __call__(self, *args):
        """Call function.

        Parameters
        ----------
        *args : int | float | list
            Inputs to Function

        Returns
        -------
        Array
            Array of dimension `self.dims`

        """
        f_args = [Array(arg) if isinstance(arg, list) else Array([arg]) for arg in args]
        result = Array(self.fn(*f_args))
        if result.shape == (1,):
            result = result[0]
        return result

    def __differentiate(self, *data):
        data = list(data)

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

    def __integrate_left(self, lower_bound, upper_bound):
        "This doesn't work in the case of integrating f: x^2 + y^2. Think about what to do to get the reimann sum of that thing"
        lower_bound = lower_bound[0]
        upper_bound = upper_bound[0]

        cum_sum = 0
        integral = []
        current_point = Array([lower_bound] * self.input_dim)
        while current_point[0] < upper_bound:
            print(current_point)
            at_point = self(*current_point) * self.e

            if not isinstance(at_point, list):
                cum_sum += at_point
            else:
                cum_sum += sum(at_point)

            integral.append(cum_sum)
            current_point = current_point + self.e
        return integral

    @property
    def differentiate(self):
        """Differentiate function.

        Returns
        -------
        Function
            f -> f'(x)

        """
        output_dims = self.output_dims

        if output_dims is None:
            output_dims = (self.input_dim,)

        else:
            output_dims += (self.input_dim,)

        return Function(
            function_def=self.__differentiate, num_inputs=self.input_dim, output_dims=output_dims
        )

    @property
    def integrate(self):
        """Integrate function.

        Returns
        -------
        Function
            f -> F(x)

        """
        return Function(self.__integrate_left, num_inputs=self.input_dim, output_dims=None)


class DimensionMismatchExpcetion(Exception):
    """Raised when the number of arguments don't equal the dimension of the function."""
