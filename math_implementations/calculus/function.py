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

    def __repr__(self):
        """Human-readable representation of Function."""
        input_dimension = f"R^{self.input_dim}" if self.input_dim > 1 else "R"
        output_dimension = None

        if self.output_dims is None:
            output_dimension = "R"
        elif len(self.output_dims) == 1:
            output_dimension = f"R^{self.output_dims[0]}"
        else:
            output_dimension = f"R^{self.output_dims}"

        return f"f: {input_dimension} --> {output_dimension}"

    def __call__(self, *args):
        """Call function.

        Parameters
        ----------
        *args : int | float | list
            Inputs to Function

        Returns
        -------
        Array
            Array of dimension `self.input_dim`

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

        partials = []
        for dim_number in range(self.input_dim):
            new_data = [
                data[idx] + self.e if idx == dim_number else data[idx]
                for idx in range(self.input_dim)
            ]
            partials.append((self(*new_data) - self(*data)) / self.e)

        return partials

    def __integrate_left(self, lower_bound, upper_bound):
        """Integrate function from R^N -> R, where N does not equal 2.

        This integrates f over the hypercube defined by [`lower_bound`, `upper_bound`]^N. This
        function is also used to integrate gradients.
        """
        lower_bound = lower_bound[0]
        upper_bound = upper_bound[0]

        cum_sum = 0
        current_point = Array([lower_bound] * self.input_dim)
        while current_point[0] < upper_bound:
            at_point = self(*current_point) * self.e

            if not isinstance(at_point, list):
                cum_sum += at_point
            else:
                cum_sum += sum(at_point)

            current_point = current_point + self.e

        return [cum_sum]

    def __integrate_r2_to_r(self, lower_bound, upper_bound):
        """Integrate a function from R^2 -> R, over a cube."""
        eps = 1e-2
        lower_bound = lower_bound[0]
        upper_bound = upper_bound[0]

        cum_sum = 0
        current_x = lower_bound
        current_y = lower_bound
        while current_x < upper_bound:
            while current_y < upper_bound:
                at_point = self(current_x, current_y) * (eps ** 2)
                cum_sum += at_point
                current_y += eps

            current_y = lower_bound
            current_x += eps

        return [cum_sum]

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
        if self.input_dim == 2 and self.output_dims is None:
            return Function(self.__integrate_r2_to_r, num_inputs=self.input_dim, output_dims=None)

        return Function(self.__integrate_left, num_inputs=self.input_dim, output_dims=None)
