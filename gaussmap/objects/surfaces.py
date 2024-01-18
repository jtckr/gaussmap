"""Contains classes for a parameterized surface and its Gauss map."""
import manim
import numpy as np
import numpy.typing as npt

from gaussmap.typing import Range, Vectorization
from gaussmap.utils import utils


class OriginalSurface(manim.Surface):
    """A manim surface of the parameterization for the manimation."""

    def __init__(
        self,
        vectorization: Vectorization,
        u_range: Range,
        v_range: Range,
        max_radius: float = 8,
        **kwargs,
    ) -> None:
        """Generate the parameterized surface for the manimation.

        Args:
            vectorization: A tuple of vectorized functions that describes
                the parameterized surface.
            u_range: A tuple containing the starting and ending u
                coordinates.
            v_range: A tuple containing the staring and ending v
                coordinates.
            max_radius: The maximum radius of vector that will be rendered.
        """
        self.__max_radius = max_radius
        self.__vectorization = vectorization

        super().__init__(self.func, u_range=u_range, v_range=v_range, **kwargs)

    def func(self, u_value: float, v_value: float) -> npt.NDArray[np.float64]:
        """Evaluate the function at u and v to generate the surface."""
        evaluation = utils.evaluate(self.__vectorization, u_value, v_value)

        # Limit graphs to sphere with radius r_max
        # a spherical boundary looks better than a cubic one
        radius = np.linalg.norm(evaluation)
        if radius > self.__max_radius:
            evaluation = np.divide(np.multiply(self.__max_radius, evaluation), radius)

        return evaluation


class GaussMapSurface(manim.Surface):
    """A manim surface of the Gauss map for the manimation."""

    def __init__(
        self,
        normal_vectorization: Vectorization,
        u_range: Range,
        v_range: Range,
        **kwargs,
    ) -> None:
        """Generate the Gauss map surface for the manimation.

        Args:
            normal_vectorization: A tuple of vectorized functions that
                describes the normal vectors of the original surface.
            u_range: A tuple containing the starting and ending u
                coordinates.
            v_range: A tuple containing the starting and ending v
                coordinates.
        """
        self.__normal_vectorization = normal_vectorization

        super().__init__(self.func, u_range=u_range, v_range=v_range, **kwargs)

    def func(self, u_value: float, v_value: float) -> npt.NDArray[np.float64]:
        """Evaluate the normal vectors at u and v to generate the surface."""
        normal_evaluation = utils.evaluate(
            self.__normal_vectorization, u_value, v_value
        )

        return utils.normalize(normal_evaluation)


class GaussMapParametricFunction(manim.ParametricFunction):
    """A manim parametric function of the Gauss map for the manimation."""

    def __init__(
        self,
        normal_vectorization: Vectorization,
        t_range: Range,
        is_u_function: bool,
        **kwargs,
    ) -> None:
        """Generate the Gauss map parametric function for the manimation.

        Args:
            normal_vectorization: A tuple of vectorized functions that
                describes the normal vectors of the original surface.
            t_range: A tuple containing the starting and ending t
                coordinates.
            is_u_function: A bool that is true if the normal function is a
                function of only u and false if it is a function of only v.
        """
        self.__normal_vectorization = normal_vectorization
        self.__t_range = t_range
        self.__is_u_function = is_u_function

        super().__init__(self.func, t_range=t_range, **kwargs)

    def func(self, t_value: float) -> npt.NDArray[np.float64]:
        """Evaluate the normal vectors at t to generate the function."""
        # The evaluation is done at 1 so that the denominator is
        # unlikely to be zero when normalizing.
        if self.__is_u_function:
            normal_evaluation = utils.evaluate(self.__normal_vectorization, t_value, 1)
        else:
            normal_evaluation = utils.evaluate(self.__normal_vectorization, 1, t_value)

        return utils.normalize(normal_evaluation)
