"""Contains a class for the normal vector field of a surface."""
import manim
import numpy as np
import numpy.typing as npt

from gaussmap.typing import Range, Vectorization
from gaussmap.utils import utils


class VectorField(manim.VGroup):
    """The normal vectors as a group of manim vectors for the manimation."""

    def __init__(
        self,
        vectorization: Vectorization,
        normal_vectorization: Vectorization,
        u_range: Range,
        v_range: Range,
        max_radius: float = 8,
        amount: int = 20,
        **kwargs,
    ) -> None:
        """Generate the normal vector field for the manimation.

        Args:
            vectorization: A tuple of vectorized functions that describes
                the original surface.
            normal_vectorization: A tuple of vectorized functions that
                describes the normal vectors of the original surface.
            u_range: A tuple containing the starting and ending u
                coordinates.
            v_range: A tuple containing the starting and ending v
                coordinates.
            max_radius: The maximum radius from the origin where vectors
                are generated.
            amount: The number of vectors to generate. They will be
                dispersed evenly throughout the surface.
        """
        super().__init__(**kwargs)
        self.__max_radius = max_radius

        u_values = np.linspace(u_range[0], u_range[1], amount)
        v_values = np.linspace(v_range[0], v_range[1], amount)

        for u_value in u_values:
            for v_value in v_values:
                evaluation = utils.evaluate(vectorization, u_value, v_value)
                normal_evaluation = utils.evaluate(
                    normal_vectorization, u_value, v_value
                )

                # Ignore vectors that start outside a sphere of radius r_max
                radius = np.linalg.norm(evaluation)
                if radius < self.__max_radius:
                    self.add(self.__create_vector(evaluation, normal_evaluation))

    @staticmethod
    def __create_vector(
        position: npt.NDArray[np.float64], normal_vector: npt.NDArray[np.float64]
    ) -> manim.Vector:
        """Evaluate the function at u and v to generate the vector field."""
        normal_vector = utils.normalize(normal_vector)

        # Move vector away from the surface a bit so it does not clip through
        position = position + 0.1 * normal_vector

        vector = manim.Vector(direction=normal_vector).shift(position)

        # shade_in_3d and depth_test prevent vectors and their tips from being
        # visible when they are behind 3d objects. shade_in_3d is for the
        # Cairo renderer and depth_test is for the OpenGL Renderer.
        if hasattr(vector, "shade_in_3d"):
            vector.shade_in_3d = True
        if hasattr(vector.get_tip(), "shade_in_3d"):
            vector.get_tip().shade_in_3d = True

        if hasattr(vector, "depth_test"):
            vector.depth_test = True
        if hasattr(vector.get_tip(), "depth_test"):
            vector.get_tip().depth_test = True

        return vector
