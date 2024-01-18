"""Contains a class for generating a Gauss map manimation."""
import manim
import numpy as np
import sympy as sym

from gaussmap.objects import surfaces, vector_field
from gaussmap.typing import Expression, Range
from gaussmap.utils import utils


class GaussMapScene(manim.ThreeDScene):
    """An animated scene that demonstrates the Gauss map of a surface."""

    def __init__(
        self, expression: Expression, u_range: Range, v_range: Range, *args, **kwargs
    ) -> None:
        """Set up the surfaces in the manimation scene."""
        self.gauss_map_surface = None
        self.gauss_map_function = None
        partial_u_expression, partial_v_expression = utils.compute_partials(expression)

        normal_expression = partial_u_expression.cross(partial_v_expression)

        vectorization = utils.sympy_to_numpy(expression)
        normal_vectorization = utils.sympy_to_numpy(normal_expression)

        if utils.is_inward_field(vectorization, normal_vectorization, u_range, v_range):
            normal_expression = -1 * normal_expression
            normal_vectorization = utils.sympy_to_numpy(normal_expression)

        print(f"x = {expression}")
        print(f"x_u = {partial_u_expression}")
        print(f"x_v = {partial_v_expression}")
        print(f"N = {normal_expression}")

        self.original_surface = surfaces.OriginalSurface(
            vectorization, u_range, v_range
        )

        normal_expression_norm = sym.sqrt(normal_expression.dot(normal_expression))

        unit_normal_expression = normal_expression / normal_expression_norm

        unit_normal_vectorization = utils.sympy_to_numpy(unit_normal_expression)

        is_u_function = utils.is_u_function(unit_normal_vectorization, u_range, v_range)
        is_v_function = utils.is_v_function(unit_normal_vectorization, u_range, v_range)

        if is_u_function and is_v_function:
            self.gauss_map_surface = surfaces.GaussMapSurface(
                normal_vectorization, u_range, v_range
            )
        elif is_u_function:
            self.gauss_map_function = surfaces.GaussMapParametricFunction(
                normal_vectorization, u_range, True
            )
        elif is_v_function:
            self.gauss_map_function = surfaces.GaussMapParametricFunction(
                normal_vectorization, v_range, False
            )

        self.vector_field = vector_field.VectorField(
            vectorization, normal_vectorization, u_range, v_range
        )

        super().__init__(*args, **kwargs)

    def construct(self) -> None:
        """Create the objects in the scene and plays the animation."""
        axes = manim.ThreeDAxes()

        self.begin_ambient_camera_rotation(rate=0.1)
        self.set_camera_orientation(phi=45 * manim.DEGREES, theta=30 * manim.DEGREES)
        self.add(self.original_surface)
        self.wait(5)
        self.play(manim.FadeIn(self.vector_field))
        self.wait(5)
        self.play(manim.FadeOut(self.original_surface))
        self.remove(self.original_surface)

        animations = []
        for vector in self.vector_field:
            middle_vector = np.array(vector.get_vector()) / 2
            animations.append(
                manim.ApplyMethod(
                    vector.move_to, manim.ORIGIN + middle_vector, run_time=0.2
                )
            )

        self.play(*animations, run_time=3.0)
        self.wait(2)
        if self.gauss_map_surface is not None:
            self.play(manim.FadeIn(self.gauss_map_surface))
        if self.gauss_map_function is not None:
            self.play(manim.FadeIn(self.gauss_map_function))
        self.play(manim.FadeOut(self.vector_field))
        self.remove(self.vector_field)
        self.wait(5)
