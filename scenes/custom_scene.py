"""Gauss map manimation generator"""
import sys

from sympy.core.sympify import SympifyError

from gaussmap import scene
from gaussmap.utils import parameterization_input


class CustomScene(scene.GaussMapScene):
    """A custom scene that renders the Gauss map transformation from a
    parameterization provided by a user.
    """

    def __init__(self, *args, **kwargs) -> None:
        try:
            expression, u_range, v_range = parameterization_input.get_function()
        except (TypeError, ValueError, SyntaxError, AttributeError, SympifyError) as e:
            sys.exit(f"Unable to parse parameterization {e}")

        print("Generate a Gauss map manimation for the following" " parameterization?")
        print(f"x: {expression}")
        print(f"u: {u_range}, v: {v_range}")
        response = input("[y/n]: ").lower()
        if not response == "y":
            sys.exit("Exiting")

        super().__init__(expression, u_range, v_range, *args, **kwargs)
