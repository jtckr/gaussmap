import manim
import numpy as np
import pytest
import sympy as sym
from sympy.abc import u, v

from gaussmap import parameterizations, scene
from gaussmap.objects import surfaces


def test_scene(monkeypatch: pytest.MonkeyPatch) -> None:
    amount = 20

    def skip(*args, **kwargs) -> None:
        pass

    monkeypatch.setattr(manim.ThreeDScene, "play", skip)
    monkeypatch.setattr(manim.ThreeDScene, "add", skip)
    monkeypatch.setattr(manim.ThreeDScene, "remove", skip)
    monkeypatch.setattr(manim.ThreeDScene, "wait", skip)
    monkeypatch.setattr(manim.ThreeDScene, "begin_ambient_camera_rotation", skip)
    monkeypatch.setattr(manim.ThreeDScene, "set_camera_orientation", skip)

    for parameterization in parameterizations.parameterization_list:
        expression = sym.Matrix(parameterization.expression)
        u_range = parameterization.u_range
        v_range = parameterization.v_range

        gauss_map_scene = scene.GaussMapScene(expression, u_range, v_range)

        gauss_map_scene.construct()

        normal_vectorization = parameterization.normal_vectorization

        gauss_map_surface = gauss_map_scene.gauss_map_surface
        gauss_map_function = gauss_map_scene.gauss_map_function

        if parameterization.is_gauss_map_1d:
            expression_v = expression.subs({u: v, v: u}, simultaneous=True)
            gauss_map_scene_v = scene.GaussMapScene(expression_v, v_range, u_range)
            gauss_map_surface_v = gauss_map_scene_v.gauss_map_surface
            gauss_map_function_v = gauss_map_scene_v.gauss_map_function

            assert gauss_map_surface_v is None
            assert gauss_map_function_v is not None

            assert gauss_map_surface is None
            assert gauss_map_function is not None
        else:
            assert gauss_map_surface is not None
            assert gauss_map_function is None

            gauss_map_surface_positive = surfaces.GaussMapSurface(
                normal_vectorization, u_range, v_range
            )

            u_values = np.linspace(u_range[0], u_range[1], amount)
            v_values = np.linspace(v_range[0], v_range[1], amount)
            for u_value in u_values:
                for v_value in v_values:
                    test = gauss_map_surface.func(u_value, v_value)
                    if parameterization.is_gauss_map_inward:
                        expected = -1 * gauss_map_surface_positive.func(
                            u_value, v_value
                        )
                        assert np.allclose(test, expected)
