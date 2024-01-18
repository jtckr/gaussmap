from typing import List

import numpy as np
import sympy as sym
from sympy.abc import u, v

from gaussmap import parameterizations
from gaussmap.objects import surfaces
from gaussmap.utils import utils


def test_original_surface_max_radius_smaller() -> None:
    amount = 20
    max_radius = 0.5

    catenoid = parameterizations.CatenoidParameterization()
    cylinder = parameterizations.CylinderParameterization()
    hyperboloid = parameterizations.HyperboloidParameterization()
    ring_torus = parameterizations.RingTorusParameterization()
    sphere = parameterizations.SphereParameterization()

    ParameterizationList = List[parameterizations.Parameterization]

    parameterization_list: ParameterizationList = [
        catenoid,
        cylinder,
        hyperboloid,
        ring_torus,
        sphere,
    ]

    for parameterization in parameterization_list:
        vectorization = parameterization.vectorization
        u_range = parameterization.u_range
        v_range = parameterization.v_range

        original_surface = surfaces.OriginalSurface(
            vectorization, u_range, v_range, max_radius=max_radius
        )

        u_values = np.linspace(u_range[0], u_range[1], amount)
        v_values = np.linspace(v_range[0], v_range[1], amount)

        for u_value in u_values:
            for v_value in v_values:
                surface_evaluated = utils.evaluate(vectorization, u_value, v_value)
                surface_norm = np.linalg.norm(surface_evaluated)
                surface_expected = max_radius * (surface_evaluated / surface_norm)

                surface_test = original_surface.func(u_value, v_value)

                assert np.array_equal(surface_test, surface_expected)


def test_original_surface_max_radius_larger() -> None:
    amount = 20

    # sphere with radius 1
    max_radius = 1

    sphere = parameterizations.SphereParameterization()

    vectorization = sphere.vectorization
    u_range = sphere.u_range
    v_range = sphere.v_range

    original_surface = surfaces.OriginalSurface(
        vectorization, u_range, v_range, max_radius=max_radius
    )

    u_values = np.linspace(u_range[0], u_range[1], amount)
    v_values = np.linspace(v_range[0], v_range[1], amount)

    for u_value in u_values:
        for v_value in v_values:
            surface_expected = utils.evaluate(vectorization, u_value, v_value)
            surface_test = original_surface.func(u_value, v_value)

            assert np.array_equal(surface_test, surface_expected)

    # maximum radius for cylinder from z=-1 to z=1
    max_radius = np.sqrt(2)

    cylinder = parameterizations.CylinderParameterization()

    vectorization = cylinder.vectorization
    u_range = cylinder.u_range
    v_range = cylinder.v_range

    original_surface = surfaces.OriginalSurface(
        vectorization, u_range, v_range, max_radius=max_radius
    )

    u_values = np.linspace(u_range[0], u_range[1], amount)
    v_values = np.linspace(v_range[0], v_range[1], amount)

    for u_value in u_values:
        for v_value in v_values:
            surface_expected = utils.evaluate(vectorization, u_value, v_value)
            surface_test = original_surface.func(u_value, v_value)

            assert np.array_equal(surface_test, surface_expected)

    # major radius + minor radius
    max_radius = 3 + 1

    ring_torus = parameterizations.RingTorusParameterization()

    vectorization = ring_torus.vectorization
    u_range = ring_torus.u_range
    v_range = ring_torus.v_range

    original_surface = surfaces.OriginalSurface(
        vectorization, u_range, v_range, max_radius=max_radius
    )

    u_values = np.linspace(u_range[0], u_range[1], amount)
    v_values = np.linspace(v_range[0], v_range[1], amount)

    for u_value in u_values:
        for v_value in v_values:
            surface_expected = utils.evaluate(vectorization, u_value, v_value)
            surface_test = original_surface.func(u_value, v_value)

            assert np.array_equal(surface_test, surface_expected)


def test_gauss_map_surface() -> None:
    amount = 20

    for parameterization in parameterizations.parameterization_list:
        normal_vectorization = parameterization.normal_vectorization
        u_range = parameterization.u_range
        v_range = parameterization.v_range

        gauss_map_surface = surfaces.GaussMapSurface(
            normal_vectorization, u_range, v_range
        )

        u_values = np.linspace(u_range[0], u_range[1], amount)
        v_values = np.linspace(v_range[0], v_range[1], amount)

        for u_value in u_values:
            for v_value in v_values:
                normal_evaluated = utils.evaluate(
                    normal_vectorization, u_value, v_value
                )
                unit_normal_expected = utils.normalize(normal_evaluated)
                unit_normal_test = gauss_map_surface.func(u_value, v_value)

                assert np.array_equal(unit_normal_test, unit_normal_expected)


def test_gauss_map_parametric_function() -> None:
    amount = 20

    for parameterization in parameterizations.parameterization_list:
        if parameterization.is_gauss_map_1d:
            normal_expression_u = sym.Matrix(parameterization.normal_expression)
            normal_expression_v = normal_expression_u.subs(
                {u: v, v: u}, simultaneous=True
            )
            normal_vectorization_u = parameterization.normal_vectorization
            normal_vectorization_v = utils.sympy_to_numpy(normal_expression_v)
            u_range = parameterization.u_range
            v_range = parameterization.v_range

            gauss_map_function_u = surfaces.GaussMapParametricFunction(
                normal_vectorization_u, u_range, True
            )
            gauss_map_function_v = surfaces.GaussMapParametricFunction(
                normal_vectorization_v, v_range, False
            )

            u_values = np.linspace(u_range[0], u_range[1], amount)
            v_values = np.linspace(v_range[0], v_range[1], amount)

            for u_value in u_values:
                unit_normal_evaluated_u = utils.evaluate(
                    normal_vectorization_u, u_value, 1
                )
                unit_normal_expected_u = utils.normalize(unit_normal_evaluated_u)
                unit_normal_test_u = gauss_map_function_u.func(u_value)

                assert np.array_equal(unit_normal_test_u, unit_normal_expected_u)

            for v_value in v_values:
                unit_normal_evaluated_v = utils.evaluate(
                    normal_vectorization_v, 1, v_value
                )
                unit_normal_expected_v = utils.normalize(unit_normal_evaluated_v)
                unit_normal_test_v = gauss_map_function_v.func(v_value)

                assert np.array_equal(unit_normal_test_v, unit_normal_expected_v)
