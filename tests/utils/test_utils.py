import numpy as np
import sympy as sym
from sympy.abc import u, v

from gaussmap import parameterizations
from gaussmap.utils import utils

generator = np.random.default_rng(seed=321927785518592140237305164217414271323)


def test_is_u_function() -> None:
    for parameterization in parameterizations.parameterization_list:
        normal_vectorization = parameterization.normal_vectorization
        u_range = parameterization.u_range
        v_range = parameterization.v_range

        assert utils.is_u_function(normal_vectorization, u_range, v_range)
        if parameterization.is_gauss_map_1d:
            normal_expression = sym.Matrix(parameterization.normal_expression)
            normal_expression_norm = sym.sqrt(normal_expression.dot(normal_expression))
            unit_normal_expression = normal_expression / normal_expression_norm
            unit_normal_vectorization = utils.sympy_to_numpy(unit_normal_expression)
            assert not utils.is_v_function(unit_normal_vectorization, u_range, v_range)


def test_is_v_function() -> None:
    for parameterization in parameterizations.parameterization_list:
        normal_expression = sym.Matrix(parameterization.normal_expression)
        normal_expression = normal_expression.subs({u: v, v: u}, simultaneous=True)
        normal_vectorization = utils.sympy_to_numpy(normal_expression)
        u_range = parameterization.u_range
        v_range = parameterization.v_range

        assert utils.is_v_function(normal_vectorization, v_range, u_range)
        if parameterization.is_gauss_map_1d:
            normal_expression_norm = sym.sqrt(normal_expression.dot(normal_expression))
            unit_normal_expression = normal_expression / normal_expression_norm
            normal_vectorization = utils.sympy_to_numpy(unit_normal_expression)
            assert not utils.is_u_function(normal_vectorization, v_range, u_range)


def test_is_inward_field() -> None:
    for parameterization in parameterizations.parameterization_list:
        vectorization = parameterization.vectorization
        normal_vectorization = parameterization.normal_vectorization
        u_range = parameterization.u_range
        v_range = parameterization.v_range

        if parameterization.is_gauss_map_inward:
            assert utils.is_inward_field(
                vectorization, normal_vectorization, u_range, v_range
            )
        else:
            assert not utils.is_inward_field(
                vectorization, normal_vectorization, u_range, v_range
            )


def test_is_pointing_inward() -> None:
    amount = 100
    min_scalar = 1
    max_scalar = 20

    scalars = np.abs(generator.integers(min_scalar, max_scalar, size=(amount, 1)))
    points = np.abs(generator.random(size=(amount, 3)))
    inward_vectors = -1 * np.abs(generator.random(size=(amount, 3)))
    outward_vectors = np.abs(generator.random(size=(amount, 3)))

    scaled_points = scalars * points
    scaled_inward_vectors = scalars * inward_vectors
    scaled_outward_vectors = scalars * outward_vectors

    for i in range(amount):
        assert utils.is_pointing_inward(scaled_points[i], scaled_inward_vectors[i])
        assert not utils.is_pointing_inward(scaled_points[i], scaled_outward_vectors[i])

        assert utils.is_pointing_inward(
            -1 * scaled_points[i], -1 * scaled_inward_vectors[i]
        )
        assert not utils.is_pointing_inward(
            -1 * scaled_points[i], -1 * outward_vectors[i]
        )


def test_evaluate() -> None:
    amount = 100
    max_scalar = 20

    scalars = generator.integers(max_scalar, size=(amount, 1))
    u = generator.random(size=(amount, 1))
    v = generator.random(size=(amount, 1))

    scaled_u = scalars * u
    scaled_v = scalars * v

    scaled_u = np.concatenate((scaled_u, -1 * scaled_u), axis=1)
    scaled_v = np.concatenate((scaled_v, -1 * scaled_v), axis=1)

    for parameterization in parameterizations.parameterization_list:
        vectorization = parameterization.vectorization

        for u_value in scaled_u:
            for v_value in scaled_v:
                x_value = vectorization[0](u_value, v_value)
                y_value = vectorization[1](u_value, v_value)
                z_value = vectorization[2](u_value, v_value)
                evaluation_expected = [x_value, y_value, z_value]
                evaluation_test = utils.evaluate(vectorization, u_value, v_value)

                assert np.array_equal(evaluation_test, evaluation_expected)


def test_sympy_to_numpy() -> None:
    amount = 100
    max_scalar = 20

    scalars = generator.integers(max_scalar, size=(amount, 1))
    u = generator.random(size=(amount, 1))
    v = generator.random(size=(amount, 1))

    scaled_u = scalars * u
    scaled_v = scalars * v

    scaled_u = np.concatenate((scaled_u, -1 * scaled_u), axis=1)
    scaled_v = np.concatenate((scaled_v, -1 * scaled_v), axis=1)

    for parameterization in parameterizations.parameterization_list:
        expression = parameterization.expression
        vectorization_expected = parameterization.vectorization
        vectorization_test = utils.sympy_to_numpy(expression)

        for u_value in scaled_u:
            for v_value in scaled_v:
                evaluation_expected = utils.evaluate(
                    vectorization_expected, u_value, v_value
                )
                evaluation_test = utils.evaluate(vectorization_test, u_value, v_value)
                assert np.allclose(evaluation_test, evaluation_expected)


def test_normalize_zero_vector() -> None:
    zero_vector = np.array([0, 0, 0])
    zero_vector_normalized = utils.normalize(zero_vector)

    assert np.array_equal(zero_vector_normalized, zero_vector)


def test_normalize_single_axis() -> None:
    amount = 100
    min_scalar = 1
    max_scalar = 20

    x_vector = np.array([1, 0, 0])
    y_vector = np.array([0, 1, 0])
    z_vector = np.array([0, 0, 1])

    integers = generator.integers(min_scalar, max_scalar, size=(amount, 1))
    scalars = integers * (generator.random(size=(amount, 1)) + 1)

    x_vectors = scalars * x_vector
    y_vectors = scalars * y_vector
    z_vectors = scalars * z_vector

    print(x_vectors[0])

    for i in range(amount):
        norm_x = utils.normalize(x_vectors[i])
        norm_y = utils.normalize(y_vectors[i])
        norm_z = utils.normalize(z_vectors[i])
        assert np.array_equal(norm_x, x_vector)
        assert np.array_equal(norm_y, y_vector)
        assert np.array_equal(norm_z, z_vector)

        norm_x = utils.normalize(-1 * x_vectors[i])
        norm_y = utils.normalize(-1 * y_vectors[i])
        norm_z = utils.normalize(-1 * z_vectors[i])
        assert np.array_equal(norm_x, -1 * x_vector)
        assert np.array_equal(norm_y, -1 * y_vector)
        assert np.array_equal(norm_z, -1 * z_vector)


def test_normalize_dual_axis() -> None:
    min_scalar = 1
    max_scalar = 20

    vectors = np.array(
        [
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [-1, 1, 0],
            [-1, 0, 1],
            [0, -1, 1],
            [1, -1, 0],
            [1, 0, -1],
            [0, 1, -1],
            [-1, -1, 0],
            [-1, 0, -1],
            [0, -1, -1],
        ]
    ) / np.sqrt(2)

    vector_amount = vectors.shape[0]

    amount = vector_amount

    integers = generator.integers(min_scalar, max_scalar, size=(amount, 1))
    scalars = integers * (generator.random(size=(amount, 1)) + 1)

    scaled_vectors = scalars * vectors

    for i in range(amount):
        norm = utils.normalize(scaled_vectors[i])
        assert np.allclose(norm, vectors[i])


def test_normalize_dual_axis_uneven() -> None:
    min_scalar = 1
    max_scalar = 20

    vectors = np.array(
        [
            [2 * np.pi, np.pi, 0],
            [2 * np.pi, 0, np.pi],
            [0, 2 * np.pi, np.pi],
            [-2 * np.pi, np.pi, 0],
            [-2 * np.pi, 0, np.pi],
            [0, -2 * np.pi, np.pi],
            [2 * np.pi, -np.pi, 0],
            [2 * np.pi, 0, -np.pi],
            [0, 2 * np.pi, -np.pi],
            [-2 * np.pi, -np.pi, 0],
            [-2 * np.pi, 0, -np.pi],
            [0, -2 * np.pi, -np.pi],
            [np.pi, 2 * np.pi, 0],
            [np.pi, 0, 2 * np.pi],
            [0, np.pi, 2 * np.pi],
            [-np.pi, 2 * np.pi, 0],
            [-np.pi, 0, 2 * np.pi],
            [0, -np.pi, 2 * np.pi],
            [np.pi, -2 * np.pi, 0],
            [np.pi, 0, -2 * np.pi],
            [0, np.pi, -2 * np.pi],
            [-np.pi, -2 * np.pi, 0],
            [-np.pi, 0, -2 * np.pi],
            [0, -np.pi, -2 * np.pi],
        ]
    )

    vectors /= np.sqrt(5) * np.pi

    vector_amount = vectors.shape[0]

    amount = vector_amount

    integers = generator.integers(min_scalar, max_scalar, size=(amount, 1))
    scalars = integers * (generator.random(size=(amount, 1)) + 1)

    scaled_vectors = scalars * vectors

    for i in range(amount):
        norm = utils.normalize(scaled_vectors[i])
        assert np.allclose(norm, vectors[i])


def test_compute_partials() -> None:
    for parameterization in parameterizations.parameterization_list:
        expression = parameterization.expression
        partial_u_expression_expected = parameterization.partial_u_expression
        partial_v_expression_expected = parameterization.partial_v_expression

        partials_test = utils.compute_partials(expression)

        assert partials_test[0].equals(sym.Matrix(partial_u_expression_expected))
        assert partials_test[1].equals(sym.Matrix(partial_v_expression_expected))
