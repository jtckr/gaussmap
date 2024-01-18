import manim
import numpy as np

from gaussmap import parameterizations
from gaussmap.objects import vector_field


def test_vector_field_init_amount() -> None:
    cylinder = parameterizations.CylinderParameterization()

    vectorization = cylinder.vectorization
    normal_vectorization = cylinder.normal_vectorization
    u_range = cylinder.u_range
    v_range = cylinder.v_range

    max_radius = 8
    amount_list = [20, 50]
    objects_expected_list = [400, 2500]

    for amount, objects_expected in zip(amount_list, objects_expected_list):
        field = vector_field.VectorField(
            vectorization,
            normal_vectorization,
            u_range,
            v_range,
            max_radius=max_radius,
            amount=amount,
        )
        assert len(field.submobjects) == objects_expected


def test_vector_field_init_max_radius() -> None:
    cylinder = parameterizations.CylinderParameterization()

    vectorization = cylinder.vectorization
    normal_vectorization = cylinder.normal_vectorization
    u_range = cylinder.u_range
    v_range = cylinder.v_range

    amount = 20
    max_radius_list = [0.5, 1, 2]
    objects_expected_list = [0, 0, 400]

    for max_radius, objects_expected in zip(max_radius_list, objects_expected_list):
        field = vector_field.VectorField(
            vectorization,
            normal_vectorization,
            u_range,
            v_range,
            max_radius=max_radius,
            amount=amount,
        )
        assert len(field.submobjects) == objects_expected


def test_vector_field_create_vector_cairo() -> None:
    manim.config.renderer = "cairo"
    # The parameterization does not matter since __create_vector does not
    # depend on the VectorField instance.
    cylinder = parameterizations.CylinderParameterization()

    vectorization = cylinder.vectorization
    normal_vectorization = cylinder.normal_vectorization
    u_range = cylinder.u_range
    v_range = cylinder.v_range

    field = vector_field.VectorField(
        vectorization, normal_vectorization, u_range, v_range
    )

    position = np.array([2, 3, 4])
    normal_vector = np.array([5, 4, 3])

    vector = field._VectorField__create_vector(position, normal_vector)

    unit_normal_vector = normal_vector / np.linalg.norm(normal_vector)

    start_position_expected = position + 0.1 * unit_normal_vector
    end_position_expected = start_position_expected + unit_normal_vector

    start_position_test = vector.get_start()
    end_position_test = vector.get_end()

    assert np.array_equal(start_position_test, start_position_expected)

    assert np.array_equal(end_position_test, end_position_expected)

    assert vector.shade_in_3d

    assert vector.get_tip().shade_in_3d


def test_vector_field_create_vector_opengl() -> None:
    manim.config.renderer = "opengl"
    # The parameterization does not matter since __create_vector does not
    # depend on the VectorField instance.
    cylinder = parameterizations.CylinderParameterization()

    vectorization = cylinder.vectorization
    normal_vectorization = cylinder.normal_vectorization
    u_range = cylinder.u_range
    v_range = cylinder.v_range

    field = vector_field.VectorField(
        vectorization, normal_vectorization, u_range, v_range
    )

    position = np.array([2, 3, 4])
    normal_vector = np.array([5, 4, 3])

    vector = field._VectorField__create_vector(position, normal_vector)

    unit_normal_vector = normal_vector / np.linalg.norm(normal_vector)

    start_position_expected = position + 0.1 * unit_normal_vector
    end_position_expected = start_position_expected + unit_normal_vector

    start_position_test = vector.get_start()
    end_position_test = vector.get_end()

    assert np.array_equal(start_position_test, start_position_expected)

    assert np.array_equal(end_position_test, end_position_expected)

    assert vector.depth_test

    assert vector.get_tip().depth_test
