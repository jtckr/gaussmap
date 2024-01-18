"""Contains useful functions that are used throughout the Gauss map project."""
import numpy as np
import numpy.typing as npt
import sympy as sym
from sympy.abc import u, v

from gaussmap.typing import Expression, Matrices, Range, Vectorization


def is_u_function(vectorization: Vectorization, u_range: Range, v_range: Range) -> bool:
    """Determine whether a vectorized function is a function of u.

    Args:
        vectorization: A tuple of vectorized functions.
        u_range: A tuple containing the starting and ending u
        coordinates.
        v_range: A tuple containing the starting and ending v
        coordinates.

    Returns:
        A boolean that is true when the vectorization is a function of u
        and false when the vectorization is not a function of u.
    """
    u_values = np.linspace(u_range[0], u_range[1])
    v_value = (v_range[1] - v_range[0]) / 2

    evaluations = [evaluate(vectorization, u_value, v_value) for u_value in u_values]

    # Does changing u give different outputs?
    return not np.all(
        [np.allclose(evaluation, evaluations[0]) for evaluation in evaluations]
    )


def is_v_function(vectorization: Vectorization, u_range: Range, v_range: Range) -> bool:
    """Determine whether a vectorized function is a function of v.

    Args:
        vectorization: A tuple of vectorized functions.
        u_range: A tuple containing the starting and ending u
        coordinates.
        v_range: A tuple containing the starting and ending v
        coordinates.

    Returns:
        A boolean that is true when the vectorization is a function of v
        and false when the vectorization is not a function of v.
    """
    u_value = (u_range[1] - u_range[0]) / 2
    v_values = np.linspace(v_range[0], v_range[1])

    evaluations = [evaluate(vectorization, u_value, v_value) for v_value in v_values]

    # Does changing v give different outputs?
    return not np.all(
        [np.allclose(evaluation, evaluations[0]) for evaluation in evaluations]
    )


def is_inward_field(
    vectorization: Vectorization,
    normal_vectorization: Vectorization,
    u_range: Range,
    v_range: Range,
    u_amount: int = 20,
    v_amount: int = 20,
) -> bool:
    """Determine whether a normal vector field is oriented inward.

    Args:
        vectorization: A tuple of vectorized functions representing
            points on the original surface.
        normal_vectorization: A tuple of vectorized functions representing
            normal vectors of the surface.
        u_range: A tuple containing the starting and ending u coordinates.
        v_range: A tuple containing the starting and ending v coordinates.

    Returns:
        A boolean that is true when the field is oriented inward and false
        when it is oriented outwards.
    """
    u_values = np.linspace(u_range[0], u_range[1], u_amount)
    v_values = np.linspace(v_range[0], v_range[1], v_amount)

    is_pointing_inward_array = np.zeros(shape=(u_amount, v_amount))
    is_pointing_inward_negative_array = np.zeros(shape=(u_amount, v_amount))
    for i, u_value in enumerate(u_values):
        for j, v_value in enumerate(v_values):
            evaluation = evaluate(vectorization, u_value, v_value)
            normal_evaluation = evaluate(normal_vectorization, u_value, v_value)
            negative_normal_evaluation = -1 * normal_evaluation
            is_pointing_inward_array[i, j] = is_pointing_inward(
                evaluation, normal_evaluation
            )
            is_pointing_inward_negative_array[i, j] = is_pointing_inward(
                evaluation, negative_normal_evaluation
            )

    amount_inward = np.count_nonzero(is_pointing_inward_array)
    amount_inward_negative = np.count_nonzero(is_pointing_inward_negative_array)

    return amount_inward > amount_inward_negative


def is_pointing_inward(
    point: npt.NDArray[np.float64], vector: npt.NDArray[np.float64]
) -> bool:
    """Determine whether a normal vector is pointing towards the origin.

    Args:
        point: A NumPy array of the coordinates of a point.
        vector: A NumPy array of the components of the normal vector at
            that point.

    Returns:
        A boolean that is true when the vector is pointing inwards and
        false when it is pointing outwards.
    """
    norm = np.linalg.norm(point)
    vector = normalize(vector)

    # Does following the vector get us closer to the origin?
    new_point = point + 0.1 * norm * vector

    return np.linalg.norm(new_point) < norm


def evaluate(
    vectorization: Vectorization, u_value: float, v_value: float
) -> npt.NDArray[np.float64]:
    """Evaluate the vectorized function at the given coordinates.

    Args:
        V: A tuple of vectorized functions to be evaluated.
        u_i: The u coordinate to evaluate at.
        v_i: The v coordinate to evaluate at.

    Returns:
        A NumPy array of the evaluated values in x, y, and z coordinates.
    """
    x, y, z = vectorization

    return np.array([x(u_value, v_value), y(u_value, v_value), z(u_value, v_value)])


def sympy_to_numpy(expression: Expression) -> Vectorization:
    """Convert SymPy expressions to NumPy expressions.

    Args:
        expression: A sequence of 3 SymPy expressions to convert.

    Returns:
        A Tuple of NumPy vectorized equations.
    """
    # iter needed for SymPy matrix type checking
    x, y, z = iter(expression)
    x_numpy = sym.lambdify([u, v], x, "numpy")
    y_numpy = sym.lambdify([u, v], y, "numpy")
    z_numpy = sym.lambdify([u, v], z, "numpy")
    x_vectorization = np.vectorize(x_numpy)
    y_vectorization = np.vectorize(y_numpy)
    z_vectorization = np.vectorize(z_numpy)

    return x_vectorization, y_vectorization, z_vectorization


def normalize(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Divide a vector by its magnitude."""
    norm = np.linalg.norm(vector)

    if norm == 0:
        norm = np.float64(1.0)

    return np.divide(vector, norm)


def compute_partials(expression: Expression) -> Matrices:
    """Calculate the partial derivatives of the expression.

    Args:
        expression: A parametrized equation in terms of u and v where the
            elements describe the x, y, and z coordinates in that order.

    Returns:
        A tuple (partial_u, partial_v) where partial_u is the parameterized
        equation of the paritial derivative of the expression with respect to u
        in terms of u and v, and partial_v is the parameterized equations of
        the partial derivative of the expression with respect to v in terms of
        u and v.
    """
    # iter needed for SymPy Matrix type checking
    x, y, z = iter(expression)
    x_partial_u = sym.diff(x, u)
    x_partial_v = sym.diff(x, v)
    y_partial_u = sym.diff(y, u)
    y_partial_v = sym.diff(y, v)
    z_partial_u = sym.diff(z, u)
    z_partial_v = sym.diff(z, v)
    partial_u_expression = sym.Matrix([x_partial_u, y_partial_u, z_partial_u])
    partial_v_expression = sym.Matrix([x_partial_v, y_partial_v, z_partial_v])

    return partial_u_expression, partial_v_expression
