"""Contains classes describing various common parameterizations."""
from dataclasses import dataclass
from typing import List

import numpy as np
import sympy as sym
from sympy.abc import u, v

from gaussmap.typing import Expression, Range, Vectorization


@dataclass(frozen=True)
class Parameterization:
    """A particular parameterization of a surface."""

    expression: Expression
    vectorization: Vectorization
    partial_u_expression: Expression
    partial_v_expression: Expression
    normal_expression: Expression
    normal_vectorization: Vectorization
    u_range: Range
    v_range: Range
    is_gauss_map_1d: bool
    is_gauss_map_inward: bool


@dataclass(frozen=True)
class CatenoidParameterization(Parameterization):
    """A parameterization of a catenoid centered at the origin."""

    def _x(u: float, v: float) -> float:
        return 2 * np.cosh(0.5 * v) * np.cos(u)

    def _y(u: float, v: float) -> float:
        return 2 * np.cosh(0.5 * v) * np.sin(u)

    def _z(u: float, v: float) -> float:
        return v

    def _normal_x(u: float, v: float) -> float:
        return 2 * np.cos(u) * np.cosh(0.5 * v)

    def _normal_y(u: float, v: float) -> float:
        return 2 * np.sin(u) * np.cosh(0.5 * v)

    def _normal_z(u: float, v: float) -> float:
        return -1 * np.sinh(v)

    expression: Expression = (
        2 * sym.cosh(0.5 * v) * sym.cos(u),
        2 * sym.cosh(0.5 * v) * sym.sin(u),
        v,
    )
    vectorization: Vectorization = (
        np.vectorize(_x),
        np.vectorize(_y),
        np.vectorize(_z),
    )
    partial_u_expression: Expression = (
        -2 * sym.cosh(0.5 * v) * sym.sin(u),
        2 * sym.cosh(0.5 * v) * sym.cos(u),
        sym.sympify(0),
    )
    partial_v_expression: Expression = (
        sym.sinh(0.5 * v) * sym.cos(u),
        sym.sinh(0.5 * v) * sym.sin(u),
        sym.sympify(1),
    )
    normal_expression: Expression = (
        2 * sym.cos(u) * sym.cosh(0.5 * v),
        2 * sym.sin(u) * sym.cosh(0.5 * v),
        -sym.sinh(v),
    )
    normal_vectorization: Vectorization = (
        np.vectorize(_normal_x),
        np.vectorize(_normal_y),
        np.vectorize(_normal_z),
    )
    u_range: Range = (-np.pi, np.pi)
    v_range: Range = (-2, 2)
    is_gauss_map_1d: bool = False
    is_gauss_map_inward: bool = False


@dataclass(frozen=True)
class ConeParameterization(Parameterization):
    """A parameterization of an upper half cone centered along the z-axis."""

    def _x(u: float, v: float) -> float:
        return v * np.cos(u)

    def _y(u: float, v: float) -> float:
        return v * np.sin(u)

    def _z(u: float, v: float) -> float:
        return v

    def _normal_x(u: float, v: float) -> float:
        return v * np.cos(u)

    def _normal_y(u: float, v: float) -> float:
        return v * np.sin(u)

    def _normal_z(u: float, v: float) -> float:
        return v

    expression: Expression = (v * sym.cos(u), v * sym.sin(u), v)
    vectorization: Vectorization = (
        np.vectorize(_x),
        np.vectorize(_y),
        np.vectorize(_z),
    )
    partial_u_expression: Expression = (-v * sym.sin(u), v * sym.cos(u), sym.sympify(0))
    partial_v_expression: Expression = (sym.cos(u), sym.sin(u), sym.sympify(1))
    normal_expression: Expression = (v * sym.cos(u), v * sym.sin(u), -v)
    normal_vectorization: Vectorization = (
        np.vectorize(_normal_x),
        np.vectorize(_normal_y),
        np.vectorize(_normal_z),
    )
    u_range: Range = (0, 2 * np.pi)
    # 0.01 is used to avoid singular point.
    v_range: Range = (0.01, 1)
    is_gauss_map_1d: bool = True
    is_gauss_map_inward: bool = False


@dataclass(frozen=True)
class CylinderParameterization(Parameterization):
    """A parameterization of a cylinder centered along the z-axis."""

    def _x(u: float, v: float) -> float:
        return np.cos(u)

    def _y(u: float, v: float) -> float:
        return np.sin(u)

    def _z(u: float, v: float) -> float:
        return v

    def _normal_x(u: float, v: float) -> float:
        return np.cos(u)

    def _normal_y(u: float, v: float) -> float:
        return np.sin(u)

    def _normal_z(u: float, v: float) -> float:
        return 0

    expression: Expression = (sym.cos(u), sym.sin(u), v)
    vectorization: Vectorization = (
        np.vectorize(_x),
        np.vectorize(_y),
        np.vectorize(_z),
    )
    partial_u_expression: Expression = (-sym.sin(u), sym.cos(u), sym.sympify(0))
    partial_v_expression: Expression = (sym.sympify(0), sym.sympify(0), sym.sympify(1))
    normal_expression: Expression = (sym.cos(u), sym.sin(u), sym.sympify(0))
    normal_vectorization: Vectorization = (
        np.vectorize(_normal_x),
        np.vectorize(_normal_y),
        np.vectorize(_normal_z),
    )
    u_range: Range = (0, 2 * np.pi)
    v_range: Range = (-1, 1)
    is_gauss_map_1d: bool = True
    is_gauss_map_inward: bool = False


@dataclass(frozen=True)
class HyperbolicParaboloidParameterization(Parameterization):
    """A parameterization of a hyperbolic paraboloid centered at the origin."""

    def _x(u: float, v: float) -> float:
        return u

    def _y(u: float, v: float) -> float:
        return v

    def _z(u: float, v: float) -> float:
        return u * v

    def _normal_x(u: float, v: float) -> float:
        return -1 * np.cos(u) * np.sin(v) ** 2

    def _normal_y(u: float, v: float) -> float:
        return -1 * np.sin(u) * np.sin(v) ** 2

    def _normal_z(u: float, v: float) -> float:
        return -0.5 * np.sin(2 * v)

    expression: Expression = (u, v, u * v)
    vectorization: Vectorization = (
        np.vectorize(_x),
        np.vectorize(_y),
        np.vectorize(_z),
    )
    partial_u_expression: Expression = (sym.sympify(1), sym.sympify(0), v)
    partial_v_expression: Expression = (sym.sympify(0), sym.sympify(1), u)
    normal_expression: Expression = (
        -sym.cos(u) * sym.sin(v) ** 2,
        -sym.sin(u) * sym.sin(v) ** 2,
        -0.5 * sym.sin(2 * v),
    )
    normal_vectorization: Vectorization = (
        np.vectorize(_normal_x),
        np.vectorize(_normal_y),
        np.vectorize(_normal_z),
    )
    u_range: Range = (-2, 2)
    v_range: Range = (-2, 2)
    is_gauss_map_1d: bool = False
    is_gauss_map_inward: bool = False


@dataclass(frozen=True)
class HyperboloidParameterization(Parameterization):
    """A parameterization of a hyperboloid centered along the z-axis."""

    def _x(u: float, v: float) -> float:
        return np.cosh(u) * np.cos(v)

    def _y(u: float, v: float) -> float:
        return np.cosh(u) * np.sin(v)

    def _z(u: float, v: float) -> float:
        return np.sinh(u)

    def _normal_x(u: float, v: float) -> float:
        return -1 * np.cos(v) * np.cosh(u) ** 2

    def _normal_y(u: float, v: float) -> float:
        return -1 * np.sin(v) * np.cosh(u) ** 2

    def _normal_z(u: float, v: float) -> float:
        return 0.5 * np.sinh(2 * u)

    expression: Expression = (
        sym.cosh(u) * sym.cos(v),
        sym.cosh(u) * sym.sin(v),
        sym.sinh(u),
    )
    vectorization: Vectorization = (
        np.vectorize(_x),
        np.vectorize(_y),
        np.vectorize(_z),
    )
    partial_u_expression: Expression = (
        sym.sinh(u) * sym.cos(v),
        sym.sinh(u) * sym.sin(v),
        sym.cosh(u),
    )
    partial_v_expression: Expression = (
        -sym.cosh(u) * sym.sin(v),
        sym.cosh(u) * sym.cos(v),
        sym.sympify(0),
    )
    normal_expression: Expression = (
        -sym.cos(v) * sym.cosh(u) ** 2,
        -sym.sin(v) * sym.cosh(u) ** 2,
        0.5 * sym.sinh(2 * u),
    )
    normal_vectorization: Vectorization = (
        np.vectorize(_normal_x),
        np.vectorize(_normal_y),
        np.vectorize(_normal_z),
    )
    u_range: Range = (-2 * np.pi, 2 * np.pi)
    v_range: Range = (0, 2 * np.pi)
    is_gauss_map_1d: bool = False
    is_gauss_map_inward: bool = True


@dataclass(frozen=True)
class MonkeySaddleParameterization(Parameterization):
    """A parameterization of a monkey saddle centered at the origin."""

    def _x(u: float, v: float) -> float:
        return u

    def _y(u: float, v: float) -> float:
        return v

    def _z(u: float, v: float) -> float:
        return u**3 - 3 * u * v**2

    def _normal_x(u: float, v: float) -> float:
        return -3 * u**2 + 3 * v**2

    def _normal_y(u: float, v: float) -> float:
        return 6 * u * v

    def _normal_z(u: float, v: float) -> float:
        return 1

    expression: Expression = (u, v, u**3 - 3 * u * v**2)
    vectorization: Vectorization = (
        np.vectorize(_x),
        np.vectorize(_y),
        np.vectorize(_z),
    )
    partial_u_expression: Expression = (
        sym.sympify(1),
        sym.sympify(0),
        3 * u**2 - 3 * v**2,
    )
    partial_v_expression: Expression = (sym.sympify(0), sym.sympify(1), -6 * u * v)
    normal_expression: Expression = (
        -3 * u**2 + 2 * v**2,
        4 * u * v,
        sym.sympify(1),
    )
    normal_vectorization: Vectorization = (
        np.vectorize(_normal_x),
        np.vectorize(_normal_y),
        np.vectorize(_normal_z),
    )
    u_range: Range = (-3, 3)
    v_range: Range = (-3, 3)
    is_gauss_map_1d: bool = False
    is_gauss_map_inward: bool = False


@dataclass(frozen=True)
class ParaboloidParameterization(Parameterization):
    """A parameterization of a paraboloid opening towards negative z."""

    def _x(u: float, v: float) -> float:
        return v * np.cos(u)

    def _y(u: float, v: float) -> float:
        return v * np.sin(u)

    def _z(u: float, v: float) -> float:
        return -1 * v**2

    def _normal_x(u: float, v: float) -> float:
        return -2 * v**2 * np.cos(u)

    def _normal_y(u: float, v: float) -> float:
        return -2 * v**2 * np.sin(u)

    def _normal_z(u: float, v: float) -> float:
        return -1 * v

    expression: Expression = (v * sym.cos(u), v * sym.sin(u), -(v**2))
    vectorization: Vectorization = (
        np.vectorize(_x),
        np.vectorize(_y),
        np.vectorize(_z),
    )
    partial_u_expression: Expression = (-v * sym.sin(u), v * sym.cos(u), sym.sympify(0))
    partial_v_expression: Expression = (sym.cos(u), sym.sin(u), -2 * v)
    normal_expression: Expression = (
        -2 * v**2 * sym.cos(u),
        -2 * v**2 * sym.sin(u),
        -v,
    )
    normal_vectorization: Vectorization = (
        np.vectorize(_normal_x),
        np.vectorize(_normal_y),
        np.vectorize(_normal_z),
    )
    u_range: Range = (0, 2 * np.pi)
    # 0.01 is used to avoid singular point.
    v_range: Range = (0.01, 2)
    is_gauss_map_1d: bool = False
    is_gauss_map_inward: bool = True


@dataclass(frozen=True)
class RingTorusParameterization(Parameterization):
    """A parameterization of a torus with major radius 3 and minor radius 1."""

    def _x(u: float, v: float) -> float:
        return (3 + np.cos(u)) * np.cos(v)

    def _y(u: float, v: float) -> float:
        return (3 + np.cos(u)) * np.sin(v)

    def _z(u: float, v: float) -> float:
        return np.sin(u)

    def _normal_x(u: float, v: float) -> float:
        return -1 * (3 + np.cos(u)) * np.cos(u) * np.cos(v)

    def _normal_y(u: float, v: float) -> float:
        return -1 * (3 + np.cos(u)) * np.cos(u) * np.sin(v)

    def _normal_z(u: float, v: float) -> float:
        return -1 * (3 + np.cos(u)) * np.sin(u)

    expression: Expression = (
        (3 + sym.cos(u)) * sym.cos(v),
        (3 + sym.cos(u)) * sym.sin(v),
        sym.sin(u),
    )
    vectorization: Vectorization = (
        np.vectorize(_x),
        np.vectorize(_y),
        np.vectorize(_z),
    )
    partial_u_expression: Expression = (
        -sym.sin(u) * sym.cos(v),
        -sym.sin(u) * sym.sin(v),
        sym.cos(u),
    )
    partial_v_expression: Expression = (
        -(3 + sym.cos(u)) * sym.sin(v),
        (3 + sym.cos(u)) * sym.cos(v),
        sym.sympify(0),
    )
    normal_expression: Expression = (
        -(3 + sym.cos(u)) * sym.cos(u) * sym.cos(v),
        -(3 + sym.cos(u)) * sym.cos(u) * sym.sin(v),
        -(3 + sym.cos(u)) * sym.sin(u),
    )
    normal_vectorization: Vectorization = (
        np.vectorize(_normal_x),
        np.vectorize(_normal_y),
        np.vectorize(_normal_z),
    )
    u_range: Range = (0, 2 * np.pi)
    v_range: Range = (0, 2 * np.pi)
    is_gauss_map_1d: bool = False
    is_gauss_map_inward: bool = True


@dataclass(frozen=True)
class SphereParameterization(Parameterization):
    """A parameterization of a sphere with radius 1 centered at the origin."""

    def _x(u: float, v: float) -> float:
        return np.cos(u) * np.sin(v)

    def _y(u: float, v: float) -> float:
        return np.sin(u) * np.sin(v)

    def _z(u: float, v: float) -> float:
        return np.cos(v)

    def _normal_x(u: float, v: float) -> float:
        return -1 * np.sin(v) ** 2 * np.cos(u)

    def _normal_y(u: float, v: float) -> float:
        return -1 * np.sin(v) ** 2 * np.sin(u)

    def _normal_z(u: float, v: float) -> float:
        return -1 * np.cos(v) * np.sin(v)

    expression: Expression = (
        sym.cos(u) * sym.sin(v),
        sym.sin(u) * sym.sin(v),
        sym.cos(v),
    )
    vectorization: Vectorization = (
        np.vectorize(_x),
        np.vectorize(_y),
        np.vectorize(_z),
    )
    partial_u_expression: Expression = (
        -sym.sin(u) * sym.sin(v),
        sym.cos(u) * sym.sin(v),
        sym.sympify(0),
    )
    partial_v_expression: Expression = (
        sym.cos(u) * sym.cos(v),
        sym.sin(u) * sym.cos(v),
        -sym.sin(v),
    )
    normal_expression: Expression = (
        -sym.sin(v) ** 2 * sym.cos(u),
        -sym.sin(v) ** 2 * sym.sin(u),
        -sym.cos(v) * sym.sin(v),
    )
    normal_vectorization: Vectorization = (
        np.vectorize(_normal_x),
        np.vectorize(_normal_y),
        np.vectorize(_normal_z),
    )
    u_range: Range = (0, 2 * np.pi)
    # 0.01 is used to avoid the two singular points at the poles.
    v_range: Range = (0.01, np.pi - 0.01)
    is_gauss_map_1d: bool = False
    is_gauss_map_inward: bool = True


parameterization_list: List[Parameterization] = [
    CatenoidParameterization(),
    ConeParameterization(),
    CylinderParameterization(),
    HyperbolicParaboloidParameterization(),
    HyperboloidParameterization(),
    MonkeySaddleParameterization(),
    ParaboloidParameterization(),
    RingTorusParameterization(),
    SphereParameterization(),
]
