"""Describes types used throughout the Gauss map package."""
from typing import Tuple, Union

from numpy import vectorize
from sympy import Expr, Matrix

Range = Tuple[float, float]

Matrices = Tuple[Matrix, Matrix]

# Expression representing x, y, and z coordinates
Expression = Union[Tuple[Expr, Expr, Expr], Matrix]

# Vectorized function representing x, y, and z coordinates
Vectorization = Tuple[vectorize, vectorize, vectorize]
