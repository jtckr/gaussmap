"""Contains functions to validate parametric functions from user input."""
from typing import Tuple

import sympy as sym
from sympy.abc import u, v

from gaussmap.typing import Expression, Range


def _validate_bound(bound: str, bound_name: str) -> float:
    """Validate input and evaluates the expression for the boundaries.

    **Note** under the hood SymPy uses the eval
    function. This function should only be used locally.

    Args:
        bound: A string to be evaluated as a SymPy
            expression. The evaluated result is expected to be a
            real number which is then converted to a float value.
            Only numbers, operators, and the constants pi and exp
            are allowed in the string.
        bound_name: the name of the boundary to be used in
            error output i.e. u_min, u_max.

    Returns:
            The boundary expressed as a float value.

    Raises:
        ValueError: An unallowed character was found in the input
            string.
        NameError: An unallowed function or expression was found in
            the input string.
        SyntaxError: The expression could not be parsed by eval.
        SympifyError: The expression could not be parsed by SymPy.
    """
    allowed_chars = [*"pi", *"exp", *"0123456789", *".+-*/^()"]
    allowed_names = {"pi": sym.pi, "exp": sym.exp}

    if not all([char in allowed_chars for char in bound]):
        raise ValueError(f"Unallowed character found in {bound_name}")

    code = compile(bound, "<string>", "eval")
    for name in code.co_names:
        if name not in allowed_names:
            raise NameError(
                "Only pi and exp() are allowed in " f"{bound_name} got {name}"
            )

    bound_expr = sym.sympify(bound, {"__builtins__": {}}, allowed_names, evaluate=True)
    if not bound_expr.is_real:
        raise ValueError(f"{bound_name} must be a real value got " f"{bound_expr}")

    bound_float = min(max(float(bound_expr), -100.0), 100.0)

    if bound_float == 100.0 or bound_float == -100.0:
        print(f"{bound_name} too large, truncated to {bound_float}")

    return bound_float


def _validate_function(function: str, function_name: str) -> sym.Expr:
    """Validate input and evaluates the expression for the functions.

    **Note** under the hood SymPy uses the eval
    function. This function should only be used locally.

    Args:
        function: A string to be evaluated as a SymPy
            expression. The evaluated result is expected to be a
            parameterized function in terms of u and v.
            Only numbers, operators, and trigonometric, hyperbolic,
            and exponential functions are allowed in the string.
        function_name: The name of the function to be used in error
            output i.e. X.

    Returns:
            The function expressed as a SymPy expression.

    Raises:
        ValueError: An unallowed character was found in the input
            string.
        NameError: An unallowed function or expression was found in
            the input string.
        SympifyError: The expression could not be parsed by SymPy.
    """
    allowed_chars = [
        *"cos",
        *"sin",
        *"tan",
        *"csc",
        *"sec",
        *"cot",
        "h",
        *"exp",
        *"log",
        *"pi",
        "u",
        "v",
        *"0123456789",
        *".+-*/^()",
        " ",
    ]
    allowed_names = {
        "cos": sym.cos,
        "sin": sym.sin,
        "tan": sym.tan,
        "csc": sym.csc,
        "sec": sym.sec,
        "cot": sym.cot,
        "cosh": sym.cosh,
        "sinh": sym.sinh,
        "tanh": sym.tanh,
        "csch": sym.csch,
        "sech": sym.sech,
        "coth": sym.coth,
        "exp": sym.exp,
        "log": sym.log,
        "pi": sym.pi,
        "u": u,
        "v": v,
    }

    if not all([char in allowed_chars for char in function]):
        raise ValueError(f"Unallowed character found in {function_name}")

    code = compile(function, "<string>", "eval")
    for name in code.co_names:
        if name not in allowed_names:
            raise NameError(
                f"{name} not allowed. Only trigonometric, "
                "hyperbolic, exp, and log functions, "
                "variables u and v, and pi as a constant "
                f"are allowed in {function_name} got "
                f"{name}"
            )

    function_expr = sym.sympify(function, {"__builtins__": {}}, allowed_names)

    return function_expr


def get_function() -> Tuple[Expression, Range, Range]:
    """Prompt the user to input a parameterized equation.

    **Note** under the hood SymPy uses the eval
    function. This function should only be used locally.

    Returns:
        A tuple ((x, y, z), u_range, v_range, vector_amount) where
        (x, y, z) is a tuple of SymPy expressions describing the x, y, and
        z coordinates and u_range and v_range are tuples containing the
        starting and ending u and v coordinates respectively with
        vector_amount being the number of normal vectors to render.
    """
    print("Gauss map manimation generator: ")
    x_input = input("Enter x parameterization in terms of u and v: ")
    y_input = input("Enter y parameterization in terms of u and v: ")
    z_input = input("Enter z parameterization in terms of u and v: ")
    u_min_input = input("Enter minimum u value: ")
    u_max_input = input("Enter maximum u value: ")
    v_min_input = input("Enter minimum v value: ")
    v_max_input = input("Enter maximum v value: ")

    u_min = _validate_bound(u_min_input, "u_min")
    u_max = _validate_bound(u_max_input, "u_max")
    v_min = _validate_bound(v_min_input, "v_min")
    v_max = _validate_bound(v_max_input, "v_max")

    x = _validate_function(x_input, "x")
    y = _validate_function(y_input, "y")
    z = _validate_function(z_input, "z")

    u_range, v_range = ((u_min, u_max), (v_min, v_max))

    return (x, y, z), u_range, v_range
