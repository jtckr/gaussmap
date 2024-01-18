import numpy as np
import pytest
import sympy as sym
from sympy import (
    cos,
    cosh,
    cot,
    coth,
    csc,
    csch,
    exp,
    log,
    pi,
    sec,
    sech,
    sin,
    sinh,
    tan,
    tanh,
)
from sympy.abc import u, v

from gaussmap import parameterizations
from gaussmap.utils import parameterization_input


def test_validate_bounds_allowed() -> None:
    bound = parameterization_input._validate_bound(bound="-pi", bound_name="u_min")
    assert np.isclose(bound, -1 * np.pi)

    bound = parameterization_input._validate_bound(bound="pi", bound_name="u_max")
    assert np.isclose(bound, np.pi)

    bound = parameterization_input._validate_bound(bound="-exp(1)", bound_name="u_min")
    assert np.isclose(bound, -np.exp(1))

    bound = parameterization_input._validate_bound(bound="exp(1)", bound_name="u_max")
    assert np.isclose(bound, np.exp(1))

    bound = parameterization_input._validate_bound(bound="-99", bound_name="v_min")
    assert np.isclose(bound, -99)

    bound = parameterization_input._validate_bound(bound="99", bound_name="v_max")
    assert np.isclose(bound, 99)

    bound = parameterization_input._validate_bound(bound="-100", bound_name="u_min")
    assert np.isclose(bound, -100)

    bound = parameterization_input._validate_bound(bound="100", bound_name="u_max")
    assert np.isclose(bound, 100)

    bound = parameterization_input._validate_bound(bound="0", bound_name="v_min")
    assert np.isclose(bound, 0)

    bound = parameterization_input._validate_bound(bound="0", bound_name="v_max")
    assert np.isclose(bound, 0)


def test_validate_bounds_unallowed_chars() -> None:
    with pytest.raises(ValueError):
        # https://xkcd.com/327/
        parameterization_input._validate_bound(
            bound="Robert'); DROP TABLE Students;--", bound_name="u_min"
        )


def test_validate_bounds_unallowed_name() -> None:
    with pytest.raises(NameError):
        # Function call needs to only use valid characters to not raise
        # a ValueError. 'pixie' is the only word I could think of that
        # only uses characters from pi or exp.
        parameterization_input._validate_bound(bound="pixie(pi)", bound_name="u_max")


def test_validate_bounds_too_large_negative() -> None:
    bound = parameterization_input._validate_bound(bound="-101", bound_name="v_min")
    assert np.isclose(bound, -100)

    bound = parameterization_input._validate_bound(bound="-101", bound_name="v_max")
    assert np.isclose(bound, -100)
    bound = parameterization_input._validate_bound(bound="-200", bound_name="v_min")
    assert np.isclose(bound, -100)

    bound = parameterization_input._validate_bound(bound="-200", bound_name="v_max")
    assert np.isclose(bound, -100)


def test_validate_bounds_too_large() -> None:
    bound = parameterization_input._validate_bound(bound="101", bound_name="v_min")
    assert np.isclose(bound, 100)

    bound = parameterization_input._validate_bound(bound="101", bound_name="v_max")
    assert np.isclose(bound, 100)
    bound = parameterization_input._validate_bound(bound="200", bound_name="v_min")
    assert np.isclose(bound, 100)

    bound = parameterization_input._validate_bound(bound="200", bound_name="v_max")
    assert np.isclose(bound, 100)


def test_validate_bounds_real() -> None:
    with pytest.raises(ValueError):
        parameterization_input._validate_bound(bound="(-1)^(0.5)", bound_name="v_min")


def test_validate_function_allowed() -> None:
    function = parameterization_input._validate_function(
        function="cos(u)*sin(v)", function_name="x"
    )
    assert function == cos(u) * sin(v)

    function = parameterization_input._validate_function(
        function="tan(pi*u)**2", function_name="y"
    )
    assert function == tan(pi * u) ** 2
    function = parameterization_input._validate_function(
        function="exp(sinh(cosh(v) + u**2) / 1)", function_name="z"
    )
    assert function == exp(sinh(cosh(v) + u**2) / 1)
    function = parameterization_input._validate_function(
        function="tan(tanh(u))*csc(csch(v)) + " " cot(coth(u))*sec(sech(v))",
        function_name="f",
    )
    assert function == tan(tanh(u)) * csc(csch(v)) + cot(coth(u)) * sec(sech(v))
    function = parameterization_input._validate_function(
        function="log(12345)*exp(67890)", function_name="g"
    )
    assert function == log(12345) * exp(67890)


def test_validate_function_unallowed_chars() -> None:
    with pytest.raises(ValueError):
        # https://xkcd.com/327/
        parameterization_input._validate_function(
            function="Robert'); DROP TABLE Students;--", function_name="x"
        )


def test_validate_function_unallowed_names() -> None:
    with pytest.raises(NameError):
        parameterization_input._validate_function(
            function="acsc(asin(u)*v)", function_name="y"
        )

    with pytest.raises(NameError):
        parameterization_input._validate_function(
            function="exculpation(constantine)", function_name="crime"
        )


def test_validate_function_syntax_error() -> None:
    with pytest.raises(SyntaxError):
        parameterization_input._validate_function(
            function="cos)x(**2 + sin)x(**2", function_name="cursed"
        )

    with pytest.raises(SyntaxError):
        parameterization_input._validate_function(function="", function_name="empty")


def test_get_function_valid() -> None:
    monkey_patch = pytest.MonkeyPatch()
    inputs = ["cos(u)", "sin(u)", "v", "0", "2*pi", "-1", "1"]
    monkey_patch.setattr("builtins.input", lambda _: inputs.pop(0))

    cylinder = parameterizations.CylinderParameterization()

    expression_expected = sym.Matrix(cylinder.expression)
    u_range_expected = cylinder.u_range
    v_range_expected = cylinder.v_range

    expression_test, u_range_test, v_range_test = parameterization_input.get_function()

    expression_test = sym.Matrix(expression_test)

    assert expression_test.equals(expression_expected)
    assert u_range_test[0] == u_range_expected[0]
    assert u_range_test[1] == u_range_expected[1]
    assert v_range_test[0] == v_range_expected[0]
    assert v_range_test[1] == v_range_expected[1]


def test_get_function_invalid() -> None:
    monkey_patch = pytest.MonkeyPatch()
    inputs = [
        "cos)x(**2 + sin)x(**2",
        "asin(u)",
        "test",
        "pixie(pi)",
        "test",
        "-101",
        "101",
    ]
    monkey_patch.setattr("builtins.input", lambda _: inputs.pop(0))

    with pytest.raises(NameError):
        parameterization_input.get_function()
