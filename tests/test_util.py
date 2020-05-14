from sympy.parsing.sympy_parser import parse_expr
from sympy import lambdify
import numpy as np
from operator import itemgetter
import math
from ode_composer.util import MultiVariableFunction


def test_evaluate_multi_variable_function():
    # test the dictionary function evaluation
    d_f = "x1**2"
    sym_expr = parse_expr(s=d_f, evaluate=False)
    expr_variables = list(sym_expr.free_symbols)
    func = lambdify(args=expr_variables, expr=sym_expr, modules="numpy")
    multvar_fcn = MultiVariableFunction(
        arguments=expr_variables,
        fcn_pointer=func,
        symbolic_expression=sym_expr,
    )
    test_values = [1, 2, 3, 4]
    x1_values = np.array(test_values)
    A = multvar_fcn.fcn_pointer(x1_values)
    B = [sym_expr.subs("x1", k) for k in test_values]
    assert np.array_equal(A, B)

    # test the dictionary function argument order
    d_f = "x1*x2/x3"
    sym_expr = parse_expr(s=d_f, evaluate=False)
    expr_variables = list(sym_expr.free_symbols)
    func = lambdify(args=expr_variables, expr=sym_expr, modules="numpy")
    multvar_fcn = MultiVariableFunction(
        arguments=expr_variables,
        fcn_pointer=func,
        symbolic_expression=sym_expr,
    )
    data = {"x1": 2, "x2": 3, "x3": 5}
    A = (
        sym_expr.subs("x1", data["x1"])
        .subs("x2", data["x2"])
        .subs("x3", data["x3"])
    )

    str_arugment_list = [str(a) for a in multvar_fcn.arguments]
    values_list = itemgetter(*str_arugment_list)(data)
    B = multvar_fcn.fcn_pointer(*np.array(values_list))

    assert math.isclose(float(A), B, rel_tol=np.finfo(float).eps)
