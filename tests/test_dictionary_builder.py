from ode_composer.dictionary_builder import DictionaryBuilder
from sympy.parsing.sympy_parser import parse_expr
import pytest
import numpy as np


def test_dictionary_builder():
    d_f = ["x1", "x1*x2", "x2"]
    dict_builder = DictionaryBuilder(dict_fcns=d_f)
    assert len(d_f) == len(dict_builder.dict_fcns)


def test_add_dict_fcn():
    # test exponent conversion
    dict_builder = DictionaryBuilder(dict_fcns=[])
    d_f = "x1^2"
    dict_builder.add_dict_fcn(d_f=d_f)
    dict_fcn = dict_builder.dict_fcns.pop()
    d_f_replaced = d_f.replace("^", "**")
    assert str(dict_fcn.symbolic_expression) == d_f_replaced

    # test an undefined function
    d_f = ["f(x1)"]
    dict_builder = DictionaryBuilder(dict_fcns=d_f)
    data = {"x1": [5, 2]}
    with pytest.raises(NameError):
        dict_builder.evaluate_dict(input_data=data)


def test_evaluate_dict():
    d_f = ["x1*x2", "x1**2"]
    dict_builder = DictionaryBuilder(dict_fcns=d_f)
    data_x1 = [1, 5, 10]
    data_x2 = [3, 7, 15]
    data = {"x1": data_x1, "x2": data_x2}
    A = dict_builder.evaluate_dict(input_data=data)
    sym_expr_df1 = parse_expr(s=d_f[0], evaluate=False)
    df1_eval = [
        sym_expr_df1.evalf(subs=dict(x1=x1, x2=x2))
        for x1, x2 in zip(data_x1, data_x2)
    ]
    sym_expr_df2 = parse_expr(s=d_f[1], evaluate=False)
    df2_eval = [
        sym_expr_df2.evalf(subs=dict(x1=x1, x2=x2))
        for x1, x2 in zip(data_x1, data_x2)
    ]
    B = np.array([df1_eval, df2_eval]).T

    assert np.array_equal(A, B)
