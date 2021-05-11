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


def test_from_mak_generator():
    # test normal operation
    db = DictionaryBuilder.from_mak_generator(
        number_of_states=2, max_order=2, add_states=False, add_inputs=False
    )
    assert str(db) == " | ".join(["x1*x1", "x1*x2", "x2*x2"])

    db = DictionaryBuilder.from_mak_generator(
        number_of_states=2,
        max_order=2,
        number_of_inputs=1,
        add_states=False,
        add_inputs=False,
    )
    assert str(db) == " | ".join(
        [
            "x1*x1",
            "x1*x2",
            "u1*x1",
            "x2*x2",
            "u1*x2",
            "u1*u1",
        ]
    )

    # test negative arguments
    with pytest.raises(
        ValueError, match=r"Model has to have at least non-state"
    ):
        DictionaryBuilder.from_mak_generator(number_of_states=0, max_order=2)

    with pytest.raises(
        ValueError, match=r"Model has to have at least non-state"
    ):
        DictionaryBuilder.from_mak_generator(number_of_states=-5, max_order=2)

    with pytest.raises(
        ValueError, match=r"The max_order has to be at least one"
    ):
        DictionaryBuilder.from_mak_generator(number_of_states=2, max_order=0)

    with pytest.raises(
        ValueError, match=r"The max_order has to be at least one"
    ):
        DictionaryBuilder.from_mak_generator(number_of_states=2, max_order=-4)

    with pytest.raises(
        ValueError, match=r"The number of inputs cannot be negative"
    ):
        DictionaryBuilder.from_mak_generator(
            number_of_states=2, max_order=2, number_of_inputs=-1
        )


def test_dictionary_builder_addition():
    db = DictionaryBuilder.from_mak_generator(
        number_of_states=2, max_order=2, add_states=False, add_inputs=False
    )
    assert len(db) == 3
    db2 = DictionaryBuilder.from_mak_generator(
        number_of_states=1, max_order=2, add_states=False, add_inputs=False
    )
    assert len(db2) == 1

    new_dictionary = db + db2
    assert len(new_dictionary) == 4

    with pytest.raises(ValueError, match=r"Dictionary cannot be empty!"):
        db3 = db + DictionaryBuilder(dict_fcns=[])
        len(db3)


def test_positive_hill():
    db = DictionaryBuilder.from_positive_hill_generator(
        state_variable="x1", Km_range=[0, 1], cooperativity_range=[1, 2]
    )
    assert len(db) == 4
    # test positive behavior
    assert db.dict_fcns[0].symbolic_expression.subs("x1", 1) == pytest.approx(
        1
    )


def test_negative_hill():
    db = DictionaryBuilder.from_negative_hill_generator(
        state_variable="x1", Km_range=[0, 1], cooperativity_range=[1, 2]
    )
    assert len(db) == 4
    # test positive behavior
    assert db.dict_fcns[0].symbolic_expression.subs(
        "x1", 100
    ) == pytest.approx(1 / 100)
