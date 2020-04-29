import numpy as np
from sympy import *
from sympy.parsing.sympy_parser import parse_expr
from typing import List, Dict
from .util import MultiVariableFunction


class DictionaryBuilder(object):
    def __init__(self, dict_fcns: List[str]):
        self.dict_fcns: List[MultiVariableFunction] = list()
        self.regression_mtx = None
        for d_f in dict_fcns:
            self.add_dict_fcn(d_f)

    def add_dict_fcn(self, d_f):
        if "^" in d_f:
            d_f = d_f.replace("^", "**")
        sym_expr = parse_expr(s=d_f, evaluate=False)
        expr_variables = list(sym_expr.free_symbols)
        func = lambdify(args=expr_variables, expr=sym_expr, modules="numpy")
        dict_fcn = MultiVariableFunction(
            arguments=expr_variables,
            fcn_pointer=func,
            symbolic_expression=sym_expr,
        )
        self.dict_fcns.append(dict_fcn)

    def evaluate_dict(self, input_data: Dict):
        reg_mtx = [
            d_fcn.evaluate_function(measurement_data=input_data)
            for d_fcn in self.dict_fcns
        ]
        self.regression_mtx = np.transpose(np.vstack(reg_mtx))

        return self.regression_mtx
