import numpy as np
from sympy import *
from sympy.parsing.sympy_parser import parse_expr
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class DictionaryFunction:
    arguments: List[Symbol]
    # TODO restrict this better, check sympy documentation
    function: object
    expression: Mul


class DictionaryBuilder(object):
    def __init__(self, dict_fcns: List[str]):
        self.dict_fcns: List[DictionaryFunction] = list()
        self.regression_mtx = None
        for d_f in dict_fcns:
            self.add_dict_fcn(d_f)

    def add_dict_fcn(self, d_f):
        if "^" in d_f:
            d_f = d_f.replace("^", "**")
        sym_expr = parse_expr(s=d_f, evaluate=False)
        expr_variables = list(sym_expr.free_symbols)
        func = lambdify(args=expr_variables, expr=sym_expr, modules="numpy")
        dict_fcn = DictionaryFunction(
            arguments=expr_variables, function=func, expression=sym_expr
        )
        self.dict_fcns.append(dict_fcn)

    def evaluate_dict(self, measurement_data: Dict):
        reg_mtx = list()
        for d_fcn in self.dict_fcns:
            data = list()
            for key in d_fcn.arguments:
                key = str(key)
                if key not in measurement_data.keys():
                    raise KeyError(
                        "Missing data for %s in expression %s"
                        % (key, d_fcn.expression)
                    )
                data.append(np.array(measurement_data.get(key)))
            reg_mtx.append(d_fcn.function(*data))
        self.regression_mtx = np.transpose(np.vstack(reg_mtx))

    def get_regression_matrix(self):
        return self.regression_mtx
