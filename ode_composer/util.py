from sympy import *
from typing import List, Dict
from dataclasses import dataclass
import numpy as np
from sympy.parsing.sympy_parser import parse_expr


@dataclass
class MultiVariableFunction:
    arguments: List[Symbol]
    # TODO restrict this better, check sympy documentation
    fcn_pointer: object
    symbolic_expression: Mul
    constant_name: str = ""
    constant: float = 1.0

    def __repr__(self):
        return str(self.symbolic_expression)

    def evaluate_function(self, measurement_data: Dict):
        data = list()
        for key in self.arguments:
            key = str(key)
            if key not in measurement_data.keys():
                raise KeyError(
                    "Missing data for %s in expression %s"
                    % (key, self.symbolic_expression)
                )
            data.append(np.array(measurement_data.get(key)))
        return self.fcn_pointer(*data)

    @staticmethod
    def create_function(
        rhs_fcn: str, parameters: Dict[str, float], weight: float
    ):
        if "^" in rhs_fcn:
            rhs_fcn = rhs_fcn.replace("^", "**")
        sym_expr = parse_expr(s=rhs_fcn, evaluate=False, local_dict=parameters)
        expr_variables = list(sym_expr.free_symbols)
        func = lambdify(args=expr_variables, expr=sym_expr, modules="numpy")
        return MultiVariableFunction(
            arguments=expr_variables,
            fcn_pointer=func,
            symbolic_expression=sym_expr,
            constant=weight,
        )