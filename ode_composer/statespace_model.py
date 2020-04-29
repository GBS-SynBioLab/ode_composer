from sympy import *
from sympy.parsing.sympy_parser import parse_expr
from typing import List, Dict
from .util import MultiVariableFunction


class StateSpaceModel(object):
    def __init__(self, states, parameters):
        self.state_vector: Dict[str, List[MultiVariableFunction]] = dict()
        self.parameters: Dict[str, float] = parameters
        self.add_state_equation(states, parameters)

    @classmethod
    def from_string(cls, states: Dict[str, str], parameters: Dict[str, float]):
        d = list()
        for rhs in states.values():
            d.append({rhs: 1})
        states_dict = dict(zip(states.keys(), d))
        return cls(states=states_dict, parameters=parameters)

    def __repr__(self):
        # TODO replace str.join()
        ss = ""
        for state_name, rhs in self.state_vector.items():
            ss += f"d{state_name}/dt = "
            for rr in rhs:
                ss += f"+{float(rr.constant):.2f}*{rr.symbolic_expression}"
            ss += "\n"
        return ss

    def _build_righthand_side(self, rhs_fcn, parameters, weight):
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

    def add_state_equation(self, states, parameters):
        for state_var, rhs_fcns in states.items():
            func_list = list()
            for rhs_fcn, weight in rhs_fcns.items():
                weight_numeric = parse_expr(
                    s=str(weight), evaluate=True, local_dict=parameters
                )
                multi_var_fcn = self._build_righthand_side(
                    rhs_fcn=rhs_fcn,
                    parameters=self.parameters,
                    weight=weight_numeric,
                )
                func_list.append(multi_var_fcn)
            self.state_vector[state_var] = func_list

    def get_rhs(self, t, y, states):
        ret = list()
        if len(y) != len(states):
            raise ValueError(
                f"#states:{len(states)} must be equal to {len(y)}"
            )

        state_map = dict(zip(states, y))
        for state, func_dict in self.state_vector.items():
            rhs_value = 0.0
            for multi_var_fcn in func_dict:
                rhs_value += (
                    multi_var_fcn.constant
                    * multi_var_fcn.evaluate_function(state_map)
                )
            ret.append(rhs_value)
        return ret
