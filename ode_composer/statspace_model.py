from sympy import *
from sympy.parsing.sympy_parser import parse_expr
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class State:
    rhs_arguments: List[Symbol]
    # TODO restrict this better, check sympy documentation
    rhs_fcn: object
    rhs_expression: Mul
    state_var: Mul


class StateSpaceModel(object):
    def __init__(self, states, parameters):
        self.state_vector: List[State] = list()
        self.parameters: Dict[str, float] = parameters
        self.add_state_equation(states)

    def __repr__(self):
        sss = [f"{s.state_var}={s.rhs_expression}" for s in self.state_vector]
        return str(sss)

    def _build_righthand_side(self, state, rhs_fcn, parameters):
        if "^" in rhs_fcn:
            rhs_fcn = rhs_fcn.replace("^", "**")
        sym_expr = parse_expr(s=rhs_fcn, evaluate=False, local_dict=parameters)
        expr_variables = list(sym_expr.free_symbols)
        sym_var = symbols(state)
        func = lambdify(args=expr_variables, expr=sym_expr, modules="numpy")
        return State(
            rhs_arguments=expr_variables,
            rhs_fcn=func,
            rhs_expression=sym_expr,
            state_var=sym_var,
        )

    def add_state_equation(self, states):
        for state_var, rhs_fcn in states.items():
            state = self._build_righthand_side(
                state=state_var, rhs_fcn=rhs_fcn, parameters=self.parameters
            )
            self.state_vector.append(state)

    def get_rhs(self, t, y, states, parameters):
        ret = [None] * len(y)
        if len(y) != len(states):
            raise ValueError(
                f" #states:{len(states)} must be equal to {len(y)}"
            )

        state_map = dict(zip(states, y))
        for idx, state in enumerate(self.state_vector):
            s_str = list(map(str, state.rhs_arguments))
            values = list(map(state_map.get, s_str))
            ret[idx] = state.rhs_fcn(*values)
        return ret
