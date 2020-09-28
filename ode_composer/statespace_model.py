from sympy import *
from typing import List, Dict
from .util import MultiVariableFunction
from sympy.parsing.sympy_parser import parse_expr
from sympy import Matrix
from .signal_preprocessor import SignalPreprocessor
import copy
import re


class StateSpaceModel(object):
    def __init__(
        self,
        states: Dict[str, Dict[str, str]],
        parameters: Dict[str, float],
        inputs: List[str] = None,
    ):
        """

        Args:
            states:
            parameters:
        """
        self.state_vector: Dict[str, List[MultiVariableFunction]] = dict()
        self.inputs = inputs
        # TODO add handling for parameter dict merging
        self.parameters: Dict[str, float] = parameters
        if states is not None:
            self.add_state_equation(states, parameters)

    @classmethod
    def from_string(cls, states: Dict[str, str], parameters: Dict[str, float]):
        """

        Args:
            states:
            parameters:

        Returns:
            an instance of StateSpaceModel

        """
        d = list()
        for rhs in states.values():
            d.append({rhs: 1})
        states_dict = dict(zip(states.keys(), d))
        return cls(states=states_dict, parameters=parameters)

    @classmethod
    def from_sbl(cls, states_dict, parameters, inputs=None):
        ss_object = cls(states=None, parameters=parameters, inputs=inputs)
        ss_object.state_vector = states_dict
        return ss_object

    def __repr__(self):
        # TODO replace str.join()
        ss = ""
        for state_name, rhs in self.state_vector.items():
            ss += f"d{state_name}/dt = "
            for rr in rhs:
                ss += f"+{float(rr.constant):.2f}*{rr.symbolic_expression}"
            ss += "\n"
        return ss

    def add_state_equation(
        self, states: Dict[str, Dict[str, str]], parameters: Dict[str, float]
    ):
        for state_var, rhs_fcns in states.items():
            func_list = list()
            for rhs_fcn, weight in rhs_fcns.items():
                weight_numeric = parse_expr(
                    s=str(weight), evaluate=True, local_dict=parameters
                )
                multi_var_fcn = MultiVariableFunction.create_function(
                    rhs_fcn=rhs_fcn,
                    parameters=self.parameters,
                    weight=weight_numeric,
                )
                func_list.append(multi_var_fcn)
            # TODO add check for existing keys!
            self.state_vector[state_var] = func_list

    def get_rhs(self, t, y, states, inputs=None):
        ret = list()
        if len(y) != len(states):
            raise ValueError(
                f"#states:{len(states)} must be equal to {len(y)}"
            )
        # TODO Check that states are valid, but it must be fast!
        state_map = dict(zip(states, y))
        if inputs:
            # inputs are continuous functions
            processed_inputs = copy.deepcopy(inputs)
            if all(
                isinstance(u, SignalPreprocessor)
                for u in processed_inputs.values()
            ):
                for key, u in processed_inputs.items():
                    processed_inputs[key] = u.interpolate(t)
            else:
                raise NotImplementedError(
                    "only interpolated input is implemented"
                )

            state_map.update(processed_inputs)
        for state, func_dict in self.state_vector.items():
            rhs_value = 0.0
            for multi_var_fcn in func_dict:
                rhs_value += (
                    multi_var_fcn.constant
                    * multi_var_fcn.evaluate_function(state_map)
                )
            ret.append(rhs_value)
        return ret

    def compute_jacobian(self):
        # sympy does not work with chain
        state_names = list(self.state_vector.keys())
        if len(state_names) == 0:
            raise ValueError(
                "At least one state is required for Jacobian computation"
            )
        full_rhs = []
        for state, rhs in self.state_vector.items():
            full_rhs.append(
                Add(
                    *[
                        c_rhs.constant * c_rhs.symbolic_expression
                        for c_rhs in rhs
                    ]
                )
            )
        X = Matrix(full_rhs)
        jacobian_mtx = X.jacobian(state_names)
        if self.inputs:
            merged = state_names + self.inputs
        else:
            merged = state_names
        self.jacobian_mtx = lambdify(
            args=merged, expr=jacobian_mtx, modules="numpy"
        )

    def get_jacobian(self, t, y, states, inputs=None):
        if len(y) != len(states):
            raise ValueError(
                f"#states:{len(states)} must be equal to {len(y)}"
            )
            # TODO Check that states are valid, but it must be fast!
        state_map = dict(zip(states, y))
        if inputs:
            # inputs are continuous functions
            processed_inputs = copy.deepcopy(inputs)
            if all(
                isinstance(u, SignalPreprocessor)
                for u in processed_inputs.values()
            ):
                for key, u in processed_inputs.items():
                    processed_inputs[key] = u.interpolate(t)
            else:
                raise NotImplementedError(
                    "only interpolated input is implemented"
                )

            state_map.update(processed_inputs)

        return self.jacobian_mtx(*state_map.values())
