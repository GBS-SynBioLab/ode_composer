from sympy import *
from typing import List, Dict
from .util import MultiVariableFunction
from sympy.parsing.sympy_parser import parse_expr
from sympy import Matrix
from .signal_preprocessor import SignalPreprocessor
import copy
import re
import matplotlib.pyplot as plt
import math


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
                ss += f"{float(rr.constant):+.2e}*{rr.symbolic_expression}"
            ss += "\n"
        return ss

    def __len__(self) -> int:
        """The length of the state space model is the number of right hand side terms."""
        return sum([len(rhs) for rhs in self.state_vector.values()])

    def to_amigo(self):
        ode_eq = list()
        for state_name, rhs in self.state_vector.items():
            ss = ""
            ss += f"d{state_name}="
            param_dict = self.get_parameters_for_state(state_name=state_name)
            for rr, p_name in zip(rhs, param_dict.keys()):
                m = re.search("([a-z]+)([1-9]+),([1-9]+)", p_name)
                p_value = f"+{m[1]}{m[2]}_{m[3]}"
                term = rr.symbolic_expression
                ss += f"{p_value}*{str(term).replace('**','^')}"
            ode_eq.append(ss)
        return ode_eq

    def to_latex(self, filename=None, parameter_table=False):
        ss = "\\begin{eqnarray}"
        for state_name, rhs in self.state_vector.items():
            m = re.search("([a-z]+)([1-9]+)", state_name)
            ss += "\\frac{d%s_{%s}}{dt} &=& " % (m[1], m[2])
            param_dict = self.get_parameters_for_state(state_name=state_name)
            for rr, p_name in zip(rhs, param_dict.keys()):
                if parameter_table:
                    m = re.search("([a-z]+)([1-9]+),([1-9]+)", p_name)
                    p_value = f"+{m[1]}_{{{m[2]},{m[3]}}}"
                else:
                    p_value = f"\\num{{{float(rr.constant):+.2e}}}"
                ss += f"{p_value}{latex(rr.symbolic_expression)}"
            ss += "\\\ "
        ss += "\n \\end{eqnarray}"

        if parameter_table:
            ss += self.get_parameter_table(latex_format=True)

        if filename:
            with open(filename, "w") as f:
                f.writelines(ss)

        return ss

    def get_parameter_table(self, latex_format=False):
        # determine the width of the table
        max_terms = max([len(v) for v in self.state_vector.values()])
        # variable for the return string
        ss = ""
        if latex_format:
            sep = " & "
            new_line = "\\\ "
            columns = "|c" * max_terms + "|"
            preamble = (
                "\\begin{center}\\begin{tabular}{ %s } \\hline\n" % columns
            )
            postamble = "\n \\end{tabular}\\end{center}"
        else:
            sep = " | "
            new_line = "\n"
            preamble = ""
            postamble = ""

        ss += preamble
        for state_name in self.state_vector.keys():
            p_dict = self.get_parameters_for_state(
                state_name=state_name, latex_format=latex_format
            )
            max_width = 9  # using +.2e format that results 9 chars
            if latex_format:
                extra_columns = max_terms - len(p_dict)
                columns = " & " * extra_columns
            else:
                columns = ""
            ss += (
                sep.join([f"{p:<{max_width}}" for p in list(p_dict.keys())])
                + columns
                + new_line
            )
            ss += (
                sep.join([f"{v:+.2e}" for v in list(p_dict.values())])
                + columns
                + new_line
            )
            if latex_format:
                ss += "\\hline "
        ss += postamble
        return ss

    def get_parameters_for_state(
        self, state_name, latex_format=False, divider=","
    ):
        if state_name not in self.state_vector.keys():
            raise KeyError(f"Invalid state name: {state_name}")
        rhs = self.state_vector[state_name]
        state_num = list(self.state_vector.keys()).index(state_name) + 1
        param_dict = dict()
        for rr in rhs:
            m = re.search("([a-z]+)([0-9]+)", rr.constant_name)
            if latex_format:
                p_str = f"${m[1]}_{{{state_num}{divider}{m[2]}}}$"
            else:
                p_str = f"{m[1]}{state_num}{divider}{m[2]}"
            param_dict.update({p_str: rr.constant})
        return param_dict

    def plot_parameter_table(self, file_name=None):
        plt.style.use("ggplot")
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for i, state_name in zip(
            range(1, len(self.state_vector) + 1), self.state_vector.keys()
        ):
            fig.add_subplot(len(self.state_vector), 1, i)
            p_dict = self.get_parameters_for_state(
                state_name=state_name, latex_format=True
            )
            x_pos = [i for i, _ in enumerate(p_dict)]
            plt.barh(
                x_pos,
                [math.log10(abs(p)) for p in p_dict.values()],
                color="green",
            )
            plt.yticks(x_pos, [rf"{p}" for p in p_dict.keys()], fontsize=7)
            plt.xlabel(r"$\log_{10}(w)$")

        if file_name:
            plt.savefig(file_name, format="pdf")
        else:
            plt.show()

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
