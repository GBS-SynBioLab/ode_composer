import numpy as np
from typing import List, Dict
from .util import MultiVariableFunction
from itertools import combinations_with_replacement
from sympy import latex


class DictionaryBuilder(object):
    def __init__(self, dict_fcns: List[str]):
        self.dict_fcns: List[MultiVariableFunction] = list()
        self.regressor_mtx = None
        for d_f in dict_fcns:
            self.add_dict_fcn(d_f)

    def add_dict_fcn(self, d_f: str):
        dict_fcn = MultiVariableFunction.create_function(
            rhs_fcn=d_f, parameters={}, weight=1.0
        )
        self.dict_fcns.append(dict_fcn)

    def evaluate_dict(self, input_data: Dict) -> np.ndarray:
        """Evaluates the symbolic expressions stored in the dictionary with input data.

        The evaluated dictionary, referred to as regressor matrix attribute, is returned."""
        reg_mtx = []
        for idx, d_fcn in enumerate(self.dict_fcns):
            reg_mtx.append(
                d_fcn.evaluate_function(measurement_data=input_data)
            )
            # each dictionary function's weight gets a parameter name
            d_fcn.constant_name = f"w{idx+1}"

        #Ensure constants are evaluated not as a single digit, but as a vector of the same length as the input data
        for i in range(len(reg_mtx)):
            if not isinstance(reg_mtx[i], np.ndarray):
                dict_val = input_data.values()
                value_iterator = iter(dict_val)
                first_value = next(value_iterator)
                reg_mtx[i] = reg_mtx[i]*np.ones_like(first_value)

        self.regressor_mtx = np.transpose(np.vstack(reg_mtx))
        return self.regressor_mtx

    @classmethod
    def from_mak_generator(
        cls, number_of_states: int, max_order: int = 2, number_of_inputs=0
    ):
        """Build a dictionary with massaction kinetic terms.

        based on the number of states and the maximum order (or chemical complex size)
        this function generates all the possible polynomial terms.

        >>> db = DictionaryBuilder.from_mak_generator(number_of_states=2, max_order=2)
        >>> str(db)
        'x1*x1 | x1*x2 | x2*x2'

        >>> db = DictionaryBuilder.from_mak_generator(number_of_states=2, max_order=2, number_of_inputs=1)
        >>> str(db)
        'x1*x1 | x1*x2 | u1*x1 | x2*x2 | u1*x2 | u1*u1'

        Args:
            number_of_states: number of states the model has, e.g. two means (x_1,x_2)
            max_order: the maximum number of states in polynomial term, e.g. max_order is three means x_1*x_2^2,  in a two state system
            number_of_inputs: that are added to the dictionary function (as massaction kinetics terms)

        Returns: DictionaryBuilder object

        """
        if number_of_states < 1:
            raise ValueError("Model has to have at least non-state")

        if max_order < 1:
            raise ValueError("The max_order has to be at least one")

        if number_of_inputs < 0:
            raise ValueError("The number of inputs cannot be negative")

        states = []
        for s in range(1, number_of_states + 1):
            states.append(f"x{s}")
        if number_of_inputs != 0:
            for i in range(1, number_of_inputs + 1):
                states.append(f"u{i}")

        comb = combinations_with_replacement(states, max_order)

        mak_dictionary = []
        for c in comb:
            mak_dictionary.append("*".join(c))

        return cls(dict_fcns=mak_dictionary)

    @classmethod
    def from_positive_hill_generator(
        cls,
        state_variable,
        Km_range,
        cooperativity_range,
        proportional_species=None,
    ):
        """
        f(x_1,x_2) = x_2*x_1^n/(Km^n+x_1^n)
        Args:
            state_variable:
            Km_range:
            cooperativity_range:
            proportional_species:

        Returns:

        """
        term_list = []
        for Km in Km_range:
            for n in cooperativity_range:
                term_list.append(
                    f"{proportional_species+'*' if proportional_species else ''}{state_variable}^{n}/({Km}^{n} + {state_variable}^{n})"
                )

        return cls(dict_fcns=term_list)

    @classmethod
    def from_negative_hill_generator(
        cls,
        state_variable,
        Km_range,
        cooperativity_range,
        proportional_species=None,
    ):
        term_list = []
        for Km in Km_range:
            for n in cooperativity_range:
                term_list.append(
                    cls.negative_hill(
                        variable=state_variable,
                        Km=Km,
                        n=n,
                        proportional_species=proportional_species,
                    )
                )

        return cls(dict_fcns=term_list)

    @staticmethod
    def negative_hill(variable, Km, n, proportional_species=None):
        return f"{proportional_species if proportional_species else '1'}/({Km}^{n} + {variable}^{n})"

    @staticmethod
    def pos_neg_hill(activator, repressor, Km_act, n_act, Km_rep, n_rep):
        """Positive and Negative Hill representation for combinatorial promoters

        Based on equation 2.25 in BFS Version 1.0b, September 14, 2014
        http://www.cds.caltech.edu/~murray/books/AM08/pdf/bfs-coreproc_14Sep14.pdf
        """
        return f"(({activator}/{Km_act})^{n_act})/(1+({activator}/{Km_act})^{n_act}+({repressor}/{Km_rep})^{n_rep})"

    @classmethod
    def from_neg_hill_of_neg_hill_generator(
        cls,
        state1,
        Km1_range,
        cooperativity1_range,
        state2,
        Km2_range,
        cooperativity2_range,
    ):
        terms = []
        for Km in Km1_range:
            for n in cooperativity1_range:
                for Km2 in Km2_range:
                    for n2 in cooperativity2_range:
                        neg_hill = cls.negative_hill(
                            variable=state2, Km=Km2, n=n2
                        )
                        terms.append(
                            cls.negative_hill(
                                variable=f"{state1}*{neg_hill}", Km=Km, n=n
                            )
                        )

        return cls(dict_fcns=terms)

    @classmethod
    def from_pos_neg_hill_generator(
        cls,
        state1,
        Km1_range,
        cooperativity1_range,
        state2,
        Km2_range,
        cooperativity2_range,
    ):
        terms = []
        for Km in Km1_range:
            for n in cooperativity1_range:
                for Km2 in Km2_range:
                    for n2 in cooperativity2_range:
                        terms.append(
                            cls.pos_neg_hill(
                                activator=state1,
                                repressor=state2,
                                Km_act=Km,
                                n_act=n,
                                Km_rep=Km2,
                                n_rep=n2,
                            )
                        )

        return cls(dict_fcns=terms)

    @classmethod
    def from_dict_fcns(cls, dict_fcn):
        # TODO change the __init__ to accept MultiVariableFunction as a dictionary
        instance = cls(dict_fcns=[])
        instance.dict_fcns = dict_fcn
        return instance

    def __str__(self):
        """Returns the string representation of dictionary functions"""

        return " | ".join([str(df) for df in self.dict_fcns])

    def print_dictionary(self, latex_format=False):
        ss = []
        line_width = 10
        max_width = max(
            [len(str(df.symbolic_expression)) for df in self.dict_fcns]
        )
        s = []
        s2 = []
        if latex_format:
            sep = " & "
            new_line = "\\\ "
            columns = "|c" * line_width + "|"
            preamble = (
                "\\begin{center}\\begin{tabular}{ %s } \\hline\n" % columns
            )
            postamble = "\n \\end{tabular}\\end{center}"
        else:
            sep = " | "
            new_line = "\n"
            preamble = ""
            postamble = ""
        for idx, df in enumerate(self.dict_fcns):
            s.append(f"{df.constant_name:<{max_width}}")
            s2.append(f"${str(latex(df.symbolic_expression)):<{max_width}}$")
            if (idx + 1) % line_width == 0:
                ss.append(s)
                ss.append(s2)
                s = []
                s2 = []
        else:
            ss.append(s)
            ss.append(s2)

        print(preamble)
        for idx, row in enumerate(ss):
            if (idx + 1) % 2 == 0:
                hline = "\hline"
            else:
                hline = ""
            print(sep.join(row) + new_line + hline)
        print(postamble)

    def __add__(self, other):
        """Adds to DictionaryBuilder instances together"""
        if len(self.dict_fcns) == 0 or len(other.dict_fcns) == 0:
            raise ValueError("Dictionary cannot be empty!")
        # TODO ZAT: change it to chain or extend
        return DictionaryBuilder.from_dict_fcns(
            self.dict_fcns + other.dict_fcns
        )

    def __len__(self):
        return len(self.dict_fcns)