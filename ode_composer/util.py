from sympy import *
from typing import List, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class MultiVariableFunction:
    arguments: List[Symbol]
    # TODO restrict this better, check sympy documentation
    fcn_pointer: object
    symbolic_expression: Mul
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
