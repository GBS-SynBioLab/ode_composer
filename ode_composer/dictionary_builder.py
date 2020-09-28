import numpy as np
from typing import List, Dict
from .util import MultiVariableFunction


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
            d_fcn.constant_name = f"p{idx+1}"
        self.regressor_mtx = np.transpose(np.vstack(reg_mtx))
        return self.regressor_mtx
