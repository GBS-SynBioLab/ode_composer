import numpy as np
from typing import List, Dict
from .util import MultiVariableFunction


class DictionaryBuilder(object):
    def __init__(self, dict_fcns: List[str]):
        self.dict_fcns: List[MultiVariableFunction] = list()
        self.regression_mtx = None
        for d_f in dict_fcns:
            self.add_dict_fcn(d_f)

    def add_dict_fcn(self, d_f: str):
        dict_fcn = MultiVariableFunction.create_function(
            rhs_fcn=d_f, parameters={}, weight=1.0
        )
        self.dict_fcns.append(dict_fcn)

    def evaluate_dict(self, input_data: Dict):
        reg_mtx = [
            d_fcn.evaluate_function(measurement_data=input_data)
            for d_fcn in self.dict_fcns
        ]
        self.regression_mtx = np.transpose(np.vstack(reg_mtx))

        return self.regression_mtx
