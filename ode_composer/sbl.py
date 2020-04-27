import copy
import cvxpy as cp
import numpy as np
from typing import List, Dict, Union
from .linear_model import LinearModel
from .dictionary_builder import MultiVariableFunction


class SBL(object):
    def __init__(self, linear_model: LinearModel, dict_fcns: List[str] = None):
        self.linear_model = linear_model
        self.z = np.ones((self.linear_model.parameter_num, 1))
        self.w_estimates: List[float] = list()
        self.z_estimates: List[float] = list()
        self.dict_fcns: List[str] = dict_fcns

    def data_fit(self, w):
        return (1.0 / 2.0) * cp.pnorm(
            self.linear_model.data_vec - self.linear_model.dict_mtx @ w, p=2
        ) ** 2

    def regularizer(self, w):
        return np.sqrt(self.z).T * cp.atoms.elementwise.abs.abs(w)

    def objective_fn(self, w):
        return self.data_fit(
            w
        ) + self.linear_model.lambda_param * self.regularizer(w)

    def estimate_model_parameters(self):
        w_variable = cp.Variable(self.linear_model.parameter_num)
        problem = cp.Problem(cp.Minimize(self.objective_fn(w=w_variable)))

        try:
            problem.solve()
            if problem.status == "optimal":
                self.w_estimates.append(w_variable.value)
            else:
                print("opt problem")
                # TODO deal w/ opt error
        except cp.error.SolverError:
            raise cp.error.SolverError
            # TODO deal w/ solver error

    def update_z(self):
        w_actual = self.w_estimates[-1]
        Gamma = abs(w_actual) / np.sqrt(self.z)
        Gamma_diag = np.zeros((Gamma.shape[0], Gamma.shape[0]), float)
        np.fill_diagonal(Gamma_diag, Gamma)
        Sigma_y = self.linear_model.lambda_param * np.eye(
            self.linear_model.data_num
        ) + self.linear_model.dict_mtx @ Gamma_diag @ np.transpose(
            self.linear_model.dict_mtx
        )

        self.z = np.diag(
            np.transpose(self.linear_model.dict_mtx)
            @ np.linalg.pinv(Sigma_y)
            @ self.linear_model.dict_mtx
        )
        self.z_estimates.append(self.z)

    def compute_model_structure(self):
        # TODO transform this into a generator
        for _ in range(10):
            self.estimate_model_parameters()
            self.update_z()

    def get_results(
        self, zero_th: float = None
    ) -> Dict[Union[str, MultiVariableFunction], float]:
        zero_idx = list()
        if len(self.w_estimates) == 0:
            w_est = np.zeros((self.linear_model.parameter_num, 1))
        else:
            w_est = self.w_estimates[-1]
            if zero_th is not None:
                zero_idx = [
                    idx for idx, w in enumerate(w_est) if abs(w) <= zero_th
                ]

        if (
            self.dict_fcns is None
            or self.dict_fcns is isinstance(self.dict_fcns, list)
            and len(self.dict_fcns) == 0
        ):
            self.dict_fcns = [
                f"w_{i}" for i in range(self.linear_model.parameter_num)
            ]

        if len(zero_idx) > 0:
            d_fcns = copy.deepcopy(self.dict_fcns)
            new_w_est = list(w_est)
            for idx in sorted(zero_idx, reverse=True):
                del d_fcns[idx]
                del new_w_est[idx]
            return dict(zip(d_fcns, new_w_est))
        else:
            return dict(zip(self.dict_fcns, w_est))
