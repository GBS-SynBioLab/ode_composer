import cvxpy as cp
import numpy as np
from typing import List
from .linear_model import LinearModel


class SBL(object):
    def __init__(self, linear_model: LinearModel):
        self.linear_model = linear_model
        self.z = np.ones((self.linear_model.parameter_num, 1))
        self.w_estimates: List[float] = list()
        self.z_estimates: List[float] = list()

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
            pass
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
