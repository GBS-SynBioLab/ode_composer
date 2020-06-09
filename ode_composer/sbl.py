import copy
import cvxpy as cp
import numpy as np
from typing import List, Dict, Union
from .linear_model import LinearModel
from .dictionary_builder import MultiVariableFunction
import warnings


class SBL(object):
    def __init__(
        self,
        dict_mtx: np.ndarray,
        data_vec: np.ndarray,
        lambda_param: float,
        dict_fcns: List[str] = None,
    ):
        #L2-normalization D.Wipf et. al - Iterative reweighted l1 and l2 methods for finding sparse solutions
        self.norm_vect = np.linalg.norm(dict_mtx, axis=0)
        self.norm_dict = dict_mtx/self.norm_vect

        self.linear_model = LinearModel(self.norm_dict, data_vec=data_vec)
        self.z = np.ones((self.linear_model.parameter_num, 1))
        self.w_estimates: List[float] = list()
        self.z_estimates: List[float] = list()
        self.dict_fcns: List[str] = dict_fcns
        self.lambda_param = lambda_param

    @property
    def lambda_param(self) -> float:
        return self._lambda_param

    @lambda_param.setter
    def lambda_param(self, new_lambda_param: float):
        if not isinstance(new_lambda_param, float):
            raise TypeError(
                "lambda param is %s, it must be float!" % new_lambda_param
            )
        if new_lambda_param < 0:
            raise ValueError(
                "lambda param must be non-negative, not %s!" % new_lambda_param
            )

        self._lambda_param = new_lambda_param

    def data_fit(self, w):
        return (1.0 / 2.0) * cp.pnorm(
            self.linear_model.data_vec - self.linear_model.dict_mtx @ w, p=2
        ) ** 2

    def regularizer(self, w):
        return np.sqrt(self.z).T * cp.atoms.elementwise.abs.abs(w)

    def objective_fn(self, w):
        return self.data_fit(w) + 2*self.lambda_param * self.regularizer(w)

    def estimate_model_parameters(self):
        w_variable = cp.Variable(self.linear_model.parameter_num)
        problem = cp.Problem(cp.Minimize(self.objective_fn(w=w_variable)))

        try:
            problem.solve()
            if problem.status == cp.OPTIMAL:
                self.w_estimates.append(w_variable.value)
            else:
                if problem.status == cp.OPTIMAL_INACCURATE:
                    warnings.warn(
                        f"Problem with optimization accuracy: {problem.status}"
                    )
                    self.w_estimates.append(w_variable.value)
                else:
                    print(
                        f"Problem with the solution from cvxpy: {problem.status}"
                    )
        except cp.error.SolverError:
            raise cp.error.SolverError
            # TODO deal w/ solver error

    def update_z(self):
        w_actual = self.w_estimates[-1]
        Gamma = np.divide(abs(w_actual), np.sqrt(self.z.T)).T
        Gamma_diag = np.zeros((Gamma.shape[0], Gamma.shape[0]), float)
        np.fill_diagonal(Gamma_diag, Gamma)
        Sigma_y = self.lambda_param * np.eye(
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
        # TODO: stop loop upon convergence
        for _ in range(500):
            self.estimate_model_parameters()
            self.update_z()

    def get_results(
        self, zero_th: float = None
    ) -> Dict[Union[str, MultiVariableFunction], float]:
        zero_idx = list()
        if len(self.w_estimates) == 0:
            w_est = np.zeros((self.linear_model.parameter_num, 1))
        else:
            # Recover back parameters after dict normalization
            w_est = self.w_estimates[-1]/self.norm_vect
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
