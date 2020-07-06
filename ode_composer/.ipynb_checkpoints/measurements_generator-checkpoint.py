from ode_composer.statespace_model import StateSpaceModel
from scipy.integrate import solve_ivp
from typing import List, Dict, Union
import numpy as np


class MeasurementsGenerator(object):
    def __init__(
        self,
        ss: StateSpaceModel,
        time_span: List[float],
        initial_values: Dict[str, float],
        data_points: int = None,
    ):
        self.ss = ss
        states = initial_values.keys()
        if data_points is None:
            t_eval = None
        else:
            t_eval = np.linspace(time_span[0], time_span[1], data_points)
        sol = solve_ivp(
            fun=self.ss.get_rhs,
            t_span=time_span,
            y0=list(initial_values.values()),
            args=(states,),
            t_eval=t_eval,
        )

        if sol.success is not True:
            raise ValueError(f"Integration Problem {sol.message}")

        self.sol = sol

    def get_measurements(self, SNR_db=None):
        if SNR_db is not None:
            y_measured = np.zeros(shape=self.sol.y.shape)
            SNR = 10 ** (SNR_db / 10)
            for idx, y in enumerate(self.sol.y):
                Esym = np.sum(abs(y) ** 2) / len(y)
                N_PSD = (Esym) / SNR
                y_measured[idx, :] = y + np.sqrt(N_PSD) * np.random.randn(
                    len(y)
                )
        else:
            y_measured = self.sol.y

        return self.sol.t, y_measured
