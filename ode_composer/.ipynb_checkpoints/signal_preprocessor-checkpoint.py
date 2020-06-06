from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    ExpSineSquared,
    DotProduct,
    ConstantKernel,
    WhiteKernel,
)
import numpy as np
from scipy.interpolate import CubicSpline


class SignalPreprocessor(object):
    def __init__(self, t, y):
        self._dydt = None
        self.y = y
        self.t = t

    @property
    def dydt(self):
        # TODO add checks
        return self._dydt

    @dydt.setter
    def dydt(self, new_value):
        raise ValueError(
            "dydt cannot be changed from the outside of the object!"
        )


class GPSignalPreprocessor(SignalPreprocessor):
    def __init__(self, t, y, selected_kernel="RBF"):
        super().__init__(t, y)
        self.kernels = None
        self.selected_kernel = selected_kernel

        # Create different kernels that will be explored
        self.kernels = dict()

        self.kernels["RBF"] = 1.0 * RBF(
            length_scale=0.5, length_scale_bounds=(1e-1, 100.0)
        )
        self.kernels["RatQuad"] = 1.0 * RationalQuadratic(
            length_scale=1.0, alpha=0.2
        )
        self.kernels["ExpSineSquared"] = 1.0 * ExpSineSquared(
            length_scale=1.0,
            periodicity=3,
            length_scale_bounds=(0.1, 10.0),
            periodicity_bounds=(1.0, 10.0),
        )
        self.kernels["Matern"] = 1.0 * Matern(
            length_scale=1.0, length_scale_bounds=(1e-1, 100.0), nu=1.5
        )

        if selected_kernel not in self.kernels.keys():
            raise KeyError(
                f"Unknown kernel: {selected_kernel}, available kernels: {self.kernels.keys()}"
            )

        # Generate the noisy kernels
        self.noisy_kernels = dict()
        for key, kernel in self.kernels.items():
            self.noisy_kernels[key] = kernel + WhiteKernel(
                noise_level=1, noise_level_bounds=(1e-3, 1e3)
            )

    def interpolate(self):
        # Adjust the number of samples to be drawn from the fitted GP
        gp_samples = 1

        acutal_kernel = self.noisy_kernels[self.selected_kernel]
        gp = GaussianProcessRegressor(kernel=acutal_kernel)

        X = self.t[:, np.newaxis]
        gp.fit(X, self.y)

        self.A_mean, A_std = gp.predict(X, return_std=True)
        _, self.K_A = gp.predict(X, return_cov=True)
        y_samples = gp.sample_y(X, gp_samples)

        return y_samples

    def calculate_time_derivative(self):
        dA_mean = np.diff(self.A_mean)
        dTime = np.diff(self.t)
        dTime = np.append(dTime, [dTime[-1]])
        dA_mean = np.append(dA_mean, [dA_mean[-1]]) / dTime

        self._dydt = dA_mean

    def diff_matrix(self, size):
        """Differentiation matrix -- used as a linear operator"""
        A = np.zeros((size, size))
        b = np.ones(size - 1)
        np.fill_diagonal(A[0:], -b)
        np.fill_diagonal(A[:, 1:], b)
        return A


class SplineSignalPreprocessor(SignalPreprocessor):
    def __init__(self, t, y):
        super().__init__(t, y)

    def interpolate(self, t_new):
        self.cs = CubicSpline(self.t, self.y.T)

        return self.cs(t_new)

    def calculate_time_derivative(self, t_new):
        pp = self.cs.derivative()
        return pp(t_new)


class RHSEvalSignalPreprocessor(SignalPreprocessor):
    def __init__(self, t, y, rhs_function, states):
        super().__init__(t, y)
        self.rhs_function = rhs_function
        self.states = states

    def interpolate(self):
        pass

    def calculate_time_derivative(self):
        rr = list()
        for yy in self.y.T:
            rr.append(self.rhs_function(0, yy, self.states))

        self._dydt = np.array(rr).T
