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
    def __init__(self, t, y):
        super().__init__(t, y)
        self.kernels = None

        # Create different kernels that will be explored
        self.kernels = [
            1.0 * RBF(length_scale=0.5, length_scale_bounds=(1e-1, 100.0)),
            1.0 * RationalQuadratic(length_scale=1.0, alpha=0.2),
            1.0
            * ExpSineSquared(
                length_scale=1.0,
                periodicity=3,
                length_scale_bounds=(0.1, 10.0),
                periodicity_bounds=(1.0, 10.0),
            ),
            1.0
            * Matern(
                length_scale=1.0, length_scale_bounds=(1e-1, 100.0), nu=1.5
            ),
        ]

        # Generate the noisy kernels
        self.noisy_kernels = []
        for kernel in self.kernels:
            self.noisy_kernels.append(
                kernel
                + WhiteKernel(noise_level=1, noise_level_bounds=(1e-3, 1e3))
            )

    def interpolate(self):
        gp_samples = (
            1  # Adjust the number of samples to be drawn from the fitted GP
        )

        KERNEL = 0
        gp = GaussianProcessRegressor(kernel=self.noisy_kernels[KERNEL])

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

        # NUM_PLOT = self.y.shape[0]
        # dMatrix = self.diff_matrix(NUM_PLOT)

        # dK_A = dMatrix * self.K_A * np.transpose(dMatrix)
        # dA_std = np.sqrt(np.diag(dK_A))

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
    def __init__(self, t, y, rhs_function):
        super().__init__(t, y)
