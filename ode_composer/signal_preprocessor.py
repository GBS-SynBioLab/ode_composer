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
from scipy.interpolate import UnivariateSpline
from collections import defaultdict
from sklearn.model_selection import LeaveOneOut
from .util import Timer
from .cache_manager import CacheManager


class BatchSignalPreprocessor(object):
    def __init__(self, t, data, method):
        self.prepocessors = defaultdict()
        if method == "SplineSignalPreprocessor":
            for key, datum in data.items():
                self.prepocessors[key] = SplineSignalPreprocessor(t, datum)

    def interpolate(self, t_new):
        ret_data = defaultdict()
        for key, preprocessor in self.prepocessors.items():
            ret_data[key] = preprocessor.interpolate(t_new)
        return ret_data

    def calculate_time_derivative(self, t_new):
        ret_data = defaultdict()
        for key, preprocessor in self.prepocessors.items():
            ret_data[key] = preprocessor.calculate_time_derivative(t_new)
        return ret_data


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
    def __init__(self, t, y, **kwargs):
        super().__init__(t, y)
        self.cs = None

    def interpolate(self, t_new):
        self.cs = CubicSpline(self.t, self.y)

        return self.cs(t_new)

    def calculate_time_derivative(self, t_new):
        if self.cs is None:
            self.interpolate(t_new=t_new)

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


class ZeroOrderHoldPreprocessor(SignalPreprocessor):
    def __init__(self, t, y):
        super(ZeroOrderHoldPreprocessor, self).__init__(t=t, y=y)

    def calculate_time_derivative(self):
        raise NotImplementedError(
            "Time derivative calculation is not implemented for Zero order hold!"
        )

    def interpolate(self, t_new):
        # TODO ZAT support non pandas data format too!
        ret = []
        if isinstance(t_new, float):
            return self.y[abs(self.t - t_new).idxmin()]

        for t_i in t_new:
            ret.append(self.y[abs(self.t - t_i).idxmin()])
        return ret


class SmoothingSplinePreprocessor(SignalPreprocessor):
    def __init__(
        self,
        t,
        y,
        tune_smoothness=True,
        weights=None,
        spline_id=None,
        cache_folder=None,
    ):
        super().__init__(t, y)
        self.cs = None
        self.s = None
        self.weights = weights
        self.spline_id = spline_id
        self.cache_folder = cache_folder

        if tune_smoothness:
            self.tune_smoothness()

    def _cache_checked(f):
        def cache_checker(*args, **kwargs):
            self = args[0]
            cache_manager = CacheManager(
                cache_id=self.spline_id, cache_folder=self.cache_folder
            )
            if cache_manager.cache_hit():
                cached_data = cache_manager.read()
                self.s = cached_data["smoothness"]
            else:
                f(*args, **kwargs)
                data_to_cache = {"smoothness": self.s}
                cache_manager.write(data_to_cache)

        return cache_checker

    @_cache_checked
    def tune_smoothness(self):
        sum_res = [[] for _ in range(len(self.t))]

        loo = LeaveOneOut()
        sweep = [0] + list(np.logspace(-4, 4, 100)) + [len(self.t)]
        with Timer(
            f"CV loop for SmoothingSplinePreprocessor on {self.spline_id}"
        ):
            for case_idx, (train_index, test_index) in enumerate(
                loo.split(self.t)
            ):
                if self.weights is not None:
                    w = self.weights.iloc[train_index]
                else:
                    w = None
                X_train, X_test = (
                    self.t.iloc[train_index],
                    self.t.iloc[test_index],
                )
                y_train, y_test = (
                    self.y.iloc[train_index],
                    self.y.iloc[test_index],
                )
                spl = UnivariateSpline(X_train, y_train, w=w)

                for s in sweep:
                    spl.set_smoothing_factor(s=s)
                    sum_res[case_idx].append(
                        np.square(float(y_test - spl(X_test)))
                    )

        total = np.sum(np.array(sum_res), axis=0)

        s_opt_idx = np.argmin(total)
        self.s = sweep[s_opt_idx]

    def interpolate(self, t_new):
        self.cs = UnivariateSpline(self.t, self.y, s=self.s, w=self.weights)

        return self.cs(t_new)

    def calculate_time_derivative(self, t_new):
        if self.cs is None:
            self.interpolate(t_new=t_new)

        pp = self.cs.derivative()
        return pp(t_new)
