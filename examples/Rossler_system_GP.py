from ode_composer.statespace_model import StateSpaceModel
from scipy.integrate import solve_ivp
from ode_composer.dictionary_builder import DictionaryBuilder
from ode_composer.sbl import SBL
from ode_composer.measurements_generator import MeasurementsGenerator
import matplotlib.pyplot as plt
import numpy as np
import time
from ode_composer.signal_preprocessor import (
    GPSignalPreprocessor,
    RHSEvalSignalPreprocessor,
    SplineSignalPreprocessor,
)
import matplotlib
font = {'size'   : 15}

matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)


# define the model
states = {"x": "-y-z", "y": "x+a*y", "z": "b+x*z-z*c"}
parameters = {"a": 0.2, "b": 1.2, "c": 5.0}
ss = StateSpaceModel.from_string(states=states, parameters=parameters)
print("Original Model:")
print(ss)

t_span = [0, 50]
y0 = {"x": 1.0, "y": 0.0, "z": 0.0}
data_points = 500
gp_interpolation_factor = None
noisy_obs = True
SNR = 15

# simulate model
gm = MeasurementsGenerator(
    ss=ss, time_span=t_span, initial_values=y0, data_points=data_points
)
t, y = gm.get_measurements(SNR_db=SNR, fixed_seed=True)
_, y_noiseless = gm.get_measurements(SNR_db=None)
y_normfactor = np.mean(np.linalg.norm(y, axis=1))

start_time = time.time()
# Fit a GP, find the derivative and std of each
gproc = GPSignalPreprocessor(
    t=t,
    y=y[0, :],
    selected_kernel="Matern",
    interpolation_factor=gp_interpolation_factor
)
x_samples, t_gp = gproc.interpolate(return_extended_time=True, noisy_obs=noisy_obs)
gproc.calculate_time_derivative()
dxdt = gproc.dydt
dxdt_std = gproc.A_std

gproc_2 = GPSignalPreprocessor(
    t=t,
    y=y[1, :],
    selected_kernel="Matern",
    interpolation_factor=gp_interpolation_factor,
)
y_samples, _ = gproc_2.interpolate(noisy_obs=noisy_obs)
gproc_2.calculate_time_derivative()
dydt = gproc_2.dydt
dydt_std = gproc_2.A_std

gproc_3 = GPSignalPreprocessor(
    t=t,
    y=y[2, :],
    selected_kernel="RatQuad+ExpSineSquared",
    interpolation_factor=gp_interpolation_factor,
)
print(time.time() - start_time)
z_samples, _ = gproc_3.interpolate(noisy_obs=noisy_obs)
gproc_3.calculate_time_derivative()
dzdt = gproc_3.dydt
dzdt_std = gproc_3.A_std

spline_1 = SplineSignalPreprocessor(t, y[0,:])
x_spline = spline_1.interpolate(t)
dx_spline = spline_1.calculate_time_derivative(t)

spline_2 = SplineSignalPreprocessor(t, y[1,:])
y_spline = spline_2.interpolate(t)
dy_spline = spline_2.calculate_time_derivative(t)

spline_3 = SplineSignalPreprocessor(t, y[2,:])
z_spline = spline_3.interpolate(t)
dz_spline = spline_3.calculate_time_derivative(t)

rhs_preprop = RHSEvalSignalPreprocessor(
    t=t, y=y_noiseless, rhs_function=ss.get_rhs, states=states
)

plt.figure(0)
plt.plot(t, y[0,:])
plt.plot(t, y[1,:])
plt.plot(t, y[2,:])
plt.title("Observable states")
plt.ylabel("Value (AU)")
plt.xlabel("Time (s)")
plt.legend(['x', 'y', 'z'])
plt.show()
rhs_preprop.calculate_time_derivative()
dydt_rhs = rhs_preprop.dydt
dx1 = dydt_rhs[0, :]
dx2 = dydt_rhs[1, :]
dx3 = dydt_rhs[2, :]

plt.figure(1)
plt.subplot(311)
plt.suptitle('Data Fitting: Gaussian Process vs. Spline')
plt.plot(t, y[0,:], label="Original")
plt.plot(t, x_spline, c='r', label="Spline")
plt.plot(t_gp, x_samples, c='orange', label="GP")
plt.fill_between(t_gp, x_samples - 2*dxdt_std, x_samples + 2*dxdt_std, alpha=0.2, color='k')
plt.ylabel("x - Value (AU)")
plt.xlabel("time")
plt.legend(loc="best")
plt.subplot(312)
plt.plot(t, y[1,:], label="RHS")
plt.plot(t, y_spline, c='r', label="Spline")
plt.plot(t_gp, y_samples, c='orange', label="GP")
plt.fill_between(t_gp, y_samples - 2*dydt_std, y_samples + 2*dydt_std, alpha=0.2, color='k')
plt.ylabel("y - Value (AU)")
plt.xlabel("time")
plt.legend(loc="best")
plt.subplot(313)
plt.plot(t, y[2,:], label="RHS diff")
plt.plot(t, z_spline, c='r', label="spline diff")
plt.plot(t_gp, z_samples, c='orange', label="GP diff")
plt.fill_between(t_gp, z_samples - 2*dzdt_std, z_samples + 2*dzdt_std, alpha=0.2, color='k')
plt.ylabel("z - Value (AU)")
plt.xlabel("Time (s)")
plt.legend(loc="best")
plt.show()

plt.figure(6)
plt.subplot(311)
plt.suptitle('Estimated Derivative: Gaussian Process vs. Spline')
plt.plot(t, dx1, label="RHS diff")
plt.plot(t, dx_spline, c='r', label="Spline diff")
plt.plot(t_gp, dxdt, c='orange', label="GP diff")
plt.fill_between(t_gp, dxdt - 2*dxdt_std, dxdt + 2*dxdt_std, alpha=0.2, color='k')
plt.ylabel("dx - Value (AU)")
plt.xlabel("Time (s)")
plt.legend(loc="best")
plt.subplot(312)
plt.plot(t, dx2, label="RHS diff")
plt.plot(t, dy_spline, c='r', label="Spline diff")
plt.plot(t_gp, dydt, c='orange', label="GP diff")
plt.fill_between(t_gp, dydt - 2*dydt_std, dydt + 2*dydt_std, alpha=0.2, color='k')
plt.ylabel("dy - Value (AU)")
plt.xlabel("Time (s)")
plt.legend(loc="best")
plt.subplot(313)
plt.plot(t, dx3, label="RHS diff")
plt.plot(t, dz_spline, c='r', label="spline diff")
plt.plot(t_gp, dzdt, c='orange', label="GP diff")
plt.fill_between(t_gp, dzdt - 2*dzdt_std, dzdt + 2*dzdt_std, alpha=0.2, color='k')
plt.ylabel("dz - Value (AU)")
plt.xlabel("Time (s)")
plt.legend(loc="best")
plt.show()


plt.figure(23)
plt.subplot(211)
plt.plot(t, y[1,:], label="RHS")
plt.plot(t, y_spline, c='r', label="Spline")
plt.plot(t_gp, y_samples, c='orange', label="GP")
plt.fill_between(t_gp, y_samples - 2*dydt_std, y_samples + 2*dydt_std, alpha=0.2, color='k')
plt.ylabel("Value (AU)")
plt.xlabel("time (s)")
plt.title("Observable state y")
plt.legend(loc="best")
plt.subplot(212)
plt.plot(t, dx2, label="RHS diff")
plt.plot(t, dy_spline, c='r', alpha=0.4, label="Spline diff")
plt.plot(t_gp, dydt, c='orange', label="GP diff")
plt.fill_between(t_gp, dydt - 2*dydt_std, dydt + 2*dydt_std, alpha=0.2, color='k')
plt.ylabel("Value (AU)")
plt.xlabel("time (s)")
plt.title("Derivative of y")
plt.legend(loc="best")
plt.show()

# step 1 define a dictionary
d_f = ["1", "x", "y", "z", "x*z", "y*x"]
dict_builder = DictionaryBuilder(dict_fcns=d_f)
dict_functions = dict_builder.dict_fcns
# associate variables with data
data = {"x": y[0, :].T, "y": y[1, :].T, "z": y[2, :].T}
#data = {"x": x_samples.T, "y": y_samples.T, "z": z_samples.T}
A = dict_builder.evaluate_dict(input_data=data)

# step 2 define an SBL problem
# with the Lin reg model and solve it
config_dict = {
    "solver": {"name": "ECOS", "show_time": False, "settings": {}},
    "verbose": False,
}
norm_vect = np.linalg.norm(A, axis=0)
A_norm = A / norm_vect

plt.figure(23)
for col in A_norm.T:
    plt.plot(t, col)
plt.ylabel("Value (AU)")
plt.xlabel("Time (s)")
plt.title("Dictionary Evaluation")
plt.show()

start_time = time.time()
# step 2 define an SBL problem
# with the Lin reg model and solve it
lambda_param = 60.0
lambda_param_1 = (np.linalg.norm(dxdt_std)) + 0.1
print(lambda_param_1)
sbl_x1 = SBL(
    dict_mtx=A, data_vec=dxdt, lambda_param=lambda_param_1, dict_fcns=d_f
)
sbl_x1.compute_model_structure()

lambda_param_2 = (np.linalg.norm(dydt_std)) + 0.1
print(lambda_param_2)
sbl_x2 = SBL(
    dict_mtx=A, data_vec=dydt, lambda_param=lambda_param_2, dict_fcns=d_f
)
sbl_x2.compute_model_structure()

lambda_param_3 = (np.linalg.norm(dzdt_std)) + 0.1
print(lambda_param_3)
sbl_x3 = SBL(
    dict_mtx=A, data_vec=dzdt, lambda_param=2.6, dict_fcns=d_f
)
sbl_x3.compute_model_structure()

print(time.time() - start_time)


# step 4 reporting
# #build the ODE
zero_th = 1e-5

ode_model = StateSpaceModel(
    {
        "x": sbl_x1.get_results(zero_th=zero_th),
        "y": sbl_x2.get_results(zero_th=zero_th),
        "z": sbl_x3.get_results(zero_th=zero_th),
    },
    parameters=None,
)
print("Estimated ODE model:")
print(ode_model)

t_span = [0,100]
t_eval = np.linspace(t_span[0], t_span[1], 2000)
states = ["x", "y", "z"]
y0 = [1.0, 0.0, 1.0]

sol_ode = solve_ivp(fun=ode_model.get_rhs, t_span=t_span, t_eval=t_eval, y0=y0, args=(states,))
sol_real = solve_ivp(fun=ss.get_rhs, t_span=t_span, t_eval=t_eval, y0=y0, args=(states,))
plt.figure(2)
plt.subplot(221)
plt.suptitle("Estimated vs Original System: Different Time and Initial Conditions")
plt.plot(sol_real.t, sol_real.y[0, :], "b", label=r"$x(t)$")
plt.plot(sol_real.t, sol_real.y[1, :], "g", label=r"$y(t)$")
plt.plot(sol_real.t, sol_real.y[2, :], "r", label=r"$z(t)$")
plt.legend(loc="best")
plt.xlabel("time")
plt.ylabel("Observed state")
plt.title("Original")
plt.grid()
plt.subplot(223)
plt.plot(sol_ode.t, sol_ode.y[0, :], "b", label=r"$x(t)$")
plt.plot(sol_ode.t, sol_ode.y[1, :], "g", label=r"$y(t)$")
plt.plot(sol_ode.t, sol_ode.y[2, :], "r", label=r"$z(t)$")
plt.legend(loc="best")
plt.xlabel("time")
plt.ylabel("Observed state")
plt.title("Estimation")
plt.grid()
plt.subplot(222, projection='3d')
plt.plot(sol_real.y[0, :], sol_real.y[1, :], sol_real.y[2, :])
plt.xlabel(r"$x_1(t)$")
plt.ylabel(r"$x_2(t)$")
plt.title("Original")
plt.subplot(224, projection='3d')
plt.plot(sol_ode.y[0, :], sol_ode.y[1, :], sol_ode.y[2, :])
plt.xlabel(r"$\hat{x}_1(t)$")
plt.ylabel(r"$\hat{x}_2(t)$")
plt.title("Estimation")
plt.show()
