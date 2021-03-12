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

# define the model
states = {"x": "-y-z", "y": "x+a*y", "z": "b+x*z-z*c"}
parameters = {"a": 0.2, "b": 1.2, "c": 5.0}
ss = StateSpaceModel.from_string(states=states, parameters=parameters)
print(f"Original model:\n {ss}")

# Adjust model and simulation parameters
t_span = [0, 25]
y0 = {"x": 1.0, "y": 1.0, "z": 1.0}
data_points = 1000

# Adjust noise added to the measurements
gp_interpolation_factor = None
noisy_obs = True
SNR = 13

# simulate model
gm = MeasurementsGenerator(
    ss=ss, time_span=t_span, initial_values=y0, data_points=data_points
)
t, y = gm.get_measurements(SNR_db=SNR)
t, y_original = gm.get_measurements(fixed_seed=True)
y_normfactor = np.mean(np.linalg.norm(y, axis=1))

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
# Estimation of the noise variance for state 1
lambda_param_1 = (np.linalg.norm(dxdt_std)) + 0.1

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
# Estimation of the noise variance for state 2
lambda_param_2 = (np.linalg.norm(dydt_std)) + 0.1

gproc_3 = GPSignalPreprocessor(
    t=t,
    y=y[2, :],
    selected_kernel="RatQuad",
    interpolation_factor=gp_interpolation_factor,
)
z_samples, _ = gproc_3.interpolate(noisy_obs=noisy_obs)
gproc_3.calculate_time_derivative()
dzdt = gproc_3.dydt
dzdt_std = gproc_3.A_std
# Estimation of the noise variance for state 3
lambda_param_3 = (np.linalg.norm(dzdt_std)) + 0.1


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
    t=t, y=y_original, rhs_function=ss.get_rhs, states=states
)

# Plot the observable states with the gaussian noise
plt.figure(0)
plt.plot(y[0,:])
plt.plot(y[1,:])
plt.plot(y[2,:])
plt.title("Observable states")
plt.legend(['x', 'y', 'z'])
plt.show()
rhs_preprop.calculate_time_derivative()
dydt_rhs = rhs_preprop.dydt
dx1 = dydt_rhs[0, :]
dx2 = dydt_rhs[1, :]
dx3 = dydt_rhs[2, :]


# Plot the derivatives of the system and its estimates using Spline and GPs
plt.figure(1)
plt.subplot(311)
plt.plot(t, dx1, label="RHS diff")
plt.plot(t, dx_spline, c='r', label="Spline diff", alpha=0.2)
plt.plot(t_gp, dxdt, c='orange', label="GP diff")
plt.fill_between(t_gp, dxdt - 2*dxdt_std, dxdt + 2*dxdt_std, alpha=0.2, color='k')
plt.ylabel("derivative of x")
plt.xlabel("time")
plt.legend(loc="best")
plt.subplot(312)
plt.plot(t, dx2, label="RHS diff")
plt.plot(t, dy_spline, c='r', label="Spline diff", alpha=0.2)
plt.plot(t_gp, dydt, c='orange', label="GP diff")
plt.fill_between(t_gp, dydt - 2*dydt_std, dydt + 2*dydt_std, alpha=0.2, color='k')
plt.ylabel("derivative of the y")
plt.xlabel("time")
plt.legend(loc="best")
plt.subplot(313)
plt.plot(t, dx3, label="RHS diff")
plt.plot(t, dz_spline, c='r', label="spline diff", alpha=0.2)
plt.plot(t_gp, dzdt, c='orange', label="GP diff")
plt.fill_between(t_gp, dzdt - 2*dzdt_std, dzdt + 2*dzdt_std, alpha=0.2, color='k')
plt.ylabel("derivative of z")
plt.xlabel("time")
plt.legend(loc="best")
plt.show()

print("Gaussian process MSE:")
print(np.mean((dxdt - dx1)**2))

print("Spline MSE:")
print(np.mean((dx_spline - dx1)**2))

# step 1 define a dictionary
d_f = ["1", "x", "y", "z", "x*z", "y*x", "1/x", "1/y", "z**2"]
dict_builder = DictionaryBuilder(dict_fcns=d_f)
dict_functions = dict_builder.dict_fcns
# associate variables with data
data = {"x": x_samples.T, "y": y_samples.T, "z": z_samples.T}
A = dict_builder.evaluate_dict(input_data=data)

# step 2 define an SBL problem
# with the Lin reg model and solve it
config_dict = {
    "solver": {"name": "ECOS", "show_time": False, "settings": {}},
    "verbose": False,
}
lambda_param = 10.0

sbl_x1 = SBL(
    dict_mtx=A,
    data_vec=dxdt,
    lambda_param=lambda_param_1, 
    state_name="x1", 
    dict_fcns=dict_functions,
    config=config_dict
)
start_time = time.time()
sbl_x1.compute_model_structure()
elapsed_time = time.time() - start_time
print(f"x1 computation took: {elapsed_time}")

sbl_x2 = SBL(
    dict_mtx=A, 
    data_vec=dydt, 
    lambda_param=lambda_param_2, 
    state_name="x2", 
    dict_fcns=dict_functions,
    config=config_dict
)
start_time = time.time()
sbl_x2.compute_model_structure()
elapsed_time = time.time() - start_time
print(f"x2 computation took: {elapsed_time}")

sbl_x3 = SBL(
    dict_mtx=A, 
    data_vec=dzdt, 
    lambda_param=lambda_param_3, 
    state_name="x3", 
    dict_fcns=dict_functions,
    config=config_dict
)
start_time = time.time()
sbl_x3.compute_model_structure()
elapsed_time = time.time() - start_time
print(f"x3 computation took: {elapsed_time}")

# step 4 reporting
# #build the ODE
zero_th = 1e-2

ode_model = StateSpaceModel.from_sbl(
    {
        "x": sbl_x1.get_results(zero_th=zero_th),
        "y": sbl_x2.get_results(zero_th=zero_th),
        "z": sbl_x3.get_results(zero_th=zero_th),
    },
    parameters=None,
)
print(f"Estimated ODE model: \n{ode_model}")
states = ["x", "y", "z"]
y0 = [1.0, 1.0, 1.0]
sol_ode = solve_ivp(fun=ode_model.get_rhs, t_span=t_span, t_eval=t, y0=y0, args=(states,))
plt.figure(2)
plt.subplot(221)
t_orig, y_orig = gm.get_measurements()
plt.plot(t_orig, y_orig[0, :], "b", label=r"$x(t)$")
plt.plot(t_orig, y_orig[1, :], "g", label=r"$y(t)$")
plt.plot(t_orig, y_orig[2, :], "r", label=r"$z(t)$")
plt.legend(loc="best")
plt.xlabel("time")
plt.ylabel("Observed state")
plt.title("Original")
plt.grid()
plt.subplot(222)
plt.plot(sol_ode.t, sol_ode.y[0, :], "b", label=r"$x(t)$")
plt.plot(sol_ode.t, sol_ode.y[1, :], "g", label=r"$(t)$")
plt.plot(sol_ode.t, sol_ode.y[2, :], "r", label=r"$z(t)$")
plt.legend(loc="best")
plt.xlabel("time")
plt.ylabel("Observed state")
plt.title("Estimation")
plt.grid()
plt.subplot(223, projection='3d')
plt.plot(y_orig[0, :], y_orig[1, :], y_orig[2, :])
plt.xlabel(r"$x_1(t)$")
plt.ylabel(r"$x_2(t)$")
plt.subplot(224, projection='3d')
plt.plot(sol_ode.y[0, :], sol_ode.y[1, :], sol_ode.y[2, :])
plt.xlabel(r"$\hat{x}_1(t)$")
plt.ylabel(r"$\hat{x}_2(t)$")
plt.show()

print(np.mean((y_orig-sol_ode.y)**2))