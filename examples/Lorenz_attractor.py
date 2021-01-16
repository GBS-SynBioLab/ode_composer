import time

from ode_composer.statespace_model import StateSpaceModel
from scipy.integrate import solve_ivp
from ode_composer.dictionary_builder import DictionaryBuilder
from ode_composer.sbl import SBL
from ode_composer.measurements_generator import MeasurementsGenerator
import matplotlib.pyplot as plt
from ode_composer.signal_preprocessor import RHSEvalSignalPreprocessor

# building a State Space model
states = {"x": "sigma*(y-x)", "y": "x*(rho-z)-y", "z": "x*y-beta*z"}
parameters = {"sigma": 10, "beta": 8 / 3, "rho": 28}
ss = StateSpaceModel.from_string(states=states, parameters=parameters)
print(f"Original model:\n {ss}")

t_span = [0, 65]
x0 = {"x": 1.0, "y": 1.0, "z": 1.0}
data_points = 500
# simulate model
gm = MeasurementsGenerator(
    ss=ss, time_span=t_span, initial_values=x0, data_points=data_points
)
t, y = gm.get_measurements(SNR_db=15)


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
plt.plot(xs=y[0, :], ys=y[1, :], zs=y[2, :])
plt.show()

rhs_preprop = RHSEvalSignalPreprocessor(
    t=t, y=y, rhs_function=ss.get_rhs, states=states
)
rhs_preprop.calculate_time_derivative()
dydt_rhs = rhs_preprop.dydt
dx1 = dydt_rhs[0, :]
dx2 = dydt_rhs[1, :]
dx3 = dydt_rhs[2, :]
# step 1 define a dictionary
d_f = ["x", "y", "z", "x*z", "y*x", "z*y"]
dict_builder = DictionaryBuilder(dict_fcns=d_f)
dict_functions = dict_builder.dict_fcns
# associate variables with data
data = {"x": y[0, :], "y": y[1, :], "z": y[2, :]}
A = dict_builder.evaluate_dict(input_data=data)

# step 2 define an SBL problem
# with the Lin reg model and solve it
lambda_param_1 = 0.5
sbl_x1 = SBL(
    dict_mtx=A,
    data_vec=dx1,
    lambda_param=lambda_param_1,
    state_name="x1",
    dict_fcns=dict_functions,
)
start_time = time.time()
sbl_x1.compute_model_structure()
elapsed_time = time.time() - start_time
print(f"x1 computation took: {elapsed_time}")

lambda_param_2 = 5.0
sbl_x2 = SBL(
    dict_mtx=A,
    data_vec=dx2,
    lambda_param=lambda_param_2,
    state_name="x2",
    dict_fcns=dict_functions,
)
start_time = time.time()
sbl_x2.compute_model_structure()
elapsed_time = time.time() - start_time
print(f"x2 computation took: {elapsed_time}")

lambda_param_3 = 5.0
sbl_x3 = SBL(
    dict_mtx=A,
    data_vec=dx3,
    lambda_param=lambda_param_3,
    state_name="x3",
    dict_fcns=dict_functions,
)
start_time = time.time()
sbl_x3.compute_model_structure()
elapsed_time = time.time() - start_time
print(f"x3 computation took: {elapsed_time}")
# step 4 reporting
# #build the ODE
zero_th = 1e-5

ode_model = StateSpaceModel.from_sbl(
    {
        "x1": sbl_x1.get_results(zero_th=zero_th),
        "x2": sbl_x2.get_results(zero_th=zero_th),
        "x3": sbl_x3.get_results(zero_th=zero_th),
    },
    parameters=None,
)
print("Estimated ODE model:")
print(ode_model)
states = ["x", "y", "z"]
t = [0, 100]
y0 = [1, 1, 1]
sol_ode = solve_ivp(fun=ode_model.get_rhs, t_span=t, y0=y0, args=(states,))
# report
plt.subplot(221)
t_orig, y_orig = gm.get_measurements()
plt.plot(t_orig, y_orig[0, :], "b", label=r"$x_1(t)$")
plt.plot(t_orig, y_orig[1, :], "g", label=r"$x_2(t)$")
plt.legend(loc="best")
plt.xlabel("time")
plt.ylabel("population")
plt.subplot(222)
plt.plot(sol_ode.t, sol_ode.y[0, :], "b", label=r"$x_1(t)$")
plt.plot(sol_ode.t, sol_ode.y[1, :], "g", label=r"$x_2(t)$")
plt.legend(loc="best")
plt.xlabel("time")
plt.ylabel("population")
plt.grid()
plt.subplot(223)
plt.plot(y_orig[0, :], y_orig[1, :])
plt.xlabel(r"$x_1(t)$")
plt.ylabel(r"$x_2(t)$")
plt.subplot(224)
plt.plot(sol_ode.y[0, :], sol_ode.y[1, :])
plt.xlabel(r"$\hat{x}_1(t)$")
plt.ylabel(r"$\hat{x}_2(t)$")
plt.show()
