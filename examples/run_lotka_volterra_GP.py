from ode_composer.statespace_model import StateSpaceModel
from ode_composer.linear_model import LinearModel
from scipy.integrate import solve_ivp
from ode_composer.dictionary_builder import DictionaryBuilder
from ode_composer.sbl import SBL
from ode_composer.measurements_generator import MeasurementsGenerator
import numpy as np
import matplotlib.pyplot as plt
from ode_composer.signal_preprocessor import GPSignalPreprocessor


# define Lotka-Volerra model
states = {"x1": "alpha*x1-beta*x1*x2", "x2": "delta*x1*x2-gamma*x2"}
parameters = {"alpha": 2 / 3, "beta": 4 / 3, "delta": 2, "gamma": 1}
ss = StateSpaceModel.from_string(states=states, parameters=parameters)
print("Original Model:")
print(ss)

t_span = [0, 100]
x0 = {"x1": 1.2, "x2": 1.0}
# simulate model
gm = MeasurementsGenerator(ss=ss, time_span=t_span, initial_values=x0)
t, y = gm.get_measurements()

# report
# plt.plot(t, y[0, :], "b", label=r"$x_1(t)$")
# plt.plot(t, y[1, :], "g", label=r"$x_2(t)$")
# plt.legend(loc="best")
# plt.xlabel("time")
# plt.ylabel("population")
# plt.grid()
# plt.show()

gproc = GPSignalPreprocessor(t=t, y=y[0, :])
gproc.gp_regression()
gproc.calculate_time_derivative()
dydt_1 = gproc.dydt

gproc_2 = GPSignalPreprocessor(t=t, y=y[1, :])
gproc_2.gp_regression()
gproc_2.calculate_time_derivative()
dydt_2 = gproc_2.dydt

rr = list()
for yy in y.T:
    rr.append(ss.get_rhs(0, yy, states))


dx1 = list(zip(*rr))[0]
dx2 = list(zip(*rr))[1]
plt.subplot(211)
plt.plot(dx1, label="RHS diff")
plt.plot(dydt_1, label="GP diff")
plt.ylabel("derivative of the x1")
plt.xlabel("time")
plt.legend(loc="best")
plt.subplot(212)
plt.plot(dx2, label="RHS diff")
plt.plot(dydt_2, label="GP diff")
plt.ylabel("derivative of the x2")
plt.xlabel("time")
plt.legend(loc="best")
plt.show()

# step 1 define a dictionary
d_f = ["x1", "x1*x2", "x2"]
dict_builder = DictionaryBuilder(dict_fcns=d_f)
# print(dict_builder.dict_fcns)
data = {"x1": y[0, :], "x2": y[1, :]}
dict_builder.evaluate_dict(input_data=data)

A = dict_builder.get_regression_matrix()
# print(f'regressor matrix {A}')

# step 2 define a linear regression model
lambda_param_1 = 1.5
lambda_param_2 = 1.0
linmodel_x1 = LinearModel(A, np.asarray(dydt_1, dtype=float), lambda_param_1)
linmodel_x2 = LinearModel(A, np.asarray(dydt_2, dtype=float), lambda_param_2)

# step 3 define an SBL problem with the Lin reg model and solve it
sbl_x1 = SBL(linear_model=linmodel_x1, dict_fcns=d_f)
sbl_x1.compute_model_structure()
# print('results for dx1/dt')
# print(sbl_x1.get_results())

sbl_x2 = SBL(linear_model=linmodel_x2, dict_fcns=d_f)
sbl_x2.compute_model_structure()
# print('results for dx2/dt')
# print(sbl_x2.get_results())


# step 4 reporting
# print(sbl.w_estimates)
# plt.plot(sbl.w_estimates)
# plt.ylabel("w est")
# plt.xlabel("iterations")
# plt.show()
#
# plt.plot(sbl.z_estimates)
# plt.ylabel("gamma est")
# plt.xlabel("iterations")
# plt.yscale("log")
# plt.show()


# plt.subplot(211)
# est_y = A.dot(sbl_x1.w_estimates[-1].reshape(sbl_x1.w_estimates[-1].shape[0],1))
# plt.plot(sol.t,est_y)
# plt.plot(sol.t,dx1)
# plt.subplot(212)
# ddx1 = np.array(dx1)
# plt.plot(sol.t,ddx1.reshape(ddx1.shape[0],1) - est_y, 'r',label=r'$dy(t) -d\tilde{y}(t)$')
# plt.show()

#
# #build the ODE
zero_th = 1e-5
# print("SBL results for x1:")
# print(sbl_x1.get_results(zero_th=zero_th))
# print("SBL results for x2:")
# print(sbl_x2.get_results(zero_th=zero_th))


ode_model = StateSpaceModel(
    {
        "x1": sbl_x1.get_results(zero_th=zero_th),
        "x2": sbl_x2.get_results(zero_th=zero_th),
    },
    parameters=None,
)
print("estimated ODE model:")
print(ode_model)
states = ["x1", "x2"]
t = [0, 100]
y0 = [1.2, 1]
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
plt.subplot(224)
plt.plot(sol_ode.y[0, :], sol_ode.y[1, :])
plt.show()
