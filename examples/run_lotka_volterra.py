from ode_composer.statespace_model import *
from scipy.integrate import solve_ivp
from ode_composer.dictionary_builder import *
from ode_composer.sbl import SBL
from ode_composer.signal_preprocessor import RHSEvalSignalPreprocessor
from ode_composer.measurements_generator import MeasurementsGenerator


import numpy as np
import matplotlib.pyplot as plt


# define Lotka-Volerra model
states = {"x1": "alpha*x1-beta*x1*x2", "x2": "delta*x1*x2-gamma*x2"}
parameters = {"alpha": 2 / 3, "beta": 4 / 3, "delta": 1, "gamma": 1}
ss = StateSpaceModel.from_string(states=states, parameters=parameters)
print("Original Model:")
print(ss)

states = ["x1", "x2"]
t_span = [0, 15]
x0 = {"x1": 1.2, "x2": 1.0}
# simulate model
gm = MeasurementsGenerator(ss=ss, time_span=t_span, initial_values=x0)
t, y = gm.get_measurements(SNR_db=25)
# report
plt.plot(t, y[0, :], "b", label=r"$x_1(t)$")
plt.plot(t, y[1, :], "g", label=r"$x_2(t)$")
plt.legend(loc="best")
plt.xlabel("time")
plt.ylabel("population")
plt.grid()
plt.show()


rhs_preprop = RHSEvalSignalPreprocessor(
    t=t, y=y, rhs_function=ss.get_rhs, states=states
)
rhs_preprop.calculate_time_derivative()
dydt_rhs = rhs_preprop.dydt
dx1 = dydt_rhs[0, :]
dx2 = dydt_rhs[1, :]

# step 1 define a dictionary
d_f = ["x1", "x1*x2", "x2"]
dict_builder = DictionaryBuilder(dict_fcns=d_f)
# print(dict_builder.dict_fcns)
data = {"x1": y[0, :], "x2": y[1, :]}
A = dict_builder.evaluate_dict(input_data=data)

# print(f'regressor matrix {A}')

# step 2 define an SBL problem with the Lin reg model and solve it
lambda_param = 0.0
sbl_x1 = SBL(
    dict_mtx=A, data_vec=dx1, dict_fcns=d_f, lambda_param=lambda_param
)
sbl_x1.compute_model_structure()
# print('results for dx1/dt')
# print(sbl_x1.get_results())

sbl_x2 = SBL(
    dict_mtx=A, data_vec=dx2, dict_fcns=d_f, lambda_param=lambda_param
)
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
print("SBL results for x1:")
print(sbl_x1.get_results(zero_th=zero_th))
print("SBL results for x2:")
print(sbl_x2.get_results(zero_th=zero_th))


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
plt.plot(sol_ode.t, sol_ode.y[0, :], "b", label=r"$x_1(t)$")
plt.plot(sol_ode.t, sol_ode.y[1, :], "g", label=r"$x_2(t)$")
plt.legend(loc="best")
plt.xlabel("time")
plt.ylabel("population")
plt.grid()
plt.show()
