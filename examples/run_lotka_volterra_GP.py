from ode_composer.statespace_model import StateSpaceModel
from scipy.integrate import solve_ivp
from ode_composer.dictionary_builder import DictionaryBuilder
from ode_composer.sbl import SBL
from ode_composer.measurements_generator import MeasurementsGenerator
import matplotlib.pyplot as plt
from ode_composer.signal_preprocessor import (
    GPSignalPreprocessor,
    RHSEvalSignalPreprocessor,
)


# define Lotka-Volerra model
states = {"x1": "alpha*x1-beta*x1*x2", "x2": "delta*x1*x2-gamma*x2"}
parameters = {"alpha": 2 / 3, "beta": 4 / 3, "delta": 2, "gamma": 1}
ss = StateSpaceModel.from_string(states=states, parameters=parameters)
print("Original Model:")
print(ss)

t_span = [0, 15]
x0 = {"x1": 1.2, "x2": 1.0}
# simulate model
gm = MeasurementsGenerator(ss=ss, time_span=t_span, initial_values=x0)
t, y = gm.get_measurements(SNR_db=25)


gproc = GPSignalPreprocessor(t=t, y=y[0, :])
y_samples_1 = gproc.interpolate()
gproc.calculate_time_derivative()
dydt_1 = gproc.dydt

gproc_2 = GPSignalPreprocessor(t=t, y=y[1, :])
y_samples_2 = gproc_2.interpolate()
gproc_2.calculate_time_derivative()
dydt_2 = gproc_2.dydt


rhs_preprop = RHSEvalSignalPreprocessor(
    t=t, y=y, rhs_function=ss.get_rhs, states=states
)
rhs_preprop.calculate_time_derivative()
dydt_rhs = rhs_preprop.dydt
dx1 = dydt_rhs[0, :]
dx2 = dydt_rhs[1, :]

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
# associate variables with data
data = {"x1": y_samples_1.T, "x2": y_samples_2.T}
A = dict_builder.evaluate_dict(input_data=data)

# step 2 define an SBL problem
# with the Lin reg model and solve it
lambda_param_1 = 0.5
sbl_x1 = SBL(
    dict_mtx=A, data_vec=dydt_1, lambda_param=lambda_param_1, dict_fcns=d_f
)
sbl_x1.compute_model_structure()

lambda_param_2 = 5.0
sbl_x2 = SBL(
    dict_mtx=A, data_vec=dydt_2, lambda_param=lambda_param_2, dict_fcns=d_f
)
sbl_x2.compute_model_structure()


# step 4 reporting
# #build the ODE
zero_th = 1e-5

ode_model = StateSpaceModel(
    {
        "x1": sbl_x1.get_results(zero_th=zero_th),
        "x2": sbl_x2.get_results(zero_th=zero_th),
    },
    parameters=None,
)
print("Estimated ODE model:")
print(ode_model)
states = ["x1", "x2"]
t = [0, 40]
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
plt.xlabel(r"$x_1(t)$")
plt.ylabel(r"$x_2(t)$")
plt.subplot(224)
plt.plot(sol_ode.y[0, :], sol_ode.y[1, :])
plt.xlabel(r"$\hat{x}_1(t)$")
plt.ylabel(r"$\hat{x}_2(t)$")
plt.show()
