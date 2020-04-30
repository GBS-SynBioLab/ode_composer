from ode_composer.statespace_model import *
from scipy.integrate import solve_ivp
from ode_composer.dictionary_builder import *
from ode_composer.sbl import *
from ode_composer.measurements_generator import MeasurementsGenerator
from ode_composer.signal_preprocessor import (
    GPSignalPreprocessor,
    RHSEvalSignalPreprocessor,
)

import numpy as np
import matplotlib.pyplot as plt

# define a MAK model
states = {"x1": {"x1": "m11"}, "x2": {"x1": "m21", "x2": "m22"}}
parameters = {"m11": -0.5, "m12": 0.1, "m21": 0.6, "m22": -0.3}
ss = StateSpaceModel(states=states, parameters=parameters)
print(ss)

t_span = [0, 100]
x0 = {"x1": 1.2, "x2": 1.0}
# simulate model
gm = MeasurementsGenerator(ss=ss, time_span=t_span, initial_values=x0)
t, y = gm.get_measurements()


plt.plot(t, y[0, :], "b", label=r"$x_1(t)$")
plt.plot(t, y[1, :], "g", label=r"$x_2(t)$")
plt.legend(loc="best")
plt.xlabel("time")
plt.ylabel("concentration")
plt.grid()
plt.show()


rhs_preprop = RHSEvalSignalPreprocessor(
    t=t, y=y, rhs_function=ss.get_rhs, states=states
)
rhs_preprop.calculate_time_derivative()
dydt_rhs = rhs_preprop.dydt
dx1 = dydt_rhs[0, :]
dx2 = dydt_rhs[1, :]

plt.plot(t, dydt_rhs.T)
plt.ylabel("derivative of the states")
plt.xlabel("time")
plt.show()


gproc = GPSignalPreprocessor(t=t, y=y[0, :])
gproc.interpolate()
gproc.calculate_time_derivative()
dydt_2 = gproc.dydt

plt.plot(dydt_2)
plt.plot(dx1)
plt.ylabel("derivative of the states")
plt.xlabel("time")
plt.show()


# putting the workflow together
# step 1 define a dictionary
d_f = ["x1", "x1*x2", "x2", "sin(x2)^3"]
dict_builder = DictionaryBuilder(dict_fcns=d_f)
print(dict_builder.dict_fcns)
data = {"x1": y[0, :], "x2": y[1, :]}
A = dict_builder.evaluate_dict(input_data=data)

# step 2 define an SBL problem with the Lin reg model and solve it
lambda_param = 0.5
sbl = SBL(dict_mtx=A, data_vec=dx1, lambda_param=lambda_param)
sbl.compute_model_structure()

print(f"weights {sbl.w_estimates[-1]}")
# step 4 reporting
# print(sbl.w_estimates)
plt.plot(sbl.w_estimates)
plt.ylabel("w est")
plt.xlabel("iterations")
plt.show()

plt.plot(sbl.z_estimates)
plt.ylabel("gamma est")
plt.xlabel("iterations")
plt.yscale("log")
plt.show()
