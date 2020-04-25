from ode_composer.statspace_model import *
from scipy.integrate import solve_ivp
from ode_composer.dictionary_builder import *
from ode_composer.sbl import *


import numpy as np
import matplotlib.pyplot as plt

# define a MAK model
states = {"x1": {"x1": "m11"}, "x2": {"x1": "m21", "x2": "m22"}}
parameters = {"m11": -0.5, "m12": 0.1, "m21": 0.6, "m22": -0.3}
ss = StateSpaceModel(states=states, parameters=parameters)
print(ss)

# evaluate the RHS of the ODE
ss.get_rhs(t=0, y=[0, 1], states=["x1", "x2"])

states = ["x1", "x2"]
parameters = None
t = np.linspace(0, 10, 101)
y0 = [1.2, 1]
sol = solve_ivp(ss.get_rhs, [0, 100], y0, args=(states,))
# print(sol)
plt.plot(sol.t, sol.y[0, :], "b", label="x_1(t)")
plt.plot(sol.t, sol.y[1, :], "g", label="x_2(t)")
plt.legend(loc="best")
plt.xlabel("time")
plt.ylabel("concentration")
plt.grid()
plt.show()


d = sol.y
rr = list()
for y in zip(*d):
    rr.append(ss.get_rhs(0, y, states))


dx1 = list(zip(*rr))[0]
dx2 = list(zip(*rr))[1]
plt.plot(dx1)
plt.plot(dx2)
plt.ylabel("derivative of the states")
plt.xlabel("time")
plt.show()


# putting the workflow together
# step 1 define a dictionary
d_f = ["x1", "x1*x2", "x2", "sin(x2)^3"]
dict_builder = DictionaryBuilder(dict_fcns=d_f)
print(dict_builder.dict_fcns)
data = {"x1": sol.y[0, :], "x2": sol.y[1, :]}
dict_builder.evaluate_dict(input_data=data)

A = dict_builder.get_regression_matrix()
# print(f'regressor matrix {A}')

# step 2 define a linear regression model
lambda_param = 0.4
linmodel = LinearModel(A, np.asarray(dx1, dtype=float), lambda_param)

# step 3 define an SBL problem with the Lin reg model and solve it
sbl = SBL(linear_model=linmodel)
sbl.compute_model_structure()

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


print(f"weights {sbl.w_estimates[-1]}")
