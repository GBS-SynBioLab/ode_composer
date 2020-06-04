from ode_composer.sbl import *
from ode_composer.dictionary_builder import *
import matplotlib.pyplot as plt


# A = np.array([[1, 2], [3, 4], [3, 4]])
# # print(A)
# # y = np.array([1, 2, 2])
# #
# # lambda_param = 0.5
# # linmodel = LinearModel(A, y, lambda_param)
# #
# #
# # sbl = SBL(linear_model=linmodel)
# #
# # sbl.compute_model_structure()
# # print(sbl.w_estimates)
# # print(sbl.z_estimates)

A = np.array([[1, 1], [1, 5], [1, 4]])
print(f" regressor mtx\n {A}")
y = np.array([2, 2, 2])

lambda_param = 0.5

sbl = SBL(dict_mtx=A, data_vec=y, lambda_param=lambda_param)


sbl.compute_model_structure()
print(f"weights {sbl.w_estimates[-1]}")
print(f"gamma {sbl.z_estimates[-1]}")


#
# d_f = ["x1*x3*x2^2/x1", "x1*x2", "x1^2"]
# dict_builder = DictionaryBuilder(dict_fcns=d_f)
# # print(dict_builder.dict_fcns)
#
#
# data = {"x1": [1, 2, 6, 4, 5], "x2": [3, 5, 6, 7, 8], "x3": [3, 3, 2, 5, 6]}
# dict_builder.evaluate_dict(measurement_data=data)
#
# A = dict_builder.get_regression_matrix()
# print(A)
# y = np.array([1.0, 0.2, 3.4, 3.4, 4.3])
#
# lambda_param = 0.5
# linmodel = LinearModel(A, y, lambda_param)
#
# sbl = SBL(linear_model=linmodel)
#
# sbl.compute_model_structure()
#
# plt.plot(sbl.w_estimates)
# plt.ylabel("some numbers")
# plt.show()
#
# plt.plot(sbl.z_estimates)
# plt.ylabel("some numbers")
# plt.yscale("log")
# plt.show()
