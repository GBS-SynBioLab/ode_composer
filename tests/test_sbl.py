#  Copyright (c) 2020. Zoltan A Tuza, Guy-Bart Stan. All Rights Reserved.
#  This code is published under the MIT License.
#  Department of Bioengineering,
#  Centre for Synthetic Biology,
#  Imperial College London, London, UK
#  contact: ztuza@imperial.ac.uk, gstan@imperial.ac.uk

import pytest
import numpy as np
import cvxpy as cp

from ode_composer.util import MultiVariableFunction
from ode_composer.statespace_model import StateSpaceModel
from ode_composer.measurements_generator import MeasurementsGenerator
from ode_composer.signal_preprocessor import RHSEvalSignalPreprocessor
from ode_composer.dictionary_builder import DictionaryBuilder
from ode_composer.sbl import SBL


def test_solver_not_found():
    dummy_solver = "dummy_solver"
    config = {
        "nonnegative": True,
        "verbose": True,
        "solver": {"name": dummy_solver},
    }
    A = np.array([[1, 2], [3, 4], [3, 4]])
    y = np.array([1, 2, 2])
    x1 = MultiVariableFunction.create_function(
        rhs_fcn="x1", weight=1, parameters={}
    )
    x2 = MultiVariableFunction.create_function(
        rhs_fcn="x2", weight=1, parameters={}
    )
    sbl = SBL(
        dict_mtx=A,
        data_vec=y,
        lambda_param=1.0,
        dict_fcns=[x1, x2],
        state_name="x1",
        config=config,
    )
    with pytest.raises(
        cp.error.SolverError,
        match=f"The solver {dummy_solver} is not installed.",
    ):
        sbl.compute_model_structure()


def test_sparsity():
    A = np.array([[1, 2], [2, 7], [2, 5.65]])
    # standarize the columns of the regressor matrix
    A[:, 0] = A[:, 0] / np.std(A[:, 0])
    A[:, 1] = A[:, 1] / np.std(A[:, 1])
    # measurement data
    y = np.array([1, 2, 2])
    x1 = MultiVariableFunction.create_function(
        rhs_fcn="x1", weight=1, parameters={}
    )
    x2 = MultiVariableFunction.create_function(
        rhs_fcn="x2", weight=1, parameters={}
    )
    dict_fcns = [x1, x2]
    sbl = SBL(
        dict_mtx=A,
        data_vec=y,
        lambda_param=1,
        dict_fcns=dict_fcns,
        state_name="x1",
    )
    sbl.compute_model_structure()

    w_est = sbl.w_estimates[-1]
    # the second column's weights is below the zero thrreshold
    assert abs(w_est[1]) < 1e-8


def test_sbl_on_lotka_volterra():
    # define Lotka-Volerra model
    states = {"x1": "alpha*x1-beta*x1*x2", "x2": "delta*x1*x2-gamma*x2"}
    parameters = {"alpha": 2 / 3, "beta": 4 / 3, "delta": 1, "gamma": 1}
    ss = StateSpaceModel.from_string(states=states, parameters=parameters)

    states = ["x1", "x2"]
    t_span = [0, 30]
    x0 = {"x1": 1.2, "x2": 1.0}
    # simulate model
    gm = MeasurementsGenerator(ss=ss, time_span=t_span, initial_values=x0)
    t, y = gm.get_measurements()

    # compute the time derivative of the state variables
    rhs_preprop = RHSEvalSignalPreprocessor(
        t=t, y=y, rhs_function=ss.get_rhs, states=states
    )
    rhs_preprop.calculate_time_derivative()
    dydt_rhs = rhs_preprop.dydt
    dx1 = dydt_rhs[0, :]
    dx2 = dydt_rhs[1, :]

    # step 1 define a dictionary
    d_f = ["x1", "x2", "x1*x1", "x1*x2", "x2*x2"]
    dict_builder = DictionaryBuilder(dict_fcns=d_f)
    dict_functions = dict_builder.dict_fcns
    data = {"x1": y[0, :], "x2": y[1, :]}
    A = dict_builder.evaluate_dict(input_data=data)

    # step 2 define an SBL problem and solve it
    lambda_param_x1 = 0.1
    lambda_param_x2 = 0.05
    sbl_x1 = SBL(
        dict_mtx=A,
        data_vec=dx1,
        lambda_param=lambda_param_x1,
        state_name="x1",
        dict_fcns=dict_functions,
    )
    sbl_x1.compute_model_structure()

    sbl_x2 = SBL(
        dict_mtx=A,
        data_vec=dx2,
        lambda_param=lambda_param_x2,
        state_name="x2",
        dict_fcns=dict_functions,
    )
    sbl_x2.compute_model_structure()

    # test SBL results
    zero_th = 1e-8
    x1_results = [
        str(item.symbolic_expression)
        for item in sbl_x1.get_results(zero_th=zero_th)
    ]
    assert len(x1_results) == 2
    assert "x1" in x1_results
    assert "x1*x2" in x1_results

    x2_results = [
        str(item.symbolic_expression)
        for item in sbl_x2.get_results(zero_th=zero_th)
    ]
    assert len(x2_results) == 2
    assert "x2" in x2_results
    assert "x1*x2" in x2_results
