#  Copyright (c) 2020. Zoltan A Tuza, Guy-Bart Stan. All Rights Reserved.
#  This code is published under the MIT License.
#  Department of Bioengineering,
#  Centre for Synthetic Biology,
#  Imperial College London, London, UK
#  contact: ztuza@imperial.ac.uk, gstan@imperial.ac.uk

import pytest
import numpy as np
import cvxpy as cp
from ode_composer.sbl import SBL
from ode_composer.util import MultiVariableFunction


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
