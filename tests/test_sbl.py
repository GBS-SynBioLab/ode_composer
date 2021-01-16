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
