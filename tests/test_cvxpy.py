#  Copyright (c) 2021. Zoltan A Tuza, Guy-Bart Stan. All Rights Reserved.
#  This code is published under the MIT License.
#  Department of Bioengineering,
#  Centre for Synthetic Biology,
#  Imperial College London, London, UK
#  contact: ztuza@imperial.ac.uk, gstan@imperial.ac.uk

import numpy as np
import cvxpy as cp


def test_lsq():
    A = np.array([[1, 2], [2, 7], [2, 5.65]])
    # standarize the columns of the regressor matrix
    A[:, 0] = A[:, 0] / np.std(A[:, 0])
    A[:, 1] = A[:, 1] / np.std(A[:, 1])
    # measurement data
    y = np.array([1, 2, 2])

    w = cp.Variable(2)
    cost = cp.Minimize(cp.sum_squares(A @ w - y))

    problem = cp.Problem(objective=cost, constraints=[])
    problem.solve()
    assert problem.status == cp.OPTIMAL
    # check that the second column's weights is zero
    assert w.value[1] < 1e-8


def test_lasso():
    A = np.array([[1, 2], [2, 7], [2, 5.65]])
    # standarize the columns of the regressor matrix
    A[:, 0] = A[:, 0] / np.std(A[:, 0])
    A[:, 1] = A[:, 1] / np.std(A[:, 1])
    # measurement data
    y = np.array([1, 2, 2])

    w2 = cp.Variable(2)
    cost2 = cp.Minimize(
        cp.sum_squares(A @ w2 - y) + cp.abs(w2[0]) + cp.abs(w2[1])
    )

    problem2 = cp.Problem(objective=cost2, constraints=[])
    problem2.solve()
    assert problem2.status == cp.OPTIMAL
    assert w2.value[1] < 1e-8
