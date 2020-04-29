from ode_composer.linear_model import LinearModel
import pytest
import numpy as np


def test_linear_model():
    A = np.array([[1, 2], [3, 4], [3, 4]])
    y = np.array([1, 2, 2])
    lin_model = LinearModel(dict_mtx=A, data_vec=y)

    assert lin_model.dict_mtx.shape[0] == lin_model.data_vec.shape[0]
    assert lin_model.dict_mtx.shape[1] == lin_model.w.shape[0]


def test_linear_model_type_checking():
    pass


# with pytest.raises(ValueError, match=r".* 123 .*"):


def test_linear_model_lambda():
    pass
