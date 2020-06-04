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
    A = [1, 2, 4]
    y = np.array([1, 2, 2])
    with pytest.raises(TypeError):
        LinearModel(dict_mtx=A, data_vec=y)

    with pytest.raises(TypeError):
        A = np.array([[1, 2], [3, 4], [3, 4]])
        y = [1, 2, 2]
        LinearModel(dict_mtx=A, data_vec=y)


def test_linear_parameter_and_data_num():
    A = np.array([[1, 2], [3, 4], [3, 4]])
    y = np.array([1, 2, 2])
    lin_model = LinearModel(dict_mtx=A, data_vec=y)

    assert lin_model.parameter_num == lin_model.dict_mtx.shape[1]
    assert lin_model.data_num == lin_model.dict_mtx.shape[0]


def test_incompatible_parameters():
    """data vector update must check the new data vector size compatibility"""
    A = np.array([[1, 2], [3, 4], [3, 4]])
    y = np.array([1, 2, 2])
    lin_model = LinearModel(dict_mtx=A, data_vec=y)

    with pytest.raises(
        ValueError,
        match=r"data has \d+ rows, but the regressor mtx has \d+ columns$",
    ):
        lin_model.data_vec = np.array([1, 2])
