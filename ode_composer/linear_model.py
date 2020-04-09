import numpy as np


class LinearModel(object):
    def __init__(
        self, dict_mtx: np.ndarray, data_vec: np.ndarray, lambda_param: float
    ):
        self.dict_mtx = dict_mtx
        self.data_vec = data_vec
        param_num = dict_mtx.shape[1]
        self.w = np.empty((param_num, 1))
        self.w[:] = np.nan
        self.lambda_param = lambda_param

    @property
    def dict_mtx(self) -> np.ndarray:
        return self._dict_mtx

    @dict_mtx.setter
    def dict_mtx(self, new_dict_mtx: np.ndarray):
        if not isinstance(new_dict_mtx, np.ndarray):
            raise TypeError(
                "%s is not numpy ndarray, but %s"
                % (new_dict_mtx, type(new_dict_mtx))
            )

        self._dict_mtx = new_dict_mtx

    @property
    def data_vec(self) -> np.ndarray:
        return self._data_vec

    @data_vec.setter
    def data_vec(self, new_data_vec: np.ndarray):
        if not isinstance(new_data_vec, np.ndarray):
            raise TypeError(
                "%s is not numpy nd array, but %s" % new_data_vec,
                type(new_data_vec),
            )
        if self.dict_mtx.shape[0] != new_data_vec.shape[0]:
            raise ValueError(
                "data has %d rows, but the regressor mtx has %d columns"
                % (new_data_vec.shape[0], self.dict_mtx.shape[0])
            )
        self._data_vec = new_data_vec

    @property
    def lambda_param(self) -> float:
        return self._lambda_param

    @lambda_param.setter
    def lambda_param(self, new_lambda_param: float):
        if not isinstance(new_lambda_param, float):
            raise TypeError(
                "lambda param is %s, it must be int!" % new_lambda_param
            )
        if new_lambda_param < 0:
            raise ValueError(
                "lambda param must be non-negative, not %s!" % new_lambda_param
            )

        self._lambda_param = new_lambda_param

    @property
    def parameter_num(self):
        return self.dict_mtx.shape[1]

    @property
    def data_num(self):
        return self.dict_mtx.shape[0]
