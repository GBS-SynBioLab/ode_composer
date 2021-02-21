#  Copyright (c) 2020. Zoltan A Tuza, Guy-Bart Stan. All Rights Reserved.
#  This code is published under the MIT License.
#  Department of Bioengineering,
#  Centre for Synthetic Biology,
#  Imperial College London, London, UK
#  contact: ztuza@imperial.ac.uk, gstan@imperial.ac.uk

from collections.abc import Iterable
import pandas as pd
import numpy as np
from pathlib import Path
from .signal_preprocessor import SplineSignalPreprocessor
from .signal_preprocessor import SmoothingSplinePreprocessor
import warnings
import matplotlib.pyplot as plt
import math


class Database(object):
    def __init__(self, structure_config):
        self.df = None
        self.structure_config = structure_config
        self.data_file = None

    @property
    def structure_config(self):
        return self._structure_config

    @structure_config.setter
    def structure_config(self, new_config):
        # TODO add checks
        self._structure_config = new_config

    def import_data(self, data_dir, data_file):
        my_file = Path(data_dir + "/" + data_file)
        self.data_file = my_file.stem
        if not my_file.is_file():
            raise FileNotFoundError(f" file {my_file} does not exist")
        self.df = pd.read_csv(my_file, skipinitialspace=True)
        # remove trailing whitespaces in the header row
        self.df.rename(columns=lambda x: x.strip(), inplace=True)

    def _find_data_label(self, data_label):
        for data_type in self.structure_config.values():
            if data_label in data_type:
                return data_type[data_label]

        return None

    @staticmethod
    def _standardize_data(data, remove_mean=False):
        data_std = np.std(data)
        if remove_mean:
            data = data - np.mean(data)
        if data_std == 0:
            return_data = data
        else:
            return_data = data / data_std
        return return_data

    def get_data(self, data_label, exp_id, standardize=True, merge_mode=None):
        if not isinstance(exp_id, Iterable):
            exp_id = [exp_id]

        if not isinstance(data_label, str):
            raise ValueError(
                "This function only supports one data label, for multiple data labels use get_multicolumn_data!"
            )
        ret = []
        for one_exp_id in exp_id:
            one_exp = self.df[
                self.df[self.structure_config["experiment_id"]["exp_id"]]
                == one_exp_id
            ]
            df_column_name = self._find_data_label(data_label=data_label)
            if df_column_name:
                data = one_exp[df_column_name]
                if standardize:
                    data = Database._standardize_data(data)
                ret.append(data)
            else:
                raise ValueError(f"Invalid data label: {data_label}")
        if merge_mode is None:
            if len(exp_id) > 1:
                warnings.warn(
                    "No merge mode was selected, but multiple exp_id was given! Only the results for the first exp_id returned!"
                )
            return ret[0]
        elif merge_mode.lower() == "stacked":
            return np.hstack(ret)
        elif merge_mode.lower() == "list":
            return ret
        else:
            raise ValueError("Invalid merge mode was given!")

    def get_multicolumn_data(self, data_labels, exp_id, **kwargs):
        ret_dict = {}
        for data_label in data_labels:
            ret_dict.update(
                {
                    data_label: self.get_data(
                        data_label=data_label, exp_id=exp_id, **kwargs
                    )
                }
            )
        return ret_dict

    def get_preprocessed_data(
        self,
        data_label,
        exp_id,
        preprocessor,
        merge_mode=None,
        standardize=True,
    ):
        if not isinstance(exp_id, Iterable):
            exp_id = [exp_id]
        # get raw data and the corresponding time vector
        y = self.get_data(
            data_label=data_label,
            exp_id=exp_id,
            merge_mode="list",
            standardize=False,
        )
        if self._find_data_label(f"{data_label}e"):
            ystd = self.get_data(
                data_label=f"{data_label}e",
                exp_id=exp_id,
                merge_mode="list",
                standardize=False,
            )
        else:
            ystd = [None] * len(y)

        t = self.get_data(
            data_label="t", exp_id=exp_id, merge_mode="list", standardize=False
        )
        ret_dict = {}
        data_list = []
        time_derivative_list = []
        for one_t, one_y, one_ystd, one_exp_id in zip(t, y, ystd, exp_id):

            if preprocessor == "SplineSignalPreprocessor":
                preprocessor_ref = SplineSignalPreprocessor
            elif preprocessor == "SmoothingSplinePreprocessor":
                preprocessor_ref = SmoothingSplinePreprocessor
            else:
                raise ValueError(f"Unknown preprocessor: {preprocessor}")

            spline_id = (
                f"{self.data_file}_{one_exp_id}_{data_label}_{preprocessor}"
            )

            if one_ystd is not None:
                weights = 1 / one_ystd
            else:
                weights = None

            spline_preproc = preprocessor_ref(
                t=one_t, y=one_y, weights=weights, spline_id=spline_id
            )
            interpolated = spline_preproc.interpolate(t_new=one_t)
            if standardize:
                interpolated = self._standardize_data(interpolated)
            data_list.append(interpolated)
            time_derivated = spline_preproc.calculate_time_derivative(
                t_new=one_t
            )
            if standardize:
                time_derivated = self._standardize_data(time_derivated)

            time_derivative_list.append(time_derivated)

        if merge_mode is None:
            if len(exp_id) > 1:
                warnings.warn(
                    "No merge mode was selected, but multiple exp_id was given! Only the results for the first exp_id returned!"
                )
            ret_dict.update({data_label: data_list[0]})
            ret_dict.update({f"d{data_label}dt": time_derivative_list[0]})
        elif merge_mode.lower() == "stacked":
            ret_dict.update({data_label: np.hstack(data_list)})
            ret_dict.update(
                {f"d{data_label}dt": np.hstack(time_derivative_list)}
            )
        elif merge_mode.lower() == "list":
            ret_dict.update({data_label: data_list})
            ret_dict.update({f"d{data_label}dt": time_derivative_list})
        else:
            raise ValueError("Invalid merge mode was given!")

        return ret_dict

    def get_preprocessed_multicolumn_data(self, data_labels, exp_id, **kwargs):
        ret_dict = {}
        for data_label in data_labels:
            ret_dict.update(
                self.get_preprocessed_data(
                    data_label=data_label, exp_id=exp_id, **kwargs
                )
            )
        return ret_dict

    def get_multicolumn_datum(self, data_labels, exp_id, index):
        if not isinstance(exp_id, int):
            raise TypeError("experiment id must be an integer!")
        data = self.get_multicolumn_data(
            data_labels=data_labels, exp_id=exp_id
        )
        return {key: datum.iloc[index] for key, datum in data.items()}
