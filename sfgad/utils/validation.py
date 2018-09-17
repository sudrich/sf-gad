import numpy as np
import pandas as pd


def check_p_values(p_values):
    p_values = np.atleast_2d(p_values)
    p_values = np.array(p_values, dtype=np.float64)

    if p_values.ndim > 2:
        raise ValueError("Found array with dim %d, but expected <= 2." % p_values.ndim)

    if p_values.shape[1] == 0:
        raise ValueError("Found empty array.")

    return p_values


def check_meta_info_series(meta_info, required_columns=()):
    if not isinstance(meta_info, pd.Series):
        raise TypeError("Found input of type %s, but expected a Series." % type(meta_info))

    if not set(required_columns).issubset(meta_info.index):
        raise ValueError("Found Series with the required columns %s missing." % set(required_columns).difference(
            meta_info.index))

    return meta_info


def check_meta_info_dataframe(meta_info, required_columns=()):
    if not isinstance(meta_info, pd.DataFrame):
        raise TypeError("Found input of type %s, but expected a DataFrame." % type(meta_info))

    if not set(required_columns).issubset(meta_info.columns):
        raise ValueError("Found DataFrame with the required columns %s missing." % set(required_columns).difference(
            meta_info.columns))

    return meta_info
