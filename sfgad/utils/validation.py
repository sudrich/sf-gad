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


def check_observations(observations, n_features=None):
    observations = np.atleast_2d(observations)
    observations = np.array(observations, dtype=np.float64)

    if observations.ndim > 2:
        raise ValueError("Found array with dim %d, but expected <= 2." % observations.ndim)

    if observations.shape[1] != n_features:
        raise ValueError("Found array of shape %s, but expected (n, %d)." % (observations.shape, n_features))

    return observations


def check_reference_observations(ref_observations):
    ref_observations = np.atleast_2d(ref_observations)
    ref_observations = np.array(ref_observations, dtype=np.float64)

    if ref_observations.ndim > 2:
        raise ValueError("Found array with dim %d, but expected <= 2." % ref_observations.ndim)

    if ref_observations.shape[1] == 0:
        raise ValueError("Found empty array.")

    return ref_observations


def check_weights(weights, ref_observations):
    weights = np.array(weights, dtype=np.float64)

    if weights.ndim != 1:
        raise ValueError("Found array with dim %d, but expected 1." % weights.ndim)

    if len(weights) != len(ref_observations):
        raise ValueError("Length of supplied weights is not consistent with length of reference observations.")

    return weights


def check_is_fitted(estimator, attributes):
    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % estimator)

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all([hasattr(estimator, attr) for attr in attributes]):
        raise ValueError("This %s is not fitted yet. Call 'fit' with appropriate arguments before using this method."
                         % type(estimator).__name__)
