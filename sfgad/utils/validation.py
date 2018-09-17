import numpy as np


def check_p_values(p_values):
    p_values = np.atleast_2d(p_values)
    p_values = np.array(p_values, dtype=np.float64)

    if p_values.ndim > 2:
        raise ValueError("Found array with dim %d, but expected <= 2." % p_values.ndim)

    if p_values.shape[1] == 0:
        raise ValueError("Found empty array.")

    return p_values
