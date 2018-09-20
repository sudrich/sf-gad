import math

import numpy as np

from sfgad.utils.validation import check_meta_info_series, check_meta_info_dataframe
from .weighting import Weighting


class ExponentialDecayWeight(Weighting):
    """
    This weighting function assigns a weight to every record so that for every window step the weight is exponentially
    reduced
    """

    def __init__(self, half_life):
        if half_life <= 0:
            raise ValueError("The half-life period was %d, but must be a number >= 0." % half_life)
        self.decay_lambda = math.log(2) / half_life

    def compute(self, reference_meta_info, current_meta_info):
        check_meta_info_dataframe(reference_meta_info, required_columns=['time_window'])
        check_meta_info_series(current_meta_info, required_columns=['time_window'])

        return np.exp(self.decay_lambda * (reference_meta_info['time_window'] - current_meta_info['time_window']))
