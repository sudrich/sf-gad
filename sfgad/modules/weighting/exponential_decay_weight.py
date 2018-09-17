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
        self.decay_lambda = math.log(0.5) / half_life

    def compute(self, reference_meta_info, current_meta_info):
        check_meta_info_dataframe(reference_meta_info, required_columns=['time_window'])
        check_meta_info_series(current_meta_info, required_columns=['time_window'])

        return np.exp(-1 * self.decay_lambda * (reference_meta_info['time_window'] - current_meta_info['time_window']))
