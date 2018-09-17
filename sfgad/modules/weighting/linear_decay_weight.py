import numpy as np

from sfgad.utils.validation import check_meta_info_dataframe, check_meta_info_series
from .weighting import Weighting


class LinearDecayWeight(Weighting):
    """
    This weighting function assigns a weight to every record so that for every window step the weight is linearly
    reduced
    """

    def __init__(self, factor=0.5):
        self.factor = factor

    def compute(self, reference_meta_info, current_meta_info):
        check_meta_info_dataframe(reference_meta_info, required_columns=['time_window'])
        check_meta_info_series(current_meta_info, required_columns=['time_window'])

        return np.clip(1 - self.factor * (current_meta_info['time_window'] - reference_meta_info['time_window']),
                       a_min=0.0, a_max=None)
