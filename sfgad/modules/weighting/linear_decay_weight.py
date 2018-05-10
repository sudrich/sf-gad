import pandas as pd
import numpy as np

from .weighting import Weighting


class LinearDecayWeight(Weighting):
    """
    This weighting function assigns a weight to every record so that for every window step the weight is linearly
     reduced
    """

    def __init__(self, factor=0.5):
        self.factor = factor

    def compute(self, reference_feature_values, time_window):
        if not isinstance(reference_feature_values, pd.DataFrame):
            raise TypeError
        if not isinstance(time_window, int):
            raise TypeError
        if reference_feature_values.empty:
            raise ValueError
        if time_window <= 0:
            raise ValueError

        reference_feature_values.fillna(0)
        weight_df = pd.DataFrame(reference_feature_values['time_window'], columns=['time_window'], dtype=np.int)
        weight_df['weight'] = 1 - (time_window - weight_df['time_window']) * self.factor
        weight_df['weight'] = weight_df['weight'].clip(lower=0.0)
        return weight_df