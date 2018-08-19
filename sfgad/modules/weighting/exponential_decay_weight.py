import math

import numpy as np
import pandas as pd

from .weighting import Weighting


class ExponentialDecayWeight(Weighting):
    """
    This weighting function assigns a weight to every record so that for every window step the weight is exponentially
    reduced
    """

    def __init__(self, half_life, lower_threshold=0.01):
        self.decay_lambda = math.log(0.5) / half_life
        self.lower_threshold = lower_threshold

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
        weight_df['weight'] = np.exp(-1 * self.decay_lambda * (weight_df['time_window'] - time_window))
        weight_df.loc[weight_df['weight'] < self.lower_threshold, 'weight'] = 0.0

        return weight_df
