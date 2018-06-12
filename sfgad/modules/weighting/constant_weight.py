import pandas as pd
import numpy as np

from .weighting import Weighting


class ConstantWeight(Weighting):
    """
    This weighting function assigns a constant default weight of to every record
    """

    def __init__(self, default_weight=1):
        self.weight = default_weight

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
        weight_df = pd.DataFrame(reference_feature_values['time_window'],
                                 columns=['time_window'], dtype=np.int)
        weight_df['weight'] = self.weight
        return weight_df
