import pandas as pd
import numpy as np

from .weighting import Weighting


class TypeSpecificWeight(Weighting):
    """
    This weighting function assigns a weight to every record so that for every vertex type a different weight is used.
    """

    def __init__(self, type_dict):
        self.type_dict = type_dict

    def compute(self, reference_feature_values, time_window):
        if not isinstance(reference_feature_values, pd.DataFrame):
            raise TypeError
        if not isinstance(time_window, int):
            raise TypeError
        if reference_feature_values.empty:
            raise ValueError
        if time_window <= 0:
            raise ValueError

        if not set(list(reference_feature_values['type'])).issubset(set(self.type_dict.keys())):
            raise ValueError('Reference feature values contain a non specified type')

        reference_feature_values.fillna(0)
        weight_df = pd.DataFrame(reference_feature_values['time_window'], columns=['time_window'], dtype=np.int)
        weight_df['weight'] = reference_feature_values['type'].replace(self.type_dict)
        # weight_df['weight'] = weight_df['weight'].clip(lower=0.0)
        return weight_df
