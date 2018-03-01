import abc
import math
import pandas as pd
import numpy as np


class WeightingFunction:
    @abc.abstractmethod
    def compute(self, reference_feature_values, time_window):
        """
        Takes a vertex and a reference to the database.
        Returns a dataframe of all the relevant vertices that are needed for calculating p_value of the vertex
        :param reference_feature_values: List of tuples of windows and the corresponding feature values
        :param time_window: The current time window
        :return: Dataframe with the columns (window, weight)
        """


class ConstantWeight(WeightingFunction):
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
        weight_df = pd.DataFrame(reference_feature_values['time_window'], columns=['time_window'], dtype=np.int)
        weight_df['weight'] = self.weight
        return weight_df


class LinearDecayWeight(WeightingFunction):
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


class ExponentialDecayWeight(WeightingFunction):
    """
    This weighting function assigns a weight to every record so that for every window step the weight is exponentially
     reduced
    """

    def __init__(self, half_life, lower_threshold=0.01):
        self.decay_lambda = math.log(0.5) / half_life * -1
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
        weight_df[weight_df['weight'] < self.lower_threshold] = 0.0

        return weight_df
