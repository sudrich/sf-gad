import pandas as pd
import numpy as np

from .probability_estimator import ProbabilityEstimator


class Empirical(ProbabilityEstimator):

    def __init__(self, direction='right', mode="one-tailed"):
        self.direction = direction

        if mode == "one-tailed":
            self.mode = 1
        elif mode == "two-tailed":
            self.mode = 2
        else:
            raise ValueError("The supplied mode of operation for empirical calculation is not known.")

    def estimate(self, vertex_name, features, reference_feature_values, weights):

        # Add the weights to a combined dataframe of reference_values and weights
        df = pd.merge(reference_feature_values, weights, on="time_window")

        # Get a list of all the features for building an easy iterable
        features_list = features.columns.values.tolist()
        features_list.remove('name')

        p_values_list = []

        for feature_name in features_list:

            # This is the feature value for the current feature of the vertex in question
            feature_value = features.iloc[0][feature_name]

            if self.mode == 1:
                p_value = self.empirical(feature_value, df[feature_name], df['weight'], self.direction)
            else:
                p_value_right = self.empirical(feature_value, df[feature_name], df['weight'], 'right')
                p_value_left = self.empirical(feature_value, df[feature_name], df['weight'], 'left')

                p_value = 2 * min(p_value_right, p_value_left)

            # Add the calculated p_value to the list of p_values for this vertex
            p_values_list.append(p_value)

        # Return the completed list
        return p_values_list

    def empirical(self, value, references, weights, direction):
        """
        Execute the empirical p-value calculation.
        :param value: the given minimal p-value
        :param references: the minimal p_values of the reference observations
        :param weights: weights for the reference observations
        :param direction: direction for the empirical calculation.
        :return: the empirical p_value
        """
        isnan = np.isnan(references)

        if direction == 'right':
            conditions = references[~isnan] >= value
        elif direction == 'left':
            conditions = references[~isnan] <= value
        else:
            raise ValueError("The given direction for empirical calculation is not known.")


        sum_all_weights = weights[~isnan].sum()
        sum_conditional_weights = (conditions * weights[~isnan]).sum()

        if sum_all_weights == 0:
            return np.nan

        return sum_conditional_weights / sum_all_weights