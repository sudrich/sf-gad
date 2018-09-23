import numpy as np

from sfgad.utils.validation import check_is_fitted, check_weights, check_observations, check_reference_observations
from .probability_estimator import ProbabilityEstimator


class EmpiricalEstimator(ProbabilityEstimator):
    def __init__(self, direction='right-tailed'):
        if direction not in ['right-tailed', 'left-tailed', 'two-tailed']:
            raise ValueError("The given direction for probability calculation is not known! "
                             "Possible directions are: 'right-tailed', 'left-tailed' & 'two-tailed'.")

        self.direction = direction

    def fit(self, reference_observations, weights):
        reference_observations = check_reference_observations(reference_observations)
        weights = check_weights(weights, reference_observations)

        sort_index = np.argsort(reference_observations, axis=0)
        # Sort the reference observations in ascending order
        self.reference_observations = np.take_along_axis(reference_observations, sort_index, axis=0)
        # Use the same order to sort the weights and accumulate them
        self.cum_weights = np.cumsum(weights[sort_index], axis=0)
        # Insert a sentinel row with 0 to the beginning of the cumulated weights
        self.cum_weights = np.insert(self.cum_weights, 0, 0, axis=0)

    def transform(self, observations):
        check_is_fitted(self, ["reference_observations", "cum_weights"])
        observations = check_observations(observations, n_features=len(self.reference_observations.T))

        p_values = np.empty_like(observations.T)
        for i, (x, y, w) in enumerate(zip(self.reference_observations.T, observations.T, self.cum_weights.T)):
            if self.direction == 'right-tailed':
                p_values[i] = 1 - w[np.searchsorted(x, y, side="left")] / w[-1]
            elif self.direction == 'left-tailed':
                p_values[i] = w[np.searchsorted(x, y, side="right")] / w[-1]
            else:
                p_values_right = 1 - w[np.searchsorted(x, y, side="left")] / w[-1]
                p_values_left = w[np.searchsorted(x, y, side="right")] / w[-1]
                p_values[i] = np.clip(2 * np.minimum(p_values_right, p_values_left), 0.0, 1.0)

        # Fill all nan values with 1.0. This happens if the standard deviation is zero.
        p_values[np.isnan(p_values)] = 1.0

        return p_values.T

    def estimate(self, features_values, reference_features_values, weights):
        """
        Takes a vertex and a reference to the database.
        Returns a list of p_values with a p_value for each given feature value in features_values. These p_values are
        calculated based on the given reference_features_values.
        :param features_values: (1 x n) dataframe with a value for each feature. Each column refers to a feature, so n
        is the number of features (no 'name'-column).
        :param reference_features_values: (m x n+1) dataframe of tuples of windows and the corresponding feature values.
        :param weights: (m x 2) dataframe with weights for the different windows.
        :return: List of p_values.
        """

        self.fit(reference_features_values, weights)
        return self.transform(features_values)
