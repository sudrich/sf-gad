import numpy as np
import scipy.stats as st

from sfgad.utils.validation import check_is_fitted, check_weights, check_observations, check_reference_observations
from .probability_estimator import ProbabilityEstimator


class Uniform(ProbabilityEstimator):
    def __init__(self, direction='right-tailed'):
        if direction not in ['right-tailed', 'left-tailed', 'two-tailed']:
            raise ValueError("The given direction for probability calculation is unknown! "
                             "Possible directions are: 'right-tailed', 'left-tailed' & 'two-tailed'.")

        self.direction = direction

    def fit(self, reference_observations, weights):
        reference_observations = check_reference_observations(reference_observations)
        weights = check_weights(weights, reference_observations)

        self.mins = np.min(reference_observations, axis=0)
        self.maxs = np.max(reference_observations, axis=0)

    def transform(self, observations):
        check_is_fitted(self, ["mins", "maxs"])
        observations = check_observations(observations, n_features=len(self.mins))

        if self.direction == 'right-tailed':
            p_values = 1 - st.uniform.cdf(observations, self.mins, self.maxs - self.mins)
        elif self.direction == 'left-tailed':
            p_values = st.uniform.cdf(observations, self.mins, self.maxs - self.mins)
        else:
            p_values_right = 1 - st.uniform.cdf(observations, self.mins, self.maxs - self.mins)
            p_values_left = st.uniform.cdf(observations, self.mins, self.maxs - self.mins)
            p_values = 2 * np.minimum(p_values_right, p_values_left)

        # Fill all nan values with 1.0. This happens if the standard deviation is zero.
        p_values[np.isnan(p_values)] = 1.0

        return p_values

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
