import numpy as np
import pandas as pd
import scipy.stats as st

from .probability_estimator import ProbabilityEstimator


class Uniform(ProbabilityEstimator):
    def __init__(self, direction='left-tailed'):

        if direction not in ['right-tailed', 'left-tailed', 'two-tailed']:
            raise ValueError("The given direction for probability calculation is not known! "
                             "Possible directions are: 'right-tailed', 'left-tailed' & 'two-tailed'.")

        self.direction = direction

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

        ### INPUT VALIDATION

        if not isinstance(features_values, pd.DataFrame) \
                or not isinstance(reference_features_values, pd.DataFrame) \
                or not isinstance(weights, np.ndarray):
            raise ValueError("The given arguments 'feature_values', 'reference_features_values' should "
                             "be a DataFrame and 'weights' a numpy array.")

        # check that features_values has 1 row and >= 1 columns
        if features_values.shape[0] != 1:
            raise ValueError("The given argument 'feature_values' should have exactly 1 row with data and a column for "
                             "each feature!")

        # check that reference_features_values has >= 1 rows and n+1 columns
        if not reference_features_values.shape[0] >= 1 \
                or reference_features_values.shape[1] != (features_values.shape[1] + 1):
            raise ValueError("The given argument 'reference_features_values' should have >= 1 rows with data and a "
                             "column for each feature and the time_window!")

        # check that weights has m rows and 2 columns
        if weights.shape[0] != reference_features_values.shape[0]:
            raise ValueError(
                "The given argument 'weights' should have exactly the same number of rows as "
                "'reference_features_values'!")

        # check that reference_features_values has the column 'time_window'
        if 'time_window' not in reference_features_values.columns.values.tolist():
            raise ValueError("The given argument 'reference_features_values' should have the column 'time_window'!")

        # check that the feature names in features_values and reference_features_values are the same
        if set().union(*[features_values.columns.values.tolist(), ['time_window']]) != \
                set(reference_features_values.columns.values.tolist()):
            raise ValueError("The given arguments 'feature_values' & 'reference_features_values' should have the same "
                             "columns with feature names!")

        # check that the values of each feature are all floats (or integers)
        for feature_name in features_values.columns.values.tolist():
            if not isinstance(features_values.iloc[0][feature_name], (int, np.int64, float, np.float64)):
                raise ValueError(
                    "The values of each feature in features_values should all be of the type 'int' or 'float'")
            if not all(
                    isinstance(x, (int, np.int64, float, np.float64)) for x in reference_features_values[feature_name]):
                raise ValueError(
                    "The values of each feature in reference_features_values should all be of the type 'int' or "
                    "'float'")

        # check that the values of the time windows are all floats (or integers)
        if not all(isinstance(x, (int, np.int64, float, np.float64)) for x in reference_features_values['time_window']):
            raise ValueError(
                "The values of the time windows in reference_features_values should all be of the type 'int' or "
                "'float'!")

        ### FUNCTION CODE

        # Get a list of all the features for building an easy iterable
        features_list = features_values.columns.values.tolist()

        p_values_list = []

        for feature_name in features_list:

            # This is the feature value for the current feature of the vertex in question
            feature_value = features_values.iloc[0][feature_name]

            min_value = min(feature_value, min(reference_features_values[feature_name]))
            max_value = max(feature_value, max(reference_features_values[feature_name]))

            if self.direction == 'right-tailed':
                p_value = 1 - st.uniform.cdf(feature_value, min_value, max_value - min_value)

            elif self.direction == 'left-tailed':
                p_value = st.uniform.cdf(feature_value, min_value, max_value - min_value)

            else:
                p_value_right = 1 - st.uniform.cdf(feature_value, min_value, max_value - min_value)
                p_value_left = st.uniform.cdf(feature_value, min_value, max_value - min_value)

                p_value = 2 * min(p_value_right, p_value_left)

            # Add the calculated p_value to the list of p_values for this vertex
            p_values_list.append(p_value)

        # Return the completed list
        return p_values_list
