from unittest import TestCase

import numpy as np
import pandas as pd

from sfgad.modules.probability_estimation.gaussian import Gaussian


class TestGaussian(TestCase):
    def setUp(self):
        self.estimator = Gaussian()

        self.features_values = pd.DataFrame(
            data={'Feature_A': [42], 'Feature_B': [1]},
            columns=['Feature_A', 'Feature_B'])

        self.reference_features_values = pd.DataFrame(
            data={'Feature_A': [40, 45, 42, 39, 43], 'Feature_B': [0, 1, 0, 1, 2], 'time_window': [0, 1, 2, 3, 4]},
            columns=['Feature_A', 'Feature_B', 'time_window'])

        self.weights = np.ones(5, dtype=int)

    def test_estimator_output(self):
        # test the right output with direction='left-tailed'
        np.testing.assert_almost_equal(
            self.estimator.estimate(self.features_values, self.reference_features_values, self.weights),
            [0.5373, 0.6054], 4)

    def test_direction_right(self):
        self.estimator = Gaussian(direction='right-tailed')

        # test the right output with direction='right-tailed'
        np.testing.assert_almost_equal(
            self.estimator.estimate(self.features_values, self.reference_features_values, self.weights),
            [0.4627, 0.3946], 4)

    def test_direction_two_tailed(self):
        self.estimator = Gaussian(direction='two-tailed')

        # test the right output with direction='two-tailed'
        np.testing.assert_almost_equal(
            self.estimator.estimate(self.features_values, self.reference_features_values, self.weights),
            [0.9254, 0.7893], 4)

    def test_wrong_direction(self):
        # expect a value error
        self.assertRaises(ValueError, Gaussian, direction='up')

    def test_wrong_input_no_dataframe(self):
        # parameter is not a dataframe

        # expect a value error
        self.assertRaises(ValueError, self.estimator.estimate, 42, self.reference_features_values,
                          self.weights)

        # expect a value error
        self.assertRaises(ValueError, self.estimator.estimate, self.features_values, 42,
                          self.weights)

        # expect a value error
        self.assertRaises(ValueError, self.estimator.estimate, self.features_values, self.reference_features_values,
                          42)

    def test_wrong_input_features_values(self):
        # features_values has != 1 row
        features_values_1 = pd.DataFrame(
            columns=['Feature_A', 'Feature_B'])

        features_values_2 = pd.DataFrame(
            data={'Feature_A': [42, 43], 'Feature_B': [0, 2]},
            columns=['Feature_A', 'Feature_B'])

        # wrong value type
        features_values_3 = pd.DataFrame(
            data={'Feature_A': ['String'], 'Feature_B': [0]},
            columns=['Feature_A', 'Feature_B'])

        # expect a value error on features_values_1
        self.assertRaises(ValueError, self.estimator.estimate, features_values_1, self.reference_features_values,
                          self.weights)

        # expect a value error in features_values_2
        self.assertRaises(ValueError, self.estimator.estimate, features_values_2, self.reference_features_values,
                          self.weights)

        # expect a value error in features_values_3
        self.assertRaises(ValueError, self.estimator.estimate, features_values_3, self.reference_features_values,
                          self.weights)

    def test_wrong_input_reference_features_values(self):
        # reference_features_values has 0 rows
        reference_features_values_1 = pd.DataFrame(
            columns=['Feature_A', 'Feature_B'])

        # reference_features_values has too many columns
        reference_features_values_2 = pd.DataFrame(
            data={'Feature_A': [40, 45, 42, 39, 43], 'Feature_B': [0, 1, 0, 1, 2], 'Feature_C': [0, 1, 0, 1, 2],
                  'time_window': [0, 1, 2, 3, 4]},
            columns=['Feature_A', 'Feature_B', 'Feature_C', 'time_window'])

        # reference_features_values has not enough columns
        reference_features_values_3 = pd.DataFrame(
            data={'Feature_A': [40, 45, 42, 39, 43],
                  'time_window': [0, 1, 2, 3, 4]},
            columns=['Feature_A', 'time_window'])

        # reference_features_values has no 'time_window'-column
        reference_features_values_4 = pd.DataFrame(
            data={'Feature_A': [40, 45, 42, 39, 43], 'Feature_B': [0, 1, 0, 1, 2], 'no_time_window': [0, 1, 2, 3, 4]},
            columns=['Feature_A', 'Feature_B', 'no_time_window'])

        # reference_features_values has different features
        reference_features_values_5 = pd.DataFrame(
            data={'Feature_A': [40, 45, 42, 39, 43], 'Feature_C': [0, 1, 0, 1, 2],
                  'time_window': [0, 1, 2, 3, 4]},
            columns=['Feature_A', 'Feature_C', 'time_window'])

        # wrong value type
        reference_features_values_6 = pd.DataFrame(
            data={'Feature_A': [40, 45, 'String', 39, 43], 'Feature_B': [0, 1, 0, 1, 2],
                  'time_window': [0, 1, 2, 3, 4]},
            columns=['Feature_A', 'Feature_B', 'time_window'])

        # expect a value error on reference_features_values_1
        self.assertRaises(ValueError, self.estimator.estimate, self.features_values, reference_features_values_1,
                          self.weights)

        # expect a value error on reference_features_values_2
        self.assertRaises(ValueError, self.estimator.estimate, self.features_values, reference_features_values_2,
                          self.weights)

        # expect a value error on reference_features_values_3
        self.assertRaises(ValueError, self.estimator.estimate, self.features_values, reference_features_values_3,
                          self.weights)

        # expect a value error on reference_features_values_4
        self.assertRaises(ValueError, self.estimator.estimate, self.features_values, reference_features_values_4,
                          self.weights)

        # expect a value error on reference_features_values_5
        self.assertRaises(ValueError, self.estimator.estimate, self.features_values, reference_features_values_5,
                          self.weights)

        # expect a value error on reference_features_values_6
        self.assertRaises(ValueError, self.estimator.estimate, self.features_values, reference_features_values_6,
                          self.weights)

    def test_wrong_input_weights(self):
        # too many columns
        weights_1 = pd.DataFrame(
            data={'weight': np.ones(5, dtype=int), 'weight_2': np.ones(5, dtype=int),
                  'time_window': [0, 1, 2, 3, 4]},
            columns=['weight', 'weight_2', 'time_window'])

        # wrong column names
        weights_2 = pd.DataFrame(
            data={'weightS': np.ones(5, dtype=int), 'time_window': [0, 1, 2, 3, 4]},
            columns=['weightS', 'time_window'])

        # wrong column names
        weights_3 = pd.DataFrame(
            data={'weight': np.ones(5, dtype=int), 'time_windowS': [0, 1, 2, 3, 4]},
            columns=['weight', 'time_windowS'])

        # wrong number of rows
        weights_4 = pd.DataFrame(
            data={'weightS': np.ones(2, dtype=int), 'time_window': [0, 1]},
            columns=['weightS', 'time_window'])

        # wrong value type
        weights_5 = pd.DataFrame(
            data={'weight': [1, 1, 1, 'String', 1], 'time_window': [0, 1, 2, 3, 4]},
            columns=['weight', 'time_window'])

        # expect a value error on weights_1
        self.assertRaises(ValueError, self.estimator.estimate, self.features_values, self.reference_features_values,
                          weights_1)

        # expect a value error on weights_2
        self.assertRaises(ValueError, self.estimator.estimate, self.features_values, self.reference_features_values,
                          weights_2)

        # expect a value error on weights_3
        self.assertRaises(ValueError, self.estimator.estimate, self.features_values, self.reference_features_values,
                          weights_3)

        # expect a value error on weights_4
        self.assertRaises(ValueError, self.estimator.estimate, self.features_values, self.reference_features_values,
                          weights_4)

        # expect a value error on weights_5
        self.assertRaises(ValueError, self.estimator.estimate, self.features_values, self.reference_features_values,
                          weights_5)

    def test_time_windows_match(self):
        weights = pd.DataFrame(
            data={'weight': np.ones(5, dtype=int), 'time_window': [0, 1, 2, 3, 6]},
            columns=['weight', 'time_window'])

        # expect a value error
        self.assertRaises(ValueError, self.estimator.estimate, self.features_values, self.reference_features_values,
                          weights)
