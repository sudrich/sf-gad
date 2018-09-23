import math
from unittest import TestCase

import numpy as np

from sfgad.modules.probability_estimation.exponential import Exponential


class TestExponential(TestCase):
    def setUp(self):
        self.estimator = Exponential()

    def test_init_default(self):
        self.estimator = Exponential()
        self.assertEqual(self.estimator.direction, 'right-tailed')
        self.assertFalse(hasattr(self.estimator, "means"))

    def test_init_left_tailed(self):
        self.estimator = Exponential(direction='left-tailed')
        self.assertEqual(self.estimator.direction, 'left-tailed')
        self.assertFalse(hasattr(self.estimator, "means"))

    def test_init_right_tailed(self):
        self.estimator = Exponential(direction='right-tailed')
        self.assertEqual(self.estimator.direction, 'right-tailed')
        self.assertFalse(hasattr(self.estimator, "means"))

    def test_init_two_tailed(self):
        self.estimator = Exponential(direction='two-tailed')
        self.assertEqual(self.estimator.direction, 'two-tailed')
        self.assertFalse(hasattr(self.estimator, "means"))

    def test_init_unknown_direction(self):
        self.assertRaises(ValueError, Exponential, 'unknown-tailed')

    def test_fit_empty_ref_observations(self):
        ref_observations = np.array([[]])
        weights = np.array([])
        self.assertRaises(ValueError, self.estimator.fit, ref_observations, weights)

    def test_fit_inconsistent_length_ref_observations_and_weights(self):
        ref_observations = np.array([[1, 2],
                                     [2, 3]])
        weights = np.array([1, 2, 3])
        self.assertRaises(ValueError, self.estimator.fit, ref_observations, weights)

    def test_fit_single_ref_observation(self):
        ref_observations = np.array([[1, 2]])
        weights = np.array([1])
        self.estimator.fit(ref_observations, weights)
        np.testing.assert_array_equal(self.estimator.means, np.array([1, 2]))

    def test_fit_multiple_ref_observations_equal_weights(self):
        ref_observations = np.array([[1, 2],
                                     [2, 3],
                                     [3, 4]])
        weights = np.array([1, 1, 1])
        self.estimator.fit(ref_observations, weights)
        np.testing.assert_array_equal(self.estimator.means, np.array([2, 3]))

    def test_fit_multiple_ref_observations_various_weights(self):
        ref_observations = np.array([[1, 2],
                                     [2, 3],
                                     [3, 4]])
        weights = np.array([1, 2, 3])
        self.estimator.fit(ref_observations, weights)
        np.testing.assert_array_almost_equal(self.estimator.means, np.array([14 / 6, 20 / 6]))

    def test_without_fit(self):
        observations = np.array([[1, 2, 3],
                                 [2, 3, 4]])
        self.assertRaises(ValueError, self.estimator.transform, observations)

    def test_transform_inconsistent_number_of_features(self):
        ref_observations = np.array([[1, 2],
                                     [2, 3]])
        weights = np.array([1, 1])
        observations = np.array([[1, 2, 3],
                                 [2, 3, 4]])
        self.estimator.fit(ref_observations, weights)
        self.assertRaises(ValueError, self.estimator.transform, observations)

    def test_transform_zero_means(self):
        ref_observations = np.array([[0, 0]])
        weights = np.array([1])
        observations = np.array([[1, 2],
                                 [2, 3]])
        self.estimator.fit(ref_observations, weights)
        np.testing.assert_array_almost_equal(self.estimator.transform(observations), np.array([[1.0, 1.0],
                                                                                               [1.0, 1.0]]))

    def test_transform_single_observation(self):
        ref_observations = np.array([[1, 2],
                                     [2, 3],
                                     [3, 4]])
        weights = np.array([1, 1, 1])
        observations = np.array([[2 * math.log(2), 3 * math.log(2)]])
        self.estimator.fit(ref_observations, weights)
        np.testing.assert_array_almost_equal(self.estimator.transform(observations), np.array([[0.5, 0.5]]))

    def test_transform_multiple_observations_left_tailed(self):
        self.estimator = Exponential(direction='left-tailed')
        ref_observations = np.array([[1, 2],
                                     [2, 3],
                                     [3, 4]])
        weights = np.array([1, 1, 1])
        observations = np.array([[2 * math.log(4 / 3), 3 * math.log(4 / 3)],
                                 [2 * math.log(2), 3 * math.log(2)],
                                 [2 * math.log(4), 3 * math.log(4)]])
        self.estimator.fit(ref_observations, weights)
        np.testing.assert_array_almost_equal(self.estimator.transform(observations), np.array([[0.25, 0.25],
                                                                                               [0.50, 0.50],
                                                                                               [0.75, 0.75]]))

    def test_transform_multiple_observations_right_tailed(self):
        self.estimator = Exponential(direction='right-tailed')
        ref_observations = np.array([[1, 2],
                                     [2, 3],
                                     [3, 4]])
        weights = np.array([1, 1, 1])
        observations = np.array([[2 * math.log(4 / 3), 3 * math.log(4 / 3)],
                                 [2 * math.log(2), 3 * math.log(2)],
                                 [2 * math.log(4), 3 * math.log(4)]])
        self.estimator.fit(ref_observations, weights)
        np.testing.assert_array_almost_equal(self.estimator.transform(observations), np.array([[0.75, 0.75],
                                                                                               [0.50, 0.50],
                                                                                               [0.25, 0.25]]))

    def test_transform_multiple_observations_two_tailed(self):
        self.estimator = Exponential(direction='two-tailed')
        ref_observations = np.array([[1, 2],
                                     [2, 3],
                                     [3, 4]])
        weights = np.array([1, 1, 1])
        observations = np.array([[2 * math.log(4 / 3), 3 * math.log(4 / 3)],
                                 [2 * math.log(2), 3 * math.log(2)],
                                 [2 * math.log(4), 3 * math.log(4)]])
        self.estimator.fit(ref_observations, weights)
        np.testing.assert_array_almost_equal(self.estimator.transform(observations), np.array([[0.5, 0.5],
                                                                                               [1.0, 1.0],
                                                                                               [0.5, 0.5]]))
