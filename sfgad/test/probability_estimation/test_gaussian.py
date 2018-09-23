from unittest import TestCase

import numpy as np

from sfgad.modules.probability_estimation.gaussian import Gaussian


class TestGaussian(TestCase):
    def setUp(self):
        self.estimator = Gaussian()

    def test_init_default(self):
        self.estimator = Gaussian()
        self.assertEqual(self.estimator.direction, 'right-tailed')
        self.assertFalse(hasattr(self.estimator, "means"))
        self.assertFalse(hasattr(self.estimator, "stds"))

    def test_init_left_tailed(self):
        self.estimator = Gaussian(direction='left-tailed')
        self.assertEqual(self.estimator.direction, 'left-tailed')
        self.assertFalse(hasattr(self.estimator, "means"))
        self.assertFalse(hasattr(self.estimator, "stds"))

    def test_init_right_tailed(self):
        self.estimator = Gaussian(direction='right-tailed')
        self.assertEqual(self.estimator.direction, 'right-tailed')
        self.assertFalse(hasattr(self.estimator, "means"))
        self.assertFalse(hasattr(self.estimator, "stds"))

    def test_init_two_tailed(self):
        self.estimator = Gaussian(direction='two-tailed')
        self.assertEqual(self.estimator.direction, 'two-tailed')
        self.assertFalse(hasattr(self.estimator, "means"))
        self.assertFalse(hasattr(self.estimator, "stds"))

    def test_init_unknown_direction(self):
        self.assertRaises(ValueError, Gaussian, 'unknown-tailed')

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
        np.testing.assert_array_equal(self.estimator.stds, np.array([0, 0]))

    def test_fit_multiple_ref_observations_equal_weights(self):
        ref_observations = np.array([[1, 2],
                                     [1, 2],
                                     [3, 4],
                                     [3, 4]])
        weights = np.array([1, 1, 1, 1])
        self.estimator.fit(ref_observations, weights)
        np.testing.assert_array_equal(self.estimator.means, np.array([2, 3]))
        np.testing.assert_array_equal(self.estimator.stds, np.array([1, 1]))

    def test_fit_multiple_ref_observations_various_weights(self):
        ref_observations = np.array([[1, 2],
                                     [1, 2],
                                     [3, 4],
                                     [3, 4]])
        weights = np.array([1, 2, 2, 1])
        self.estimator.fit(ref_observations, weights)
        np.testing.assert_array_equal(self.estimator.means, np.array([2, 3]))
        np.testing.assert_array_equal(self.estimator.stds, np.array([1, 1]))

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

    def test_transform_zero_stds(self):
        ref_observations = np.array([[1, 2]])
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
        observations = np.array([[2, 3]])
        self.estimator.fit(ref_observations, weights)
        np.testing.assert_array_almost_equal(self.estimator.transform(observations), np.array([[0.5, 0.5]]))

    def test_transform_multiple_observations_left_tailed(self):
        self.estimator = Gaussian(direction='left-tailed')
        ref_observations = np.array([[1, 2],
                                     [1, 2],
                                     [3, 4],
                                     [3, 4]])
        weights = np.array([1, 1, 1, 1])
        observations = np.array([[1, 2],
                                 [2, 3],
                                 [3, 4]])
        self.estimator.fit(ref_observations, weights)
        np.testing.assert_array_almost_equal(self.estimator.transform(observations), np.array([[0.158655, 0.158655],
                                                                                               [0.50, 0.50],
                                                                                               [0.841345, 0.841345]]))

    def test_transform_multiple_observations_right_tailed(self):
        self.estimator = Gaussian(direction='right-tailed')
        ref_observations = np.array([[1, 2],
                                     [1, 2],
                                     [3, 4],
                                     [3, 4]])
        weights = np.array([1, 1, 1, 1])
        observations = np.array([[1, 2],
                                 [2, 3],
                                 [3, 4]])
        self.estimator.fit(ref_observations, weights)
        np.testing.assert_array_almost_equal(self.estimator.transform(observations), np.array([[0.841345, 0.841345],
                                                                                               [0.50, 0.50],
                                                                                               [0.158655, 0.158655]]))

    def test_transform_multiple_observations_two_tailed(self):
        self.estimator = Gaussian(direction='two-tailed')
        ref_observations = np.array([[1, 2],
                                     [1, 2],
                                     [3, 4],
                                     [3, 4]])
        weights = np.array([1, 1, 1, 1])
        observations = np.array([[1, 2],
                                 [2, 3],
                                 [3, 4]])
        self.estimator.fit(ref_observations, weights)
        np.testing.assert_array_almost_equal(self.estimator.transform(observations), np.array([[0.317311, 0.317311],
                                                                                               [1.0, 1.0],
                                                                                               [0.317311, 0.317311]]))
