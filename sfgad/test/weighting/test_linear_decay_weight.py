from unittest import TestCase

import numpy as np
import pandas as pd

from sfgad.modules.weighting import LinearDecayWeight


class TestLinearWeight(TestCase):
    def setUp(self):
        self.weighting_function = LinearDecayWeight(factor=0.25)

    def test_init_zero(self):
        self.weighting_function = LinearDecayWeight(factor=0.0)
        self.assertAlmostEqual(self.weighting_function.factor, 0.0)

    def test_init_custom_1(self):
        self.weighting_function = LinearDecayWeight(factor=1.0)
        self.assertAlmostEqual(self.weighting_function.factor, 1.0)

    def test_init_custom_2(self):
        self.weighting_function = LinearDecayWeight(factor=2.0)
        self.assertAlmostEqual(self.weighting_function.factor, 2.0)

    def test_ref_observations_not_as_dataframe(self):
        # Basically a valid input, but a numpy array instead of a pandas DataFrame
        # The numpy array is an invalid input since it lacks indication of columns
        ref_meta_info = np.array([[1, 2, 3, 4]])
        current_meta_info = pd.Series({'time_window': 4})

        self.assertRaises(TypeError, self.weighting_function.compute, ref_meta_info, current_meta_info)

    def test_current_observation_not_as_series(self):
        # Basically a valid input, but a numpy array instead of a pandas DataFrame
        # The numpy array is an invalid input since it lacks indication of columns
        ref_meta_info = pd.DataFrame({'time_window': [1, 2, 3, 4]})
        current_meta_info = np.array([1])

        self.assertRaises(TypeError, self.weighting_function.compute, ref_meta_info, current_meta_info)

    def test_empty_ref_observations_with_incorrect_columns(self):
        ref_meta_info = pd.DataFrame(columns=['type'])
        current_meta_info = pd.Series({'time_window': 4})

        self.assertRaises(ValueError, self.weighting_function.compute, ref_meta_info, current_meta_info)

    def test_empty_ref_observations_with_correct_columns(self):
        ref_meta_info = pd.DataFrame(columns=['time_window'])
        current_meta_info = pd.Series({'time_window': 4})

        np.testing.assert_array_equal(self.weighting_function.compute(ref_meta_info, current_meta_info), np.array([]))

    def test_current_observation_with_incorrect_columns(self):
        ref_meta_info = pd.DataFrame({'time_window': [3]})
        current_meta_info = pd.Series({'type': 4})

        self.assertRaises(ValueError, self.weighting_function.compute, ref_meta_info, current_meta_info)

    def test_single_ref_observation(self):
        ref_meta_info = pd.DataFrame({'time_window': [3]})
        current_meta_info = pd.Series({'time_window': 4})

        np.testing.assert_array_almost_equal(self.weighting_function.compute(ref_meta_info, current_meta_info),
                                             np.array([0.75]))

    def test_clipped_ref_observation(self):
        ref_meta_info = pd.DataFrame({'time_window': [0]})
        current_meta_info = pd.Series({'time_window': 5})

        np.testing.assert_array_almost_equal(self.weighting_function.compute(ref_meta_info, current_meta_info),
                                             np.array([0]))

    def test_multiple_ref_observations(self):
        ref_meta_info = pd.DataFrame({'time_window': [1, 2, 3, 4]})
        current_meta_info = pd.Series({'time_window': 4})

        np.testing.assert_array_almost_equal(self.weighting_function.compute(ref_meta_info, current_meta_info),
                                             np.array([0.25, 0.5, 0.75, 1.0]))
