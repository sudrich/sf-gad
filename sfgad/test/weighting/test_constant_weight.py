from unittest import TestCase

import numpy as np
import pandas as pd

from sfgad.modules.weighting import ConstantWeight


class TestConstantWeight(TestCase):
    def setUp(self):
        self.weighting_function = ConstantWeight(weight=1)

    def test_ref_observations_not_as_dataframe(self):
        # Basically a valid input, but a numpy array instead of a pandas DataFrame
        # The numpy array is an invalid input since it lacks indication of columns
        ref_meta_info = np.array([[1, 2, 3, 4]])
        current_meta_info = pd.Series({'time_window': 1})

        self.assertRaises(TypeError, self.weighting_function.compute, ref_meta_info, current_meta_info)

    def test_current_observation_not_as_series(self):
        # Basically a valid input, but a numpy array instead of a pandas DataFrame
        # The numpy array is an invalid input since it lacks indication of columns
        ref_meta_info = pd.DataFrame({'time_window': [1, 1, 1, 1]})
        current_meta_info = np.array([1])

        self.assertRaises(TypeError, self.weighting_function.compute, ref_meta_info, current_meta_info)

    def test_empty_ref_observations(self):
        ref_meta_info = pd.DataFrame()
        current_meta_info = pd.Series({'time_window': 1})

        np.testing.assert_array_equal(self.weighting_function.compute(ref_meta_info, current_meta_info), np.array([]))

    def test_empty_current_observation(self):
        ref_meta_info = pd.DataFrame()
        current_meta_info = pd.Series()

        np.testing.assert_array_equal(self.weighting_function.compute(ref_meta_info, current_meta_info),
                                      np.array([]))

    def test_single_ref_observation(self):
        ref_meta_info = pd.DataFrame({'time_window': [3]})
        current_meta_info = pd.Series({'time_window': 1})

        np.testing.assert_array_almost_equal(self.weighting_function.compute(ref_meta_info, current_meta_info),
                                             np.array([1]))

    def test_multiple_ref_observations(self):
        ref_meta_info = pd.DataFrame({'time_window': [1, 1, 1, 1]})
        current_meta_info = pd.Series({'time_window': 1})

        np.testing.assert_array_almost_equal(self.weighting_function.compute(ref_meta_info, current_meta_info),
                                             np.array([1, 1, 1, 1]))
