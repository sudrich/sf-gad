from unittest import TestCase

import numpy as np
import pandas as pd

from sfgad.modules.weighting import TypeSpecificWeight


class TestTypeSpecificWeight(TestCase):
    def setUp(self):
        type_dict = {'GUEST': 0.5, 'USER': 0.75, 'ADMIN': 1.0}
        self.weighting_function = TypeSpecificWeight(type_dict)

    def test_init_custom(self):
        type_dict = {'GUEST': 0.5, 'USER': 0.75, 'ADMIN': 1.0}
        self.weighting_function = TypeSpecificWeight(type_dict)
        self.assertAlmostEqual(self.weighting_function.type_dict, type_dict)

    def test_init_empty_dict(self):
        type_dict = {}
        self.weighting_function = TypeSpecificWeight(type_dict)
        self.assertAlmostEqual(self.weighting_function.type_dict, {})

    def test_ref_observations_not_as_dataframe(self):
        # Basically a valid input, but a numpy array instead of a pandas DataFrame
        # The numpy array is an invalid input since it lacks indication of columns
        ref_meta_info = np.array([['USER', 'USER', 'GUEST', 'ADMIN']])
        current_meta_info = pd.Series({})

        self.assertRaises(TypeError, self.weighting_function.compute, ref_meta_info, current_meta_info)

    def test_current_observation_not_as_series(self):
        # Basically a valid input, but a numpy array instead of a pandas DataFrame
        # The numpy array is an invalid input since it lacks indication of columns
        ref_meta_info = pd.DataFrame({'type': [1, 2, 3, 4]})
        current_meta_info = np.array([])

        self.assertRaises(TypeError, self.weighting_function.compute, ref_meta_info, current_meta_info)

    def test_empty_input_with_incorrect_columns(self):
        ref_meta_info = pd.DataFrame(columns=['name'])
        current_meta_info = pd.Series({})

        self.assertRaises(ValueError, self.weighting_function.compute, ref_meta_info, current_meta_info)

    def test_empty_input_with_correct_columns(self):
        ref_meta_info = pd.DataFrame(columns=['type'])
        current_meta_info = pd.Series({})

        np.testing.assert_array_equal(self.weighting_function.compute(ref_meta_info, current_meta_info), np.array([]))

    def test_single_ref_observation(self):
        ref_meta_info = pd.DataFrame({'type': ['USER']})
        current_meta_info = pd.Series()

        np.testing.assert_array_almost_equal(self.weighting_function.compute(ref_meta_info, current_meta_info),
                                             np.array([0.75]))

    def test_multiple_ref_observations(self):
        ref_meta_info = pd.DataFrame({'type': ['USER', 'USER', 'GUEST', 'ADMIN']})
        current_meta_info = pd.Series()

        np.testing.assert_array_almost_equal(self.weighting_function.compute(ref_meta_info, current_meta_info),
                                             np.array([0.75, 0.75, 0.5, 1.0]))

    def test_unknown_type(self):
        ref_meta_info = pd.DataFrame({'type': ['BLOCKED_USER']})
        current_meta_info = pd.Series({})

        self.assertRaises(ValueError, self.weighting_function.compute, ref_meta_info, current_meta_info)
