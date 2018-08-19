from unittest import TestCase

import pandas as pd

from sfgad.modules.weighting.weighting import Weighting


class TypeSpecificWeight(TestCase):
    def setUp(self):
        from sfgad.modules.weighting.type_specific_weight import TypeSpecificWeight
        type_dict = {'GUEST': 0.5, 'USER': 0.75, 'ADMIN': 1.0}
        self.type_spec_wgt = TypeSpecificWeight(type_dict)

    def test_class_type_check(self):
        self.assertTrue(isinstance(self.type_spec_wgt, Weighting))

    def test_type_of_reference_feature_values(self):
        reference_feature_values_lst = [None, 10, 'test_string', 2.0]
        time_window = 1
        for reference_feature_values in reference_feature_values_lst:
            with self.assertRaises(TypeError):
                self.type_spec_wgt.compute(reference_feature_values, time_window)

    def test_type_of_time_window(self):
        reference_feature_values = pd.DataFrame()
        time_window_lst = [None, 'test', pd.DataFrame(), 2.0]
        for time_window in time_window_lst:
            with self.assertRaises(TypeError):
                self.type_spec_wgt.compute(reference_feature_values, time_window)

    def test_values_of_reference_feature_values(self):
        ref_feature_val_lst = [pd.DataFrame(),
                               pd.DataFrame(columns=['name', 'type', 'age', 'time_window', 'time', 'feature']),
                               pd.DataFrame({'name': [], 'type': [], 'age': [],
                                             'time_window': [], 'time': [], 'feature': []})]
        time_window = 1
        for reference_feature_values in ref_feature_val_lst:
            with self.assertRaises(ValueError):
                self.type_spec_wgt.compute(reference_feature_values, time_window)

    def test_values_of_time_window(self):
        reference_feature_values = pd.DataFrame({'name': ['name'], 'type': ['TYPE'], 'age': [1.0],
                                                 'time_window': [0], 'time': [None], 'feature': [1.0]})
        time_window_lst = [0, -1]
        for time_window in time_window_lst:
            with self.assertRaises(ValueError):
                self.type_spec_wgt.compute(reference_feature_values, time_window)

    def test_type_mismatches(self):
        time_window = 1
        df = pd.DataFrame({'name': [None], 'type': ['BLOCKED_USER'], 'age': [None],
                           'time_window': [1], 'time': [None], 'feature': [None]})
        with self.assertRaises(ValueError):
            self.type_spec_wgt.compute(df, time_window)

    def test_functionality(self):
        time_window = 1
        dfs = [pd.DataFrame({'name': [None], 'type': ['GUEST'], 'age': [None],
                             'time_window': [1], 'time': [None], 'feature': [None]}),
               pd.DataFrame({'name': [None], 'type': ['USER'], 'age': [None],
                             'time_window': [1], 'time': [None], 'feature': [None]}),
               pd.DataFrame({'name': [None], 'type': ['ADMIN'], 'age': [None],
                             'time_window': [1], 'time': [None], 'feature': [None]})]
        weight_dfs_check = [pd.DataFrame({'time_window': [1], 'weight': [0.5]}),
                            pd.DataFrame({'time_window': [1], 'weight': [0.75]}),
                            pd.DataFrame({'time_window': [1], 'weight': [1.0]})]

        for index, df in enumerate(dfs):
            weight_df = self.type_spec_wgt.compute(df, time_window)
            pd.testing.assert_frame_equal(weight_df, weight_dfs_check[index])
