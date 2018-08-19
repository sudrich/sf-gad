from unittest import TestCase

import pandas as pd

from sfgad.modules.weighting.weighting import Weighting


class TestConstantWeight(TestCase):
    def setUp(self):
        from sfgad.modules.weighting.exponential_decay_weight import ExponentialDecayWeight
        self.exp_dec_wgt = ExponentialDecayWeight(half_life=1.0, lower_threshold=0.25)

    def test_class_type_check(self):
        self.assertTrue(isinstance(self.exp_dec_wgt, Weighting))

    def test_type_of_reference_feature_values(self):
        reference_feature_values_lst = [None, 10, 'test_string', 2.0]
        time_window = 1
        for reference_feature_values in reference_feature_values_lst:
            with self.assertRaises(TypeError):
                self.exp_dec_wgt.compute(reference_feature_values, time_window)

    def test_type_of_time_window(self):
        reference_feature_values = pd.DataFrame()
        time_window_lst = [None, 'test', pd.DataFrame(), 2.0]
        for time_window in time_window_lst:
            with self.assertRaises(TypeError):
                self.exp_dec_wgt.compute(reference_feature_values, time_window)

    def test_values_of_reference_feature_values(self):
        ref_feature_val_lst = [pd.DataFrame(),
                               pd.DataFrame(columns=['name', 'type', 'age', 'time_window', 'time', 'feature']),
                               pd.DataFrame({'name': [], 'type': [], 'age': [],
                                             'time_window': [], 'time': [], 'feature': []})]
        time_window = 1
        for reference_feature_values in ref_feature_val_lst:
            with self.assertRaises(ValueError):
                self.exp_dec_wgt.compute(reference_feature_values, time_window)

    def test_values_of_time_window(self):
        reference_feature_values = pd.DataFrame({'name': ['name'], 'type': ['TYPE'], 'age': [1.0],
                                                 'time_window': [0], 'time': [None], 'feature': [1.0]})
        time_window_lst = [0, -1]
        for time_window in time_window_lst:
            with self.assertRaises(ValueError):
                self.exp_dec_wgt.compute(reference_feature_values, time_window)

    def test_functionality(self):
        time_window_lst = [1, 2, 3, 4]
        dfs = [pd.DataFrame({'name': [None], 'type': [None], 'age': [None],
                             'time_window': [1], 'time': [None], 'feature': [None]}),
               pd.DataFrame({'name': [None, None], 'type': [None, None], 'age': [None, None],
                             'time_window': [1, 2], 'time': [None, None], 'feature': [None, None]}),
               pd.DataFrame({'name': [None, None, None], 'type': [None, None, None], 'age': [None, None, None],
                             'time_window': [1, 2, 3], 'time': [None, None, None], 'feature': [None, None, None]}),
               pd.DataFrame({'name': [None, None, None, None],
                             'type': [None, None, None, None],
                             'age': [None, None, None, None],
                             'time_window': [1, 2, 3, 4],
                             'time': [None, None, None, None],
                             'feature': [None, None, None, None]})]

        weight_dfs_check = [pd.DataFrame({'time_window': [1], 'weight': [1.0]}),
                            pd.DataFrame({'time_window': [1, 2], 'weight': [0.5, 1.0]}),
                            pd.DataFrame({'time_window': [1, 2, 3], 'weight': [0.25, 0.5, 1.0]}),
                            pd.DataFrame({'time_window': [1, 2, 3, 4], 'weight': [0.0, 0.25, 0.5, 1.0]})]

        for index, time_window in enumerate(time_window_lst):
            weight_df = self.exp_dec_wgt.compute(dfs[index], time_window)
            pd.testing.assert_frame_equal(weight_df, weight_dfs_check[index])
