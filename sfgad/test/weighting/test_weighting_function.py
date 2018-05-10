from unittest import TestCase
import pandas as pd


class TestConstantWeight(TestCase):
    def test_compute(self):
        from sfgad.modules import ConstantWeight, WeightingFunction
        constant_weight = ConstantWeight()

        # Class type check
        self.assertTrue(isinstance(constant_weight, WeightingFunction))

        # TypeError check
        reference_feature_values_lst = [None, 10, 'test_string', 2.0]
        time_window = 1
        for reference_feature_values in reference_feature_values_lst:
            with self.assertRaises(TypeError):
                constant_weight.compute(reference_feature_values, time_window)

        reference_feature_values = pd.DataFrame()
        time_window_lst = [None, 'test', pd.DataFrame(), 2.0]
        for time_window in time_window_lst:
            with self.assertRaises(TypeError):
                constant_weight.compute(reference_feature_values, time_window)

        # ValueError check
        reference_feature_values_lst = [pd.DataFrame(),
                                        pd.DataFrame(columns=['name', 'type', 'age',
                                                              'time_window', 'time', 'feature']),
                                        pd.DataFrame({'name': [], 'type': [], 'age': [],
                                                      'time_window': [], 'time': [], 'feature': []})
                                        ]

        time_window = 1
        for reference_feature_values in reference_feature_values_lst:
            with self.assertRaises(ValueError):
                constant_weight.compute(reference_feature_values, time_window)

        time_window_lst = [0, -1]
        for time_window in time_window_lst:
            with self.assertRaises(ValueError):
                constant_weight.compute(reference_feature_values, time_window)

        # Functionality and shape check
        time_window_lst = [2, 3, 4, 4]
        for index, time_window in enumerate(time_window_lst):
            df = pd.read_csv('test_ref_feat_val_df{}.csv'.format(index + 1))
            weight_df = constant_weight.compute(df, time_window)
            weight_df_check = pd.read_csv('test_const_w_fct_results_df{}.csv'.format(index + 1))
            pd.testing.assert_frame_equal(weight_df, weight_df_check)


class TestLinearDecayWeight(TestCase):
    def test_compute(self):
        from sfgad.modules import LinearDecayWeight, WeightingFunction
        linear_decay_weight = LinearDecayWeight()

        # Class type check
        self.assertTrue(isinstance(linear_decay_weight, WeightingFunction))

        # TypeError check
        reference_feature_values_lst = [None, 10, 'test_string', 2.0]
        time_window = 1
        for reference_feature_values in reference_feature_values_lst:
            with self.assertRaises(TypeError):
                linear_decay_weight.compute(reference_feature_values, time_window)

        reference_feature_values = pd.DataFrame()
        time_window_lst = [None, 'test', pd.DataFrame(), 2.0]
        for time_window in time_window_lst:
            with self.assertRaises(TypeError):
                linear_decay_weight.compute(reference_feature_values, time_window)

        # ValueError check
        reference_feature_values_lst = [pd.DataFrame(),
                                        pd.DataFrame(columns=['name', 'type', 'age',
                                                              'time_window', 'time', 'feature']),
                                        pd.DataFrame({'name': [], 'type': [], 'age': [],
                                                      'time_window': [], 'time': [], 'feature': []})
                                        ]

        time_window = 1
        for reference_feature_values in reference_feature_values_lst:
            with self.assertRaises(ValueError):
                linear_decay_weight.compute(reference_feature_values, time_window)

        time_window_lst = [0, -1]
        for time_window in time_window_lst:
            with self.assertRaises(ValueError):
                linear_decay_weight.compute(reference_feature_values, time_window)

        # Functionality and shape check
        time_window_lst = [2, 3, 4, 4]
        for index, time_window in enumerate(time_window_lst):
            df = pd.read_csv('test_ref_feat_val_df{}.csv'.format(index + 1))
            weight_df = linear_decay_weight.compute(df, time_window)
            weight_df_check = pd.read_csv('test_lin_w_fct_results_df{}.csv'.format(index + 1))
            pd.testing.assert_frame_equal(weight_df, weight_df_check)


class TestExponentialDecayWeight(TestCase):
    def test_compute(self):
        from sfgad.modules import ExponentialDecayWeight, WeightingFunction
        exponential_decay_weight = ExponentialDecayWeight(half_life=0.5)

        # Class type check
        self.assertTrue(isinstance(exponential_decay_weight, WeightingFunction))

        # TypeError check
        reference_feature_values_lst = [None, 10, 'test_string', 2.0]
        time_window = 1
        for reference_feature_values in reference_feature_values_lst:
            with self.assertRaises(TypeError):
                exponential_decay_weight.compute(reference_feature_values, time_window)

        reference_feature_values = pd.DataFrame()
        time_window_lst = [None, 'test', pd.DataFrame(), 2.0]
        for time_window in time_window_lst:
            with self.assertRaises(TypeError):
                exponential_decay_weight.compute(reference_feature_values, time_window)

        # ValueError check
        reference_feature_values_lst = [pd.DataFrame(),
                                        pd.DataFrame(columns=['name', 'type', 'age',
                                                              'time_window', 'time', 'feature']),
                                        pd.DataFrame({'name': [], 'type': [], 'age': [],
                                                      'time_window': [], 'time': [], 'feature': []})
                                        ]

        time_window = 1
        for reference_feature_values in reference_feature_values_lst:
            with self.assertRaises(ValueError):
                exponential_decay_weight.compute(reference_feature_values, time_window)

        time_window_lst = [0, -1]
        for time_window in time_window_lst:
            with self.assertRaises(ValueError):
                exponential_decay_weight.compute(reference_feature_values, time_window)

        # Functionality and shape check
        time_window_lst = [2, 3, 4, 4]
        for index, time_window in enumerate(time_window_lst):
            df = pd.read_csv('test_ref_feat_val_df{}.csv'.format(index + 1))
            weight_df = exponential_decay_weight.compute(df, time_window)
            weight_df_check = pd.read_csv('test_exp_w_fct_results_df{}.csv'.format(index + 1))
            pd.testing.assert_frame_equal(weight_df, weight_df_check)


if __name__ == '__main__':
    import unittest
    unittest.main()
