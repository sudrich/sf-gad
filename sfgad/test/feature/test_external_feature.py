import pandas as pd
import numpy as np

from unittest import TestCase
from pandas.util.testing import assert_frame_equal, assert_series_equal
from sfgad.modules.features.external_feature import ExternalFeature


class TestExternalFeature(TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'TIMESTAMP': ['2018-01-01 00:00:00', '2018-01-01 00:00:05', '2018-01-01 00:00:09'],
                                'SRC_NAME': ['A', 'A', 'B'],
                                'SRC_TYPE': ['NODE', 'NODE', 'NODE'],
                                'DST_NAME': ['B', 'C', 'C'],
                                'DST_TYPE': ['NODE', 'NODE', 'NODE']})
        self.df['TIMESTAMP'] = pd.to_datetime(self.df['TIMESTAMP'])

        # the target output of the feature after 1. time step
        self.target_df = pd.DataFrame(
            data={'name': ['A', 'B', 'C'], 'ExternalFeature': [42, 24, 0]},
            columns=['name', 'ExternalFeature']
        )

        # the dictionary for the external feature

        self.values_dict = {
            'A': [(pd.to_datetime('2018-01-01 00:00:01'), -1), (pd.to_datetime('2018-01-01 00:00:11'), 42)],
            'B': [(pd.to_datetime('2018-01-01 00:00:12'), 24)],
            'C': [(pd.to_datetime('2018-01-01 00:00:11'), 0), (pd.to_datetime('2018-01-01 00:00:12'), -1)]
        }

        self.feature = ExternalFeature(self.values_dict)

    def test_init(self):
        self.assertEqual(self.feature.names, ['ExternalFeature'])
        self.assertEqual(self.feature.values_dict, self.values_dict)

    def test_result_df_shape(self):
        result_df = self.feature.process_vertices(self.df, 1)

        self.assertEqual(result_df.shape, self.target_df.shape)

    def test_result_df_columns(self):
        result_df_1 = self.feature.process_vertices(self.df, 1)

        self.assertEqual(result_df_1.columns.tolist(), ['name', 'ExternalFeature'])

    def test_result_df_dtypes(self):
        result_df_1 = self.feature.process_vertices(self.df, 1)

        assert_series_equal(result_df_1.dtypes, self.target_df.dtypes)

    def test_result_df_values(self):
        result_df_1 = self.feature.process_vertices(self.df, 1)

        assert_series_equal(result_df_1['name'], self.target_df['name'])
        assert_series_equal(result_df_1['ExternalFeature'], self.target_df['ExternalFeature'])

    def test_overall_processing(self):
        # test the calculation of vertex degree on the example data frame 'df'
        assert_frame_equal(self.feature.process_vertices(self.df, 1), self.target_df)

    # TEST SPECIAL CASES

    def test_unknown_node(self):
        # remove 'A' from dictionary
        self.values_dict.pop('A', None)

        feature = ExternalFeature(self.values_dict)

        # the target output of the feature after 1. time step
        target_df = pd.DataFrame(
            data={'name': ['A', 'B', 'C'], 'ExternalFeature': [np.nan, 24, 0]},
            columns=['name', 'ExternalFeature']
        )

        assert_frame_equal(feature.process_vertices(self.df, 1), target_df)

    def test_outdated_node_values(self):
        del self.values_dict['A'][-1]

        feature = ExternalFeature(self.values_dict)

        # the target output of the feature after 1. time step
        target_df = pd.DataFrame(
            data={'name': ['A', 'B', 'C'], 'ExternalFeature': [np.nan, 24, 0]},
            columns=['name', 'ExternalFeature']
        )

        assert_frame_equal(feature.process_vertices(self.df, 1), target_df)

    def test_wrong_input(self):

        # the given argument is not a dictionary
        values_dict = 42

        self.assertRaises(ValueError, ExternalFeature, values_dict)

    def test_dict_wrong_scheme_1(self):

        # nor all keys are strings
        values_dict = {
            'A': [(pd.to_datetime('2018-01-01 00:00:01'), -1), (pd.to_datetime('2018-01-01 00:00:11'), 42)],
            42: [(pd.to_datetime('2018-01-01 00:00:12'), 24)],
            'C': [(pd.to_datetime('2018-01-01 00:00:11'), 0), (pd.to_datetime('2018-01-01 00:00:12'), -1)]
        }

        self.assertRaises(ValueError, ExternalFeature, values_dict)

    def test_dict_wrong_scheme_2(self):

        # not all values are lists
        values_dict = {
            'A': [(pd.to_datetime('2018-01-01 00:00:01'), -1), (pd.to_datetime('2018-01-01 00:00:11'), 42)],
            'B': [(pd.to_datetime('2018-01-01 00:00:12'), 24)],
            'C': 42
        }

        self.assertRaises(ValueError, ExternalFeature, values_dict)

    def test_dict_wrong_scheme_3(self):

        # not all list elements are tuples
        values_dict = {
            'A': [42, (pd.to_datetime('2018-01-01 00:00:11'), 42)],
            'B': [(pd.to_datetime('2018-01-01 00:00:12'), 24)],
            'C': [(pd.to_datetime('2018-01-01 00:00:11'), 0), (pd.to_datetime('2018-01-01 00:00:12'), -1)]
        }

        self.assertRaises(ValueError, ExternalFeature, values_dict)

    def test_dict_wrong_scheme_4(self):

        # a list element with wrong time-type
        values_dict = {
            'A': [(pd.to_datetime('2018-01-01 00:00:01'), -1), (pd.to_datetime('2018-01-01 00:00:11'), 42)],
            'B': [('2018-01-01 00:00:12', 24)],
            'C': [(pd.to_datetime('2018-01-01 00:00:11'), 0), (pd.to_datetime('2018-01-01 00:00:12'), -1)]
        }

        self.assertRaises(ValueError, ExternalFeature, values_dict)

    def test_dict_wrong_scheme_5(self):

        # a list is not sorted by time
        values_dict = {
            'A': [(pd.to_datetime('2018-01-01 00:00:11'), 42), (pd.to_datetime('2018-01-01 00:00:01'), -1)],
            'B': [(pd.to_datetime('2018-01-01 00:00:12'), 24)],
            'C': [(pd.to_datetime('2018-01-01 00:00:11'), 0), (pd.to_datetime('2018-01-01 00:00:12'), -1)]
        }

        self.assertRaises(ValueError, ExternalFeature, values_dict)
