from unittest import TestCase

import pandas as pd
from pandas.util.testing import assert_frame_equal, assert_series_equal

from sfgad.modules.features.vertex_degree_difference import VertexDegreeDifference


class TestVertexDegreeDifference(TestCase):
    def setUp(self):
        # 1. time step
        self.df_1 = pd.DataFrame({'TIMESTAMP': ['2018-01-01 00:00:00', '2018-01-01 00:00:01', '2018-01-01 00:00:05'],
                                  'SRC_NAME': ['A', 'A', 'B'],
                                  'SRC_TYPE': ['NODE', 'NODE', 'NODE'],
                                  'DST_NAME': ['B', 'C', 'C'],
                                  'DST_TYPE': ['NODE', 'NODE', 'NODE']})
        self.df_1['TIMESTAMP'] = pd.to_datetime(self.df_1['TIMESTAMP'])

        # 2. time step
        self.df_2 = pd.DataFrame({'TIMESTAMP': ['2018-01-01 00:00:11', '2018-01-01 00:00:14', '2018-01-01 00:00:16'],
                                  'SRC_NAME': ['A', 'A', 'A'],
                                  'SRC_TYPE': ['NODE', 'NODE', 'NODE'],
                                  'DST_NAME': ['B', 'D', 'D'],
                                  'DST_TYPE': ['NODE', 'NODE', 'NODE']})
        self.df_2['TIMESTAMP'] = pd.to_datetime(self.df_2['TIMESTAMP'])

        # the target output of the feature after 1. (2.) time step for all nodes
        self.target_df_1 = pd.DataFrame(data={'name': ['A', 'B', 'C'], 'VertexDegreeDifference': [2, 2, 2]},
                                        columns=['name', 'VertexDegreeDifference'])
        self.target_df_2 = pd.DataFrame(data={'name': ['A', 'B', 'D', 'C'], 'VertexDegreeDifference': [1, -1, 2, -2]},
                                        columns=['name', 'VertexDegreeDifference'])

        self.feature = VertexDegreeDifference()

    def test_init(self):
        self.assertEqual(self.feature.names, ['VertexDegreeDifference'])
        self.assertEqual(self.feature.only_active_nodes, False)

    def test_result_df_shape(self):
        result_df_1 = self.feature.process_vertices(self.df_1, 1)

        self.assertEqual(result_df_1.shape, self.target_df_1.shape)

    def test_result_df_columns(self):
        result_df_1 = self.feature.process_vertices(self.df_1, 1)

        self.assertEqual(result_df_1.columns.tolist(), ['name', 'VertexDegreeDifference'])

    def test_result_df_dtypes(self):
        result_df_1 = self.feature.process_vertices(self.df_1, 1)

        assert_series_equal(result_df_1.dtypes, self.target_df_1.dtypes)

    def test_result_df_values(self):
        result_df_1 = self.feature.process_vertices(self.df_1, 1)

        assert_series_equal(result_df_1['name'], self.target_df_1['name'])
        assert_series_equal(result_df_1['VertexDegreeDifference'], self.target_df_1['VertexDegreeDifference'])

    def test_overall_processing_all_nodes(self):
        # test the calculation of vertex degree difference for all nodes

        # test the calculation of vertex degree difference in the 1. time step ('df_1')
        assert_frame_equal(self.feature.process_vertices(self.df_1, 1), self.target_df_1)

        # test the calculation of vertex degree difference in the 2. time step ('df_2')
        assert_frame_equal(self.feature.process_vertices(self.df_2, 1), self.target_df_2)

    def test_overall_processing_active_nodes(self):
        # test the calculation of vertex degree difference only for active nodes
        self.feature = VertexDegreeDifference(only_active_nodes=True)

        # test the calculation of vertex degree difference in the 1. time step ('df_1')
        assert_frame_equal(self.feature.process_vertices(self.df_1, 1), self.target_df_1)

        # test the calculation of vertex degree difference in the 2. time step ('df_2')
        target_df_2 = pd.DataFrame(data={'name': ['A', 'B', 'D'], 'VertexDegreeDifference': [1, -1, 2]},
                                   columns=['name', 'VertexDegreeDifference'])
        assert_frame_equal(self.feature.process_vertices(self.df_2, 1), target_df_2)
