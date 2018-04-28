import pandas as pd

from unittest import TestCase
from pandas.util.testing import assert_frame_equal
from sfgad.modules.feature.vertex_degree_difference import VertexDegreeDifference


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

        self.feature = VertexDegreeDifference()

    def test_init(self):
        self.assertEqual(self.feature.names, ['VertexDegreeDifference'])

    def test_process_vertices(self):
        # test the calculation of vertex degree difference in the 1. time step ('df_1')
        result_df_1 = pd.DataFrame(data={'name': ['A', 'B', 'C'], 'VertexDegreeDifference': [2, 2, 2]},
                                   columns=['name', 'VertexDegreeDifference'])
        assert_frame_equal(self.feature.process_vertices(self.df_1, 1), result_df_1)

        # test the calculation of vertex degree difference in the 2. time step ('df_1')
        result_df_2 = pd.DataFrame(data={'name': ['A', 'B', 'D'], 'VertexDegreeDifference': [1, -1, 2]},
                                   columns=['name', 'VertexDegreeDifference'])
        assert_frame_equal(self.feature.process_vertices(self.df_2, 1), result_df_2)