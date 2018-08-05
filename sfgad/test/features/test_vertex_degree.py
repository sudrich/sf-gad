import pandas as pd

from unittest import TestCase
from pandas.util.testing import assert_frame_equal, assert_series_equal
from sfgad.modules.features.vertex_degree import VertexDegree


class TestVertexDegree(TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'TIMESTAMP': ['2018-01-01 00:00:00', '2018-01-01 00:00:01', '2018-01-01 00:00:05'],
                                'SRC_NAME': ['A', 'A', 'B'],
                                'SRC_TYPE': ['NODE', 'NODE', 'NODE'],
                                'DST_NAME': ['B', 'C', 'C'],
                                'DST_TYPE': ['NODE', 'NODE', 'NODE']})
        self.df['TIMESTAMP'] = pd.to_datetime(self.df['TIMESTAMP'])

        # the target output of the feature after 1. time step
        self.target_df = pd.DataFrame(data={'name': ['A', 'B', 'C'], 'VertexDegree': [2, 2, 2]},
                                      columns=['name', 'VertexDegree'])

        self.feature = VertexDegree()

    def test_init(self):
        self.assertEqual(self.feature.names, ['VertexDegree'])

    def test_result_df_shape(self):
        result_df = self.feature.process_vertices(self.df, 1)

        self.assertEqual(result_df.shape, self.target_df.shape)

    def test_result_df_columns(self):
        result_df_1 = self.feature.process_vertices(self.df, 1)

        self.assertEqual(result_df_1.columns.tolist(), ['name', 'VertexDegree'])

    def test_result_df_dtypes(self):
        result_df_1 = self.feature.process_vertices(self.df, 1)

        assert_series_equal(result_df_1.dtypes, self.target_df.dtypes)

    def test_result_df_values(self):
        result_df_1 = self.feature.process_vertices(self.df, 1)

        assert_series_equal(result_df_1['name'], self.target_df['name'])
        assert_series_equal(result_df_1['VertexDegree'], self.target_df['VertexDegree'])

    def test_overall_processing(self):
        # test the calculation of vertex degree on the example data frame 'df'
        assert_frame_equal(self.feature.process_vertices(self.df, 1), self.target_df)
