from unittest import TestCase

import pandas as pd
from pandas.util.testing import assert_frame_equal, assert_series_equal

from sfgad.modules.features.vertex_degree_by_type import VertexDegreeByType


class TestVertexDegreeByType(TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'TIMESTAMP': ['2018-01-01 00:00:00', '2018-01-01 00:00:01', '2018-01-01 00:00:05'],
                                'E_TYPE': ['LIKE', 'LIKE', 'MESSAGE'],
                                'SRC_NAME': ['A', 'A', 'B'],
                                'SRC_TYPE': ['NODE', 'NODE', 'NODE'],
                                'DST_NAME': ['B', 'C', 'C'],
                                'DST_TYPE': ['NODE', 'NODE', 'NODE']})
        self.df['TIMESTAMP'] = pd.to_datetime(self.df['TIMESTAMP'])

        # the target output of the feature after 1. time step
        self.target_df = pd.DataFrame(data={'name': ['A', 'B', 'C'], 'VertexDegreeByLIKE': [2, 1, 1],
                                            'VertexDegreeByMESSAGE': [0, 1, 1], 'VertexDegreeByFRIENDSHIP': [0, 0, 0]},
                                      columns=['name', 'VertexDegreeByLIKE', 'VertexDegreeByMESSAGE',
                                               'VertexDegreeByFRIENDSHIP'])

        self.feature = VertexDegreeByType(edge_types=['LIKE', 'MESSAGE', 'FRIENDSHIP'])

    def test_init(self):
        self.assertEqual(self.feature.names, ['VertexDegreeByLIKE', 'VertexDegreeByMESSAGE',
                                              'VertexDegreeByFRIENDSHIP'])

    def test_result_df_shape(self):
        result_df = self.feature.process_vertices(self.df, 1)

        self.assertEqual(result_df.shape, self.target_df.shape)

    def test_result_df_columns(self):
        result_df_1 = self.feature.process_vertices(self.df, 1)

        self.assertEqual(result_df_1.columns.tolist(), ['name', 'VertexDegreeByLIKE', 'VertexDegreeByMESSAGE',
                                                        'VertexDegreeByFRIENDSHIP'])

    def test_result_df_dtypes(self):
        result_df_1 = self.feature.process_vertices(self.df, 1)

        assert_series_equal(result_df_1.dtypes, self.target_df.dtypes)

    def test_result_df_values(self):
        result_df_1 = self.feature.process_vertices(self.df, 1)

        assert_series_equal(result_df_1['name'], self.target_df['name'])
        assert_series_equal(result_df_1['VertexDegreeByLIKE'], self.target_df['VertexDegreeByLIKE'])
        assert_series_equal(result_df_1['VertexDegreeByMESSAGE'], self.target_df['VertexDegreeByMESSAGE'])
        assert_series_equal(result_df_1['VertexDegreeByFRIENDSHIP'], self.target_df['VertexDegreeByFRIENDSHIP'])

    def test_overall_processing(self):
        # test the calculation of vertex degree by type on the example data frame 'df'
        assert_frame_equal(self.feature.process_vertices(self.df, 1), self.target_df)
