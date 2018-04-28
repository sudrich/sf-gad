import pandas as pd

from unittest import TestCase
from pandas.util.testing import assert_frame_equal
from sfgad.modules.feature.vertex_degree_by_type import VertexDegreeByType


class TestVertexDegreeByType(TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'TIMESTAMP': ['2018-01-01 00:00:00', '2018-01-01 00:00:01', '2018-01-01 00:00:05'],
                                'E_TYPE': ['LIKE', 'LIKE', 'MESSAGE'],
                                'SRC_NAME': ['A', 'A', 'B'],
                                'SRC_TYPE': ['NODE', 'NODE', 'NODE'],
                                'DST_NAME': ['B', 'C', 'C'],
                                'DST_TYPE': ['NODE', 'NODE', 'NODE']})
        self.df['TIMESTAMP'] = pd.to_datetime(self.df['TIMESTAMP'])

        self.feature = VertexDegreeByType(edge_types=['LIKE', 'MESSAGE', 'FRIENDSHIP'])

    def test_init(self):
        self.assertEqual(self.feature.names, ['VertexDegreeByLIKE', 'VertexDegreeByMESSAGE',
                                              'VertexDegreeByFRIENDSHIP'])

    def test_process_vertices(self):
        # test the calculation of vertex degree by type on the example data frame 'df'
        result_df = pd.DataFrame(data={'name': ['A', 'B', 'C'], 'VertexDegreeByLIKE': [2, 1, 1],
                                       'VertexDegreeByMESSAGE': [0, 1, 1], 'VertexDegreeByFRIENDSHIP': [0, 0, 0]},
                                 columns=['name', 'VertexDegreeByLIKE', 'VertexDegreeByMESSAGE',
                                          'VertexDegreeByFRIENDSHIP'])
        assert_frame_equal(self.feature.process_vertices(self.df, 1), result_df)
