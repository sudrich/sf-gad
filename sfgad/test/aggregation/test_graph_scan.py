from unittest import TestCase

import pandas as pd

from sfgad.aggregation import graph_scan as gs


class TestGraphScan(TestCase):
    def setUp(self):
        pass

    def test_empty_graph(self):
        df_edges = pd.DataFrame(
            columns=['TIMESTAMP', 'E_NAME', 'E_TYPE', 'SRC_NAME', 'SRC_TYPE', 'DST_NAME', 'DST_TYPE'])
        df_p = pd.DataFrame(columns=['name', 'p'])
        subgraph, score = gs.scan(df_edges, df_p, alpha_max=0.1, K=5)
        self.assertEqual(len(subgraph), 0)
        self.assertEqual(score, 0)

    def test_none_significant(self):
        df_edges = pd.DataFrame({'TIMESTAMP': ['2018-01-01 00:00:00', '2018-01-01 00:00:05', '2018-01-01 00:00:09'],
                                 'E_NAME': ['0_1', '0_2', '1_2'],
                                 'E_TYPE': ['0_1', '0_1', '1_1'],
                                 'SRC_NAME': ['0', '0', '1'],
                                 'SRC_TYPE': ['0', '0', '1'],
                                 'DST_NAME': ['1', '2', '2'],
                                 'DST_TYPE': ['0', '1', '1'],
                                 })
        df_p = pd.DataFrame({'name': ['0', '1', '2'],
                             'p_value': [0.5, 0.75, 1.0]
                             })
        subgraph, score = gs.scan(df_edges, df_p, alpha_max=0.2, K=5)
        self.assertEqual(len(subgraph), 0)

    def test_all_equally_significant(self):
        df_edges = pd.DataFrame({'TIMESTAMP': ['2018-01-01 00:00:00', '2018-01-01 00:00:05', '2018-01-01 00:00:09'],
                                 'E_NAME': ['0_1', '0_2', '1_2'],
                                 'E_TYPE': ['0_1', '0_1', '1_1'],
                                 'SRC_NAME': ['0', '0', '1'],
                                 'SRC_TYPE': ['0', '0', '1'],
                                 'DST_NAME': ['1', '2', '2'],
                                 'DST_TYPE': ['0', '1', '1'],
                                 })
        df_p = pd.DataFrame({'name': ['0', '1', '2'],
                             'p_value': [0.1, 0.1, 0.1]
                             })
        subgraph, score = gs.scan(df_edges, df_p, alpha_max=0.2, K=5, Z=5)
        self.assertEqual(len(subgraph), 3)

    def test_unknown_edge(self):
        df_edges = pd.DataFrame({'TIMESTAMP': ['2018-01-01 00:00:00', '2018-01-01 00:00:05', '2018-01-01 00:00:09'],
                                 'E_NAME': ['0_1', '0_2', '1_2'],
                                 'E_TYPE': ['0_1', '0_1', '1_1'],
                                 'SRC_NAME': ['0', '0', '1'],
                                 'SRC_TYPE': ['0', '0', '1'],
                                 'DST_NAME': ['1', '2', '2'],
                                 'DST_TYPE': ['0', '1', '1'],
                                 })
        df_p = pd.DataFrame({'name': ['0', '1'],
                             'p_value': [0.1, 0.5]
                             })
        with self.assertRaises(ValueError):
            subgraph, score = gs.scan(df_edges, df_p, alpha_max=0.2, K=5, Z=5)

    def test_distinct_anomaly(self):
        df_edges = pd.DataFrame({'TIMESTAMP': ['2018-01-01 00:00:00'] * 7,
                                 'E_NAME': ['0_1', '1_2', '2_3', '3_4', '4_5', '0_5', '1_4'],
                                 'E_TYPE': ['0_1', '1_0', '0_1', '1_0', '0_1', '0_1', '1_0'],
                                 'SRC_NAME': ['0', '1', '2', '3', '4', '0', '1'],
                                 'SRC_TYPE': ['0', '1', '0', '1', '0', '0', '1'],
                                 'DST_NAME': ['1', '2', '3', '4', '5', '5', '4'],
                                 'DST_TYPE': ['1', '0', '1', '0', '1', '1', '0'],
                                 })
        df_p = pd.DataFrame({'name': ['0', '1', '2', '3', '4', '5'],
                             'p_value': [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
                             })
        subgraph, score = gs.scan(df_edges, df_p, alpha_max=0.2, K=5, Z=5)
        self.assertEqual(set(subgraph), {'0', '1', '2'})
