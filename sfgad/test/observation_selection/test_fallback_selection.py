from unittest import TestCase

import pandas as pd
from pandas.util.testing import assert_frame_equal

from sfgad.modules.observation_selection.fallback_selection import FallbackSelection
# from sfgad.modules.observation_selection.helper.external_sql_database import ExternalSQLDatabase
from sfgad.modules.observation_selection.helper.in_memory_database import InMemoryDatabase
from sfgad.modules.observation_selection.historic_same_selection import HistoricSameSelection
from sfgad.modules.observation_selection.historic_similar_selection import HistoricSimilarSelection


class TestFallbackSelection(TestCase):
    def setUp(self):
        # establish a connection to the database
        # self.db = ExternalSQLDatabase(user='root', password='root', host='localhost', database='sfgad',
        #                              table_name='historic_data', feature_names=['feature_A', 'feature_B'])
        self.db = InMemoryDatabase(feature_names=['feature_A', 'feature_B'])
        self.db.insert_record('Vertex_A', 'PERSON', 1, [24, 42])
        self.db.insert_record('Vertex_B', 'PERSON', 1, [124, 142])
        self.db.insert_record('Vertex_C', 'PICTURE', 1, [224, 242])
        self.db.insert_record('Vertex_D', 'POST', 1, [324, 342])
        self.db.insert_record('Vertex_A', 'PERSON', 2, [12, 24])

        # init a selection rule
        self.sel_rule = FallbackSelection(first_rule=HistoricSameSelection(),
                                          second_rule=HistoricSimilarSelection(),
                                          threshold=3)

    def tearDown(self):
        # close db connection
        self.db.close_connection()

    def test_gather(self):
        target_df = pd.DataFrame(data={'name': ['Vertex_A', 'Vertex_B', 'Vertex_A'],
                                       'type': ['PERSON', 'PERSON', 'PERSON'],
                                       'time_window': [2, 1, 1], 'feature_A': [12.0, 124.0, 24.0],
                                       'feature_B': [24.0, 142.0, 42.0]},
                                 columns=['name', 'type', 'time_window', 'feature_A', 'feature_B'])

        assert_frame_equal(self.sel_rule.gather('Vertex_B', 'PERSON', None, self.db), target_df)

    def test_gather_with_limit(self):
        target_df = pd.DataFrame(data={'name': ['Vertex_A'], 'type': ['PERSON'],
                                       'time_window': [2], 'feature_A': [12.0],
                                       'feature_B': [24.0]},
                                 columns=['name', 'type', 'time_window', 'feature_A', 'feature_B'])

        self.sel_rule = FallbackSelection(first_rule=HistoricSameSelection(),
                                          second_rule=HistoricSimilarSelection(),
                                          threshold=1, limit=1)
        assert_frame_equal(self.sel_rule.gather('Vertex_A', 'PERSON', None, self.db), target_df)