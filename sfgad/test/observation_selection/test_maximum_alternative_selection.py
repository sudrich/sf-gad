import pandas as pd

from unittest import TestCase
from pandas.util.testing import assert_frame_equal

from sfgad.modules.observation_selection.helper.database import Database
from sfgad.modules.observation_selection.maximum_alternative_selection import MaximumAlternativeSelection
from sfgad.modules.observation_selection.historic_same_selection import HistoricSameSelection
from sfgad.modules.observation_selection.historic_similar_selection import HistoricSimilarSelection


class TestMaximumAlternativeSelection(TestCase):

    def setUp(self):
        # establish a connection to the database
        self.db = Database(user='root', password='root', host='localhost', database='sfgad', table_name='historic_data',
                           feature_names=['feature_A', 'feature_B'])
        self.db.insert_record('Vertex_A', 'PERSON', 1, [24, 42])
        self.db.insert_record('Vertex_B', 'PERSON', 1, [124, 142])
        self.db.insert_record('Vertex_C', 'PICTURE', 1, [224, 242])
        self.db.insert_record('Vertex_D', 'POST', 1, [324, 342])
        self.db.insert_record('Vertex_A', 'PERSON', 2, [12, 24])

        # init a selection rule
        self.sel_rule = MaximumAlternativeSelection(first_rule=HistoricSameSelection(),
                                            second_rule=HistoricSimilarSelection())

    def tearDown(self):
        # close db connection
        self.db.close_connection()

    def test_gather(self):
        target_df = pd.DataFrame(data={'name': ['Vertex_A', 'Vertex_A', 'Vertex_B'],
                                       'type': ['PERSON', 'PERSON', 'PERSON'],
                                       'time_window': [2, 1, 1], 'feature_A': [12.0, 24.0, 124.0],
                                       'feature_B': [24.0, 42.0, 142.0]},
                                 columns=['name', 'type', 'time_window', 'feature_A', 'feature_B'])

        assert_frame_equal(self.sel_rule.gather('Vertex_A', 'PERSON', self.db), target_df)

    def test_gather_with_limit(self):
        target_df = pd.DataFrame(data={'name': ['Vertex_A'],
                                       'type': ['PERSON'],
                                       'time_window': [2], 'feature_A': [12.0],
                                       'feature_B': [24.0]},
                                 columns=['name', 'type', 'time_window', 'feature_A', 'feature_B'])

        self.sel_rule = MaximumAlternativeSelection(first_rule=HistoricSameSelection(),
                                            second_rule=HistoricSimilarSelection(),
                                            limit=1)

        assert_frame_equal(self.sel_rule.gather('Vertex_A', 'PERSON', self.db), target_df)
