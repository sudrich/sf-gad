import pandas as pd

from unittest import TestCase
from pandas.util.testing import assert_frame_equal

from sfgad.modules.observation_selection.helper.database import Database
from sfgad.modules.observation_selection.historic_all_selection import HistoricAllSelection


class TestHistoricAllSelection(TestCase):

    def setUp(self):
        # establish a connection to the database
        self.db = Database(user='root', password='root', host='localhost', database='sfgad', table_name='historic_data',
                           feature_names=['feature_A', 'feature_B'])
        self.db.insert_record('Vertex_A', 'PERSON', 1, [24, 42])
        self.db.insert_record('Vertex_B', 'PERSON', 1, [124, 142])
        self.db.insert_record('Vertex_C', 'PICTURE', 1, [224, 242])
        self.db.insert_record('Vertex_D', 'POST', 1, [324, 342])

        # init a selection rule
        self.sel_rule = HistoricAllSelection()

    def tearDown(self):
        # close db connection
        self.db.close_connection()

    def test_gather(self):
        target_df = pd.DataFrame(data={'name': ['Vertex_A', 'Vertex_B', 'Vertex_C', 'Vertex_D'],
                                       'type': ['PERSON', 'PERSON', 'PICTURE', 'POST'],
                                       'time_window': [1, 1, 1, 1], 'feature_A': [24.0, 124.0, 224.0, 324.0],
                                       'feature_B': [42.0, 142.0, 242.0, 342.0]},
                                 columns=['name', 'type', 'time_window', 'feature_A', 'feature_B'])

        assert_frame_equal(self.sel_rule.gather(None, None, None, self.db), target_df)

    def test_gather_with_limit(self):
        target_df = pd.DataFrame(data={'name': ['Vertex_A'], 'type': ['PERSON'],
                                       'time_window': [1], 'feature_A': [24.0],
                                       'feature_B': [42.0]},
                                 columns=['name', 'type', 'time_window', 'feature_A', 'feature_B'])

        self.sel_rule = HistoricAllSelection(limit=1)
        assert_frame_equal(self.sel_rule.gather('Vertex_A', 'PERSON', None, self.db), target_df)
