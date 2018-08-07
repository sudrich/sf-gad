import pandas as pd
import numpy as np

from .database import Database


class InMemoryDatabase(Database):
    """
    The format of the data-table should be: ['name', 'type', 'time_window', 'feature_1', ..., 'feature_n']
    """

    def __init__(self, feature_names):
        # create a new dataframe
        self.database = pd.DataFrame(
            columns=['name', 'type', 'time_window'] + feature_names)

        self.feature_names = feature_names

    def insert_record(self, vertex_name, vertex_type, time_window, feature_values):
        """
        Inserts a record to the database.
        :param vertex_name: The name of the vertex.
        :param vertex_type: The type of the vertex.
        :param time_window: The time step.
        :param feature_values: The corresponding feature values in a list.
        """
        self.database = self.database.append(
            pd.DataFrame(
                data=np.array([[vertex_name, vertex_type, time_window] + feature_values]),
                columns=['name', 'type', 'time_window'] + self.feature_names), ignore_index=True)

    def select_all(self):
        """
        Selects all rows in the database.
        :return a dataframe with all historic data.
        """
        return self.database

    def select_by_vertex_name(self, vertex_name):
        """
        Selects all rows in the database where name=vertex_name.
        :param vertex_name: The given vertex_name.
        :return a dataframe with all historic data of the given vertex.
        """
        return self.database[self.database['name'] == vertex_name]

    def select_by_vertex_type(self, vertex_type):
        """
        Selects all rows in the database where type=vertex_type.
        :param vertex_type: The given vertex type.
        :return a dataframe with all historic data of vertices, which are of the given type.
        """
        return self.database[self.database['type'] == vertex_type]

    def select_by_time_step(self, time_window):
        """
        Selects all rows in the database where time_window=time_window.
        :param time_window: The given time window.
        :return a dataframe with all historic data of vertices, which have the same time window entry.
        """
        return self.database[self.database['time_window'] == time_window]
