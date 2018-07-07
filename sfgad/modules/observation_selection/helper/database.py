import pandas as pd

import mysql.connector
import gc

from mysql.connector import errorcode


class Database:
    """
    The format of the data-table should be: ['name', 'type', 'time_window', 'feature_1', ..., 'feature_n']
    """

    def __init__(self, user='root', password='root', host='localhost', database='sfgad', table_name='historic_data'):
        # close any old MySQL connections
        gc.collect()
        # create a new connection
        try:
            self.cnn = mysql.connector.connect(
                user=user,
                password=password,
                host=host,
                database=database
            )
            print("Connection to the database established!")

        except mysql.connector.Error as e:
            if e.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("The given username or password is wrong!")
            elif e.errno == errorcode.ER_BAD_DB_ERROR:
                print("The given database does not exist!")
            else:
                print(e)
        self.table_name = table_name
        self.cursor = self.cnn.cursor()

    def select_all(self):
        """
        Selects all rows in the database.
        :return a dataframe with all historic data.
        """
        return pd.read_sql('SELECT * FROM ' + self.table_name, con=self.cnn)


    def select_by_vertex_name(self, vertex_name):
        """
        Selects all rows in the database where name=vertex_name.
        :param vertex_name: The given vertex_name.
        :return a dataframe with all historic data of the given vertex.
        """
        return pd.read_sql('SELECT * FROM ' + self.table_name + ' WHERE name=%s', con=self.cnn, params=[vertex_name])

    def select_by_vertex_type(self, vertex_type):
        """
        Selects all rows in the database where type=vertex_type.
        :param vertex_type: The given vertex type.
        :return a dataframe with all historic data of vertices, which are of the given type.
        """
        return pd.read_sql('SELECT * FROM ' + self.table_name + ' WHERE type=%s', con=self.cnn, params=[vertex_type])


    def close_connection(self):
        """
        Closes the database connection.
        """
        self.cnn.close()


