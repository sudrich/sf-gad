import pandas as pd

import mysql.connector
import gc

from mysql.connector import errorcode


class Database:
    """
    The format of the data-table should be: ['name', 'type', 'time_window', 'feature_1', ..., 'feature_n']
    The primary key is the tiple ('name', 'type', 'time_window')
    """
    def __init__(self, user, password, host, database, table_name, feature_names):
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

        self.cursor = self.cnn.cursor()
        self.table_name = table_name
        self.feature_names = feature_names

        # create a new table with given table name (but delete an existing table first)
        self.cursor.execute('DROP TABLE IF EXISTS ' + self.table_name)

        # create the sql query for table creation
        sql_query = 'CREATE TABLE ' + self.table_name + '(' \
            'name VARCHAR(255) NOT NULL, ' \
            'type VARCHAR(255) NOT NULL, ' \
            'time_window INT NOT NULL, '
        for name in feature_names:
            sql_query += name + ' FLOAT(16,4), '
        sql_query += 'PRIMARY KEY (name, type, time_window))'

        self.cursor.execute(sql_query)

    def insert_record(self, vertex_name, vertex_type, time_window, feature_values):
        try:
            values = [vertex_name, vertex_type, time_window] + feature_values

            sql_query = 'INSERT INTO ' + self.table_name + ' VALUES ('
            for i in range(len(values)):
                if i + 1 != len(values):
                    sql_query += '%s, '
                else:
                    sql_query += '%s)'

            self.cursor.execute(sql_query, values)

            self.cnn.commit()
        except:
            self.cnn.rollback()
            raise ValueError("Record insertion failed!")

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

    def select_by_time_step(self, time_window):
        """
        Selects all rows in the database where time_window=time_window.
        :param vertex_type: The given vertex type.
        :return a dataframe with all historic data of vertices, which are of the given type.
        """
        return pd.read_sql('SELECT * FROM ' + self.table_name + ' WHERE time_window=%s', con=self.cnn,
                           params=[time_window])


    def close_connection(self):
        """
        Closes the database connection.
        """
        self.cursor.close()
        self.cnn.close()


