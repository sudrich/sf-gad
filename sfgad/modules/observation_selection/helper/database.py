import abc


class Database(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def insert_record(self, vertex_name, vertex_type, time_window, feature_values):
        """
        Inserts a record to the database.
        :param vertex_name: The name of the vertex.
        :param vertex_type: The type of the vertex.
        :param time_window: The time step.
        :param feature_values: The corresponding feature values in a list.
        """

    @abc.abstractmethod
    def insert_records(self, records):
        """
        Inserts a record to the database.
        :param records: DataFrame where each row is a record with meta information about the vertex and its features.
        """

    @abc.abstractmethod
    def select_all(self):
        """
        Selects all rows in the database.
        :return a dataframe with all historic data.
        """

    @abc.abstractmethod
    def select_by_vertex_name(self, vertex_name):
        """
        Selects all rows in the database where name=vertex_name.
        :param vertex_name: The given vertex_name.
        :return a dataframe with all historic data of the given vertex.
        """

    @abc.abstractmethod
    def select_by_vertex_type(self, vertex_type):
        """
        Selects all rows in the database where type=vertex_type.
        :param vertex_type: The given vertex type.
        :return a dataframe with all historic data of vertices, which are of the given type.
        """

    @abc.abstractmethod
    def select_by_time_step(self, time_window):
        """
        Selects all rows in the database where time_window=time_window.
        :param time_window: The given time window.
        :return a dataframe with all historic data of vertices, which have the same time window entry.
        """
