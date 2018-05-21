import abc


class ObservationGatherer:

    @abc.abstractmethod
    def gather(self, vertex_name, database):
        """
        Takes a vertex and a reference to the database.
        Returns a dataframe of all the relevant vertices that are needed for calculating p_value of the vertex.
        :param vertex_name: The name of a vertex
        :param database: The reference to the Database
        :return: Dataframe of the relevant entries in the database
        """