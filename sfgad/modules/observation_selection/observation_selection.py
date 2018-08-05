import abc


class ObservationSelection:

    @abc.abstractmethod
    def gather(self, vertex_name, vertex_type, current_time_step, database):
        """
        Takes a vertex and a reference to the database.
        Returns a dataframe of all the relevant vertices that are needed for calculating p_value of the vertex.
        :param vertex_name: The name of the vertex
        :param vertex_type: The type of the vertex
        :param current_time_step: The current time step
        :param database: The reference to the Database
        :return: Dataframe of the relevant entries in the database
        """