from .observation_selection import ObservationSelection


class HistoricAgeSimilarSelection(ObservationSelection):
    """
    This observation selection gathers all historic observations of all vertices with the same age and same type.
    Age is defined as the difference between the current time window and the first occurrence of an observation.
    The results can be limited by providing a limit parameter.
    """

    def __init__(self, limit=None):
        # check whether limit is given and set self.limit if thats the case
        if limit is not None:
            if not isinstance(limit, int) or not limit >= 1:
                raise ValueError("The given parameter 'limit' should be an integer and >= 1!")

        self.limit = limit

    def gather(self, vertex_name, vertex_type, current_time_window, database):
        """
        Takes a vertex and a reference to the database.
        Returns a dataframe of all the relevant vertices that are needed for calculating p_value of the vertex.
        :param vertex_name: The name of the vertex
        :param vertex_type: The type of the vertex
        :param current_time_window: The current time step
        :param database: The reference to the Database
        :return: Dataframe of the relevant entries in the database
        """

        # determine vertices with the same age
        relevant_vertices = database.get_vertices_same_age(vertex_name, vertex_type)

        # add all historic observations of the given vertex
        result = database.select_by_vertex_name(vertex_name)

        # add all historic observations of all vertices with the same age
        for other_name, other_type in relevant_vertices:
            if other_type == vertex_type:
                result = result.append(database.select_by_vertex_name(other_name))

        # sort the records by time_window descending AND reset index
        result = result.sort_values(['time_window'], ascending=False).reset_index(drop=True)

        if self.limit is not None:
            result = result.head(self.limit)

        return result