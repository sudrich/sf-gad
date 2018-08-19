from .observation_selection import ObservationSelection


class CurrentAllSelection(ObservationSelection):
    """
    This observation selection gathers all current observations of all vertices.
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

        result = database.select_by_time_step(current_time_window).reset_index(drop=True)

        # filter the given vertex from the results
        result = result[result['name'] != vertex_name].reset_index(drop=True)

        if self.limit is not None:
            result = result.head(self.limit)

        return result
