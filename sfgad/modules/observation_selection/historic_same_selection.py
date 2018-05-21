from .observation_selection import ObservationSelection


class HistoricSameSelection(ObservationSelection):
    """
    This observation selection gathers all historic observations of the requested vertex.
    The results can be limited by providing a limit parameter.
    """

    def __init__(self, limit=None):
        # check whether limit is given and set self.limit if thats the case
        if limit is not None:
            if not isinstance(limit, int) or not limit >= 1:
                raise ValueError("The given parameter 'limit' should be an integer and >= 1!")

        self.limit = limit

    def gather(self, vertex_name, database):
        """
        Takes a vertex and a reference to the database.
        Returns a dataframe of all the relevant vertices that are needed for calculating p_value of the vertex.
        :param vertex_name: The name of a vertex
        :param database: The reference to the Database
        :return: Dataframe of the relevant entries in the database
        """
        # TODO: implement the function
