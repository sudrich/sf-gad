from .observation_selection import ObservationSelection


class MaximumAlternativeSelection(ObservationSelection):
    """
    This observation selection includes 2 selection rules. This selection apply both rules and returns the result with
    more observations.
    """

    def __init__(self, first_rule, second_rule, limit=None):

        # check whether limit is given and whether its greater than threshold
        if limit is not None:
            if not isinstance(limit, int) or not limit >= 1:
                raise ValueError("The given parameter 'limit' should be an integer and >= 1!")

        self.first_rule = first_rule
        self.second_rule = second_rule
        self.limit = limit

    def gather(self, vertex_name, vertex_type, database):
        """
        Takes a vertex and a reference to the database.
        Returns a dataframe of all the relevant vertices that are needed for calculating p_value of the vertex.
        :param vertex_name: The name of a vertex
        :param vertex_type: The type of the vertex
        :param database: The reference to the Database
        :return: Dataframe of the relevant entries in the database
        """
        # get the results from the first rule
        result_first_rule = self.first_rule.gather(vertex_name, vertex_type, database)

        # get the results from the second rule
        result_second_rule = self.second_rule.gather(vertex_name, vertex_type, database)

        # decide which rule yielded more observations
        if result_second_rule.shape[0] > result_first_rule.shape[0]:
            result = result_second_rule
        else:
            result = result_first_rule

        if self.limit is not None:
            result = result.head(self.limit)

        return result
