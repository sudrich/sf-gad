from .observation_selection import ObservationSelection


class AlternativeSelection(ObservationSelection):
    """
    This observation selection includes 2 selection rules. If the first selection rule fails to provide enough
    observations, this selection falls back on a secondary rule and returns only the observations of the second
    selection rule.
    """

    def __init__(self, first_rule, second_rule, threshold, limit=None):
        # check whether limit is given and set self.limit if this is the case
        if not isinstance(threshold, int) or not threshold >= 1:
            raise ValueError("The given parameter 'threshold' should be an integer and >= 1!")

        # check whether limit is given and whether its greater than threshold
        if limit is not None:
            if not isinstance(limit, int) or not limit >= 1:
                raise ValueError("The given parameter 'limit' should be an integer and >= 1!")
            if limit < threshold:
                raise ValueError("The given parameter 'limit' should be greater than the given parameter 'threshold'")

        self.first_rule = first_rule
        self.second_rule = second_rule
        self.threshold = threshold
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
        results = self.first_rule.gather(vertex_name, vertex_type, database)

        # get the results from the second rule, if the first rule fails to provide enough observations
        if results.shape[0] < self.threshold:
            results_second_rule = self.second_rule.gather(vertex_name, vertex_type, database)
            results = results_second_rule

        return results.head(self.limit)