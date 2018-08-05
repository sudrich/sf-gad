from .observation_selection import ObservationSelection


class FallbackSelection(ObservationSelection):
    """
    This observation selection includes 2 selection rules. If the first selection rule fails to provide enough
    observations, this selection falls back on a secondary rule and returns combined observations of the two
    selection rules.
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

        # get the results from the first rule
        result = self.first_rule.gather(vertex_name, vertex_type, current_time_step, database)

        # get and append the results from the second rule if the first rule fails to provide enough observations
        if result.shape[0] < self.threshold:
            result_second_rule = self.second_rule.gather(vertex_name, vertex_type, current_time_step, database)
            result = result.append(result_second_rule, ignore_index=True)

            # drop duplicate results
            result.drop_duplicates(inplace=True)

        if self.limit is not None:
            result = result.head(self.limit)

        # sort the records by time_window descending AND reset index
        result = result.sort_values(['time_window'], ascending=False).reset_index(drop=True)

        return result
