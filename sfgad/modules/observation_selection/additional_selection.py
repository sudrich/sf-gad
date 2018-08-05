from .observation_selection import ObservationSelection


class AdditionalSelection(ObservationSelection):
    """
    This observation selection includes 2 selection rules. This selection apply both rules and returns the combination
    of the results.
    """

    def __init__(self, first_rule, second_rule, limit=None):

        # check whether limit is given and whether its greater than threshold
        if limit is not None:
            if not isinstance(limit, int) or not limit >= 1:
                raise ValueError("The given parameter 'limit' should be an integer and >= 1!")

        self.first_rule = first_rule
        self.second_rule = second_rule
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
        result_first_rule = self.first_rule.gather(vertex_name, vertex_type, current_time_step, database)

        # get the results from the second rule
        result_second_rule = self.second_rule.gather(vertex_name, vertex_type, current_time_step, database)

        # combine the results and drop duplicates
        result = result_first_rule.append(result_second_rule, ignore_index=True)
        result.drop_duplicates(inplace=True)

        # sort the records by time_window descending AND reset index
        result = result.sort_values(['time_window'], ascending=False).reset_index(drop=True)

        if self.limit is not None:
            result = result.head(self.limit)

        return result
