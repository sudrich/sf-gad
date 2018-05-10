import abc


class Weighting:
    @abc.abstractmethod
    def compute(self, reference_feature_values, time_window):
        """
        Takes a vertex and a reference to the database.
        Returns a dataframe of all the relevant vertices that are needed for calculating p_value of the vertex
        :param reference_feature_values: List of tuples of windows and the corresponding feature values
        :param time_window: The current time window
        :return: Dataframe with the columns (window, weight)
        """