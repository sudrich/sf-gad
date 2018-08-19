import abc


class Feature(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def process_vertices(self, df_edges, n_jobs, update_activity=True):
        """
        Generates vertices list from given edge frame.
        Splits the given vertices list for multiprocessoring.
        Calls the compute function for every vertex and combines the results.
        :param df_edges: The edge_frame of the current window.
        :param n_jobs: The number of cores that are supported for multiprocessing.
        :param update_activity: True, if the feature should consider the new edges for future computations (if needed),
            false otherwise
        :return: a dataframe containing the column name and then additional columns for every every feature.
        """

    @abc.abstractmethod
    def compute(self, node_name, t):
        """
        Takes a vertex and computes the new feature value.
        :param node_name: The name of the node, whose feature should be computed.
        :param t: The time point at which the new feature value should be computed.
        :return: the new feature value.
        """

    @abc.abstractmethod
    def reset(self):
        """
        Resets the feature to its initial state.
        This is especially relevant for dynamic features that preserve an internal state.
        """
