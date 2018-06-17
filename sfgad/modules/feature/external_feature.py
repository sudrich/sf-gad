import pandas as pd
import numpy as np

from .feature import Feature
from itertools import dropwhile


class ExternalFeature(Feature):
    """
    The external feature maps a given node to a measured value based on a given dictionary.
    The idea behind this feature is, that nodes could be monitoring stations and there exists a mapping function with
    nodes as keys and lists of measured data (time, value) as values.
    """

    def __init__(self, values_dict):
        self.names = ['ExternalFeature']

        # node -> list of tuples (time, value) sorted by time
        self.values_dict = values_dict

    def process_vertices(self, df_edges, n_jobs, update_activity=True):
        """
        Iterates over the current data frame and maps each node to its measured value at the current timestamp, if there
        is an entry in the given dictionary.
        :param df_edges: The data frame to process.
        :param n_jobs: The number of cores that are supported for multiprocessing.
        :param update_activity: True, if the feature should consider the new edges for future computations (if needed),
            false otherwise.
        :return a data frame with the columns ['name', 'ExternalFeature'] and the measured values at the current
            timestamp for all vertices in the given df_edges.
        """

        # read out the active nodes and the current time
        active_nodes = list(pd.unique(df_edges[['SRC_NAME', 'DST_NAME']].values.ravel()))
        current_time = max(df_edges["TIMESTAMP"])

        # map each node to its measured value
        external_values = []

        for node in active_nodes:
            # check if there is an entry in the given dictionary for the current node
            if node in self.values_dict:
                node_values = self.values_dict[node]
                node_values = list(dropwhile(lambda record: record[0] < current_time, node_values))

                if len(node_values) > 0:
                    # read out the latest measured value after the current timestamp and add it to the results
                    _, latest_value = node_values[0]
                    external_values.append(latest_value)
                else:
                    # append NaN if there are no more measured values at the current time
                    external_values.append(np.nan)

                # update the dictionary
                self.values_dict[node] = node_values
            else:
                # append NaN if the node is not known
                external_values.append(np.nan)

        # create the result dataframe
        result_df = pd.DataFrame(data={'name': active_nodes, 'ExternalFeature': external_values},
                                 columns=['name', 'ExternalFeature'])

        return result_df

    def compute(self, node_name, t):
        # Not needed here, since this feature is to simple for multiprocessing
        pass